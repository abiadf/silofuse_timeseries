import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
from typing import List, Tuple
import RNN_params as disc_param

def extract_time_info(timeseries_data: np.ndarray) -> Tuple[np.ndarray, int]:
    max_seq_len = timeseries_data.shape[1]  # go for dim 2
    time        = np.arange(max_seq_len)  # Create time indices from 0 to max_seq_len
    return time, max_seq_len

def batch_generator(timeseries_data, time_stamps, batch_size, max_seq_len):
    indices = np.random.permutation(len(timeseries_data))[:batch_size]
    X_mb    = np.zeros((batch_size, max_seq_len, timeseries_data.shape[2]), dtype=np.float32)
    T_mb    = np.zeros(batch_size, dtype=np.int32)

    for i, idx in enumerate(indices):
        seq     = timeseries_data[idx]
        seq_len = min(len(seq), max_seq_len)
        X_mb[i, :seq_len, :] = seq[:seq_len]
        T_mb[i] = seq_len  # Store actual sequence length

    return torch.tensor(X_mb, dtype=torch.float32), torch.tensor(T_mb, dtype=torch.int32)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)
        self._loop_counter = 0  # initialize counter

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        timestamps = timestamps.to(x.device)

        mask = (timestamps.unsqueeze(-1) > torch.arange(x.shape[1], device=x.device)).float()
        mask = mask.unsqueeze(-1)  # Now: (batch_size, seq_len, 1)
        mask = mask.expand(-1, -1, x.shape[2])  # Correctly expands to (batch_size, seq_len, input_dim)

        assert mask.shape == x.shape, f"Shape mismatch: x={x.shape}, mask={mask.shape}"

        x_masked = x * mask
        _, h_n   = self.gru(x_masked)
        logits   = self.fc(h_n.squeeze(0))

        if self._loop_counter % 10 == 0:
            pass
            # print(f"Logits before sigmoid: {logits}")
            # print(f"x.shape: {x.shape}, x.dtype: {x.dtype}")
            # print(f"timestamps.shape: {timestamps.shape}, timestamps.dtype: {timestamps.dtype}")
            # print(f"mask.shape: {mask.shape}, mask.dtype: {mask.dtype}")
            # print(f"x_masked.shape: {x_masked.shape}, x_masked.dtype: {x_masked.dtype}")
            # print(f"h_n.shape: {h_n.shape}, h_n.dtype: {h_n.dtype}")
            # print(f"logits.shape: {logits.shape}, logits.dtype: {logits.dtype}")
        self._loop_counter += 1

        return logits, torch.sigmoid(logits)

def compute_discriminator_loss(y_logit_real: torch.Tensor, y_logit_fake: torch.Tensor) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss()
    loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
    loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
    return loss_real + loss_fake

def train_discriminator(discriminator: Discriminator, optimizer: optim.Optimizer,
                        train_x: np.ndarray, train_t: np.ndarray, train_x_hat: np.ndarray,
                        train_t_hat: np.ndarray, batch_size: int, iterations: int,
                        max_seq_len: int) -> List[float]:
    losses = []

    # for i in tqdm(range(iterations), desc='Training'):
    for _ in range(iterations):
        X_mb, T_mb         = batch_generator(train_x, train_t, batch_size, max_seq_len)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size, max_seq_len)

        device   = next(discriminator.parameters()).device
        X_mb     = X_mb.to(device)
        T_mb     = T_mb.to(device)
        X_hat_mb = X_hat_mb.to(device)
        T_hat_mb = T_hat_mb.to(device)

        y_logit_real, _ = discriminator(X_mb, T_mb)
        y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)

        discriminator_loss = compute_discriminator_loss(y_logit_real, y_logit_fake)

        optimizer.zero_grad()
        discriminator_loss.backward()
        optimizer.step()
        losses.append(discriminator_loss.item())

    return losses

def compute_binary_accuracy(y_label_final: np.ndarray, y_pred_final: np.ndarray) -> float:
    "Also called 'discriminative score'"

    accuracy = accuracy_score(y_label_final, (y_pred_final > 0.5))
    return np.abs(accuracy - 0.5)

def _split_single_sample_into_subsamples(original_data, num_splits):
    """This function splits a dataset of shape (z,x,y)=(1,x,y) along the rows (x axis), for a desired # of samples.
    Say data has shape (1, 100, 4) and we want 5 samples, the output is then (5, 20, 4)
    original_data: original dataset, usually np, of shape (1, x, y)
    num_splits: # of desired samples we want"""

    if original_data.shape[0] == 1 and original_data.shape[1] > 1:
        x = original_data.shape[1]
        trim_len = int((x // num_splits) * num_splits)
        trimmed = original_data[:, :trim_len, :]
        split_data = np.split(trimmed[0], num_splits, axis=0)  # now guaranteed equal shapes
        return np.stack(split_data)
    return original_data

def discriminative_score_metrics(original_data: np.ndarray, generated_data: np.ndarray) -> float:
    """For this we assume a 80%-20% train-test split"""

    compute_device                = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples, n_rows, n_features = np.asarray(original_data).shape
    
    largest_dim = np.max([n_samples, n_rows, n_features])
    hidden_dim  = int(n_features * disc_param.hidden_dim_fraction)

    if original_data.shape[0] == 1 and generated_data.shape[0] == 1:  # if our dataset has 1 sample
        num_splits = 1 // (1 - disc_param.train_set_size)  # =5. for a 80-20 split, we /5
        print(f"Data has 1 sample, will split to {num_splits} samples")
        original_data  = _split_single_sample_into_subsamples(original_data, num_splits)
        generated_data = _split_single_sample_into_subsamples(generated_data, num_splits)

    # if dataset has shape (z, x, y), for 80-20% train-test split, the output will be (0.8z, x, y)+(0.2z, x, y), so each page is full in train or test set
    train_x, test_x, train_t, test_t = train_test_split(original_data, np.arange(len(original_data)),
                                                        train_size = disc_param.train_set_size, random_state = disc_param.random_state)
    train_x_hat, test_x_hat, train_t_hat, test_t_hat = train_test_split(generated_data, np.arange(len(generated_data)),
                                                                        train_size = disc_param.train_set_size, random_state = disc_param.random_state)
    max_seq_len   = max(train_x.shape[1], train_x_hat.shape[1])

    discriminator = Discriminator(n_features, hidden_dim).to(compute_device)
    optimizer     = optim.Adam(discriminator.parameters(), lr = disc_param.learning_rate)
    train_discriminator(discriminator, optimizer, train_x, train_t, train_x_hat, train_t_hat,
                        disc_param.batch_size, disc_param.iterations, max_seq_len)

    with torch.no_grad():
        # since it is done on test set, both y_logit_* have shape (0.2z, 1)
        y_logit_real, _ = discriminator(torch.tensor(test_x, dtype=torch.float32).to(compute_device),
                                        torch.tensor(test_t, dtype=torch.int32).to(compute_device))
        y_logit_fake, _ = discriminator(torch.tensor(test_x_hat, dtype=torch.float32).to(compute_device),
                                        torch.tensor(test_t_hat, dtype=torch.int32).to(compute_device))

    y_pred_final  = torch.cat([torch.sigmoid(y_logit_real), torch.sigmoid(y_logit_fake)], dim=0).cpu().numpy()
    y_label_final = np.concatenate([np.ones(len(y_logit_real)), np.zeros(len(y_logit_fake))])

    discriminative_score = compute_binary_accuracy(y_label_final, y_pred_final)  # output is a single float value

    return discriminative_score
