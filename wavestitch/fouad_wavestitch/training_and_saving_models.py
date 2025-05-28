"""Train and save autoencoder"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import sqrt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import autoencoder as ae
from training_utils import MyDataset, fetchDiffusionConfig, fetchModel

def train_and_save_autoencoder(train_dataset, test_dataset, ae_args, dataset_name, device):
    """input_size = # dataset features"""
    input_size   = train_dataset[0][0].shape[0]
    train_loader = DataLoader(train_dataset, batch_size=ae_args.ae_batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=ae_args.ae_batch_size, shuffle=False)
    autoencoder  = ae.Autoencoder(input_size, ae_args.ae_layer1_dim, ae_args.ae_layer2_dim, ae_args.ae_latent_dim, ae_args.ae_dropout_prob)
    optimizer    = torch.optim.AdamW(autoencoder.parameters(), lr=ae_args.ae_optimizer_lr, weight_decay=ae_args.ae_weight_decay)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ae_args.ae_scheduler_mode, patience=ae_args.ae_scheduler_patience, factor=ae_args.ae_scheduler_factor)

    trainer   = ae.TrainAutoencoder()
    best_loss = trainer.train_autoencoder(device, autoencoder, ae_args.ae_training_epochs, train_loader, optimizer, scheduler,
                                          validation_loader=test_loader, patience=ae_args.ae_training_patience)
    path = 'saved_models/autoencoder/'
    os.makedirs(path, exist_ok=True)
    filename = f"{dataset_name}_autoencoder.pth"
    filepath = os.path.join(path, filename)
    torch.save(autoencoder.state_dict(), filepath)
    return filepath, input_size


def train_and_save_diffusion(training_df, hierarchical_column_indices, diffusion_args, dataset, device):
    d_vals_tensor    = torch.from_numpy(training_df.values.astype(np.float32))
    training_samples = d_vals_tensor.unfold(0, diffusion_args.window_size, 1).transpose(1, 2)
    in_dim           = training_df.shape[1]
    out_dim          = in_dim - len(hierarchical_column_indices)
    training_dataset = MyDataset(training_samples.float())
    model            = fetchModel(in_dim, out_dim, diffusion_args).to(device)
    diffusion_config = fetchDiffusionConfig(diffusion_args)
    optimizer        = optim.Adam(model.parameters(), lr=diffusion_args.lr)
    criterion        = nn.MSELoss()
    dataloader       = DataLoader(training_dataset, batch_size=diffusion_args.diff_batch_size, shuffle=True)
    all_indices      = torch.arange(len(training_df.columns))
    remaining_indices= [i for i in range(len(training_df.columns)) if i not in hierarchical_column_indices]
    non_hier_cols    = torch.tensor(remaining_indices)

    def training_loop():
        for epoch in range(diffusion_args.epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch     = batch.to(device)
                t         = torch.randint(diffusion_config['T'], (batch.shape[0],), device=device)
                sigmas    = torch.randn(batch.shape, device=device)
                alpha_bars= diffusion_config['alpha_bars'].to(device)
                coeff_1   = sqrt(alpha_bars[t]).reshape(len(t), 1, 1)
                coeff_2   = sqrt(1 - alpha_bars[t]).reshape(len(t), 1, 1)

                conditional_mask = torch.ones(batch.shape, device=device)
                conditional_mask[:, :, non_hier_cols] = 0

                batch_noised    = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
                t               = t.reshape(-1, 1)
                sigmas_predicted= model(batch_noised, t)
                optimizer.zero_grad()
                sigmas_permuted = sigmas[:, :, non_hier_cols].permute(0, 2, 1).to(device)
                loss            = criterion(sigmas_predicted, sigmas_permuted)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}/{diffusion_args.epochs}, total loss: {total_loss}')
        return total_loss

    total_loss= training_loop()
    path      = f'saved_models/{dataset}/'
    os.makedirs(path, exist_ok=True)
    filename  = "model_prop.pth" if diffusion_args.propCycEnc else "model.pth"
    filepath  = os.path.join(path, filename)
    torch.save(model.state_dict(), filepath)

    return filepath, total_loss, in_dim, out_dim


class LDMTrainer:
    def __init__(self, autoencoder, LDM_args, dataset_name, device):
        self.autoencoder     = autoencoder
        self.LDM_args        = LDM_args
        self.dataset_name    = dataset_name
        self.device          = device
        self.LDM_model       = None
        self.diffusion_config= fetchDiffusionConfig(LDM_args)

        # Move all tensors in diffusion_config to the correct device
        for k, v in self.diffusion_config.items():
            if isinstance(v, torch.Tensor):
                self.diffusion_config[k] = v.to(device)

        self.criterion    = nn.MSELoss()
        self.latent_dim   = None # Will be set after data preparation
        self.latent_scaler= StandardScaler()

    def _prepare_latent_data(self, df):
        """Encodes df data to latent space and prepares for DataLoader."""
        with torch.no_grad():
            tensor_input    = torch.tensor(df.values, dtype=torch.float32).to(self.device)
            latent_features = self.autoencoder.encode_to_latent(tensor_input)
        latent_samples = latent_features.cpu()
        # latent_samples = latent_samples.unsqueeze(-1) # Shape: (batch_size, latent_dim, 1)
        return latent_samples

    def latent_diffusion_training_loop(self, dataloader) -> float:
        """Training loop for the Diffusion part of the LDM."""
        self.LDM_model.train()
        total_loss = 0.0
        for epoch in range(self.LDM_args.latent_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                t = torch.randint(self.diffusion_config['T'], (batch.shape[0],), device=self.device).long().unsqueeze(1)
                
                # Coefficients for noise addition
                alpha_bars = self.diffusion_config['alpha_bars']
                coeff_1    = torch.sqrt(alpha_bars[t]).reshape(len(t), 1, 1)
                coeff_2    = torch.sqrt(1 - alpha_bars[t]).reshape(len(t), 1, 1)
                
                # Add noise to batch
                sigmas       = torch.randn_like(batch, device=self.device)
                batch_noised = coeff_1 * batch + coeff_2 * sigmas
                
                # Model predicts noise (permute for model's expected input shape)
                sigmas_predicted = self.LDM_model(batch_noised.permute(0, 2, 1), t)
                
                loss = self.criterion(sigmas_predicted, sigmas)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                total_loss += loss.item() # Accumulate overall loss

            if (epoch + 1) % 20 == 0:
                print(f'Epoch: {epoch+1}/{self.LDM_args.latent_epochs}, Avg Epoch Loss: {epoch_loss / len(dataloader):.4f}')
        return total_loss / (len(dataloader) * self.LDM_args.latent_epochs) # Return average loss per batch over all epochs

    def latent_diffusion_test_loop(self, dataloader) -> float:
        """Evaluation loop for the Diffusion part of the LDM."""
        self.LDM_model.eval() # Set model to evaluation mode
        total_test_loss = 0.0
        with torch.no_grad(): # No gradient calculation during evaluation
            for batch in dataloader:
                batch = batch.to(self.device)
                t     = torch.randint(self.diffusion_config['T'], (batch.shape[0],), device=self.device).long().unsqueeze(1)
                
                # Coefficients for noise addition
                alpha_bars = self.diffusion_config['alpha_bars']
                coeff_1    = torch.sqrt(alpha_bars[t]).reshape(len(t), 1, 1)
                coeff_2    = torch.sqrt(1 - alpha_bars[t]).reshape(len(t), 1, 1)
                
                # Add noise to batch
                sigmas       = torch.randn_like(batch, device=self.device)
                batch_noised = coeff_1 * batch + coeff_2 * sigmas
                
                # Model predicts noise (permute for model's expected input shape)
                sigmas_predicted = self.LDM_model(batch_noised.permute(0, 2, 1), t)
                
                loss = self.criterion(sigmas_predicted, sigmas)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(dataloader)
        print(f'Test Loss: {avg_test_loss:.4f}')
        self.LDM_model.train() # Set model back to training mode
        return avg_test_loss

    def get_latent_scaler(self):
        """Returns the fitted StandardScaler for inverse transformation."""
        if self.latent_scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call train_and_save_model first.")
        return self.latent_scaler

    def train_and_save_model(self, training_df, test_df, hierarchical_column_indices=None):
        # 1. Prepare Data
        # latent_samples  = self._prepare_latent_data(training_df)
        unscaled_latent_features_train = self._prepare_latent_data(training_df)
        self.latent_dim = unscaled_latent_features_train.shape[1]
        
        # Fit and transform training latents, then unsqueeze for DataLoader
        scaled_latent_samples_train_np = self.latent_scaler.fit_transform(unscaled_latent_features_train.numpy())
        scaled_latent_samples_train    = torch.tensor(scaled_latent_samples_train_np, dtype=torch.float32).unsqueeze(-1) # UNSQUEEZE HERE
        print(f"Scaled Latent samples (training): {scaled_latent_samples_train.shape}")
        training_dataset = MyDataset(scaled_latent_samples_train)

        # Get unscaled test latent features (N, latent_dim)
        unscaled_latent_features_test = self._prepare_latent_data(test_df)
        
        # Transform test latents (do NOT fit), then unsqueeze for DataLoader
        scaled_latent_samples_test_np = self.latent_scaler.transform(unscaled_latent_features_test.numpy())
        scaled_latent_samples_test    = torch.tensor(scaled_latent_samples_test_np, dtype=torch.float32).unsqueeze(-1) # UNSQUEEZE HERE
        print(f"Scaled Latent samples (test): {scaled_latent_samples_test.shape}")
        test_dataset = MyDataset(scaled_latent_samples_test)

        # 2. Initialize Model and Optimizer
        in_dim          = self.latent_dim
        out_dim         = self.latent_dim
        self.LDM_model  = fetchModel(in_dim, out_dim, self.LDM_args).to(self.device)
        self.optimizer  = optim.AdamW(self.LDM_model.parameters(), lr=self.LDM_args.latent_lr)
        dataloader      = DataLoader(training_dataset, batch_size=self.LDM_args.latent_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.LDM_args.latent_batch_size, shuffle=False)

        # 3. Train and Evaluate
        print("Starting LDM training...")
        total_train_loss = self.latent_diffusion_training_loop(dataloader)
        print("Starting LDM evaluation...")
        total_test_loss = self.latent_diffusion_test_loop(test_dataloader)

        # 4. Save Model
        save_path = f'saved_models/{self.dataset_name}/'
        filename  = "ldm_model_prop.pth" if self.LDM_args.propCycEnc else "latent_model.pth"
        filepath  = os.path.join(save_path, filename)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.LDM_model.state_dict(), filepath)

        print(f"LDM Training complete. Final Training Loss: {total_train_loss:.4f}, Final Test Loss: {total_test_loss:.4f}")
        print(f"LDM model saved to: {filepath}")

        return filepath, total_train_loss, total_test_loss, in_dim, out_dim, self.latent_dim
