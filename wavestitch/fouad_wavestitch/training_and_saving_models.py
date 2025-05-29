"""Train and save models"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import sqrt, from_numpy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import autoencoder as ae
from training_utils import MyDataset, fetchDiffusionConfig, fetchModel
from time import perf_counter as timer # Using perf_counter for accurate timing
from metasynth import metadataMask

from timeit import default_timer as timer



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


class Synthesizer:
    def __init__(self, df, preprocessor, diffusion_args, dataset_name):
        self.df           = df
        self.preprocessor = preprocessor
        self.args         = diffusion_args
        self.dataset      = dataset_name
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = None
        self.test_loader  = None
        self.mask_loader  = None
        self.test_dataset = None
        self.mask_dataset = None
        self.non_hier_cols= None
        self.hierarchical_column_indices = None
        self.diffusion_config            = None
        self.path = f'generated/{self.dataset}/{self.args.synth_mask}/'
        os.makedirs(self.path, exist_ok=True)

    @staticmethod
    def decimal_places(series):
        return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()

    def create_pipelined_noise(self, batch_shape):
        total_rows = self.args.stride * (batch_shape[0] - 1) + self.args.window_size
        sampled    = torch.normal(0, 1, (total_rows, batch_shape[2])).to(self.device)
        return sampled.unfold(0, self.args.window_size, self.args.stride).transpose(1, 2)

    def prepare_data_and_model(self):
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)

        end        = self.preprocessor.test_indices[-1]
        start      = self.preprocessor.test_indices[0]
        count      = ((end + 1 - self.args.window_size - start) // self.args.stride) + 1
        tilde_start= end + 1 - self.args.window_size - (count * self.args.stride)
        additional = start - tilde_start

        test_df         = self.df.loc[self.preprocessor.train_indices[-additional:] + self.preprocessor.test_indices]
        test_decoded    = self.preprocessor.cyclicDecode(test_df)
        decimal_accuracy= {k: self.decimal_places(self.preprocessor.df_orig[k]) for k in test_decoded.columns}
        metadata        = test_decoded[self.preprocessor.hierarchical_features_uncyclic]
        rows_to_synth   = metadataMask(metadata, self.args.synth_mask, self.dataset)
        real_df         = test_decoded[rows_to_synth]
        real_df_out     = self.preprocessor.rescale(real_df).reset_index(drop=True).round(decimal_accuracy)
        real_path       = os.path.join(self.path, 'real.csv')
        if not os.path.exists(real_path):
            real_df_out.to_csv(real_path)

        data_tensor = from_numpy(test_df.values)
        mask_tensor = from_numpy(rows_to_synth.values)
        windows     = data_tensor.unfold(0, self.args.window_size, self.args.stride).transpose(1, 2)
        masks       = mask_tensor.unfold(0, self.args.window_size, self.args.stride)

        self.hierarchical_column_indices = test_df.columns.get_indexer(self.preprocessor.hierarchical_features_cyclic)
        self.non_hier_cols= np.setdiff1d(np.arange(len(test_df.columns)), self.hierarchical_column_indices)
        self.test_dataset = MyDataset(windows.float())
        self.mask_dataset = MyDataset(masks)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=self.args.diff_batch_size)
        self.mask_loader  = DataLoader(self.mask_dataset, batch_size=self.args.diff_batch_size)

        in_dim     = len(test_df.columns)
        out_dim    = len(self.non_hier_cols)
        self.model = fetchModel(in_dim, out_dim, self.args).to(self.device)
        self.diffusion_config = fetchDiffusionConfig(self.args)

        model_path = f'saved_models/{self.dataset}/model_prop.pth' if self.args.propCycEnc else f'saved_models/{self.dataset}/model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        return rows_to_synth, decimal_accuracy

    @torch.no_grad()
    def synthesize(self, n_trials=1):
        rows_to_synth, decimal_accuracy = self.prepare_data_and_model()
        num_ops    = 0
        exec_times = []

        for trial in range(n_trials):
            start_time   = timer()
            synth_tensor = torch.empty(0, self.test_dataset.inputs.shape[2]).to(self.device)
            for idx, (test_batch, mask_batch) in enumerate(zip(self.test_loader, self.mask_loader)):
                test_batch = test_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                noise      = self.create_pipelined_noise(test_batch.shape)
                x          = noise.clone()
                x[:, :, self.hierarchical_column_indices] = test_batch[:, :, self.hierarchical_column_indices]
                for step in reversed(range(self.diffusion_config['T'])):
                    times = torch.full((test_batch.shape[0], 1), step).to(self.device)
                    ab_t  = self.diffusion_config['alpha_bars'][step].to(self.device)
                    ab_t1 = self.diffusion_config['alpha_bars'][step - 1].to(self.device) if step > 0 else None
                    at    = self.diffusion_config['alphas'][step].to(self.device)
                    b_t   = self.diffusion_config['betas'][step].to(self.device)
                    sampled_noise = self.create_pipelined_noise(test_batch.shape)
                    cached= torch.sqrt(ab_t) * test_batch + torch.sqrt(1 - ab_t) * sampled_noise
                    mask_expanded = torch.zeros_like(test_batch, dtype=bool)
                    for c in self.non_hier_cols:
                        mask_expanded[:, :, c] = mask_batch
                    x[~mask_expanded] = cached[~mask_expanded]
                    x[:, :, self.hierarchical_column_indices] = test_batch[:, :, self.hierarchical_column_indices]
                    eps = self.model(x, times).permute(0, 2, 1)
                    variance = 0.0
                    if step > 0:
                        variance = b_t * ((1 - ab_t1) / (1 - ab_t)) * torch.normal(0, 1, size=eps.shape).to(self.device)
                    norm_denoised = self.create_pipelined_noise(test_batch.shape)
                    norm_denoised[:, :, self.non_hier_cols] = (x[:, :, self.non_hier_cols] - ((b_t / torch.sqrt(1 - ab_t)) * eps)) / torch.sqrt(at)
                    norm_denoised[:, :, self.non_hier_cols] += variance
                    x[mask_expanded]  = norm_denoised[mask_expanded]
                    x[~mask_expanded] = test_batch[~mask_expanded]
                    x[1:, : (self.args.window_size - self.args.stride), :] = x.roll(1, 0)[1:, self.args.stride: self.args.window_size, :]
                    if trial == 0:
                        num_ops += 1
                if idx == 0:
                    generated = torch.cat((x[0], x[1:, (self.args.window_size - self.args.stride):, :].reshape(-1, x.shape[2])), dim=0)
                else:
                    generated = x[:, (self.args.window_size - self.args.stride):, :].reshape(-1, x.shape[2])
                synth_tensor = torch.cat((synth_tensor, generated), dim=0)

            exec_times.append(timer() - start_time)
            df_synth    = pd.DataFrame(synth_tensor.cpu().numpy(), columns=self.df.columns)
            decoded     = self.preprocessor.decode(df_synth, rescale=True)
            mask        = rows_to_synth.to_numpy() if hasattr(rows_to_synth, "to_numpy") else np.array(rows_to_synth)
            synth_df_out= decoded[mask].round(decimal_accuracy)
            suffix      = 'cycProp' if self.args.propCycEnc else 'cycStd'
            synth_df_out.to_csv(f'{self.path}synth_hyacinth_pipeline_stride_{self.args.stride}_trial_{trial}_{suffix}.csv')
            print(f"File {trial+1}/{n_trials} done")

            if trial == 0:
                with open(f'{self.path}denoiser_calls_pipeline_stride_{self.args.stride}_{suffix}.txt', 'w') as f:
                    f.write(str(num_ops))

        return synth_df_out, exec_times



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
