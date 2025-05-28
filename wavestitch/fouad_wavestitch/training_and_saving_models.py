"""Train and save autoencoder"""

import os
import numpy as np
import torch
from torch import nn, optim, sqrt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
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


def train_and_save_ldm_model(training_df, autoencoder, LDM_args, dataset, device, hierarchical_column_indices):
    with torch.no_grad():
        tensor_input    = torch.tensor(training_df.values, dtype=torch.float32).to(device)
        latent_features = autoencoder.encode_to_latent(tensor_input)
    latent_samples   = latent_features.cpu()
    latent_samples   = latent_samples.unsqueeze(-1)
    print("latent_samples:", latent_samples.shape)
    latent_dim       = latent_samples.shape[1]
    training_dataset = MyDataset(latent_samples)

    in_dim    = latent_dim
    out_dim   = latent_dim # if still masking hierarchicals in latent
    LDM_model = fetchModel(in_dim, out_dim, LDM_args).to(device)

    diffusion_config = fetchDiffusionConfig(LDM_args)
    # Move all tensors in diffusion_config to the correct device
    for k, v in diffusion_config.items():
        if isinstance(v, torch.Tensor):
            diffusion_config[k] = v.to(device)

    optimizer  = optim.AdamW(LDM_model.parameters(), lr=LDM_args.latent_lr)
    criterion  = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=LDM_args.latent_batch_size, shuffle=True)

    # Get non-hierarchical columns (though commented out in the training loop, keeping for completeness if needed)
    all_indices      = np.arange(len(training_df.columns))
    remaining_indices= np.setdiff1d(all_indices, hierarchical_column_indices)
    non_hier_cols    = np.array(remaining_indices) # not used in the loop, remove if  unused

    def latent_diffusion_training_loop(epochs, dataloader, device, diffusion_config, model, optimizer, criterion) -> float:
        """Training loop of the Diffusion part of the LDM. NOTE: NO CONDITIONAL MASK for now"""
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                t           = torch.randint(diffusion_config['T'], (batch.shape[0],), device=device).long()
                t           = t.unsqueeze(1)
                alpha_bars  = diffusion_config['alpha_bars'].to(device)
                coeff_1     = torch.sqrt(alpha_bars[t]).reshape(len(t), 1, 1)
                coeff_2     = torch.sqrt(1 - alpha_bars[t]).reshape(len(t), 1, 1)
                sigmas      = torch.randn_like(batch, device=device)
                batch_noised= coeff_1 * batch + coeff_2 * sigmas
                sigmas_predicted= model(batch_noised.permute(0, 2, 1), t)
                loss = criterion(sigmas_predicted, sigmas)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f'epoch: {epoch+1}/{epochs}, total loss: {total_loss:.4f}')
        return total_loss

    total_loss = latent_diffusion_training_loop(LDM_args.latent_epochs, dataloader, device, diffusion_config, LDM_model, optimizer, criterion)

    # Save model
    path     = f'saved_models/{dataset}/'
    filename = "ldm_model_prop.pth" if LDM_args.propCycEnc else "latent_model.pth"
    filepath = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    torch.save(LDM_model.state_dict(), filepath)

    return filepath, total_loss, in_dim, out_dim, latent_dim