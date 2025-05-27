"""Hyperparameter search for LDM diffusion"""

from itertools import product
import torch.optim as optim
import copy
import os # Ensure os is imported for saving models

# --- Define the Hyperparameter Grid for LDM ---
# Focus on parameters most likely to impact latent diffusion performance
param_grid_ldm = {
    'latent_lr': [8e-4, 1e-4, 15e-4],  # Learning rate is crucial
    'latent_batch_size': [128], # 64, Batch size affects training stability and speed
    'latent_num_res_layers': [2], # 4],    # Depth of the denoiser
    'latent_res_channels': [128], #[32, 64, 128], # Width of the denoiser
    'latent_s4_dstate': [32],#, 64],       # S4 state dimension for memory/expressiveness
    'latent_s4_dropout': [0.0],    # Regularization
    # e.g., 'latent_beta_0', 'latent_beta_T', 'latent_timesteps', etc.
}
# latent_lr = 1e-4 (middle = better)
# latent_batch_size = 128 (bigger = better)
# latent_num_res_layers = 2 (2 = sometimes better)
# latent_res_channels = 128 (bigger = better)
# latent_s4_dstate = 32 (lower=better)
# latent_s4_dropout = 0 (0 is best)
# {'latent_lr': 0.0001, 'latent_batch_size': 128, 'latent_num_res_layers': 2, 'latent_res_channels': 128, 'latent_s4_dstate': 32, 'latent_s4_dropout': 0.0}


grid_ldm = list(product(*param_grid_ldm.values()))
param_keys_ldm = list(param_grid_ldm.keys())

early_patience_ldm = 10 # More patience might be needed for LDM, adjust as needed

best_loss_ldm = float('inf')
best_config_ldm = None
best_model_state_ldm = None

print(f"Starting LDM Hyperparam Search with {len(grid_ldm)} combinations")

for values_ldm in grid_ldm:
    current_config_ldm = dict(zip(param_keys_ldm, values_ldm))
    print(f"\nTesting LDM config: {current_config_ldm}")

    # Create a *new* args object for this trial, updating only the parameters
    # being searched, and pulling defaults for others from YAML.
    # This ensures only the relevant parameters are varied.
    trial_args = SimpleNamespace(
        backbone           = wavestitch_yaml['shared']['backbone'],
        s4_bidirectional   = wavestitch_yaml['shared']['s4_bidirectional'],
        s4_layernorm       = wavestitch_yaml['shared']['s4_layernorm'],
        latent_epochs      = wavestitch_yaml['latent_diffusion']['latent_epochs'], # Not searched, but needed by training loop
        latent_lr          = wavestitch_yaml['latent_diffusion']['latent_lr'],
        beta_0             = wavestitch_yaml['latent_diffusion']['latent_beta_0'], # Fixed for search, can add to grid
        beta_T             = wavestitch_yaml['latent_diffusion']['latent_beta_T'], # Fixed for search, can add to grid
        timesteps          = wavestitch_yaml['latent_diffusion']['latent_timesteps'], # Fixed for search, can add to grid
        latent_batch_size  = wavestitch_yaml['latent_diffusion']['latent_batch_size'],
        num_res_layers     = wavestitch_yaml['latent_diffusion']['latent_num_res_layers'],
        res_channels       = wavestitch_yaml['latent_diffusion']['latent_res_channels'],
        skip_channels      = wavestitch_yaml['latent_diffusion']['latent_skip_channels'],
        diff_step_embed_in = wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_in'], # Fixed for search
        diff_step_embed_mid= wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_mid'], # Fixed for search
        diff_step_embed_out= wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_out'], # Fixed for search
        s4_lmax            = wavestitch_yaml['latent_diffusion']['latent_s4_lmax'], # Fixed for search, related to latent_dim's length
        s4_dstate          = wavestitch_yaml['latent_diffusion']['latent_s4_dstate'],
        s4_dropout         = wavestitch_yaml['latent_diffusion']['latent_s4_dropout'],
    )

    # Re-initialize model, optimizer, and dataloader for each trial
    model_ldm = fetchModel(in_dim, out_dim, trial_args).to(device)
    # Re-fetch diffusion config as beta_0, beta_T, timesteps could be part of search
    diffusion_config_ldm = fetchDiffusionConfig(trial_args)
    optimizer_ldm = optim.AdamW(model_ldm.parameters(), lr=trial_args.latent_lr)
    dataloader_ldm = DataLoader(training_dataset, batch_size=trial_args.latent_batch_size, shuffle=True)
    criterion_ldm = nn.MSELoss()
    best_val_ldm = float('inf')
    patience_ldm = 0

    # Training loop for the current LDM config
    for epoch in range(trial_args.latent_epochs):
        model_ldm.train()
        total_loss_ldm = 0.0
        for batch in dataloader_ldm:
            batch = batch.to(device)
            optimizer_ldm.zero_grad()
            t = torch.randint(diffusion_config_ldm['T'], (batch.shape[0],), device=device).long()
            t = t.unsqueeze(1) # Ensure t is (batch_size, 1)
            alpha_bars = diffusion_config_ldm['alpha_bars'].to(device)
            coeff_1 = torch.sqrt(alpha_bars[t]).reshape(len(t), 1, 1)
            coeff_2 = torch.sqrt(1 - alpha_bars[t]).reshape(len(t), 1, 1)
            sigmas = torch.randn_like(batch, device=device)
            batch_noised = coeff_1 * batch + coeff_2 * sigmas
            sigmas_predicted = model_ldm(batch_noised.permute(0, 2, 1), t)
            loss_ldm = criterion_ldm(sigmas_predicted, sigmas)
            loss_ldm.backward()
            optimizer_ldm.step()
            total_loss_ldm += loss_ldm.item()
        avg_loss_ldm = total_loss_ldm / len(dataloader_ldm)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: loss = {avg_loss_ldm:.6f}")
        # Early stopping logic
        if avg_loss_ldm < best_val_ldm:
            best_val_ldm = avg_loss_ldm
            patience_ldm = 0
        else:
            patience_ldm += 1
            if patience_ldm >= early_patience_ldm:
                print("  Early stopping for current LDM config.")
                break
    # Track best overall configuration
    if best_val_ldm < best_loss_ldm:
        best_loss_ldm = best_val_ldm
        best_config_ldm = current_config_ldm
        best_model_state_ldm = copy.deepcopy(model_ldm.state_dict())
        print(f"  New best LDM loss found: {best_loss_ldm:.6f} with config: {best_config_ldm}")

# --- End of LDM Hyperparameter Search ---
print(f"\nLDM Hyperparameter Search Complete!")
print(f"Best LDM config: {best_config_ldm}")
print(f"Best LDM loss: {best_loss_ldm:.6f}")

# Save the best LDM model
# Ensure the path and filename are appropriate for your LDM model
ldm_save_path = f'saved_models/{wavestitch_yaml["shared"]["dataset"]}/hparam_search_ldm'
os.makedirs(ldm_save_path, exist_ok=True)
torch.save(best_model_state_ldm, os.path.join(ldm_save_path, "best_latent_model.pth"))
print(f"Best LDM model saved to: {os.path.join(ldm_save_path, 'best_latent_model.pth')}")