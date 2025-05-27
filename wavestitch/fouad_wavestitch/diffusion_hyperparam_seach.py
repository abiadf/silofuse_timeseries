"""Hyperparameter search for diffusion"""

from itertools import product
import torch
import os
import torch.optim as optim
import copy

# --- Param grid ---
param_grid = {
    'hdim': [64, 128],
    'lr': [1e-4, 5e-4],
    'diff_batch_size': [32, 64],
    'layers': [2, 3],
    'res_channels': [32, 64],}
# Best config: {'hdim': 64, 'lr': 0.0005, 'diff_batch_size': 64, 'layers': 2, 'res_channels': 32}

grid = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())

# --- Early stopping ---
early_patience = 5

# --- Tracking ---
best_loss = float('inf')
best_config = None
best_model_state = None

# --- Search loop ---
for values in grid:
    config = dict(zip(param_keys, values))
    print(f"\nTesting config: {config}")

    model      = fetchModel(in_dim, out_dim, args).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    dataset    = MyDataset(training_samples.float())
    dataloader = DataLoader(dataset, batch_size= diff_batch_size, shuffle=True)
    criterion  = nn.MSELoss()
    best_val   = float('inf')
    patience   = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch  = batch.to(device)
            t      = torch.randint(diffusion_config['T'], (batch.shape[0],), device=device)
            sigmas = torch.randn(batch.shape, device=device)
            alpha_bars = diffusion_config['alpha_bars'].to(device)
            coeff_1 = torch.sqrt(alpha_bars[t]).reshape((len(t), 1, 1))
            coeff_2 = torch.sqrt(1 - alpha_bars[t]).reshape((len(t), 1, 1))
            conditional_mask = torch.ones(batch.shape, device=device)
            conditional_mask[:, :, non_hier_cols] = 0
            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
            t = t.reshape((-1, 1))
            sigmas_pred  = model(batch_noised, t)
            optimizer.zero_grad()
            sigmas_perm= sigmas[:, :, non_hier_cols].permute((0, 2, 1))
            loss       = criterion(sigmas_pred, sigmas_perm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        if epoch%5 ==0:
            print(f"Epoch {epoch+1}: loss = {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_val:
            best_val = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_patience:
                print("Early stopping")
                break

    # Track best overall
    if best_val < best_loss:
        best_loss = best_val
        best_config = config
        best_model_state = copy.deepcopy(model.state_dict())

# --- Done ---
print(f"\nBest config: {best_config}")
print(f"Best loss: {best_loss:.6f}")

# Save best model
os.makedirs("saved_models/hparam_search", exist_ok=True)
torch.save(best_model_state, "saved_models/hparam_search/best_model.pth")
