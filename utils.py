
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.stats import kstest
from scipy.stats import wasserstein_distance as wasserstein
from statsmodels.tsa.stattools import acf
from tslearn.metrics import dtw


def compute_reconstruction_loss(x_input: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
    """Computes reconstruction loss (MSE) between input and reconstructed output
    - x_in (torch.Tensor): Original input
    - x_out (torch.Tensor): Reconstructed input
    - torch.Tensor: Scalar loss value"""

    # MSE
    return torch.nn.functional.mse_loss(x_reconstructed, x_input)

    # MAE
    return torch.nn.functional.l1_loss(x_reconstructed, x_input)


def compute_mse_loss(predicted_noise, true_noise):
    """Computes MSE loss given 2 inputs"""
    if predicted_noise.shape != true_noise.shape:
        true_noise = true_noise.unsqueeze(1)  # [B, 1, 128]
        true_noise = F.interpolate(true_noise, size=(predicted_noise.shape[-1],), mode='nearest')
        true_noise = true_noise.squeeze(1)  # back to [B, 512]
    return torch.mean((predicted_noise - true_noise) ** 2)


def evaluate_and_plot_autoencoder_metrics(X_scaled, X_reconstructed, should_we_plot):

    X_tensor= torch.tensor(X_scaled, dtype=torch.float32)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reconstructed_array= np.array(X_reconstructed)
    n_features         = X_scaled.shape[1]  # Number of features
    ks_stats           = []
    wasserstein_dists  = []
    real_acfs          = []
    generated_acfs     = []
    dtw_distances      = []

    reconstruction_error = compute_reconstruction_loss(X_tensor.to(device),
        torch.tensor(reconstructed_array, dtype=torch.float32).to(device))

    for i in range(n_features):
        real_flat     = X_scaled[:, i].flatten()
        reconstr_flat = reconstructed_array[:, i].flatten()

        # Kolg-Smir Test
        ks_statistic, _ = kstest(real_flat, reconstr_flat)  # We only need the statistic
        ks_stats.append(ks_statistic)

        # Wasserstein dist (= how much work to move from 1 distr. to another)
        wasserstein_dist = wasserstein(np.sort(real_flat), np.sort(reconstr_flat))
        wasserstein_dists.append(wasserstein_dist)

        # Autocorr. (only calculate once and store)
        if i == 0:
            real_acfs      = acf(X_scaled[:, i], nlags=20)
            generated_acfs = acf(reconstructed_array[:, 0], nlags=20)

        # DTW
        dtw_distance = dtw(X_scaled[:, i], reconstructed_array[:, i])
        dtw_distances.append(dtw_distance)

    # Print the results, calculating the averages here
    print(f"Reconst. error: {reconstruction_error:.4f}")
    print(f"Avg Kolg-Smir Statistic: {np.mean(ks_stats):.4f}")
    print(f"Avg Wasserstein Distance: {np.mean(wasserstein_dists):.4f}")
    print(f"Real ACF (first 5 lags): {real_acfs[:5]}")
    print(f"Generated ACF (first 5 lags): {generated_acfs[:5]}")
    print(f"Avg DTW Distance: {np.mean(dtw_distances):.2f}")


    if should_we_plot == True:
        # Plotting the metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # KS Statistic per feature
        axs[0, 0].bar(range(n_features), ks_stats)
        axs[0, 0].set_title('Kolmogorov-Smirnov Statistic per Feature')
        axs[0, 0].set_xlabel('Feature')
        axs[0, 0].set_ylabel('KS Statistic')

        # Wasserstein Distance per feature
        axs[0, 1].bar(range(n_features), wasserstein_dists, color='orange')
        axs[0, 1].set_title('Wasserstein Distance per Feature')
        axs[0, 1].set_xlabel('Feature')
        axs[0, 1].set_ylabel('Distance')

        # DTW Distance per feature
        axs[1, 0].bar(range(n_features), dtw_distances, color='green')
        axs[1, 0].set_title('DTW Distance per Feature')
        axs[1, 0].set_xlabel('Feature')
        axs[1, 0].set_ylabel('DTW')

        # ACF comparison (first feature only)
        lags = np.arange(len(real_acfs))
        axs[1, 1].plot(lags, real_acfs, label='Real', marker='o')
        axs[1, 1].plot(lags, generated_acfs, label='Reconstructed', marker='x')
        axs[1, 1].set_title('Autocorrelation (Feature 0)')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('ACF')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    return ks_stats, wasserstein_dists, real_acfs, generated_acfs, dtw_distances

