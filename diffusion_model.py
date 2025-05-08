"""Module about diffusion model"""

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Sequence, Tuple
from utils import compute_mse_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Diffusion():

    def forward_diffusion(x_0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """Given original data x_0 at t=0, compute x_t for ANY timestep directly, using DDPM formula:
            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        Efficiently generate a noisy version of data at specific timestep
        Used during training instead of looping through all diffusion steps"""
        device      = x_0.device
        alphas      = 1. - betas
        alpha_bars  = torch.cumprod(alphas, dim=0).to(x_0.device)
        alpha_bar_t = alpha_bars[t].reshape(-1, 1, 1)  # (batch, 1, 1) for correct broadcasting
        noise       = torch.randn_like(x_0).to(x_0.device)
        x_t         = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def reverse_diffusion(model: Callable[[Tensor, int], Tensor], x_t: Tensor,
                          betas: Sequence[float], diff_steps: int, stochastic: bool = True) -> Tensor:
        """Reverse diffusion (denoising) using a trained model, and DDPM (full posterior mean + stochastic noise) or DDIM
        - model (Callable): model that predicts noise given (x_t, t)
        - x_t (Tensor): noised input at timestep `diff_steps`
        - betas (Sequence[float]): noise schedule used during forward process
        - diff_steps (int): total number of diffusion steps
        - stochastic (bool): true=DDPM, false=DDIM
        - x_t output (Tensor): reconstructed sample after denoising"""
        
        betas      = torch.tensor(betas, device=x_t.device)
        alphas     = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(x_t.device)

        for t in reversed(range(diff_steps)):
            alpha_t    = alphas[t]
            alpha_bar_t= alpha_bars[t]
            beta_t     = betas[t]
            # noise_pred = model(x_t, t)
            t_tensor   = torch.tensor([t] * x_t.shape[0], device=x_t.device).long() # Convert t to a tensor
            noise_pred = model(x_t, t_tensor)

            noise_pred = torch.clamp(noise_pred, -10.0, 10.0) # Add this line

            # # check last few steps
            # if t > diff_steps - 5: # Check the last few steps
            #     print(f"Timestep {t}, noise_pred stats: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}, min={noise_pred.min().item():.4f}, max={noise_pred.max().item():.4f}")
            #     if torch.isnan(noise_pred).any():
            #         print(f"NaN detected in noise_pred at timestep {t}")
            #         break

            # Temporary padding to match lengths (so code doesnt break)
            if noise_pred.shape[-1] < x_t.shape[-1]:
                padding = torch.zeros_like(x_t[..., :x_t.shape[-1] - noise_pred.shape[-1]])
                noise_pred = torch.cat([noise_pred, padding], dim=-1)
            elif noise_pred.shape[-1] > x_t.shape[-1]:
                noise_pred = noise_pred[..., :x_t.shape[-1]]

            coef1 = 1 / alpha_t.sqrt()
            coef2 = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
            mean  = coef1 * (x_t - coef2 * noise_pred)

            if t > 0 and stochastic:
                noise = torch.randn_like(x_t).to(x_t.device)
                sigma = beta_t.sqrt()
                x_t   = mean + sigma * noise
            else:
                x_t = mean
        return x_t

    def get_noise_schedule(start_val: float, end_val: float, diff_steps: int,
                           cos_start_offset: float, noise_profile: str = 'l') -> torch.Tensor:
        """Generate a noise schedule (linear, cosine, quadratic) given start/end values and number of steps"""
        
        if noise_profile == 'l': # linear schedule
            return torch.linspace(start_val, end_val, diff_steps).to(device)

        elif noise_profile == 'c': # cos schedule
            steps      = torch.linspace(0, 1, diff_steps + 1)  # Normalize steps to [0, 1]
            f          = lambda t: torch.cos((t + cos_start_offset) * math.pi / 2) ** 2
            alphas_bar = f(steps) / f(0)  # Compute cumulative product
            betas      = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return betas.clamp(max=0.999).to(device) # ensure betas <1

        elif noise_profile == 'q': # quadratic schedule
            steps = torch.linspace(0, 1, diff_steps)
            betas = start_val + (end_val - start_val) * (steps ** 2)
            return betas.to(device)

        else:
            raise ValueError(f"Unknown noise profile: {noise_profile}")


# TODO: change bottleneck line from `self.CHANNEL_MUL**2` to *2
# consider larger or non-square `UPSAMPLE_SCALE`
# add temporal attention (or give them different weights) to focus on important timesteps
# add multi-input layer to account for multivariate timeseries
# use right loss function; MSE for timeseries regression, crossentropyloss for classification
class UNet(nn.Module):
    """UNet NN MODEL for 1D, contains skip connections, downsampling, and upsampling
    NOTE: used torch functions tailored to timeseries (1D), 1d is in their name"""

    KERNEL_SIZE    = 3 # Convolution layer kernel size
    PADDING        = 1 # Padding for same-size output
    OUTPUT_KERNEL  = 1 # Final conv kernel size

    CAT_DIM        = 1 # Channel dim for skip connections
    UPSAMPLE_SCALE = 2 # Pooling/upsample scale
    CHANNEL_MUL    = 2 # Channel scaling factor

    def __init__(self, input_channels: int, dropout_prob: float, embedding_dim: int, base_channels: int) -> None:
        """Initialize UNet with input and base channel sizes. The `input_channels` parameter
        specifies the # of independent features (timeseries) in the input data"""
        super().__init__()

        self.dropout_prob  = dropout_prob
        self.embedding_dim = embedding_dim

        # derive widths
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # encoder
        self.down1 = self._conv_block(input_channels, c1)
        self.down2 = self._conv_block(c1, c2)
        self.down3 = self._conv_block(c2, c3)

        # bottleneck
        self.bottleneck = self._conv_block(c3, c3)

        # time-step MLP → c3 channels
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, c3),
            nn.ReLU(),
            nn.Linear(c3, c3),)

        # Upsampling layers
        self.up1 = self._conv_block(c3 + c2, c2)
        self.up2 = self._conv_block(c2 + c1, c1)
        self.output = nn.Conv1d(c1, input_channels, kernel_size=self.OUTPUT_KERNEL)


    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Returns 2-layer convolutional block for 1D timeseries data. This block consists of
        2 consecutive convolutional layers, each followed by BatchNorm, ReLU activation, and
        Dropout. Padding is set to maintain the sequence length
            - in_channels (int): # of input channels for 1st convolutional layer
            - out_channels (int): # of output channels for both convolutional layers in this block
            - output (nn.Sequential): sequential container holding the 2 convolutional layers
            with their associated BatchNorm, ReLU, and Dropout layers"""

        conv_block_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=self.KERNEL_SIZE, padding=self.PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.KERNEL_SIZE, padding=self.PADDING),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),)
        return conv_block_module


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet architecture for denoising; this method implements the
        encoding (downsampling), bottleneck, and decoding (upsampling) stages of the UNet. Skip
        connections are used to pass feature maps from the downsampling path to the corresponding
        upsampling path, allowing the decoder to recover fine-grained details lost during
        downsampling. To handle potential size mismatches in the sequence length during concatenation
        in the skip connections, padding is applied to ensure compatible tensor shapes
        - x (torch.Tensor): The input tensor with shape [batch_size, num_channels, sequence_length],
        where num_channels represents the number of features (timeseries)
        - t (torch.Tensor, optional): timestep tensor (used in diffusion models)
                                    Defaults to None, but can be used by the convolutional blocks if needed
        - output (torch.Tensor): The output tensor, representing the denoised prediction
                          Shape: [batch_size, num_channels, sequence_length]"""
        # encode
        x1 = self.down1(x)        # → (B, c1, L)
        x2 = self.down2(x1)       # → (B, c2, L)
        x3 = self.down3(x2)       # → (B, c3, L)

        # time-step embedding
        emb = get_timestep_embedding(t, self.embedding_dim)  # (B, emb)
        emb = self.time_mlp(emb)[..., None]                 # (B, c3, 1)
        
        # Add time embedding to the bottleneck layer output
        x3 = x3 + emb # broadcast → (B, c3, L)

        # Bottleneck
        x3 = self.bottleneck(x3)      # (batch, 256, seq_len)

        # decode
        u1 = F.interpolate(x3, scale_factor=2, mode='nearest')
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1)          # → (B, c2, 2L)

        u2 = F.interpolate(u1, scale_factor=2, mode='nearest')
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2)          # → (B, c1, 4L)
        return self.output(u2)

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Generate sinusoidal timestep embeddings for time-conditioned models.
        - timesteps (torch.Tensor): Tensor of timesteps.
        - embedding_dim (int): Embedding dimension, must be even.
    Returns:
        torch.Tensor: Timestep embeddings.
    Motivation:
        The constant `10000.0` is a scaling factor ensuring a range of frequencies in the embedding, derived from the original Transformer architecture. It balances high- and low-frequency signals for effective time-step representation"""

    half = embedding_dim // 2
    freq = torch.exp(-torch.arange(half, device=timesteps.device) * (math.log(10000) / (half - 1)))
    emb  = timesteps[:, None].float() * freq[None]
    emb  = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2: emb = F.pad(emb, (0,1))
    return emb

# def _prepare_batch(batch: Tuple[torch.Tensor, ...], device: torch.device, num_channels: int) -> torch.Tensor:
#     """Prepares a batch of data for training/validation"""
#     x_0 = batch[0].to(device)
#     if x_0.ndim == 2:
#         x_0 = x_0.unsqueeze(-1)  # [B, num_features, 1]
#     elif x_0.ndim == 3 and x_0.shape[1] != num_channels:
#         x_0 = x_0.permute(0, 2, 1)  # [B, 1, L] -> [B, C, L]
#     return x_0
def _prepare_batch(batch: Tuple[torch.Tensor, ...], device: torch.device, num_channels: int) -> torch.Tensor:
    """Prepares a batch of data for training/validation"""
    x_0 = batch[0].to(device)
    return x_0

def _train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, betas: torch.Tensor,
                 diffusion_steps: int, device: torch.device, num_channels: int,) -> float:
    """Performs 1 training epoch"""

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x_0 = _prepare_batch(batch, device, num_channels)
        t   = torch.randint(0, diffusion_steps, (x_0.size(0),), device=device)
        x_t, true_noise = Diffusion.forward_diffusion(x_0, t, betas)
        predicted_noise = model(x_t, t)
        loss = compute_mse_loss(predicted_noise, true_noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevents exploding gradients
        optimizer.step()

        total_loss += loss.item() * x_0.size(0)
    return total_loss / len(train_loader.dataset)

def _validation_epoch(model: nn.Module, validation_loader: DataLoader, betas: torch.Tensor,
                      diffusion_steps: int, device: torch.device, num_channels: int,) -> float:
    """Performs 1 validation epoch"""

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in validation_loader:
            x_0 = _prepare_batch(batch, device, num_channels)
            t   = torch.randint(0, diffusion_steps, (x_0.size(0),), device=device)
            x_t, true_noise = Diffusion.forward_diffusion(x_0, t, betas)
            predicted_noise = model(x_t, t)
            val_loss += compute_mse_loss(predicted_noise, true_noise).item() * x_0.size(0)
    return val_loss / len(validation_loader.dataset)

def train_diffusion(device: torch.device, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler, betas: torch.Tensor,
                    diffusion_steps: int, epochs: int, validation_loader: DataLoader = None, patience: int = 5) -> None:
    """Trains diffusion model with given training dataset, and optional validation dataset (and avg loss
    over validation dataset). We also add early stopping. Args:
        - model (nn.Module): The diffusion model to be trained
        - train_loader (DataLoader): DataLoader for the training dataset
        - optimizer (optim.Optimizer): Optimizer for model parameters
        - betas (torch.Tensor): Noise schedule for the forward diffusion process
        - diffusion_steps (int): Number of diffusion steps for the process
        - epochs (int): Number of training epochs
        - validation_loader (DataLoader, optional): DataLoader for the validation dataset. Defaults to None
        - patience (int, optional): Number of epochs to wait before early stopping if validation loss doesn't improve. Defaults to 5
        - output (None): function performs in-place training and prints training progress"""

    model.to(device)
    best_val_loss    = float('inf')
    patience_counter = 0

    num_channels = model.output.in_channels if hasattr(model, 'output') and hasattr(model.output, 'in_channels') else model.down1[0].in_channels

    for epoch in range(epochs):
        avg_loss = _train_epoch(model, train_loader, optimizer, betas, diffusion_steps, device, num_channels)
        print(f"Epoch [{epoch+1}/{epochs}], loss: {avg_loss:.4f}", end="")

        if validation_loader:
            avg_val_loss = _validation_epoch(model, validation_loader, betas, diffusion_steps, device, num_channels)
            print(f", Validation loss: {avg_val_loss:.4f}")

            if avg_val_loss   < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                # print(f"Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        scheduler.step()


def train_multi_client_diffusion_ldm(client_data_list, client_feature_counts, autoencoder_list, diffusion_model,
                                     optimizer_diffusion, betas, diffusion_steps, num_epochs_diff=10, batch_size=64):
    print("Training Diffusion Model on Latent Representations...")
    combined_latent_representations = []
    for i, client_data in enumerate(client_data_list):
        autoencoder   = autoencoder_list[i] if len(autoencoder_list) > 1 else autoencoder_list[0]
        dataloader    = DataLoader(TensorDataset(client_data), batch_size=batch_size, shuffle=False)
        client_latents= []
        with torch.no_grad():
            for batch in dataloader:
                latents = autoencoder.encode_to_latent(batch[0])
                client_latents.append(latents)
        combined_latent_representations.append(torch.cat(client_latents, dim=1)) # Concatenate along feature dimension

    combined_latent_tensor= torch.cat(combined_latent_representations, dim=1)
    latent_dataloader     = DataLoader(TensorDataset(combined_latent_tensor), batch_size=batch_size, shuffle=True)
    num_latent_features   = combined_latent_tensor.shape[1] # Get the total number of latent features

    train_diffusion(device, diffusion_model, latent_dataloader, optimizer_diffusion,
                    None, betas, diffusion_steps, num_epochs_diff, num_channels=num_latent_features) # Pass num_latent_features as num_channels


def sample_new_data(model: nn.Module, betas: torch.Tensor, diff_steps: int, shape: tuple[int, ...]) -> Tensor:
    """Samples new data by reversing the diffusion process, starting from random noise. Args:
        - model (nn.Module): trained diffusion model used for denoising
        - betas (torch.Tensor): noise schedule for the diffusion process
        - diff_steps (int): # of diffusion steps to reverse
        - shape (tuple[int, ...]): shape of the generated sample (e.g., (batch_size, channels, height, width))
        - output (torch.Tensor): generated sample after reversing the diffusion process"""

    x_t = torch.randn(shape).to(device)
    for _ in reversed(range(diff_steps)):
        x_t = Diffusion.reverse_diffusion(model, x_t, betas, diff_steps)
    return x_t


# TODO: options to improve the diffusion model:
# - conditional diffusion model
# - guided diffusion model
# consider using:
# Exponential Moving Average to stabilize training, or
# gradient clipping to prevent exploding gradients