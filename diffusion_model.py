"""Module about diffusion model"""

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Sequence, Tuple

# Determine if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
        noise_pred = model(x_t, t)

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
# add batchnorm
# consider larger or non-square `UPSAMPLE_SCALE`
# add temporal attention (or give them different weights) to focus on important timesteps
# add multi-input layer to account for multivariate timeseries
# use right loss finction; MSE for timeseries regression, crossentropyloss for classification
class UNet(nn.Module):
    """UNet NN MODEL for 1D, contains skip connections, downsampling, and upsampling
    NOTE: used torch functions tailored to timeseries (1D), 1d is in their name"""

    # non-tunable
    CAT_DIM        = 1 # Channel dim for skip connections
    PADDING        = 1 # Padding for same-size output

    # tunable
    KERNEL_SIZE    = 3 # Conv kernel size
    UPSAMPLE_SCALE = 2 # Pooling/upsample scale
    CHANNEL_MUL    = 2 # Channel scaling factor
    OUTPUT_KERNEL  = 1 # Final conv kernel size

    def __init__(self, input_channels: int, dropout_prob: float, embedding_dim: int = 128, base_channels: int = 64) -> None:
        """Initialize UNet with input and base channel sizes. The `input_channels` parameter
        specifies the # of independent features (timeseries) in the input data"""
        super().__init__()

        self.dropout_prob  = dropout_prob
        self.embedding_dim = embedding_dim

        # Downsample
        self.down1 = self.conv_block(input_channels, base_channels)
        self.down2 = self.conv_block(base_channels, base_channels * self.CHANNEL_MUL)
        self.pool  = nn.MaxPool1d(self.UPSAMPLE_SCALE)
        self.pad   = nn.ConstantPad1d((0, 1), 0)  # manually add right padding if needed

        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * self.CHANNEL_MUL, base_channels * self.CHANNEL_MUL**2)

        # Upsample
        self.up1      = nn.ConvTranspose1d(base_channels * self.CHANNEL_MUL**2, base_channels * self.CHANNEL_MUL, kernel_size=self.UPSAMPLE_SCALE, stride=self.UPSAMPLE_SCALE)
        self.conv_up1 = self.conv_block(base_channels * self.CHANNEL_MUL**2, base_channels * self.CHANNEL_MUL)
        self.up2      = nn.ConvTranspose1d(base_channels * self.CHANNEL_MUL, base_channels, kernel_size=self.UPSAMPLE_SCALE, stride=self.UPSAMPLE_SCALE)
        self.conv_up2 = self.conv_block(base_channels * self.CHANNEL_MUL, base_channels)

        # Output
        self.output   = nn.Conv1d(base_channels, input_channels, kernel_size=self.OUTPUT_KERNEL)

        # Time embedding (Conditioning on timestep)
        self.time_embed = nn.Sequential(
                            nn.Linear(embedding_dim, base_channels * self.CHANNEL_MUL),
                            nn.ReLU(),
                            nn.Linear(base_channels * self.CHANNEL_MUL, base_channels * self.CHANNEL_MUL))
        self.to(device)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Returns 2-layer convolutional block for 1D timeseries data. This block consists of
        2 consecutive convolutional layers, each followed by BatchNorm, ReLU activation, and
        Dropout. Padding is set to maintain the sequence length
            - in_channels (int): # of input channels for 1st convolutional layer
            - out_channels (int): # of output channels for both convolutional layers in this block
            - output (nn.Sequential): sequential container holding the 2 convolutional layers
            with their associated BatchNorm, ReLU, and Dropout layers"""

        print(f"conv_block: in_channels = {in_channels}, out_channels = {out_channels}")  # Debugging
        unet = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.KERNEL_SIZE, padding=self.PADDING),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob),
                nn.Conv1d(out_channels, out_channels, kernel_size=self.KERNEL_SIZE, padding=self.PADDING),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob))
        return unet

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

        # Down path (encoder)
        x1 = self.down1(x)
        x2 = self.down2(self.pool(self.pad(x1)))
        x3 = self.pool(self.pad(x2))

        # Time embedding for conditioning
        if t is not None:
            t_emb = get_timestep_embedding(t, self.embedding_dim)
            t_emb = self.time_embed(t_emb).unsqueeze(-1)  # Add channel dimension
            x3    = x3 + t_emb  # Inject timestep info into bottleneck features

        # Bottleneck ('latent')
        x3 = self.bottleneck(self.pool(self.pad(x3)))

        # Up path (decoder)
        x = self.up1(x3)

        # Skip connection 1
        # x2_padded = F.pad(x2, [0, x.shape[-1] - x2.shape[-1]], "constant", 0)
        x2_padded = F.pad(x2, [0, x.shape[-1] - x2.shape[-1]]) if x2.shape[-1] != x.shape[-1] else x2
        x = self.conv_up1(torch.cat([x, x2_padded], dim=self.CAT_DIM))

        # Skip connection 2
        x = self.up2(x)
        x1_padded = F.pad(x1, [0, x.shape[-1] - x1.shape[-1]], "constant", 0)
        x = self.conv_up2(torch.cat([x, x1_padded], dim=self.CAT_DIM))

        return self.output(x)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Generate sinusoidal timestep embeddings for time-conditioned models.
        - timesteps (torch.Tensor): Tensor of timesteps.
        - embedding_dim (int): Embedding dimension, must be even.
    Returns:
        torch.Tensor: Timestep embeddings.
    Motivation:
        The constant `10000.0` is a scaling factor ensuring a range of frequencies in the embedding, derived from the original Transformer architecture. It balances high- and low-frequency signals for effective time-step representation"""

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def compute_mse_loss(predicted_noise, true_noise):
    """Computes MSE loss given 2 inputs"""
    return torch.mean((predicted_noise - true_noise) ** 2)


def _prepare_batch(batch: Tuple[torch.Tensor, ...], device: torch.device, num_channels: int) -> torch.Tensor:
    """Prepares a batch of data for training/validation"""
    x_0 = batch[0].to(device)
    if x_0.ndim == 2:
        x_0 = x_0.unsqueeze(-1)  # [B, num_features, 1]
    elif x_0.ndim == 3 and x_0.shape[1] != num_channels:
        x_0 = x_0.permute(0, 2, 1)  # [B, 1, L] -> [B, C, L]
    return x_0

def _train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, betas: torch.Tensor,
                 diffusion_steps: int, device: torch.device, num_channels: int,) -> float:
    """Performs 1 training epoch"""

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x_0 = _prepare_batch(batch, device, num_channels)
        t   = torch.randint(0, diffusion_steps, (x_0.size(0),), device=device)
        x_t, true_noise = forward_diffusion(x_0, t, betas)
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
            x_t, true_noise = forward_diffusion(x_0, t, betas)
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

def sample_new_data(model: nn.Module, betas: torch.Tensor, diff_steps: int, shape: tuple[int, ...]) -> Tensor:
    """Samples new data by reversing the diffusion process, starting from random noise. Args:
        - model (nn.Module): trained diffusion model used for denoising
        - betas (torch.Tensor): noise schedule for the diffusion process
        - diff_steps (int): # of diffusion steps to reverse
        - shape (tuple[int, ...]): shape of the generated sample (e.g., (batch_size, channels, height, width))
        - output (torch.Tensor): generated sample after reversing the diffusion process"""

    x_t = torch.randn(shape).to(device)
    for _ in reversed(range(diff_steps)):
        x_t = reverse_diffusion(model, x_t, betas, diff_steps)
    return x_t


# TODO: options to improve the diffusion model:
# - conditional diffusion model
# - guided diffusion model
# consider using:
# Exponential Moving Average to stabilize training, or
# gradient clipping to prevent exploding gradients