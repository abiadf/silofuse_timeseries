"""From HierarchicalTS/TSImputers/SSSDS4Imputer.py. despite the original file name, this current file does diffusion+synthesis instead of imputation"""

import math
import torch
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from scipy import special as ss
from einops import rearrange, repeat
import opt_einsum as oe
from S4Model import SSKernelNPLR, Activation, LinearActivation

contract = oe.contract
contract_expression = oe.contract_expression

def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda() if torch.cuda.is_available() else torch.exp(torch.arange(half_dim) * -_embed)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1, 0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization"""
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]
        L = np.exp(.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1)))
        A = (1. / L[:, None]) * A * L[None, :]
        B = (1. / L[:, None]) * B * np.exp(-.5 * ss.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N // 2, N // 2)))
        B = embed_c2r(np.ones((N // 2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0)  # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5 ** .5 * torch.ones(1, N, dtype=dtype)
    elif measure == 'fourier':
        P = torch.ones(N, dtype=dtype)  # (N)
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0)  # (2 N)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)  # (rank N)
    return P


def nplr(measure, N, rank=1, dtype=torch.float):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    assert dtype == torch.float or torch.cfloat
    if measure == 'random':
        dtype = torch.cfloat if dtype == torch.float else torch.cdouble
        # w = torch.randn(N//2, dtype=dtype)
        w = -torch.exp(torch.randn(N // 2)) + 1j * torch.randn(N // 2)
        P = torch.randn(rank, N // 2, dtype=dtype)
        B = torch.randn(N // 2, dtype=dtype)
        V = torch.eye(N, dtype=dtype)[..., :N // 2]  # Only used in testing
        return w, P, B, V

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)  # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]  # (N,)

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(AP)  # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()

    V_inv = V.conj().transpose(-1, -2)

    B = contract('ij, j -> i', V_inv, B.to(V))  # V^* B
    P = contract('ij, ...j -> ...i', V_inv, P.to(V))  # V^* P

    return w, P, B, V


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        self.S41 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        # self.cond_conv = Conv(2 * in_channels, 2 * self.res_channels, kernel_size=1)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.parametrizations.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.parametrizations.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        # x, cond, diffusion_step_embed = input_data
        x, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.conv_layer(h)
        h = self.S41(h.permute(2, 0, 1)).permute(1, 2, 0)

        # assert cond is not None
        # cond = self.cond_conv(cond)
        # h += cond

        h = self.S42(h.permute(2, 0, 1)).permute(1, 2, 0)

        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

    def forward(self, input_data):
        # noise, conditional, diffusion_steps = input_data
        noise, diffusion_steps = input_data
        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            # h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))
            h, skip_n = self.residual_blocks[n]((h, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSDS4Imputer(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data, diffusion_steps):
        """NOTE: fixed a type issue about permuting a tuple, see commented line"""
        # noise, conditional, mask, diffusion_steps = input_data
        # noised_data, diffusion_steps = input_data.permute((0, 2, 1)), diffusion_steps
        x = input_data.permute(0, 2, 1)

        # conditional = conditional * mask
        # conditional = torch.cat([conditional, mask.float()], dim=1)
        # x = noise
        # x = self.init_conv(x)
        # x = self.residual_layer((x, conditional, diffusion_steps))
        # y = self.final_conv(x)
        x = self.init_conv(x)
        # x = self.residual_layer((x, diffusion_steps))
        x = self.residual_layer((x, diffusion_steps))
        y = self.final_conv(x)
        return y


class SSSDS4Weaver(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Weaver, self).__init__()

        self.fc_embed = nn.Linear(in_features=in_channels, out_features=32)
        self.attn = nn.MultiheadAttention(embed_dim=32, batch_first=True, num_heads=2)

        self.init_conv = nn.Sequential(Conv(32, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data, timesteps):
        # noise, conditional, mask, diffusion_steps = input_data
        x = self.fc_embed(input_data)
        x, diffusion_steps = x.permute((0, 2, 1)), timesteps

        # conditional = conditional * mask
        # conditional = torch.cat([conditional, mask.float()], dim=1)
        #
        # x = noise
        # x = self.init_conv(x)
        # x = self.residual_layer((x, conditional, diffusion_steps))
        # y = self.final_conv(x)
        x, _ = self.attn(x, x, x)
        x = self.init_conv(x)
        x = self.residual_layer((x, diffusion_steps))
        y = self.final_conv(x)
        return y


class HippoSSKernel(nn.Module):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
            self,
            H,
            N=64,
            L=1,
            measure="legs",
            rank=1,
            channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
            dt_min=0.001,
            dt_max=0.1,
            trainable=None,  # Dictionary of options to train various HiPPO parameters
            lr=None,  # Hook to set LR of hippo parameters differently
            length_correction=True,
            # Multiply by I-A|^L after initialization; can be turned off for initialization speed
            hurwitz=False,
            tie_state=False,  # Tie parameters of HiPPO ODE across the H features
            precision=1,  # 1 (single) or 2 (double) for the kernel
            resample=False,
            # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
            verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L, w, p, B, C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def forward(self, L=None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class S4(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1,
            # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1,  # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu',  # activation in between SS and FF
            postact=None,  # activation after FF
            initializer=None,  # initializer on FF
            weight_norm=False,  # weight normalization on FF
            hyper_act=None,  # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True,  # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            # SSM Kernel arguments
            **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h * self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        # self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                + F.pad(k1.flip(-1), (L, 0))
            k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, 'b (s c) h l -> s b c h l', s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        # ysize = b, k, l, requieres l, b, k
        # y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)

        return y, None

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


class S4Layer(nn.Module):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True):
        super().__init__()
        self.s4_layer = S4(d_model=features,
                           d_state=N,
                           l_max=lmax,
                           bidirectional=bidirectional)

        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x has shape seq, batch, feature
        x = x.permute((1, 2, 0))  # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x  # skip connection   # batch, feature, seq
        xout = xout.permute((2, 0, 1))  # seq, batch, feature
        return self.norm_layer(xout)