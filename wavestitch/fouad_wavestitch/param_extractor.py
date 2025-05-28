"""Set dataset and get parameters from yaml file for each of the autoencoder, diffusion, and LDM diffusion"""

from types import SimpleNamespace
import yaml

dataset_name = "RossmanSales" # "RossmanSales", "BeijingAirQuality", "", "", "PanamaElectricity"

filenames_dict = {"RossmanSales": "rossmansales_params.yaml",
                  "BeijingAirQuality": "beijingairquality_params.yaml",
                  "PanamaElectricity": "panama_electricity_params.yaml",}

with open(filenames_dict[dataset_name], 'r') as wavestitch_file:
    wavestitch_yaml = yaml.safe_load(wavestitch_file)

ae_args = SimpleNamespace(
    ae_layer1_dim         = wavestitch_yaml['autoencoder']['layer1_dim'],
    ae_layer2_dim         = wavestitch_yaml['autoencoder']['layer2_dim'],
    ae_latent_dim         = wavestitch_yaml['autoencoder']['latent_dim'],
    ae_dropout_prob       = wavestitch_yaml['autoencoder']['dropout_prob'],
    ae_training_epochs    = wavestitch_yaml['autoencoder']['ae_training_epochs'],
    ae_batch_size         = wavestitch_yaml['autoencoder']['ae_batch_size'],
    ae_optimizer_lr       = wavestitch_yaml['autoencoder']['ae_optimizer_lr'],
    ae_weight_decay       = wavestitch_yaml['autoencoder']['weight_decay'],
    ae_training_patience  = wavestitch_yaml['autoencoder']['training_patience'],
    ae_scheduler_patience = wavestitch_yaml['autoencoder']['scheduler_patience'],
    ae_scheduler_mode     = wavestitch_yaml['autoencoder']['scheduler_mode'],
    ae_scheduler_factor   = wavestitch_yaml['autoencoder']['scheduler_factor'],)

diffusion_args = SimpleNamespace(
    backbone            = wavestitch_yaml['shared']['backbone'],
    random_seed         = wavestitch_yaml['shared']['random_seed'],
    data_split          = wavestitch_yaml['shared']['data_split'],
    beta_0              = wavestitch_yaml['shared']['beta_0'],
    beta_T              = wavestitch_yaml['shared']['beta_T'],
    timesteps           = wavestitch_yaml['shared']['timesteps'],
    hdim                = wavestitch_yaml['shared']['hdim'],
    lr                  = wavestitch_yaml['shared']['lr'],
    diff_batch_size     = wavestitch_yaml['shared']['diff_batch_size'],
    layers              = wavestitch_yaml['shared']['layers'],
    window_size         = wavestitch_yaml['shared']['window_size'],
    stride              = wavestitch_yaml['shared']['stride'],
    num_res_layers      = wavestitch_yaml['shared']['num_res_layers'],
    res_channels        = wavestitch_yaml['shared']['res_channels'],
    skip_channels       = wavestitch_yaml['shared']['skip_channels'],
    diff_step_embed_in  = wavestitch_yaml['shared']['diff_step_embed_in'],
    diff_step_embed_mid = wavestitch_yaml['shared']['diff_step_embed_mid'],
    diff_step_embed_out = wavestitch_yaml['shared']['diff_step_embed_out'],
    s4_lmax             = wavestitch_yaml['shared']['s4_lmax'],
    s4_dstate           = wavestitch_yaml['shared']['s4_dstate'],
    s4_dropout          = wavestitch_yaml['shared']['s4_dropout'],
    s4_bidirectional    = wavestitch_yaml['shared']['s4_bidirectional'],
    s4_layernorm        = wavestitch_yaml['shared']['s4_layernorm'],
    propCycEnc          = wavestitch_yaml['shared']['propCycEnc'],
    epochs              = wavestitch_yaml['training']['epochs'],
    synth_mask          = wavestitch_yaml['synthesis']['synth_mask'],
    n_trials            = wavestitch_yaml['synthesis']['n_trials'])

LDM_args = SimpleNamespace(
    backbone           = wavestitch_yaml['shared']['backbone'],
    s4_bidirectional   = wavestitch_yaml['shared']['s4_bidirectional'],
    s4_layernorm       = wavestitch_yaml['shared']['s4_layernorm'],
    latent_epochs      = wavestitch_yaml['latent_diffusion']['latent_epochs'],
    latent_lr          = wavestitch_yaml['latent_diffusion']['latent_lr'],
    beta_0             = wavestitch_yaml['latent_diffusion']['latent_beta_0'],
    beta_T             = wavestitch_yaml['latent_diffusion']['latent_beta_T'],
    timesteps          = wavestitch_yaml['latent_diffusion']['latent_timesteps'],
    latent_batch_size  = wavestitch_yaml['latent_diffusion']['latent_batch_size'],
    num_res_layers     = wavestitch_yaml['latent_diffusion']['latent_num_res_layers'],
    res_channels       = wavestitch_yaml['latent_diffusion']['latent_res_channels'],
    skip_channels      = wavestitch_yaml['latent_diffusion']['latent_skip_channels'],
    diff_step_embed_in = wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_in'],
    diff_step_embed_mid= wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_mid'],
    diff_step_embed_out= wavestitch_yaml['latent_diffusion']['latent_diff_step_embed_out'],
    s4_lmax            = wavestitch_yaml['latent_diffusion']['latent_s4_lmax'],
    s4_dstate          = wavestitch_yaml['latent_diffusion']['latent_s4_dstate'],
    s4_dropout         = wavestitch_yaml['latent_diffusion']['latent_s4_dropout'],
    propCycEnc         = wavestitch_yaml['latent_diffusion']['propCycEnc'],
    num_samples        = wavestitch_yaml['latent_diffusion']['num_samples'],
    )
