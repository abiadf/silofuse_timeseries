"""Synthesize samples using Wavestitch [from wavestitch_synthesizer.py]"""

from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
from metasynth import metadataMask

class Args:
    backbone            = backbone
    beta_0              = beta_0
    beta_T              = beta_T
    timesteps           = timesteps
    num_res_layers      = num_res_layers
    res_channels        = res_channels
    skip_channels       = skip_channels
    diff_step_embed_in  = diff_step_embed_in
    diff_step_embed_mid = diff_step_embed_mid
    diff_step_embed_out = diff_step_embed_out
    s4_lmax             = s4_lmax
    s4_dstate           = s4_dstate
    s4_dropout          = s4_dropout
    s4_bidirectional    = s4_bidirectional
    s4_layernorm        = s4_layernorm

args   = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper ---
def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()

def create_pipelined_noise(test_batch, stride, window_size):
    sampled = torch.normal(0, 1, (stride * (test_batch.shape[0]-1) + window_size, test_batch.shape[2]))
    return sampled.unfold(0, window_size, stride).transpose(1, 2)

# --- Load Data ---
np.random.seed(random_seed)
torch.manual_seed(random_seed)

preprocessor      = Preprocessor(dataset, propCycEnc)
df                = preprocessor.df_cleaned
end               = preprocessor.test_indices[-1]
start             = preprocessor.test_indices[0]
window_count      = ((end + 1 - window_size - start) // stride) + 1
tilde_start       = end + 1 - window_size - (window_count * stride)
additional_indices= start - tilde_start
test_df           = df.loc[preprocessor.train_indices[-additional_indices:] + preprocessor.test_indices]
test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
decimal_accuracy  = {k: decimal_places(preprocessor.df_orig[k])
                     for k in test_df_with_hierarchy.columns}
metadata     = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
rows_to_synth= metadataMask(metadata, synth_mask, dataset)
real_df      = test_df_with_hierarchy[rows_to_synth]
df_synth     = test_df.copy()

# --- Model Prep ---
d_vals   = df_synth.values
m_vals   = rows_to_synth.values
d_tensor = from_numpy(d_vals)
m_tensor = from_numpy(m_vals)
windows  = d_tensor.unfold(0, window_size, stride).transpose(1, 2)
masks    = m_tensor.unfold(0, window_size, stride)
hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)

non_hier_cols   = np.setdiff1d(np.arange(len(df_synth.columns)), hierarchical_column_indices)
test_dataset    = MyDataset(windows.float())
mask_dataset    = MyDataset(masks)
model           = fetchModel(len(df_synth.columns), len(non_hier_cols), args).to(device)
diffusion_config= fetchDiffusionConfig(args)
test_loader     = DataLoader(test_dataset, batch_size=batch_size)
mask_loader     = DataLoader(mask_dataset, batch_size=batch_size)
model_path      = f'saved_models/{dataset}/model_prop.pth' if propCycEnc else f'saved_models/{dataset}/model.pth'
saved_params    = torch.load(model_path, map_location=device)

with torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(saved_params[name])
        param.requires_grad = False
model.eval()

real_df_out = preprocessor.rescale(real_df).reset_index(drop=True).round(decimal_accuracy)
path        = f'generated/{dataset}/{synth_mask}/'
os.makedirs(path, exist_ok=True)
real_path   = os.path.join(path, 'real.csv')
if not os.path.exists(real_path):
    real_df_out.to_csv(real_path)

# --- Inference ---
num_ops    = 0
exec_times = []

for trial in range(n_trials):
    start_time   = timer()
    synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
    for idx, (test_batch, mask_batch) in enumerate(zip(test_loader, mask_loader)):
        test_batch = test_batch.to(device)
        mask_batch = mask_batch.to(device)
        x = create_pipelined_noise(test_batch, stride, window_size).to(device)
        x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
        for step in range(diffusion_config['T'] - 1, -1, -1):
            times = torch.full((test_batch.shape[0], 1), step).to(device)
            ab_t  = diffusion_config['alpha_bars'][step].to(device)
            ab_t1 = diffusion_config['alpha_bars'][step - 1].to(device) if step > 0 else None
            at    = diffusion_config['alphas'][step].to(device)
            b_t   = diffusion_config['betas'][step].to(device)
            sampled_noise= create_pipelined_noise(test_batch, stride, window_size).to(device)
            cached       = torch.sqrt(ab_t) * test_batch + torch.sqrt(1 - ab_t) * sampled_noise
            mask_expanded     = torch.zeros_like(test_batch, dtype=bool)
            for c in non_hier_cols:
                mask_expanded[:, :, c] = mask_batch
            x[~mask_expanded] = cached[~mask_expanded]
            x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
            eps  = model(x, times).permute(0, 2, 1)
            vari = 0.0
            if step > 0:
                vari = b_t * ((1 - ab_t1) / (1 - ab_t)) * torch.normal(0, 1, size=eps.shape).to(device)
            norm_denoised = create_pipelined_noise(test_batch, stride, window_size).to(device)
            norm_denoised[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - ((b_t / torch.sqrt(1 - ab_t)) * eps)) / torch.sqrt(at)
            norm_denoised[:, :, non_hier_cols] += vari
            x[mask_expanded]  = norm_denoised[mask_expanded]
            x[~mask_expanded] = test_batch[~mask_expanded]
            x[1:, : (window_size - stride), :] = x.roll(1, 0)[1:, stride: window_size, :]
            if trial == 0:
                num_ops += 1
        if idx == 0:
            generated = torch.cat((x[0], x[1:, (window_size - stride):, :].reshape(-1, x.shape[2])), dim=0)
        else:
            generated = x[:, (window_size - stride):, :].reshape(-1, x.shape[2])
        synth_tensor  = torch.cat((synth_tensor, generated), dim=0)

    exec_times.append(timer() - start_time)
    df_synthesized = pd.DataFrame(synth_tensor.cpu().numpy(), columns=df.columns)
    # synth_df_out   = preprocessor.decode(df_synthesized, rescale=True)[rows_to_synth.reset_index(drop=True)].round(decimal_accuracy)
    mask         = rows_to_synth.to_numpy() if hasattr(rows_to_synth, "to_numpy") else np.array(rows_to_synth)
    synth_df_out = preprocessor.decode(df_synthesized, rescale=True)[mask].round(decimal_accuracy)
    file_suffix  = 'cycProp' if propCycEnc else 'cycStd'
    synth_df_out.to_csv(f'{path}synth_hyacinth_pipeline_stride_{stride}_trial_{trial}_{file_suffix}.csv')
    if trial == 0:
        with open(f'{path}denoiser_calls_pipeline_stride_{stride}_{file_suffix}.txt', 'w') as f:
            f.write(str(num_ops))

with open(f'generated/{dataset}/{synth_mask}/denoiser_calls_pipeline_stride_{stride}_cycStd.txt', 'a') as f:
    arr_time = np.array(exec_times)
    f.write('\n' + str(np.mean(arr_time)) + '\n' + str(np.std(arr_time)))
