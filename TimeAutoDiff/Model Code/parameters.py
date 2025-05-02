# %% Auto-encoder Training

# most important
# lat_dim     = 7 # dimensionality of the latent space, representing the compressed representation of the input data
# hidden_size = 250 # number of neurons in the hidden layers of the GRU encoders and the decoder MLP, controlling the model's capacity
# lr          = 5e-4 # learning rate for the optimizer, determining the step size during weight updates

# # important
# num_layers  = 1 # number of stacked layers in the GRU encoders. More layers can capture more complex temporal dependencies
# emb_dim     = 128 # dimensionality of the embeddings used to represent categorical features
# weight_decay= 1e-5 # strength of L2 regularization applied to the model's weights to prevent overfitting
# batch_size  = 50 # number of samples processed in each training iteration

# # other
# n_epochs    = 1_500 # Number of training iterations (epochs) the model will go through, loss stabilizes after 1k
# eps         = 1e-5 # A small constant added for numerical stability in certain calculations (though not directly used in the provided training loop)
# channels    = 64 # Likely related to the architecture of an internal convolutional or similar layer within the `Embedding_data` or other parts of the model, controlling the number of feature maps
# min_beta    = 1e-5 # minimum value for the beta parameter, which was originally intended to weight the KL divergence loss (though it's currently unused)
# max_beta    = 0.1 # initial maximum value for the beta parameter, intended for the KL divergence loss weighting (currently unused)
# time_dim    = 8 # dimensionality of the time embedding used as input to the `time_encode` layer (currently commented out in the encoder)
# seq_col     = 'Symbol' # name of the column in the input DataFrame that represents the sequence identifier (not a direct model hyperparameter)

# BEST
# loss = 0.0035
# lat_dim: 7 # best
# hidden_size: 250 # best
# lr: 0.0005 # best
# num_layers: 1 # best
# emb_dim: 128 # best
# weight_decay: 1e-05 # best
# batch_size: 50 # best


# # ORIGINAL
lat_dim     = 7
hidden_size = 200
lr          = 2e-4
num_layers  = 1
emb_dim     = 128
weight_decay= 1e-6
batch_size  = 50
n_epochs    = 1500#50_000
eps         = 1e-5
channels    = 64
min_beta    = 1e-5
max_beta    = 0.1
time_dim    = 8
seq_col     = 'Symbol'

# %% Diffusion Training
hidden_dim      = 200
num_layers_diff = 2
diffusion_steps = 100
n_epochs_diff   = 600 # loss stabilizes after ~600

# # ORIGINAL
# hidden_dim      = 200
# num_layers_diff = 2
# diffusion_steps = 100
# n_epochs_diff   = 50_000
