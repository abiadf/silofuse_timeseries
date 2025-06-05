"""Set all parameters for the discriminative score here for ease of access"""

hidden_dim_fraction = 1/2 # <1, fraction of features
# hidden_dim     = 64

# Discriminative params
random_state   = 42
iterations     = 2000 # Timewak: 2000
batch_size     = 128
train_set_size = 0.8
learning_rate  = 0.001
steps_to_print_output = 200

# Predictive_params
pred_iterations = 500 # Timewak: 5000
pred_batch_size = 128 # Timewak: 128
