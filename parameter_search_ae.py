"""Hyperparameter search for autoencoder. Can copy-paste to notebook"""
from itertools import product
import autoencoder as ae
import torch

param_grid = {
    'layer1_dim':     [12, 16, 20],
    'layer2_dim':   [6, 8, 10],
    'latent_dim':     [3, 4, 5],
    'dropout_prob':   [0.05, 0.1, 0.15],
    'optimizer_lr':   [0.0001, 0.0005]
}

# (16, 8, 3, 0.1, 0.0001)
results = []
for hd, ed, ld, dp, lr in product(param_grid['layer1_dim'],
                                  param_grid['layer2_dim'],
                                  param_grid['latent_dim'],
                                  param_grid['dropout_prob'],
                                  param_grid['optimizer_lr']):
    autoencoder = ae.Autoencoder(input_size, hd, ed, ld, dp).to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, scheduler_mode,
                                                           patience=scheduler_patience,
                                                           factor=scheduler_factor)
    trainer = ae.TrainAutoencoder()
    val_loss = trainer.train_autoencoder(device, autoencoder, ae_training_epochs, train_loader, optimizer, scheduler,
                                         validation_loader=val_loader, patience=training_patience)
    results.append(((hd, ed, ld, dp, lr), val_loss))

best_params = min(results, key=lambda x: x[1])[0]
print(f"Best AE params: {best_params}")
