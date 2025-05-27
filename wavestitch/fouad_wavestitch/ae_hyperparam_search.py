"""Hyperparameter search for AE. Copy this into the notebook after the AE is called"""

import torch
from itertools import product

param_grid = {
    'layer1_dim':     [21],
    'layer2_dim':     [9],
    'latent_dim':     [4],
    'dropout_prob':   [0.15],
    'optimizer_lr':   [0.0011],
    'batch_size':     [122, 124,126, 128, 130, 132, 134]
}
# Best AE params: (21, 9, 4, 0.15, 0.0011, 128)

results = []
for hd, ed, ld, dp, lr, bs in product(param_grid['layer1_dim'],
                                      param_grid['layer2_dim'],
                                      param_grid['latent_dim'],
                                      param_grid['dropout_prob'],
                                      param_grid['optimizer_lr'],
                                      param_grid['batch_size']):
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    autoencoder = ae.Autoencoder(input_size, hd, ed, ld, dp).to(device)
    optimizer   = torch.optim.AdamW(autoencoder.parameters(), lr=lr, weight_decay=ae_weight_decay)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, ae_scheduler_mode,
                                                             patience=ae_scheduler_patience,
                                                             factor=ae_scheduler_factor)
    trainer  = ae.TrainAutoencoder()
    val_loss = trainer.train_autoencoder(device, autoencoder, ae_training_epochs,
                                         train_loader, optimizer, scheduler,
                                         validation_loader=val_loader,
                                         patience=ae_training_patience)
    results.append(((hd, ed, ld, dp, lr, bs), val_loss))

best_params = min(results, key=lambda x: x[1])[0]
print(f"Best AE params: {best_params}")
