from plotter import Plotter
from trainer import *
from Data  import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time


end_name=''
sector = 1
filename = f"run5197_sector{sector}.csv"

doTraining=True

print('\n\nLoading data...')
startT_data = time.time()

events = read_sector_file(filename)
plotter = Plotter(print_dir="plots/", end_name=end_name,sector=sector)
# Just plot first few events
for i, (all_hits, tb_hits) in enumerate(events[:3]):
    plotter.plot_hits(all_hits, tb_hits, event_idx=i)


dataset = HitsDataset(events)
val_size = 100000
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print('\n\nTrain size:',train_size)
print('Test size:',val_size)


train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

X_sample, Y_sample = next(iter(train_loader))
print('X_sample:',X_sample.shape)  # should print: torch.Size([32, 1, 36, 112])
print('Y_sample:',Y_sample.shape,'\n\n') 

endT_data = time.time()
T_data = (endT_data - startT_data)
print(f'Loading data took {T_data:.2f}s \n\n')

model = LitConvAutoencoder()

loss_tracker = LossTracker()

trainer = pl.Trainer(
    max_epochs=50,
    enable_progress_bar=True,
    log_every_n_steps=1,
    enable_checkpointing=False,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0,
    callbacks=[loss_tracker]
)

if doTraining:
    print('\n\nTraining...')
    startT_train = time.time()
    trainer.fit(model, train_loader, val_loader)

    endT_train = time.time()
    T_train = (endT_train - startT_train)/60
    print(f'Training took {T_train:.2f}min \n\n')

    plotter.plotTrainLoss(loss_tracker)

    # Save model
    
    example_input = torch.randn(1, X_sample.shape[1], X_sample.shape[2], X_sample.shape[3])
    #print("Input shape:", example_input.shape)
    #print("Output shape:", output.shape)
    torchscript_model = torch.jit.trace(model, example_input)
    torchscript_model.save(f"nets/cnn_autoenc_sector{sector}{end_name}.pt")

model = torch.jit.load(f"nets/cnn_autoenc_sector{sector}{end_name}.pt")
model.eval()

all_hits_val = []
tb_hits_val = []
predictions = []

model.eval()

print('\n\nTesting...')
startT_test = time.time()

with torch.no_grad():
    for X_batch, Y_batch in val_loader:
        Y_hat = model(X_batch)
        all_hits_val.append(X_batch.squeeze(1).numpy())
        tb_hits_val.append(Y_batch.squeeze(1).numpy())
        predictions.append(Y_hat.squeeze(1).numpy())

endT_test = time.time()
Rate_test = (val_size / (endT_test - startT_test)) / 1000.
print(f'Testing took {endT_test-startT_test:.2f}s, Eg rate: {Rate_test:.2f} kHz\n\n')

all_hits_val = np.concatenate(all_hits_val, axis=0)
tb_hits_val = np.concatenate(tb_hits_val, axis=0)
predictions = np.concatenate(predictions, axis=0)

# Compute efficiency and background rejection
thresholds, signal_eff, background_rej, best_threshold = compute_efficiency_background(
    all_hits_val, tb_hits_val, predictions
)

# Plot efficiency vs threshold
plotter.plot_efficiency_background(thresholds, signal_eff, background_rej)

# Apply chosen threshold to predictions
pred_masked = (predictions >= best_threshold).astype(float)

# Mask pred_masked to only keep hits that exist in all_hits
pred_masked = pred_masked.astype(int) + all_hits_val.astype(int) - 1  

# Plot first few events
for i in range(min(5, len(all_hits_val))):
    plotter.plot_hits(all_hits_val[i], tb_hits_val[i], pred_masked[i], event_idx=i)
