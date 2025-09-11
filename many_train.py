import os
import csv
import time
import warnings

from plotter import Plotter
from trainer import *
from Data import *

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

def run_experiment(params, train_loader, val_loader, nEpoch=20, save_name="_test"):
    # Train
    model = LitConvAutoencoder(**params)
    trainer = pl.Trainer(
        max_epochs=nEpoch,  # reduce for speed
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False
    )

    trainer.fit(model, train_loader, val_loader)

    #save model
    X_sample, Y_sample = next(iter(train_loader))
    example_input = torch.randn(1, X_sample.shape[1], X_sample.shape[2], X_sample.shape[3])
    #print("Input shape:", example_input.shape)
    #print("Output shape:", output.shape)
    torchscript_model = torch.jit.trace(model, example_input)
    torchscript_model.save("nets/cnn_autoenc"+save_name+".pt")

    # Test
    model.eval()
    predictions = []
    all_hits_val = []
    tb_hits_val = []

    startT = time.time()
    with torch.no_grad():
      for X_batch, Y_batch in val_loader:
          Y_hat = model(X_batch)
          all_hits_val.append(X_batch.squeeze(1).numpy())
          tb_hits_val.append(Y_batch.squeeze(1).numpy())
          predictions.append(Y_hat.squeeze(1).numpy())

    endT = time.time()
    Rate_test = (len(val_loader.dataset) / (endT - startT)) / 1000.0

    all_hits_val = np.concatenate(all_hits_val, axis=0)
    tb_hits_val = np.concatenate(tb_hits_val, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    best_eff, best_rej, best_thr = compute_efficiency_background(
        all_hits_val, tb_hits_val, predictions,return_best=True
    )

    # Save to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            params["nBlocks"], params["nFilters"], params["kernel_size"],
            best_thr, Rate_test, best_eff, best_rej
        ])

    return best_thr, Rate_test, best_eff, best_rej


# Parameter grid to sweep
param_grid = [
    
    {"nBlocks": 1, "nFilters": 48, "kernel_size":(4,6)},

    {"nBlocks": 1, "nFilters": 24, "kernel_size":(4,6)},
    {"nBlocks": 1, "nFilters": 12, "kernel_size":(4,6)},
    {"nBlocks": 1, "nFilters": 96, "kernel_size":(4,6)},

    {"nBlocks": 4, "nFilters": 48, "kernel_size":(4,6)},
    {"nBlocks": 2, "nFilters": 48, "kernel_size":(4,6)},
    {"nBlocks": 0, "nFilters": 48, "kernel_size":(4,6)},

    {"nBlocks": 1, "nFilters": 48, "kernel_size":(8,6)},
    {"nBlocks": 1, "nFilters": 48, "kernel_size":(2,6)},

    {"nBlocks": 1, "nFilters": 48, "kernel_size":(4,12)},
    {"nBlocks": 1, "nFilters": 48, "kernel_size":(4,3)}
]

# Default parameters, don't put in grid
default_params = dict(
    nBlocks=1,
    nFilters=48,
    kernel_size=(4, 6)
)

csv_file = "model_results.csv"

end_name=''
sector = 1
nEpoch = 20
filename = f"run5197_sector{sector}.csv"
plotter = Plotter(print_dir="plots/", end_name=end_name,sector=sector)

save_names=['_sector'+str(sector)+'_default'+end_name,
            '_sector'+str(sector)+'_nFilters24'+end_name,
            '_sector'+str(sector)+'_nFilters12'+end_name,
            '_sector'+str(sector)+'_nFilters96'+end_name,
            '_sector'+str(sector)+'_nBlocks4'+end_name,
            '_sector'+str(sector)+'_nBlocks2'+end_name,
            '_sector'+str(sector)+'_nBlocks0'+end_name,
            '_sector'+str(sector)+'_kernelsize86'+end_name,
            '_sector'+str(sector)+'_kernelsize26'+end_name,
            '_sector'+str(sector)+'_kernelsize412'+end_name,
            '_sector'+str(sector)+'_kernelsize43'+end_name
]

# Create CSV if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "nBlocks", "nFilters", "kernel_size",
            "best_threshold", "prediction_rate_kHz", "best_signal_eff", "best_background_rej"
        ])

events = read_sector_file(filename)

dataset = HitsDataset(events)
val_size = 100000
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

all_results = {}
i=0

print('\n\nDo All Tests...')
startT_test = time.time()
for p in param_grid:
    print(f"\n\n ***** Running params: {p}")
    endT_test = time.time()
    print(f'Took {((endT_test-startT_test)/60):.2f}mins so far... *****\n\n')

    thresholds,  Rate_test, signal_eff, background_rej, = run_experiment(
        p, train_loader, val_loader, nEpoch, save_names[i]
    )
    all_results[tuple(p.items())] = (thresholds, Rate_test, signal_eff, background_rej)

    # plot every iteration so don't have to wait til end to see stuff
    # plot each metric vs param (others at default)
    # Signal Efficiency and Background Rejection plotted together

    plotter.plot_metric_vs_param(all_results, "nBlocks", 2, "Signal Efficiency",default_params)
    plotter.plot_metric_vs_param(all_results, "nBlocks", 1, "Prediction Rate (kHz)",default_params)

    plotter.plot_metric_vs_param(all_results, "nFilters", 2, "Signal Efficiency",default_params)
    plotter.plot_metric_vs_param(all_results, "nFilters", 1, "Prediction Rate (kHz)",default_params)

    plotter.plot_metric_vs_param(all_results, "kernel_size", 2, "Signal Efficiency",default_params)
    plotter.plot_metric_vs_param(all_results, "kernel_size", 1, "Prediction Rate (kHz)",default_params)

    i=i+1



endT_test = time.time()
print(f'\n\n ***** Took {((endT_test-startT_test)/60):.2f}mins in total *****\n\n')