from plotter import Plotter
from trainer import *
from Data  import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
import argparse
import random

def parse_args():
    """
    Parse command-line arguments for training or inference.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including device, input files, hyperparameters, and output options.
    """
    parser = argparse.ArgumentParser(description="Transformer Masked Autoencoder Training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("--sector", type=int, default=1,
                        help="sector of drift chambers")
    parser.add_argument("input", type=str,
                        help="Input CSV file (required)")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs/local",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="default",
                        help="Optional suffix to append to output files")
    parser.add_argument("--nBlocks", type=int, default=2,
                        help="Number of blocks in the convolutional auto-encoder")
    parser.add_argument("--nFilters", type=int, default=48,
                        help="Number of filters in a block of the convolutional auto-encoder")
    parser.add_argument("--kernel_size", type=int, nargs=2, default=[4, 6],
                        help="Kernel size (height width) for convolutional layers, e.g. --kernel_size 4 6 (default: 4 6)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    parser.add_argument("--enable_progress_bar", action="store_true",
                        help="Enable progress bar during training (default: disabled)")
    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    set_seed(42)

    args = parse_args()

    sector = args.sector
    filename = args.input if args.input else f"run5197_sector{sector}.csv"
    outDir = args.outdir
    end_name = args.end_name
    doTraining = not args.no_train
    os.makedirs(outDir, exist_ok=True)
    maxEpochs = args.max_epochs
    batchSize = args.batch_size

    print('\n\nLoading data...')
    startT_data = time.time()

    events = read_sector_file(filename)
    plotter = Plotter(print_dir=outDir, end_name=end_name,sector=sector)
    # Just plot first few events
    for i, (all_hits, tb_hits) in enumerate(events[:3]):
        plotter.plot_hits(all_hits, tb_hits, event_idx=i)


    dataset = HitsDataset(events)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    print('\n\nTrain size:',train_size)
    print('Test size:',val_size)


    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    X_sample, Y_sample = next(iter(train_loader))
    print('X_sample:',X_sample.shape)  # should print: torch.Size([32, 1, 36, 112])
    print('Y_sample:',Y_sample.shape,'\n\n')

    endT_data = time.time()
    T_data = (endT_data - startT_data)
    print(f'Loading data took {T_data:.2f}s \n\n')

    model = LitConvAutoencoder(
        nBlocks=args.nBlocks,
        nFilters=args.nFilters,
        kernel_size=tuple(args.kernel_size),
        lr=args.lr
    )

    loss_tracker = LossTracker()

    if doTraining:
        if args.device == "cpu":
            accelerator = "cpu"; devices = 1
        elif args.device == "gpu":
            if torch.cuda.is_available(): accelerator="gpu"; devices=1
            else: print("GPU not available. Falling back to CPU."); accelerator="cpu"; devices=1
        elif args.device == "auto":
            if torch.cuda.is_available(): accelerator="gpu"; devices="auto"
            else: accelerator="cpu"; devices=1
        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="auto",
            max_epochs=maxEpochs,
            enable_progress_bar=args.enable_progress_bar,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            logger=False,
            callbacks=[loss_tracker]
        )

        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)

        endT_train = time.time()
        T_train = (endT_train - startT_train)/60
        print(f'Training took {T_train:.2f}min \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save model
        model.eval()
        model.to("cpu")
        torchscript_model = torch.jit.script(model)
        torchscript_model.save(f"{outDir}/cnn_autoenc_sector{sector}_{end_name}.pt")

    model_file = f"{outDir}/cnn_autoenc_sector{sector}_{end_name}.pt" if doTraining else "nets/cnn_autoenc_sector1_default.pt"
    model = torch.jit.load(model_file)
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
    plotter.save_efficiency_results(thresholds, signal_eff, background_rej)
    plotter.plot_efficiency_background(thresholds, signal_eff, background_rej)

    # Apply chosen threshold to predictions
    pred_masked = (predictions >= best_threshold).astype(float)

    # Mask pred_masked to only keep hits that exist in all_hits
    pred_masked = pred_masked.astype(int) + all_hits_val.astype(int) - 1

    # Plot first few events
    for i in range(min(5, len(all_hits_val))):
        plotter.plot_hits(all_hits_val[i], tb_hits_val[i], pred_masked[i], event_idx=i)

if __name__ == "__main__":
    main()
