import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pytorch_lightning.callbacks import Callback

class HitsDataset(Dataset):
    def __init__(self, events):
        # events is a list of (all_hits, tb_hits) numpy arrays
        X_list, Y_list = zip(*events)
        X = np.array(X_list, dtype=np.float32)  # (N, 36, 112)
        Y = np.array(Y_list, dtype=np.float32)

        # Add channel dimension (N, 1, 36, 112)
        self.X = X[:, np.newaxis, :, :]
        self.Y = Y[:, np.newaxis, :, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

class LitConvAutoencoder(pl.LightningModule):
    """PyTorch Lightning convolutional autoencoder"""
    def __init__(self, nBlocks=2,nFilters=48,kernel_size=(4, 6),lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        

        #if nBlocks =0, just have 1 encoder and 1 decoder
        self.nBlocks = nBlocks
        self.nFilters = nFilters
        self.kernel_size=kernel_size
        self.lr = lr

        #fix poolsize, as we're pooling/upsampling
        #with 36 layers, quickly becomes messy for larger 
        # poolsize with several blocks
        self.poolsize=2


        self.pool = nn.MaxPool2d((self.poolsize, self.poolsize))
        self.up = nn.Upsample(scale_factor=self.poolsize, mode='nearest')

        self.encoders = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # has channel 1
        self.first=nn.Conv2d(1, nFilters, kernel_size=self.kernel_size, padding='same')
        # Dropout for first encoder
        self.firstdp=nn.Dropout2d(p=0.1)

        # Dropout for second encoder (#might not use it)
        self.seconddp=nn.Dropout2d(p=0.1)

        # Additional encoder blocks (same in/out channels)
        for _ in range(nBlocks):
            self.encoders.append(nn.Conv2d(nFilters, nFilters, kernel_size=self.kernel_size, padding='same'))

        #last encoder, has no pool after, needed for symmetry
        self.middle=nn.Conv2d(nFilters, nFilters, kernel_size=self.kernel_size, padding='same')

        # Additional decoder blocks (same in/out channels)
        self.decoders = nn.ModuleList()
        for _ in range(nBlocks):
            self.decoders.append(nn.Conv2d(nFilters, nFilters, kernel_size=self.kernel_size, padding='same'))

        # Final output layer: back to 1 channel
        self.final = nn.Conv2d(nFilters, 1, kernel_size=self.kernel_size, padding='same')

        self.criterion = nn.BCELoss()

    def forward(self, x):
        # Encoder
        x = F.relu(self.first(x))
        x = self.firstdp(x)
        x = self.pool(x)

        for i, enc in enumerate(self.encoders):
            x = F.relu(enc(x))
            #only put dropout & pool for second encoder (eg first block)
            if i==0:
                x = self.seconddp(x)
                x = self.pool(x)

        #have at least one layer after pool
        x = F.relu(self.middle(x))

        # Decoder (no dropout)
        for i, dec in enumerate(self.decoders):
            # only last decoder block upsamples
            if i == len(self.decoders) - 1:  
              x = self.up(x)
            x = F.relu(dec(x))

        # symmetry with encoder
        x = self.up(x)  
        x = torch.sigmoid(self.final(x))
        return x
    
    # def forward(self, x): 
    #   x = F.relu(self.enc1(x)) 
    #   x = self.dropout1(x) 
    #   x = self.pool(x)

    #   x = F.relu(self.enc2(x))
    #   x = self.dropout2(x)
    #   x = self.pool(x)

    #   x = F.relu(self.enc3(x))

    #   x = self.up(x)
    #   x = F.relu(self.dec1(x))

    #   x = self.up(x)
    #   x = torch.sigmoid(self.dec2(x)) 
    #   return x

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        loss = self.criterion(Y_hat, Y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LossTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())