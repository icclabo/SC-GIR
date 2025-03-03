## core encoder and projection head models
from functools import partial
from typing import Sequence, Tuple, Union

import lightning as L
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet34
from torchvision.utils import make_grid
from torchvision import models
from pytorch_lightning.loggers import CSVLogger
from loss import BarlowTwinsLoss
from utils import linear_warmup_decay, fn   
import wandb

##projection_head archtechture
class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, hidden_features=1024, out_feature=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_feature),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)



## enoder archtechture
class BarlowTwins(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        projection_hidden_dim=1024,
        projection_out_dim=2048,
        learning_rate=3e-4,
        warmup_epochs=10,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(in_features=encoder_out_dim, hidden_features=projection_hidden_dim, out_feature=projection_out_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=projection_out_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        loss, on_diag, off_diag= self.loss_fn(z1, z2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar= True, logger=True)
        wandb.log({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"train_loss": loss})  # Log directly to WandB
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]