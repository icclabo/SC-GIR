import torch 
from pytorch_lightning import LightningModule
import lightning as L
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from channel import Channels


class DownstreamTask(nn.Module):
    def __init__(self, in_features: int, hideen_features_channel=1024, hidden_features_classi=1024, num_classes = 10, compressed_dimension=2048, channel_type='AWGN', SNR=10):
        super().__init__()
        self.encoder_out = in_features
        self.channel_type = channel_type
        self.SNR = SNR
        ## channel Encoder and decoder 
        self.channel_encoder = nn.Sequential(nn.Linear(self.encoder_out, hideen_features_channel),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hideen_features_channel, compressed_dimension))
        

        self.channel = Channels()
        
        self.channel_decoder = nn.Sequential(nn.Linear(compressed_dimension, hideen_features_channel),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hideen_features_channel, self.encoder_out))
        
        self.classifier = Classifier(in_features, hidden_features_classi, num_classes)
 
    def forward(self, x: Tensor) -> Tensor:
        z = self.channel_encoder(x) ## channel encoded version

        if self.channel_type == 'AWGN':
            z_hat = self.channel.AWGN(z, self.SNR) ## channel
        elif self.channel_type == 'Rayleigh':
            z_hat = self.channel.Rayleigh(z, self.SNR) ## channel
        elif self.channel_type == 'Rician':
            z_hat = self.channel.Rician(z,  self.SNR)
        else:
            raise ValueError('Invalid channel type')
        
        z = self.channel_decoder(z_hat) ## channel decoded version
        return self.classifier(z)





### dwonstream task classifier
class Classifier(nn.Module):
    def __init__(self, in_features: int, hidden_features=1024, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)