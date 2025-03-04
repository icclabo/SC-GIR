from functools import partial
from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models.resnet import resnet18
from torchvision.utils import make_grid
import argparse
from torchvision.models.resnet import resnet34, resnet18, resnet50

from transformation import BarlowTwinsTransform, cifar10_normalization, mnist_normalization
from model import BarlowTwins
from loss import BarlowTwinsLoss
import wandb
from pytorch_lightning.loggers import WandbLogger


wandb.init(project="TMC_SemCom", name="barlow_twins_training")
# WandB Logger for PyTorch Lightning
wandb_logger = WandbLogger(project="TMC_SemCom", name="barlow_twins_training")


parser = argparse.ArgumentParser(description='Contrastive Sem Com Training')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=3e-4, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lambd', default=0.0055, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--output_dim', default=2048, type=int,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--dataset', default='cifar10', type=str,
                    metavar='data', help='dataset type')
parser.add_argument('--encoder', default='resnet34', type=str,
                    metavar='encoder', help='encoder model')
parser.add_argument('--num_workers', default=8, type=int,
                    metavar='num_work', help='Number of GPu workers')


def main():
    args = parser.parse_args()
    print(args)

    if args.dataset == 'cifar10':
        train_transform = BarlowTwinsTransform(
        train=True, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization())

        train_dataset = CIFAR10(root=".", train=True, download=True, transform=train_transform)

        val_transform = BarlowTwinsTransform(
            train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=cifar10_normalization()
        )
        val_dataset = CIFAR10(root=".", train=False, download=True, transform=val_transform)

    if args.dataset == 'mnist':
        train_transform = BarlowTwinsTransform(
        train=True, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=mnist_normalization())

        train_dataset = MNIST(root=".", train=True, download=True, transform=train_transform)

        val_transform = BarlowTwinsTransform(
            train=False, input_height=32, gaussian_blur=False, jitter_strength=0.5, normalize=mnist_normalization()
        )
        val_dataset = MNIST(root=".", train=False, download=True, transform=val_transform)


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)



    if args.encoder == 'resnet34':
        encoder = resnet34(pretrained=False)
        if args.dataset == 'cifar10':
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        elif args.dataset == 'mnist':
            encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        
        encoder_out = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif args.encoder == 'resnet18':
        encoder = resnet18(pretrained=False)
        if args.dataset == 'cifar10':
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
            
        elif args.dataset == 'mnist':
            encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        encoder.fc = nn.Identity()
        encoder_out = encoder.fc.in_features
    
    elif args.encoder == 'resnet50':
        encoder = resnet18(pretrained=False)
        if args.dataset == 'cifar10':
            encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
            
        elif args.dataset == 'mnist':
            encoder.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        encoder_out = encoder.fc.in_features
        encoder.fc = nn.Identity()
        

    else:
        raise ValueError('Encoder not found')


    ## model generation     
    model = BarlowTwins(
        encoder = encoder,
        encoder_out_dim = encoder_out,
        num_training_samples = len(train_dataset),
        batch_size = args.batch_size,
        lambda_coeff = args.lambd,
        projection_hidden_dim = int(args.output_dim / 2),
        projection_out_dim = args.output_dim,
        learning_rate = args.lr,
        warmup_epochs = 10,
        max_epochs = args.epochs,
    )
    checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",   # Directory to save checkpoints
    filename="checkpoint_epoch-{epoch}-loss-{val_loss:.2f}-batch-{args.batch_size}-proj-{args.out_dim}-tot_epo-{args.epochs}",  # Naming format
    save_top_k=1,  # Keep only the best model
    monitor="val_loss",  # Track validation loss
    mode="min",  # Save the model with the lowest validation loss
    save_last=True,  # Also save the last model
    verbose=True
)
    
    trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    devices=[0,1,2,3,4,5] if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[#online_finetuner,
        checkpoint_callback],
    logger=wandb_logger,
)
    print('===================Training model====================')

    trainer.fit(model, train_loader,val_loader)#val_loader)
    print('==================Training Finished==================')

    print('==================Saving Model==================')
    best_model_path = checkpoint_callback.best_model_path
    #best_model_path = 'checkpoints/' + 'epoch=230-val_loss=48.2402.ckpt'
    checkpoint = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])


    save_str = f'{args.dataset}_{args.encoder}_outDim-{args.output_dim}_Epo{args.epochs}_barlowtwins.pth'
    torch.save(model.state_dict(), save_str)



    print(f'Model saved as {save_str}')


if __name__ == '__main__':
    main()