import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# from autoaugment import CIFAR10Policy

import matplotlib.pyplot as plt


def get_cifar_dataloaders(batch_size):
    # transforms
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    train_trans = transforms.Compose([
        transforms.Pad(4, padding_mode="reflect"),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.3, .3, .3),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

    # datasets
    root = "../data"
    train_ds = datasets.CIFAR10(root, train=True, transform=train_trans, download=True)
    val_ds = datasets.CIFAR10(root, train=False, transform=val_trans, download=True)

    # dataloader

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return train_dl, val_dl