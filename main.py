from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from library.CatDogDataset import *
from library.models import *
from library.train import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

torch.manual_seed(69)

NUM_EPOCH = 10
BATCH_SIZE = 64

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
assert TRAIN_SPLIT + VAL_SPLIT == 1

CLASS_NAME = ["Cat", "Dog"]

full_dataset = CatDogDataset(".\data\PetImages", True)

train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader   = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

model = BatchNormConvNet(device, 2)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, optimizer, criterion, CLASS_NAME, NUM_EPOCH, 
            project_name = "Komugi-Analyzing-Project", 
            ident_str = "BatchNormConv2dBasic")

data_file_name = f"{model.__class__.__name__}.txt"
torch.save(model.state_dict(), data_file_name)

