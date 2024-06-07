from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
from tqdm import tqdm
import wandb
import torchmetrics
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

def plot_confusion_matrix(cm, class_names):
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    model.train()
    train_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes).to(device)
    UAR_train = torchmetrics.Recall(task = "multiclass", num_classes = num_classes, average = 'macro').to(device)
    average_epoch_loss = torchmetrics.MeanMetric().to(device)

    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, lbls)
        average_epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        average_epoch_loss(loss)
        train_accuracy(outputs, lbls)
        UAR_train(outputs, lbls)

    metrics_dict = {
        'Loss_train': average_epoch_loss.compute(),
        'Accuracy_train': train_accuracy.compute(),
        'UAR_train': UAR_train.compute().detach().cpu().numpy()
    }

    return metrics_dict

def val_epoch(epoch, model, criterion, loader, num_classes, device):
    model.eval()

    train_accuracy = torchmetrics.Accuracy(task = "multiclass", num_classes = num_classes).to(device)
    UAR_train = torchmetrics.Recall(task = "multiclass", num_classes = num_classes, average = 'macro').to(device)
    average_epoch_loss = torchmetrics.MeanMetric().to(device)

    confused_matrix = torchmetrics.ConfusionMatrix(task = 'multiclass', num_classes = num_classes).to(device)
    
    for inputs, lbls in loader:
        inputs, lbls = inputs.to(device), lbls.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, lbls)

        average_epoch_loss(loss)
        train_accuracy(outputs, lbls)
        UAR_train(outputs, lbls)
    
        confused_matrix(outputs, lbls)
         
    metrics_dict = {
        'Loss_val': average_epoch_loss.compute(),
        'Accuracy_val': train_accuracy.compute(),
        'UAR_val': UAR_train.compute().detach().cpu().numpy()
    }

    cm = confused_matrix.compute().detach().cpu().numpy()

    return metrics_dict, cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)
    
    if ident_str is None:
      ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)

    try:
        for epoch in tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
                    
            val_metrics_dict, cm = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)
            wandb.log({**train_metrics_dict, **val_metrics_dict})
    finally:
        run.finish()

    plot_confusion_matrix(cm = cm, class_names = class_names)
