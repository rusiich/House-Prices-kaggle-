import torch
from src.utils import get_device
import numpy as np


device = get_device()

def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader
    Returns: accuracy as a float value between 0 and 1
    """
    model.eval() 
    correct_samples = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            _, indices = torch.max(prediction, 1)
            correct_samples += (indices == y).sum().item()
            total_samples += y.shape[0]
    accuracy = float(correct_samples) / total_samples


    return accuracy

def compute_rmse(model, loader):
    model.eval() 
    rmse = 0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            preds.append(prediction.detach().cpu())
            targets.append(y.detach().cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    rmse = torch.sqrt(torch.mean((preds - targets)**2)).item()


    return rmse