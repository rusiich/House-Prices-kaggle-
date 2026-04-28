import torch
from src.utils import get_device


device = get_device()

def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval() # Evaluation mode
    # TODO: Implement the inference of the model on all of the batches from loader,
    #       and compute the overall accuracy.
    # Hint: PyTorch has the argmax function!
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