
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import joblib
from configs import config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_plots(train_losses, val_losses, metrics, lr_changes):

    # Learning rate changes
    plt.plot(range(len(lr_changes)), lr_changes, label='Learning Rate')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Changes of validation and train losses')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()


def save_NN_model(config, artifact):
    """Save PyTorch model."""
    config.paths.path_to_checkpoints.mkdir(parents=True, exist_ok=True)

    name = f"{config.training.model_name}_{config.general.experiment_name}.pth"
    save_path = config.paths.path_to_checkpoints / name

    torch.save(artifact, save_path)

    print(f"NN model saved to: {save_path}")

def save_classic_model(config, randomized_search):
    '''Save Classic model.'''
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints, exist_ok=True)

    name = f"{config.training.model_name}_{config.general.experiment_name}.joblib"
    model_path = config.paths.path_to_checkpoints / name  
    joblib.dump(randomized_search, model_path)

    print(f"randomized_search сохранен: {model_path}")