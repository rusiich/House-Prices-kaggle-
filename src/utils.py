
import numpy as np
import pandas as pd
import random
import torch
import os
import matplotlib.pyplot as plt
import joblib
from configs import config
from pathlib import Path
import seaborn as sns

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_dirs():
    for path in config.paths.values():
        p = Path(path)

        if not p.suffix:
            p.mkdir(parents=True, exist_ok=True)


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


def save_NN_model(artifact):
    """Save PyTorch model."""
    config.paths.path_to_NN_model.mkdir(parents=True, exist_ok=True)

    name = f"{config.training.model_name}_{config.general.experiment_name}.pth"
    save_path = config.paths.path_to_NN_model / name

    torch.save(artifact, save_path)

    print(f"NN model saved to: {save_path}")

def save_classic_model(randomized_search):
    '''Save Classic model.'''
    if not os.path.exists(config.paths.path_to_classic_model):
        os.makedirs(config.paths.path_to_classic_model, exist_ok=True)

    name = f"{config.training.model_name}_{config.general.experiment_name}.joblib"
    model_path = config.paths.path_to_classic_model / name  
    joblib.dump(randomized_search, model_path)

    print(f"randomized_search сохранен: {model_path}")

def log_result(record: dict):
    
    record_df = pd.DataFrame([record])

    file_exists =  os.path.exists(config.paths.path_to_leaderboard)
    
    record_df.to_csv(
        config.paths.path_to_leaderboard, 
        mode='a',
        header= not file_exists,
        index=False)


def draw_phik_matrix(df, drop_columns=None):
    num_columns =  df.select_dtypes(include='number').columns.tolist()
    df = df[num_columns]
    plt.figure(figsize=(25, 25))
    if drop_columns is not None:
        df = df.drop(drop_columns, axis=1)
    sns.heatmap(df.phik_matrix(interval_cols=num_columns, bins=20), annot=True, fmt='.2f')
    plt.title('Матрица корреляции')
    plt.show()    