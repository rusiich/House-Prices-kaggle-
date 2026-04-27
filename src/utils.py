
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
import joblib

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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

def visualize_confusion_matrix(predictions, ground_truth):
    """
    Visualizes confusion matrix
                     
    """
    # Adapted from 
    # https://stackoverflow.com/questions/2897826/confusion-matrix-with-number-of-classified-misclassified-instances-on-it-python
    

    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    size = max(len(np.unique(predictions)), len(np.unique(ground_truth)))
    confusion_matrix = np.zeros((size, size), dtype='int64')

    
    for pred, true in zip(predictions, ground_truth):
        confusion_matrix[true, pred] += 1

    fig = plt.figure(figsize=(size, size))
    plt.title("Confusion matrix")
    plt.ylabel("predicted")
    plt.xlabel("ground truth")
    res = plt.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
    cb = fig.colorbar(res)
    plt.xticks(np.arange(size))
    plt.yticks(np.arange(size))
    for i, row in enumerate(confusion_matrix):
        for j, count in enumerate(row):
            plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')



def save_NN_model(config, model, current_metric, optimizer, 
                       name, scheduler):
    '''Save PyTorch model.'''

    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints, exist_ok=True)

    torch.save({
        'model': model.state_dict(),
        # 'epoch': epoch,
        'metric': current_metric,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, os.path.join(config.paths.path_to_checkpoints, name))

def save_classic_model(config, randomized_search):
    '''Save Classic model.'''
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints, exist_ok=True)
    
    search_path = config.paths.path_to_checkpoints + '/' + config.training.model_name + '_' + config.general.experiment_name + ".joblib"
    joblib.dump(randomized_search, search_path)