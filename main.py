from configs import config
from src.utils import set_seed
from src.train import train, run
from src.data import get_data
import pandas as pd

def fit(config):
    set_seed(config.general.seed)
    X_train, X_val, y_train, y_val = get_data()
    
    if config.training.model_name != 'DNN':
        randomized_search = train(X_train, y_train)
    else:
        run()
        


    

if __name__ == '__main__':
    fit(config)