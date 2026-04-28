from configs import config
from src.utils import set_seed
from src.train import train, run_NN, fit_final_dnn, predict_test_dnn
from src.data import get_data
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def fit(config):
    set_seed(config.general.seed)


    
    if config.training.model_name != 'DNN':
        train()
    else:
        cv_result = run_NN()
        final_artifact = fit_final_dnn(cv_result)
        submission_df = predict_test_dnn(final_artifact)
        
        
if __name__ == '__main__':
    fit(config)