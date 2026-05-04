from configs import config
from src.utils import set_seed, make_dirs
from src.classic_runner import train_classic
from src.dnn_runner import  run_NN, fit_final_dnn, predict_test_dnn
from src.ensemble import average_proba_ensemble, voting_ensemble

CLASSIC_MODEL_NAMES = [
    'RandomForest', 
    'GradientBoosting', 
    'CatBoost', 
    'KNN',
    'Linear', 
    'Ridge',  
    'Lasso', 
    'ElasticNet',
]


def fit():
    set_seed(config.general.seed)
    make_dirs()

    model_name = config.training.model_name

    if model_name == 'All':
        for classic_name in CLASSIC_MODEL_NAMES:
            config.training.model_name = classic_name
            train_classic()
        return
    
    if model_name == "Ensemble_AVG":
        average_proba_ensemble(["CatBC_v2", "LogR_v2", "KNN_v2", "RFC_v2"])
        return
    
    if model_name == "Ensemble_voting":
        voting_ensemble(["CatBC_v2", "LogR_v2", "KNN_v2", "RFC_v2", "DNN_v2", "GradBC_v2"])
        return
    
    if model_name == "DNN":
        cv_result = run_NN()
        final_artifact = fit_final_dnn(cv_result)
        predict_test_dnn(final_artifact)
        return

    train_classic()
        
        
if __name__ == '__main__':
    fit()