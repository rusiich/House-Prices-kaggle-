from configs import config
from src.utils import set_seed, make_dirs
from src.classic_runner import train_classic
from src.dnn_runner import  run_NN, fit_final_dnn, predict_test_dnn
from src.ensemble import average_ensemble
from src.search_spaces import get_random_param_DNN
import time

CLASSIC_MODEL_NAMES = [
    'RandomForest', 
    'GradientBoosting', 
    # 'CatBoost', 
    'KNN',
    'Linear', 
    'Ridge',  
    'Lasso', 
    'ElasticNet',
]


def fit():
    start = time.perf_counter()
    set_seed(config.general.seed)
    make_dirs()

    model_name = config.training.model_name

    if model_name == 'All':
        for classic_name in CLASSIC_MODEL_NAMES:
            config.training.model_name = classic_name
            start = time.perf_counter()
            train_classic()
            end = time.perf_counter()
            print(f"Время выполнения: {(end - start)/60:.2f} мин")
        return
    
    if model_name == "Ensemble_AVG":
        average_ensemble(
            [
                "CatBoost_v4", 
                'Linear_v4', 
                'Ridge_v4',  
                'Lasso_v4', 
                'ElasticNet_v4',
                # 'KNN_v4', 
                'GradientBoosting_v4', 
                'RandomForest_v4'],
                # weight=[0.4, 0.15, 0.05, 0.2, 0.2,]
            )
        return
    

    
    if model_name == "DNN":
        cv_result = run_NN()
        final_artifact = fit_final_dnn(cv_result)
        predict_test_dnn(final_artifact)
        return
    

    if model_name == "DNN_RANDOM":
        used_params = set()
        count = 0

        while count < config.training.search_n_iter:
            conf = get_random_param_DNN()

            params_key = (
                conf.training.num_layers,
                conf.training.hidden_dim,
                conf.training.p_dropout,
                conf.training.batch_size,
                conf.training.num_epochs,
                conf.training.lr,
                conf.training.weight_decay,
            )

            if params_key in used_params:
                continue
            print(params_key)

            used_params.add(params_key)
            count += 1

            cv_result = run_NN()
            final_artifact = fit_final_dnn(cv_result)
            predict_test_dnn(final_artifact)
            return    


    train_classic()
    end = time.perf_counter()
    print(f"Время выполнения: {(end - start)/60:.2f} мин")
        
if __name__ == '__main__':
    fit()