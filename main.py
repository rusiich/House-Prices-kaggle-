from configs import config
from src.utils import set_seed, make_dirs
from src.classic_runner import train
from src.dnn_runner import  run_NN, fit_final_dnn, predict_test_dnn
from src.search_spaces import param_grid
from src.ensemble import average_proba_ensemble

def fit(config):
    set_seed(config.general.seed)
    make_dirs()
    if config.training.training_all_models:
        for model_name in param_grid:
            config.training.model_name = model_name
            if config.training.model_name != 'DNN':
                train()
            elif config.training.model_name != 'Ensemble_AVG':
                models_name = [
                'CatBC_baseline',
                    'LogR_baseline',
                    'KNN_baseline',
                    'RFC_baseline',
                ]

                average_proba_ensemble(models_name)
            else:
                cv_result = run_NN()
                final_artifact = fit_final_dnn(cv_result)
                predict_test_dnn(final_artifact)
    else:
        if config.training.model_name == 'Ensemble_AVG':
            models_name = [
            'CatBC_baseline',
            'LogR_baseline',
            'KNN_baseline',
            'RFC_baseline',
            ]
            average_proba_ensemble(models_name)
        elif config.training.model_name != 'DNN':
            train()
        else:
            cv_result = run_NN()
            final_artifact = fit_final_dnn(cv_result)
            predict_test_dnn(final_artifact)
        
        
if __name__ == '__main__':
    fit(config)