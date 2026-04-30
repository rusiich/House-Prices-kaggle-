from configs import config
from src.utils import set_seed, make_dirs
from src.classic_runner import train
from src.dnn_runner import  run_NN, fit_final_dnn, predict_test_dnn

def fit(config):
    set_seed(config.general.seed)
    make_dirs()

    if config.training.model_name != 'DNN':
        train()
    else:
        cv_result = run_NN()
        final_artifact = fit_final_dnn(cv_result)
        predict_test_dnn(final_artifact)

        
        
if __name__ == '__main__':
    fit(config)