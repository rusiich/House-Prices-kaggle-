from configs import config
from src.utils import set_seed
from src.classic_runner import train
from src.dnn_runner import  run_NN, fit_final_dnn, predict_test_dnn


def fit(config):
    set_seed(config.general.seed)
    if config.training.model_name != 'DNN':
        train()
    else:
        cv_result = run_NN()
        final_artifact = fit_final_dnn(cv_result)
        submission_df = predict_test_dnn(final_artifact)
        submission_df.to_csv(config.paths.path_to_submission / 'predictions.csv', index=False)
        
        
if __name__ == '__main__':
    fit(config)