from configs import config
from src.utils import set_seed
from src.train import train
from src.data import get_data
import pandas as pd

def fit(config):
    set_seed(config.general.seed)
    X_train, X_val, y_train, y_val = get_data()
    
    randomized_search = train(X_train, y_train)
    best_model = randomized_search.best_estimator_

    print('Лучшая модель и её параметры:\n\n', best_model)
    print ('Метрика лучшей модели на кросс-валидации:', randomized_search.best_score_)

    result = pd.DataFrame(randomized_search.cv_results_)
    print(result[
        ['rank_test_score', 'param_models', 'mean_test_score','params']
    ].sort_values('rank_test_score')[:10])
    

if __name__ == '__main__':
    fit(config)