
from configs import config
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.pipeline import build_pipeline
from src.features import get_feature_groups
from src.search_spaces import get_param_grid
import pandas as pd
from src.utils import save_classic_model,  get_device, log_result
from src.data import get_data
from datetime import datetime



device = get_device()

def train():
    X, y = get_data()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.general.seed)
    ohe_columns, ord_columns, num_columns = get_feature_groups()
    pipe_final = build_pipeline(ohe_columns, ord_columns, num_columns)
    param_grid = get_param_grid(model_name=config.training.model_name)

    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=cv,
        scoring=config.training.scoring,
        random_state=config.general.seed,
        n_jobs=config.training.search_n_jobs,
        verbose=config.training.search_verbose,
        n_iter=config.training.search_n_iter
    )
    randomized_search.fit(X, y)

    best_model = randomized_search.best_estimator_

    print('Лучшая модель и её параметры:\n\n', best_model)
    print ('Метрика лучшей модели на кросс-валидации:', randomized_search.best_score_)

    result = pd.DataFrame(randomized_search.cv_results_)
    print(result[
        ['rank_test_score', 'param_models', 'mean_test_score','params']
    ].sort_values('rank_test_score')[:10])
    

    save_classic_model(randomized_search)

    record = {
        "created_at": datetime.now().isoformat(),
        "experiment_name": config.general.experiment_name,
        "model_name": config.training.model_name,
        "score": float(randomized_search.best_score_),
        "params": str(randomized_search.best_params_),
        "scoring": config.training.scoring,
        "seed": config.general.seed,
    }

    log_result(record)

    return randomized_search