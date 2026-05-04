
from configs import config
from sklearn.model_selection import  RandomizedSearchCV, KFold
from src.pipeline import build_pipeline
from src.schema import get_feature_groups, TARGET_COLUMN, ID_COLUMN
from src.search_spaces import get_param_grid
import pandas as pd
from src.utils import save_classic_model,  get_device, log_result
from src.data import get_data
from datetime import datetime


import os
import wandb
import numpy as np




device = get_device()

def train_classic():
    print(f"start training {config.training.model_name}")
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project_name,
            name=f"{config.training.model_name}_{config.general.experiment_name}",
            mode="offline",
            config={
                "model_name": config.training.model_name,
                "seed": config.general.seed,
            }
        )
        print("wandb run started")

    X, y = get_data()

    y = np.log(y)

    cv = KFold(n_splits=5, shuffle=True, random_state=config.general.seed)
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
    predict_test(randomized_search)

    if config.logging.use_wandb:
        wandb.log({
            "cv_best_score": float(randomized_search.best_score_),
        })
        wandb.finish()

    return randomized_search

def predict_test(randomized_search):
    test_df = get_data(test_data=True)
    passenger_ids = test_df[ID_COLUMN]

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(test_df)
    y_pred = np.exp(y_pred)

    submission_df = pd.DataFrame({
        ID_COLUMN: passenger_ids,
        TARGET_COLUMN: y_pred,
    })
        
    name = f"{config.training.model_name}_{config.general.experiment_name}_prediction.csv"
    submission_df.to_csv(config.paths.path_to_submission / name, index=False)
    return submission_df
