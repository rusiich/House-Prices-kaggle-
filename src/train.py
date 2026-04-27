
from configs import config
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.pipeline import build_pipeline
from src.features import get_feature_groups
from src.pipeline import get_param_grid


def train(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.general.seed)
    ohe_columns, ord_columns, num_columns = get_feature_groups(X_train)
    pipe_final = build_pipeline(ohe_columns, ord_columns, num_columns)
    param_grid = get_param_grid(model_name='LogisticRegression')

    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=cv,
        scoring='accuracy',
        random_state=config.general.seed,
        n_jobs=-1,
        verbose=1,
        n_iter=1
    )
    randomized_search.fit(X_train, y_train)

    return randomized_search