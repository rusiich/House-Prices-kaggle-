import random
import pprint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from configs import config


RANDOM_STATE = config.general.seed

param_grid = {
    'RandomForest': {
        'models': [RandomForestRegressor(
            random_state=RANDOM_STATE,
        )],
        'models__n_estimators': [200, 400, 800, 1200],
        'models__max_depth': [None, 8, 12, 16, 24],
        'models__max_features': [1.0, 'sqrt', 0.5, 0.7],
        'models__min_samples_split': [2, 5, 10, 20],
        'models__min_samples_leaf': [1, 2, 4, 8],
        'models__bootstrap': [True],
        'models__max_samples': [None, 0.7, 0.85],
        'preprocessor__num__scaler': ['passthrough'],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'Linear':

    {
        'models': [LinearRegression()],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'Ridge': {
        'models': [Ridge(random_state=RANDOM_STATE)],
        'models__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            'passthrough'
        ],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'Lasso': {
        'models': [Lasso(random_state=RANDOM_STATE, max_iter=10000)],
        'models__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler()
        ],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'ElasticNet': {
        'models': [ElasticNet(random_state=RANDOM_STATE, max_iter=10000)],
        'models__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'models__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler()
        ],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'GradientBoosting': {
        'models': [GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            criterion='friedman_mse'
        )],
        'models__loss': ['squared_error', 'huber'],
        'models__n_estimators': [300, 500, 800, 1200, 1600],
        'models__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'models__max_depth': [2, 3, 4, 5],
        'models__subsample': [0.7, 0.85, 1.0],
        'models__min_samples_leaf': [1, 3, 5, 10],
        'models__min_samples_split': [2, 5, 10, 20],
        'models__max_features': [None, 'sqrt', 0.7],
        'preprocessor__num__scaler': ['passthrough'],
        # 'feature_selection__k': range(30, 80, 10),
    },

    'CatBoost': {
        'models': [CatBoostRegressor(
            random_state=RANDOM_STATE,
            verbose=0,
            loss_function='RMSE'
        )],
        'models__iterations': [1000, 2000, 4000, 6000],
        'models__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'models__depth': [4, 5, 6, 7, 8],
        'models__l2_leaf_reg': [1, 3, 5, 7, 10],
        'models__random_strength': [0, 1, 2, 5],
        'models__subsample': [0.66, 0.8, 1.0],
        'models__rsm': [0.7, 0.9, 1.0],
        'preprocessor__num__scaler': ['passthrough'],
        # 'feature_selection__k': range(30, 80, 10),
    },
  
    'KNN': 
    {
        'models': [KNeighborsRegressor()],
        'models__n_neighbors': [3, 5, 7, 9, 15],
        'models__weights': ['uniform', 'distance'],
        'models__p': [1, 2],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()],
        # 'feature_selection__k': range(30, 80, 10),
    },

    # 'LGBM':
    # {
    #     'models': [LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)],
    #     'models__n_estimators': range(100, 1000, 100),
    #     'models__learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'models__max_depth': [-1, 3, 5, 7],
    #     'models__num_leaves': [15, 31, 63],
    #     'models__subsample': [0.7, 1.0],
    #     'models__colsample_bytree': [0.7, 1.0],
    #     'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']
    # },
    'DNN':
    {
       
    },

    
    'DNN_RANDOM': 
    {
        'num_layers': [2, 3, 4, 5, 6],
        'hidden_dim': [64, 128, 256],
        'p_dropout': [0.0,  0.2, ],
        'batch_size': [16, 32, 64, 128],
        'num_epochs': [200],
        'lr': [1e-2, 3e-3, 1e-3],
        'weight_decay': [0.0, 1e-5],
    }
}

def get_param_grid(model_name='LogisticRegression'):
  return [param_grid[model_name]]

def get_random_param_DNN():
    params = get_param_grid(model_name='DNN_RANDOM')[0]


    config.training.num_layers = random.choice(params['num_layers'])
    config.training.hidden_dim = random.choice(params['hidden_dim'])
    config.training.p_dropout = random.choice(params['p_dropout'])
    config.training.batch_size = random.choice(params['batch_size'])
    config.training.num_epochs = random.choice(params['num_epochs'])
    config.training.lr = random.choice(params['lr'])
    config.training.weight_decay = random.choice(params['weight_decay'])

    print(config.training)

    return config

