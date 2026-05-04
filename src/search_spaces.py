
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
    'RandomForest':
    {
        'models': [RandomForestRegressor(random_state=RANDOM_STATE)],
        'models__n_estimators': range(50, 300, 50),
        'models__max_depth': range(3, 15, 3),
        'preprocessor__num__scaler': ['passthrough']
    },

    'Linear':

    {
        'models': [LinearRegression()],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()]
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
        ]
    },

    'Lasso': {
        'models': [Lasso(random_state=RANDOM_STATE, max_iter=10000)],
        'models__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'models__fit_intercept': [True, False],
        'preprocessor__num__scaler': [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler()
        ]
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
        ]
    },

    'GradientBoosting':

    {
        'models': [GradientBoostingRegressor(random_state=RANDOM_STATE)],
        'models__n_estimators': range(100, 1000, 100),
        'models__learning_rate': [0.01, 0.5, 0.1, 0.2],
        'models__max_depth': [3, 5, 7],
        'models__subsample': [0.7, 1.0],
        'preprocessor__num__scaler': ['passthrough']
    },

    'CatBoost': 
    {
        'models': [CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)],
        'models__iterations': range(100, 2000, 100),
        'models__learning_rate': [0.01, 0.5, 0.1, 0.2],
        'models__depth': [4, 6, 8, 10],
        'models__subsample': [0.7, 1.0],
        'preprocessor__num__scaler': ['passthrough']
    },
  
    'KNN': 
    {
        'models': [KNeighborsRegressor()],
        'models__n_neighbors': [3, 5, 7, 9, 15],
        'models__weights': ['uniform', 'distance'],
        'models__p': [1, 2],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()]
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
      
    }
}

def get_param_grid(model_name='LogisticRegression'):
  return [param_grid[model_name]]