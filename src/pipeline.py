
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.features import FeatureEngineer
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from configs import config

RANDOM_STATE = config.general.seed

def build_preprocessor(ohe_columns, ord_columns, num_columns):

  ohe_pipe = Pipeline(
    [('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ohe', OneHotEncoder(drop=None, handle_unknown='ignore'))
    ]
    )
  
  ord_pipe = Pipeline(
    [('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ord',  OrdinalEncoder(
                categories=[
                    [1, 2, 3],
                ],
                handle_unknown='use_encoded_value', unknown_value=np.nan
            )
        ),
     ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
    ]
  )

  num_pipe = Pipeline(
    [('simpleImputer_num', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())
    ]
  )

  data_preprocessor = ColumnTransformer(
      [('ohe', ohe_pipe, ohe_columns),
      ('ord', ord_pipe, ord_columns),
      ('num', num_pipe, num_columns)
      ],
      remainder='drop'
  )
  
  return data_preprocessor



def build_pipeline(ohe_columns, ord_columns, num_columns):
  data_preprocessor = build_preprocessor(ohe_columns, ord_columns, num_columns)

  pipe_final = Pipeline([
      ('feature_engineer', FeatureEngineer()),
      ('preprocessor', data_preprocessor),
      # ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
      ('models', RandomForestClassifier(random_state=RANDOM_STATE)),
  ])
  return pipe_final


param_grid = {
  'RFC':
    
    {
        'models': [RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')],
        'models__max_depth': [4, 5, 6],
        'models__n_estimators': [50, 100, 200],
        'models__min_samples_split': [2, 5],
        'models__min_samples_leaf': [1, 2],
        'models__class_weight': ['balanced', None],
        'preprocessor__num__scaler': ['passthrough']
    },

    # Словарь для модели LogisticRegression

  'LogR':
    {
        'models': [LogisticRegression(random_state=RANDOM_STATE)],
        'models__solver': ['liblinear', 'lbfgs'],
        'models__C': [0.01, 0.1, 1, 10, 100],
        'models__class_weight': [None, 'balanced'],
        'models__max_iter': [500, 1000, 2000],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'],
    },
  'LogR_l1':
    {
        'models': [LogisticRegression(random_state=RANDOM_STATE, penalty='l1')],
        'models__solver': ['liblinear', 'saga'],
        'models__C': [0.01, 0.1, 1, 10, 100],
        'models__class_weight': [None, 'balanced'],
        'models__max_iter': [500, 1000, 2000],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'],
    },

  'LogR_elasticnet':
    {
        'models': [LogisticRegression(random_state=RANDOM_STATE, solver='saga', penalty='elasticnet')],
        'models__C': [0.01, 0.1, 1, 10],
        'models__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'models__class_weight': [None, 'balanced'],
        'models__max_iter': [500, 1000, 2000],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough'],
    },
  
  'KNN':
    {
      'models': [KNeighborsClassifier()],
      'models__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
      'models__weights': ['uniform', 'distance'],
      'models__metric': ['euclidean', 'manhattan', 'minkowski'],
      'models__p': [1, 2],
    },

    # Словарь для модели SVC (Support Vector Classification)
  'SVC':
    {
        'models': [SVC()],
        'models__C': [0.1, 1, 10],
        'models__kernel': ['linear', 'rbf', 'poly'],
        'models__gamma': ['scale', 'auto'],
        'models__degree': [3, 4],  
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough']
    },

    # Словарь для модели LightGBM (Gradient Boosting Classifier)
  # 'LGBMClassifier':
  #   {
  #       'models': [LGBMClassifier(random_state=RANDOM_STATE, class_weight='balanced')],
  #       'models__num_leaves': [31, 70],
  #       'models__max_depth': [5, 10],
  #       'models__learning_rate': [0.01, 0.1, 0.2],
  #       'models__n_estimators': [50, 100, 200],
  #       'models__subsample_for_bin': [200000, 300000],
  #       'models__min_data_in_leaf': [20, 30],
  #       'models__class_weight': ['balanced',  None],
  #       # 'feature_selection__k': [10, 20, 30],
  #       'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough']
  #   },
  'GradBC':
    {
        'models': [GradientBoostingClassifier(random_state=RANDOM_STATE)],
        'models__learning_rate': [0.01, 0.1, 0.2],
        'models__n_estimators': [50, 100, 200],
        'models__max_depth': [3, 5, 10],
        'models__min_samples_split': [2, 5, 10],
        'models__min_samples_leaf': [1, 2, 4],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler(), 'passthrough']
    },

    # CatBoost for Classification
  'CatBC':
    {
        'models': [CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)],
        'models__iterations': range(100, 2000, 100),
        'models__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'models__depth': [4, 6, 8, 10],
        'models__subsample': [0.7, 1.0],
        'preprocessor__num__scaler': [StandardScaler(), MinMaxScaler()]
    },
    'DNN':
    {
      
    }
}

def get_param_grid(model_name='LogisticRegression'):
  return [param_grid[model_name]]