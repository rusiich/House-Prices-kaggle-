
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.features import FeatureEngineer
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


