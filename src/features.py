from sklearn.base import BaseEstimator, TransformerMixin
from src.schema import DROP_COLUMNS, LOG_FEATURES
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        X = X.copy()

        X['LotFrontage'] = X['LotFrontage'].fillna(0)
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        X['GarageYrBlt'] = X['GarageYrBlt'].fillna(-1)
        X = X.drop(columns=DROP_COLUMNS, errors="ignore")

        for col in LOG_FEATURES:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        return X