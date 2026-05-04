from sklearn.base import BaseEstimator, TransformerMixin
from src.schema import DROP_COLUMNS

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()
        return self

    def transform(self, X):
        X = X.copy()

        X['LotFrontage'] = X['LotFrontage'].fillna(0)
        X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        X['GarageYrBlt'] = X['GarageYrBlt'].fillna(-1)
        X = X.drop(columns=DROP_COLUMNS)
        return X