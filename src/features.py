from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    TITLE_MAPPING = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Doctor",
        "Rev": "Religious",
        "Col": "Military",
        "Major": "Military",
        "Capt": "Military",
        "Sir": "Nobility",
        "Lady": "Nobility",
        "Countess": "Nobility",
        "Jonkheer": "Nobility",
        "Don": "Nobility",
        "Mlle": "Miss",
        "Mme": "Mrs",
    }

    def _extract_title(self, X):
        title = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title = title.replace(self.TITLE_MAPPING)
        return title

    def fit(self, X, y=None):
        X = X.copy()

        X['Title'] = self._extract_title(X)

        self.title_age_medians_ = X.groupby('Title')['Age'].median().to_dict()
        self.age_median_ = X['Age'].median()
        self.fare_median_ = X['Fare'].median()
        self.embarked_mode_ = X['Embarked'].mode()[0]

        return self

    def transform(self, X):
        X = X.copy()

        X['Cabin'] = X['Cabin'].str[:1].fillna('unknown')
        X['Family'] = X['SibSp'] + X['Parch']
        X['Is_alone'] = (X['Family'] == 0).astype(int)
        X['Big_family'] = (X['Family'] > 3).astype(int)
        X['Title'] = self._extract_title(X)

        X['Age_was_missing'] = X['Age'].isna().astype(int)
        X['Age'] = X['Age'].fillna(X['Title'].map(self.title_age_medians_))
        X['Age'] = X['Age'].fillna(self.age_median_)
        X['AgeGroup'] = pd.cut(
            X['Age'],
            bins=[0, 12, 18, 35, 50, 65, 100],
            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'MiddleAged', 'Senior']
        )

        X['WomanOrChild'] = ((X['Sex'] == 'female') | (X['Age'] < 12)).astype(int)

        X['Fare'] = X['Fare'].fillna(self.fare_median_)
        X['Embarked'] = X['Embarked'].fillna(self.embarked_mode_)

        X['Fare_per_person'] = X['Fare'] / (X['Family'] + 1)
        X['Age_class'] = X['Age'] * X['Pclass']
        X['Fare_class'] = X['Fare'] * X['Pclass']
        X['Name_length'] = X['Name'].str.len()
        X['Fare_log'] = np.log1p(X['Fare'])

        return X