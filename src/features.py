from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    X = X.copy()
    self.age_median = X['Age'].median()
    self.fare_median = X['Fare'].median()
    self.embarked_mode = X['Embarked'].mode()[0]
    return self
    
  
  def transform(self, X):
    X = X.copy()

    X['Cabin'] = X['Cabin'].str[:1].fillna('unknown')
    X['Family'] = X['SibSp'] + X['Parch']
    X['Is_alone'] = (X['Family'] == 0).astype(int)
    X['Big_family'] = (X['Family'] > 3).astype(int)
    X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.')


    title_mapping = {
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

    X['Title'] = X['Title'].replace(title_mapping)
    X['Age'] = X['Age'].fillna(self.age_median)
    X['Fare'] = X['Fare'].fillna(self.fare_median)
    X['Embarked'] = X['Embarked'].fillna(self.embarked_mode)

    X['Fare_per_person'] = X['Fare'] / (X['Family'] + 1)
    X['Age_class'] = X['Age'] * X['Pclass']
    X['Fare_class'] = X['Fare'] * X['Pclass']

    X['Name_length'] = X['Name'].apply(len)
    X['Fare_log'] = X['Fare'].apply(lambda x: np.log1p(x))

    return X

# src/features/schema.py

DROP_COLUMNS = [
    "PassengerId",
    # "Name",
    "Ticket",
    'Survived'
    # "Surname",
]

FORCE_CATEGORICAL = [
    'Sex',
    'Embarked',
    'Cabin',
    'Pclass',
    'Is_alone',
    'Big_family',
    'Title',
]

FORCE_NUMERICAL = [
    'Age',
    'Fare',
    'Fare_log',
    'Parch',
    'SibSp',
    'Family',
    'Fare_per_person',
    'Age_class',
    'Fare_class',
    'Name_length',
]

FORCE_ORDINAL =[
   
]

def get_feature_groups(df):
    cat_columns = [col for col in df.columns if col in FORCE_CATEGORICAL]
    ord_columns = [col for col in df.columns if col in FORCE_ORDINAL]
    num_columns = [col for col in df.columns if col in FORCE_NUMERICAL]
    return cat_columns, ord_columns,num_columns