import  pandas as pd
from src.features import DROP_COLUMNS
from sklearn.model_selection import train_test_split


def get_data(config = None):
    data = pd.read_csv('data/train.csv')
    X = data.drop(DROP_COLUMNS, axis=1)
    y = data['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val