import  pandas as pd
from src.features import DROP_COLUMNS
from sklearn.model_selection import train_test_split
from src.features import FeatureEngineer, FORCE_CATEGORICAL, FORCE_NUMERICAL, FORCE_ORDINAL
from src.pipeline import build_preprocessor
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data(config = None):
    data = pd.read_csv('data/train.csv')
    X = data.drop(DROP_COLUMNS, axis=1)
    y = data['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_val, y_train, y_val


def get_loaders(X_train_df, y_train, X_val_df, y_val, batch_size=64):

    fe = FeatureEngineer()
    prepr = build_preprocessor(FORCE_CATEGORICAL, FORCE_ORDINAL, FORCE_NUMERICAL)

    X_train = fe.fit_transform(X_train_df)
    X_val = fe.transform(X_val_df)

    X_train = prepr.fit_transform(X_train)
    X_val = prepr.transform(X_val)

    input_size = X_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, input_size, fe, prepr