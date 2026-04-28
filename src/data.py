import  pandas as pd
from src.features import FeatureEngineer, FORCE_CATEGORICAL, FORCE_NUMERICAL, FORCE_ORDINAL
from src.pipeline import build_preprocessor
import torch
from torch.utils.data import DataLoader, TensorDataset
from configs import config


def get_data(test_data=False):
    
    path = config.paths.path_to_test_data if test_data else  config.paths.path_to_train_data

    data = pd.read_csv(path)
    if 'Survived' in data.columns:
        X = data.drop('Survived', axis=1)
        y = data['Survived']

        assert 'Survived' not in X.columns
        return X, y
    
    return data


def get_loaders(X_train_df, y_train, X_val=None, y_val=None, fe=None, prepr=None, batch_size=config.training.batch_size):
    
    if fe is None:
        fe = FeatureEngineer()
    
    if prepr is None:
        prepr = build_preprocessor(FORCE_CATEGORICAL, FORCE_ORDINAL, FORCE_NUMERICAL)

    X_train = fe.fit_transform(X_train_df)
    X_train = prepr.fit_transform(X_train)

    input_size = X_train.shape[1]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    if X_val is not None:
        X_val = fe.transform(X_val)
        X_val = prepr.transform(X_val)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, input_size, fe, prepr
    
    return train_loader, input_size, fe, prepr