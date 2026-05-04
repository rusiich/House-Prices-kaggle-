
from configs import config
from sklearn.model_selection import KFold
import pandas as pd
from src.utils import  save_NN_model, get_device, log_result
from src.metrics import compute_accuracy, compute_rmse
import torch
from src.data import get_data, get_loaders
from src.model import DNN
from src.schema import ID_COLUMN, TARGET_COLUMN
import torch.nn as nn
import numpy as np
import copy
from datetime import datetime
import os



device = get_device()

def train_model(model, train_loader, val_loader, loss, optimizer,scheduler=None, num_epochs=10):
    loss_history = []
    train_history = []
    val_history = []
    best_val_metric = float("inf")
    best_epoch = -1
    best_model_state = None
    for epoch in range(num_epochs):
        model.train() # Enter train mode

        loss_accum = 0
        for i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            loss_accum += loss_value.item()

        ave_loss = loss_accum / (i_step + 1)
        train_rmse = compute_rmse(model, train_loader)
        val_rmse = compute_rmse(model, val_loader)

        if val_rmse < best_val_metric:
            best_val_metric = val_rmse
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        loss_history.append(float(ave_loss))
        train_history.append(train_rmse)
        val_history.append(val_rmse)

        if scheduler is not None:
            scheduler.step(val_rmse)

        print("Average loss: %f, Train rrmse: %f, Val rmse: %f" % (ave_loss, train_rmse, val_rmse))

    return {
                "loss_history": loss_history,
                "train_history": train_history,
                "val_history": val_history,
                "best_val_metric": best_val_metric,
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
            }



def fit_final_dnn(cv_result):

    X, y = get_data()
    y = np.log(y)

    loader, input_size, fe, prepr = get_loaders(X, y, )
    
    model = DNN(
        input_size=input_size, 
        output_size=config.training.output_size, 
        p_dropout=config.training.p_dropout,
        ).to(device)
    
    num_epochs = round(cv_result["mean_best_epoch"])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.lr, 
        weight_decay=config.training.weight_decay
        )


    for epoch in range(num_epochs):
        model.train() # Enter train mode
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        print(f'epoch № {epoch}, loss_value {loss_value}, ')

    artifact = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "fe": fe,
        "prepr": prepr,
        "input_size": input_size,
        "output_size": config.training.output_size,
        "p_dropout": config.training.p_dropout,
    }

    save_NN_model(artifact)

    return artifact


 

def run_NN():
    print('start DNN model training')
    X, y = get_data()
    y = np.log(y)

    skf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=config.general.seed
    )


    fold_results = []
    best_fold_result = None
    best_fold_metric = float("inf")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        train_loader, val_loader, input_size, fe, prepr = get_loaders(
                                                            X_train_fold, 
                                                            y_train_fold, 
                                                            X_val=X_val_fold, 
                                                            y_val=y_val_fold
                                                            )
        
        model = DNN(
                input_size=input_size, 
                output_size=config.training.output_size, 
                p_dropout=config.training.p_dropout,
                ).to(device)
        
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(
                            model.parameters(), 
                            lr=config.training.lr,
                            weight_decay=config.training.weight_decay,
                            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )

        train_result = train_model(
            model,
            train_loader,
            val_loader,
            loss,
            optimizer,
            scheduler=scheduler,
            num_epochs= config.training.num_epochs
            )
        
        fold_result = {
            "fold": fold,
            "input_size": input_size,
            "best_val_metric": train_result["best_val_metric"],
            "best_epoch": train_result["best_epoch"],
            "best_model_state": train_result["best_model_state"],
        }
        fold_results.append(fold_result)

        if best_fold_metric > fold_result["best_val_metric"] :
            best_fold_metric = fold_result["best_val_metric"]
            best_fold_result = {
                **fold_result,
                "fe": fe,
                "prepr": prepr,
            }
            
        print(f"Fold {fold}: best_rmse={fold_result['best_val_metric']:.4f}, best_epoch={fold_result['best_epoch']}")

    metrics = []
    epochs = []

    for fold in fold_results:
        metrics.append(fold['best_val_metric'])
        epochs.append(fold['best_epoch'])
    
    dnn_params = {
        "batch_size": config.training.batch_size,
        "mean_best_epoch": np.mean(epochs),
        "num_layers": config.training.num_layers,
        "lr": config.training.lr,
        "weight_decay": config.training.weight_decay,
        "p_dropout": config.training.p_dropout,
        "scheduler_factor": config.training.scheduler_factor,
        "scheduler_patience": config.training.scheduler_patience,
    }
    
    record = {
        "created_at": datetime.now().isoformat(),
        "experiment_name": config.general.experiment_name,
        "model_name": config.training.model_name,
        "score": np.mean(metrics),
        "params": dnn_params,
        "scoring": config.training.scoring,
        "seed": config.general.seed,       

    }

    log_result(record)
        
    return {
        "fold_results": fold_results,
        "cv_mean_rmse": np.mean(metrics),
        "cv_std_rmse": np.std(metrics),
        "mean_best_epoch": np.mean(epochs),
        "best_fold_result": best_fold_result,
    }

    
def predict_test_dnn(artifact):
    test_df = get_data(test_data=True)
    passenger_ids = test_df[ID_COLUMN]

    fe = artifact['fe']
    prepr = artifact['prepr']
    X = fe.transform(test_df)
    X = prepr.transform(X)
    X = X.toarray() if hasattr(X, "toarray") else X


    X = torch.tensor(X, dtype=torch.float32).to(device)

    model = DNN(
        input_size=artifact["input_size"], 
        output_size=artifact["output_size"], 
        p_dropout=artifact["p_dropout"]
        ).to(device)
    
    model.load_state_dict(artifact['model_state_dict'])
    model.eval()

    with torch.no_grad(): 
        preds = model(X).squeeze().cpu().numpy()
    
    preds = np.exp(preds)

    submission_df = pd.DataFrame({
        ID_COLUMN: passenger_ids,
        TARGET_COLUMN: preds,
    })
        
    name = f"{config.training.model_name}_{config.general.experiment_name}_prediction.csv"
    submission_df.to_csv(config.paths.path_to_submission / name, index=False)
    return submission_df





