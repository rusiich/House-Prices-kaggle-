
from configs import config
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.pipeline import build_pipeline
from src.features import get_feature_groups
from src.pipeline import get_param_grid
import pandas as pd
from src.utils import save_classic_model, save_NN_model, get_device
from src.metrics import compute_accuracy
import torch
from sklearn.model_selection import train_test_split
from src.data import get_data, get_loaders
from src.model import DNN
import torch.nn as nn
import numpy as np
import copy



device = get_device()

def train():
    X, y = get_data()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.general.seed)
    ohe_columns, ord_columns, num_columns = get_feature_groups(X)
    pipe_final = build_pipeline(ohe_columns, ord_columns, num_columns)
    param_grid = get_param_grid(model_name=config.training.model_name)

    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=cv,
        scoring='accuracy',
        random_state=config.general.seed,
        n_jobs=-1,
        verbose=1,
        n_iter=50
    )
    randomized_search.fit(X, y)

    best_model = randomized_search.best_estimator_

    print('Лучшая модель и её параметры:\n\n', best_model)
    print ('Метрика лучшей модели на кросс-валидации:', randomized_search.best_score_)

    result = pd.DataFrame(randomized_search.cv_results_)
    print(result[
        ['rank_test_score', 'param_models', 'mean_test_score','params']
    ].sort_values('rank_test_score')[:10])
    
    save_classic_model(config, randomized_search)
    return randomized_search

def train_model(model, train_loader, val_loader, loss, optimizer,scheduler=None, num_epochs=10):
    loss_history = []
    train_history = []
    val_history = []
    best_val_accuracy = -1
    best_epoch = -1
    best_model_state = None
    for epoch in range(num_epochs):
        model.train() # Enter train mode

        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            _, indices = torch.max(prediction, 1)
            correct_samples += (indices == y).sum().item()
            total_samples += y.shape[0]

            loss_accum += loss_value.item()

        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        if scheduler is not None:
            scheduler.step(val_accuracy)

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return {
                "loss_history": loss_history,
                "train_history": train_history,
                "val_history": val_history,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "best_model_state": best_model_state,
            }



def fit_final_dnn(cv_result):

    X, y = get_data()

    loader, input_size, fe, prepr = get_loaders(X, y, )
    
    model = DNN(input=input_size, output=2, p_dropout=0.2).to(device)
    num_epochs = int(round(cv_result["mean_best_epoch"]))
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)


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
        "output_size": 2,
        "p_dropout": 0.2,
    }
    return artifact


 

def run_NN():
    X, y = get_data()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.general.seed
    )


    fold_results = []
    best_fold_result = None
    best_fold_acc = -1

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
        
        model = DNN(input=input_size, output=2, p_dropout=0.2).to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
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
            "best_val_accuracy": train_result["best_val_accuracy"],
            "best_epoch": train_result["best_epoch"],
            "best_model_state": train_result["best_model_state"],
        }
        fold_results.append(fold_result)

        if best_fold_acc < fold_result["best_val_accuracy"] :
            best_fold_acc = fold_result["best_val_accuracy"]
            best_fold_result = {
                **fold_result,
                "fe": fe,
                "prepr": prepr,
            }
            
        print(f"Fold {fold}: best_acc={fold_result['best_val_accuracy']:.4f}, best_epoch={fold_result['best_epoch']}")

    accuracies = []
    epochs = []

    for fold in fold_results:
        accuracies.append(fold['best_val_accuracy'])
        epochs.append(fold['best_epoch'])
        
    return {
        "fold_results": fold_results,
        "cv_mean_accuracy": np.mean(accuracies),
        "cv_std_accuracy": np.std(accuracies),
        "mean_best_epoch": np.mean(epochs),
        "best_fold_result": best_fold_result,
    }

    
def predict_test_dnn(artifact):
    test_df = get_data(test_data=True)
    passenger_ids = test_df["PassengerId"]

    fe = artifact['fe']
    prepr = artifact['prepr']
    X = fe.transform(test_df)
    X = prepr.transform(X)

    X = torch.tensor(X, dtype=torch.float32).to(device)

    model = DNN(
        input=artifact["input_size"], 
        output=artifact["output_size"], 
        p_dropout=artifact["p_dropout"]
        ).to(device)
    
    model.load_state_dict(artifact['model_state_dict'])
    model.eval()

    with torch.no_grad(): 
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()

    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds.astype(int),
    })

    return submission_df





