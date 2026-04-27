
from configs import config
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from src.pipeline import build_pipeline
from src.features import get_feature_groups
from src.pipeline import get_param_grid
import pandas as pd
from src.utils import save_classic_model
from src.metrics import compute_accuracy
import torch
from sklearn.model_selection import train_test_split
from src.data import get_data, get_loaders
from src.model import DNN
import torch.nn as nn



device = "cuda" if torch.cuda.is_available() else "cpu"

def train(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.general.seed)
    ohe_columns, ord_columns, num_columns = get_feature_groups(X_train)
    pipe_final = build_pipeline(ohe_columns, ord_columns, num_columns)
    param_grid = get_param_grid(model_name='LogisticRegression')

    randomized_search = RandomizedSearchCV(
        pipe_final,
        param_grid,
        cv=cv,
        scoring='accuracy',
        random_state=config.general.seed,
        n_jobs=-1,
        verbose=1,
        n_iter=10
    )
    randomized_search.fit(X_train, y_train)

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
            correct_samples += torch.sum(indices == y)
            total_samples += y.shape[0]

            loss_accum += loss_value

        ave_loss = loss_accum / (i_step + 1)
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)
        if scheduler is not None:
            scheduler.step(val_accuracy)

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

    return loss_history, train_history, val_history


def run():
    X_train, X_val, y_train, y_val = get_data()

    train_loader, val_loader, input_size, fe, prepr = get_loaders(X_train, y_train, X_val, y_val)


    model = DNN(input=input_size, output=2, p_dropout=0.2).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )


    loss_history, train_history, val_history = train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        scheduler=scheduler,
        num_epochs= 100
        )