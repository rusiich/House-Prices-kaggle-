from src.data import get_data
from configs import config 
import joblib
import pandas as pd
import numpy as np
import torch
from src.model import DNN

def average_proba_ensemble(models_name: list, weight:list = None):
    test_df = get_data(test_data=True)
    passenger_ids = test_df["PassengerId"]

    X_test = test_df.drop(columns=["PassengerId"], errors="ignore")

    preds=[]
    
    for model_name in models_name:
        name = f"{model_name}.joblib"
        model_path = config.paths.path_to_classic_model / name  
        rs = joblib.load(model_path)
        best_model = rs.best_estimator_ if hasattr(rs, "best_estimator_") else rs
        preds.append(best_model.predict_proba(X_test)[:, 1])
    
    preds = np.array(preds)

    if weight is None:
        pred_proba = preds.mean(axis=0)
    else:
        weight = np.asarray(weight, dtype=float)

        assert len(weight)==len(models_name)
        assert np.isclose(weight.sum(),1.0)

        pred_proba = np.average(preds, axis=0, weights=weight)

    pred = (pred_proba >= 0.5).astype(int)

    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": pred,
    })
    name = f"average_proba_ensemble_{config.general.experiment_name}_prediction.csv"
    submission_df.to_csv(config.paths.path_to_submission / name, index=False)
    print('average_proba_ensemble_ensemble')
    return submission_df


def voting_ensemble(models_name: list):
    test_df = get_data(test_data=True)
    passenger_ids = test_df["PassengerId"]

    X_test = test_df.drop(columns=["PassengerId"], errors="ignore")

    preds=[]
    
    for model_name in models_name:
        if model_name != 'DNN_v2':

            name = f"{model_name}.joblib"
            model_path = config.paths.path_to_classic_model / name
            rs = joblib.load(model_path)
            model = rs.best_estimator_ if hasattr(rs, "best_estimator_") else rs
            pred = model.predict(X_test)
            pred = np.asarray(pred).ravel()
        else:
            name = f"DNN_v2.pth"
            model_path = config.paths.path_to_NN_model / name
            artifact = torch.load(model_path, map_location="cpu")

            fe = artifact["fe"]
            prepr = artifact["prepr"]

            X_nn = fe.transform(X_test)

            X_nn = prepr.transform(X_nn)

            if hasattr(X_nn, "toarray"):
                X_nn = X_nn.toarray()

            X_nn = np.asarray(X_nn, dtype=np.float32)

            model =  DNN(
                input_size=artifact["input_size"],
                output_size=artifact["output_size"],
                p_dropout=artifact["p_dropout"],
            )
            model.load_state_dict(artifact["model_state_dict"])
            model.eval()

            with torch.no_grad():
                X_tensor = torch.tensor(X_nn, dtype=torch.float32)
                logits = model(X_tensor)
                pred = torch.argmax(logits, dim=1).cpu().numpy().ravel()

        preds.append(pred)

    preds = np.vstack(preds) 
    pred = (np.mean(preds, axis=0) >= 0.5).astype(int)


    submission_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": pred,
    })

    name = f"voting_ensemble_{config.general.experiment_name}_prediction.csv"
    submission_df.to_csv(config.paths.path_to_submission / name, index=False)
    print('Saved voting_ensemble')
    return submission_df
    

