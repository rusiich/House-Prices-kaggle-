from src.data import get_data
from configs import config 
import joblib
import pandas as pd
import numpy as np

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
    name = "average_proba_ensemble_prediction.csv"
    submission_df.to_csv(config.paths.path_to_submission / name, index=False)
    return submission_df



    

