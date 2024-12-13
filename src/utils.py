import pandas as pd
from src.consts import OUTPUTDIRECTORY
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import numpy as np


def prediction_on_folds(cv_obj, df, target):
    targets = []
    preds = []
    for est, indx in zip(cv_obj["estimator"], cv_obj["indices"]["test"]):
        df_test = df.iloc[indx, :]
        target_test = target.iloc[indx]
        pred = est.predict(df_test)
        targets.append(pd.DataFrame(target_test))
        preds.append(pd.DataFrame(pred))
    preds = pd.concat(preds)
    targets = pd.concat(targets)
    return preds, targets


def train_nocv(est, df, target):
    est.fit(df, target)
    return est


def compute_metrics_by_method(csvpath, substrate="implicit", filename=None):
    results = pd.read_csv(csvpath)
    results = results[results["substrate"] == substrate]
    labels = results["label"].unique()
    if filename:
        file = open(os.path.join(OUTPUTDIRECTORY, filename), "w")
    for label in labels:
        results_label = results[results["label"] == label]
        r2 = r2_score(results_label["target"], results_label["pred"])
        mserr = mean_squared_error(results_label["target"], results_label["pred"])
        rmserr = np.sqrt(mserr)
        maerr = mean_absolute_error(results_label["target"], results_label["pred"])
        txt = f"{substrate} {label}\tMAErr={maerr:.4f}\tRMSErr={rmserr:.4f}\t" + r"$R^2$" + f"\t={r2:.2f}"
        print(txt)
        if filename:
            file.write(txt + "\n")
    if filename:
        file.close()
