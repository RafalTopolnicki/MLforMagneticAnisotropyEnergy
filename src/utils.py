import pandas as pd


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
