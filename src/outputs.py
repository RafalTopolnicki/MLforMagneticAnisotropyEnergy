import pandas as pd
from src.utils import prediction_on_folds


def save_fold_predictions(cv_objs, labels, dfs, targets, filenamebase="", estimator_id=0, substrate="implicit"):
    assert len(cv_objs) == len(labels) == len(dfs) == len(targets), "Different lengths"
    txtoutput_label = []
    txtoutput_target = []
    txtoutput_preds = []
    txtoutput_substrate = []
    for cv_obj, label, df, target in zip(cv_objs, labels, dfs, targets):
        preds, target = prediction_on_folds(cv_obj, df=df, target=target)
        n_preds = len(preds)
        txtoutput_label += [label] * n_preds
        txtoutput_target += target.iloc[:, estimator_id].to_list()
        txtoutput_preds += preds.iloc[:, estimator_id].to_list()
        txtoutput_substrate += [substrate] * n_preds
    txt_output = pd.DataFrame(
        {
            "label": txtoutput_label,
            "target": txtoutput_target,
            "pred": txtoutput_preds,
            "substrate": txtoutput_substrate,
        }
    )
    txtfilepath = f"txtoutput/{filenamebase}.csv"
    txt_output.to_csv(txtfilepath, index=False)
    return txtfilepath


def save_nofolds_predictions(estimators, labels, dfs, targets, filenamebase="", substrate="implicit"):
    assert len(estimators) == len(labels) == len(dfs) == len(targets), "Different lengths"
    txtoutput_label = []
    txtoutput_target = []
    txtoutput_preds = []
    txtoutput_substrate = []
    for est, label, df, target in zip(estimators, labels, dfs, targets):
        preds = est.predict(df)
        n_preds = len(preds)
        txtoutput_label += [label] * n_preds
        txtoutput_target += (
            target.to_numpy()
            .reshape(
                -1,
            )
            .tolist()
        )
        txtoutput_preds += preds.reshape(
            -1,
        ).tolist()
        txtoutput_substrate += [substrate] * n_preds
    txt_output = pd.DataFrame(
        {
            "label": txtoutput_label,
            "target": txtoutput_target,
            "pred": txtoutput_preds,
            "substrate": txtoutput_substrate,
        }
    )
    txtfilepath = f"txtoutput/{filenamebase}.csv"
    txt_output.to_csv(txtfilepath, index=False)
    return txtfilepath
