"""
This script is a direct copy of TrainModels.ipynb notebook. We recommend using the notebook for better experience

The script print all the metrics to the screen and saves them in results/ dir
"""

import pandas as pd
from catboost import CatBoostRegressor
from catboost import EShapCalcType, EFeaturesSelectionAlgorithm

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso

from src.utils import prediction_on_folds, train_nocv, compute_metrics_by_method
from src.outputs import save_fold_predictions, save_nofolds_predictions
from src.manipulations import categorical_features
from src.labels_paper import labels

import shap
import os

from src.consts import DATADIRECTORY, OUTPUTDIRECTORY
from src.plot import plot_with_histogram


# # Read data
# ### Read data for implicit substrates
print("Read data: implicit substrates")
df = pd.read_csv(os.path.join(DATADIRECTORY, "implicit_dataset.csv"))
target = pd.read_csv(os.path.join(DATADIRECTORY, "implicit_target_mae.csv"))
target_soc = pd.read_csv(os.path.join(DATADIRECTORY, "implicit_target_soc.csv"))
target_socm = pd.read_csv(os.path.join(DATADIRECTORY, "implicit_target_socm.csv"))

print(f"Total number of features: {len(df.columns)}")
print(f"Total number of systems: {len(df)}")

print("Read data: explicit substrates substrates")
df_es = pd.read_csv(os.path.join(DATADIRECTORY, "explicit_dataset.csv"))
target_es = pd.read_csv(os.path.join(DATADIRECTORY, "explicit_target_mae.csv"))
target_soc_es = pd.read_csv(os.path.join(DATADIRECTORY, "explicit_target_soc.csv"))
target_socm_es = pd.read_csv(os.path.join(DATADIRECTORY, "explicit_target_socm.csv"))

print(f"Total number of features: {len(df_es.columns)}")
print(f"Total number of systems: {len(df_es)}")

# prepare data for LinearRegression and Lasso
# remove possible NaNs and perform one-hot encodings
df_nonans = df.dropna(axis=1)
df_es_nonans = df_es.dropna(axis=1)
no_nans_common_columns = list(set(df_nonans.columns).intersection(df_es_nonans.columns))
df_nonans = df_nonans[no_nans_common_columns]
df_es_nonans = df_es_nonans[no_nans_common_columns]

cat_features = categorical_features(df)
cat_features_nonans = categorical_features(df_nonans)
df_nonans_onehot = pd.get_dummies(df_nonans, columns=cat_features_nonans)
df_es_nonans_onehot = pd.get_dummies(df_es_nonans, columns=cat_features_nonans)


scoring = {
    "mean_squared_error": "neg_mean_squared_error",
    "mean_absolute_error": "neg_mean_absolute_error",
    "r2": "r2",
}

cb_hyperparams = {"max_depth": 5, "n_estimators": 2000, "eta": 0.05}

cb = CatBoostRegressor(verbose=False, cat_features=cat_features, **cb_hyperparams)
cv_cb = cross_validate(
    cb, df, target, cv=5, scoring=scoring, return_train_score=True, return_estimator=True, return_indices=True
)

lasso = Lasso(alpha=0.2)
cv_lasso = cross_validate(
    lasso,
    df_nonans_onehot,
    target,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)
lr = LinearRegression(fit_intercept=True)
cv_lr = cross_validate(
    lr,
    df_nonans_onehot,
    target,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)

# Train on all data without CV for better prediction for Explicit Substrates
nocv_cb = train_nocv(cb, df, target)
nocv_lasso = train_nocv(lasso, df_nonans_onehot, target)
nocv_lr = train_nocv(lr, df_nonans_onehot, target)

# Make predictions on whole dataset
csv_path_all_features = save_fold_predictions(
    cv_objs=[cv_lr, cv_lasso, cv_cb],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_nonans_onehot, df_nonans_onehot, df],
    targets=[target, target, target],
    filename=f"predictions_fit_all_features_implicit",
    substrate="implicit",
)


# ## Prediction metrics for all features: reproduce part of Table 1
# Metrics for implicit substrates
# Model use all features
# This metrics are given in Table 1 in the manuscript (Columns named All features)
print("Prediction metrics for all features: reproduce part of Table 1")
compute_metrics_by_method(csv_path_all_features, substrate="implicit", filename="Table1_all_features_implicit.txt")

# Metrics for explicit substrates
# Models uses all features
csv_path_all_features_es = save_nofolds_predictions(
    estimators=[nocv_lr, nocv_lasso, nocv_cb],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_es_nonans_onehot, df_es_nonans_onehot, df_es],
    targets=[target_es, target_es, target_es],
    filename=f"predictions_fit_all_features_explicit",
    substrate="explicit",
)

compute_metrics_by_method(csv_path_all_features_es, substrate="explicit", filename="Table1_all_features_explicit.txt")

# Reproduce Figure 2A
plot_with_histogram(
    csv_path=csv_path_all_features,
    cv_path_explicit=csv_path_all_features_es,
    colors=["black", "blue", "red"],
    filename=f"Figure_2A",
    corner_label="a",
    xlabel="DFT-Computed MAE (meV)",
    ylabel="Predicted MAE (meV)",
)

# Feature selection with CV
print("Future selection started. This will take a while")
num_features_to_selects = [2, 3, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]

mse_feature = []
mae_feature = []
r2_feature = []
features_selection = {}

for num_features_to_select in num_features_to_selects:
    mod = CatBoostRegressor(cat_features=cat_features)
    summary = mod.select_features(
        df,
        target,
        features_for_select=list(range(df.shape[1])),
        num_features_to_select=num_features_to_select,
        steps=20,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        plot=False,
        logging_level="Silent",
    )

    features_selection[num_features_to_select] = summary["selected_features_names"]
    df_selected = df[summary["selected_features_names"]]
    cb_mod = CatBoostRegressor(**cb_hyperparams, cat_features=cat_features, verbose=False)
    cv_cb_fetures = cross_validate(
        cb_mod,
        df_selected,
        target,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
    )
    y_pred_folds, y_test_folds = prediction_on_folds(cv_cb_fetures, df_selected, target)
    mae = mean_absolute_error(y_test_folds, y_pred_folds)
    mse = mean_squared_error(y_test_folds, y_pred_folds)
    r2 = r2_score(y_test_folds, y_pred_folds)
    mse_feature.append(mse)
    mae_feature.append(mae)
    r2_feature.append(r2)
    print("****")
    print(f"REQUIRED NUMBER OF FEATURES {num_features_to_select}")
    print(summary["selected_features_names"])
    print(f"MAErr={mae} MSErr={mse} RMSErr={np.sqrt(mse)} R2={r2}")
    print("****")

fig, axs = plt.subplots(1, 3, figsize=(16, 3.5))
axs[0].plot(num_features_to_selects, np.sqrt(mse_feature), "o-", color="black")
axs[0].set_ylabel("Root Mean Square Error, RMSErr", fontsize=14)
axs[0].set_xlim((1, 60))
axs[0].text(0.05, 0.9, "a", fontsize=17, transform=axs[0].transAxes)
# axs[0].grid()

axs[1].plot(num_features_to_selects, mae_feature, "o-", color="black")
axs[1].set_ylabel("Mean Aboslute Error, MAErr", fontsize=14)
axs[1].set_xlim((1, 60))
axs[1].text(0.05, 0.9, "b", fontsize=17, transform=axs[1].transAxes)
# axs[1].grid()

axs[2].plot(num_features_to_selects, r2_feature, "o-", color="black")
axs[2].set_ylabel(r"$R^2$", fontsize=14)
axs[2].set_xlim((1, 60))
axs[2].text(0.05, 0.9, "c", fontsize=17, transform=axs[2].transAxes)
# axs[2].grid()

axs[1].set_xlabel("Number of features in the model", fontsize=14)
#fig.savefig(os.path.join(OUTPUTDIRECTORY, "Figure_S4_part.pdf"))


# # Build models for list selected 25 features

N_FEATURES = 25
FEATURES = features_selection[N_FEATURES]
print(f"FEATURES: {FEATURES}")
df_selected = df[FEATURES].copy()

df_es_selected = df_es[FEATURES]
# prepare data for LinearRegression
# remove NaNs
# do one-hot encodings
df_nonans_selected = df_selected.dropna(axis=1)
df_es_nonans_selected = df_es_selected.dropna(axis=1)
no_nans_common_columns = list(set(df_nonans_selected.columns).intersection(df_es_nonans_selected.columns))
df_nonans_selected = df_nonans_selected[no_nans_common_columns]
df_es_nonans_selected = df_es_nonans_selected[no_nans_common_columns]

cat_features_selected = categorical_features(df_selected)
cat_features_nonans_selected = categorical_features(df_nonans_selected)
df_nonans_onehot_selected = pd.get_dummies(df_nonans_selected, columns=cat_features_nonans_selected)
df_es_nonans_onehot_selected = pd.get_dummies(df_es_nonans_selected, columns=cat_features_nonans_selected)

cat_features_nonans_selected = categorical_features(df_nonans_selected)
df_nonans_selected_onehot = pd.get_dummies(df_nonans_selected, columns=cat_features_nonans_selected)


# print selected features in LaTeX format
[labels(f) for f in FEATURES]


lasso = Lasso(alpha=0.1)
cv_lasso_selected = cross_validate(
    lasso,
    df_nonans_selected_onehot,
    target,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)

cv_lr_selected = cross_validate(
    lr,
    df_nonans_selected_onehot,
    target,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)

cb = CatBoostRegressor(verbose=False, cat_features=cat_features_selected)
cv_cb_selected = cross_validate(
    cb, df_selected, target, cv=5, scoring=scoring, return_train_score=True, return_estimator=True, return_indices=True
)


nocv_cb_selected = train_nocv(cb, df_selected, target)
nocv_lasso_selected = train_nocv(lasso, df_nonans_selected_onehot, target)
nocv_lr_selected = train_nocv(lr, df_nonans_selected_onehot, target)

csv_path_selected = save_fold_predictions(
    cv_objs=[cv_lr_selected, cv_lasso_selected, cv_cb_selected],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_nonans_selected_onehot, df_nonans_selected_onehot, df_selected],
    targets=[target, target, target],
    filename=f"predictions_fit_selected_features_implicit",
    substrate="implicit",
)


# ## Trainig metrics: reproduce Table 1

# Metrics for Implicit Substrates
# Models uses 25 features
# This metrics are given in Table 1 in the manuscript (Columns named Twenty-five features, Implicit Substrates)
compute_metrics_by_method(csv_path_selected, substrate="implicit", filename="Table1_selected_features_implicit.txt")


csv_path_selected_es = save_nofolds_predictions(
    estimators=[nocv_lr_selected, nocv_lasso_selected, nocv_cb_selected],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_es_nonans_onehot_selected, df_es_nonans_onehot_selected, df_es_selected],
    targets=[target_es, target_es, target_es],
    filename=f"predictions_fit_selected_features_explicit",
    substrate="explicit",
)


# Metrics for Explicit Substrates
# Models uses 25 features
# This metrics are given in Table 1 in the manuscript (Columns named Twenty-five features, Explicit Substrates)
compute_metrics_by_method(csv_path_selected_es, substrate="explicit", filename="Table1_selected_features_explicit.txt")

# Reproduce Figure 2B
plot_with_histogram(
    csv_path=csv_path_selected,
    cv_path_explicit=csv_path_selected_es,
    colors=["black", "blue", "red"],
    filename=f"Figure_2B",
    corner_label="a",
    xlabel="DFT-Computed MAE (meV)",
    ylabel="Predicted MAE (meV)",
)


# ## SHAP values: reproduce Table 2
cb_selected = nocv_cb_selected

explainer = shap.TreeExplainer(cb_selected)
shap_values = explainer(df_selected)
mean_shap_values = np.mean(np.abs(shap_values.values), axis=0)
df_shap_dict = {}
for val, col in zip(mean_shap_values, df_selected.columns):
    df_shap_dict[labels(col)] = val
df_shap = pd.DataFrame.from_dict(df_shap_dict, orient="index", columns=["shap"]).sort_values(by="shap", ascending=False)
# Average absolute SHAP values
# This metrics are given in Table 2 in the Manuscript
df_shap.to_csv(os.path.join(OUTPUTDIRECTORY, "Table2.csv"))
print(df_shap)


# # E_SOC predictions
# Here we reproduce results regarding the E_SOC energy prediction for top and bottom adatoms
lasso = Lasso(alpha=0.0001)
cv_lasso_selected_soc = cross_validate(
    lasso,
    df_nonans_selected,
    target_soc,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)
lr = LinearRegression()
cv_lr_selected_soc = cross_validate(
    lr,
    df_nonans_selected,
    target_soc,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)

cb = CatBoostRegressor(verbose=False, cat_features=cat_features_selected, loss_function="MultiRMSE")
cv_cb_selected_soc = cross_validate(
    cb,
    df_selected,
    target_soc,
    cv=5,
    scoring=scoring,
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)
csv_path_selected_soc_top = save_fold_predictions(
    cv_objs=[cv_lr_selected_soc, cv_lasso_selected_soc, cv_cb_selected_soc],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_nonans_selected_onehot, df_nonans_selected_onehot, df_selected],
    targets=[target_soc, target_soc, target_soc],
    estimator_id=0,
    filename=f"predictions_Esoc_top",
    substrate="implicit",
)
csv_path_selected_soc_bottom = save_fold_predictions(
    cv_objs=[cv_lr_selected_soc, cv_lasso_selected_soc, cv_cb_selected_soc],
    labels=["MLR", "LASSO", "ML"],
    dfs=[df_nonans_selected_onehot, df_nonans_selected_onehot, df_selected],
    targets=[target_soc, target_soc, target_soc],
    estimator_id=1,
    filename=f"predictions_Esoc_bottom",
    substrate="implicit",
)


# ## Reproduce Figure 3a
# Metrics E_soc prediction for top atom (Implicit Substrates)
# This metrics corresponds to Figure 3a in the Manuscript
compute_metrics_by_method(csv_path_selected_soc_top, substrate="implicit")
plot_with_histogram(
    csv_path=csv_path_selected_soc_top,
    cv_path_explicit=None,
    colors=["black", "blue", "red"],
    filename=f"Figure_3A",
    limit_y_axis=False,
    xlim_hist=(-0.1, 0.1),
    legend_metric="r2",
    xlabel=r"DFT-Computed $E_{SOC}$ (meV)",
    ylabel="Predicted $E_{SOC}$ (meV)",
    corner_label="a",
    inset_size=[0.18, 0.68, 0.25, 0.18],
)
# ## Reproduce Figure 3b
# Metrics E_soc prediction for bottom atom (Implicit Substrates)
# This metrics corresponds to Figure 3b in the Manuscript
compute_metrics_by_method(csv_path_selected_soc_bottom, substrate="implicit")
plot_with_histogram(
    csv_path=csv_path_selected_soc_bottom,
    cv_path_explicit=None,
    colors=["black", "blue", "red"],
    filename=f"Figure_3B",
    limit_y_axis=False,
    xlim_hist=(-0.1, 0.1),
    legend_metric="r2",
    xlabel=r"DFT-Computed $E_{SOC}$ (meV)",
    ylabel="Predicted $E_{SOC}$ (meV)",
    corner_label="b",
    inset_size=[0.18, 0.68, 0.25, 0.18],
)

print("*" * 50)
print("All the resoluts were written to results dir")
print(
    "This script is a direct copy of TrainModels.ipynb notebook. We recommend using the notebook for better experience"
)
print("*" * 50)
