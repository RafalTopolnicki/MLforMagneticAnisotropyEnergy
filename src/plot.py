import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
from src.consts import OUTPUTDIRECTORY

cm2in = 1 / 2.54
plot_params = {
    "text.usetex": False,
    "font.size": 11,
    "font.family": "Nimbus Sans",
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "axes.linewidth": 0.75,
    "legend.fontsize": 10,
    "legend.handlelength": 1.0,
    "legend.frameon": True,
    "legend.labelspacing": 0.2,
    "legend.handletextpad": 0.2,
    "legend.borderaxespad": 0.2,
    "legend.framealpha": 1,
    "legend.facecolor": "w",
    "figure.figsize": (13 * cm2in, 10 * cm2in),
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
}
plt.rcParams.update(plot_params)


def plot_with_histogram(
    csv_path,
    colors,
    filename="plot",
    corner_label=None,
    limit_y_axis=True,
    cv_path_explicit=None,
    xlabel="DFT-Computed Magnetic Anisotropy Energy (meV)",
    ylabel="Predicted Magnetic Anisotropy Energy (meV)",
    xlim_hist=(-130, 130),
    ylim=(-130, 250),
    xlim=None,
    legend_metric="rmse",
    inset_loc="top",
    inset_size=[0.165, 0.6, 0.3, 0.25],
    fontsizefactor=1.0,
    legend_loc="lower right",
):
    df = pd.read_csv(csv_path)
    if cv_path_explicit:
        df_explicit = pd.read_csv(cv_path_explicit)

    hist_bins = 50
    markers = ["o", "v", "s", "*", "P", "D"]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    left, bottom, width, height = inset_size
    if inset_loc == "bottom":
        left, bottom, width, height = [0.165, 0.20, inset_size[2], inset_size[3]]

    ax2 = fig.add_axes([left, bottom, width, height])
    labels = df["label"].unique()

    for label, color, marker in zip(labels, colors, markers):
        df_label = df[df["label"] == label]
        target = df_label["target"]
        pred = df_label["pred"]
        mae = mean_absolute_error(target, pred)
        r2 = r2_score(target, pred)
        mse = mean_squared_error(target, pred)
        rmse = np.sqrt(mse)
        txt = f"{label}\tMAErr={mae:.4f}\tRMSErr={rmse:.4f}\t" + r"$R^2$" + f"\t={r2:.2f}"
        if legend_metric == "r2":
            txt = f"{label} " + r"$R^2$" + f"={r2:.2f}"
        else:
            txt = f"{label} RMSErr={rmse:.1f}"
        ax1.plot(target, pred, marker, color=color, alpha=0.8, label=txt)
        # add histogram
        errs = np.array(target) - np.array(pred)
        # errs = errs[np.abs(errs) < 130]
        _ = ax2.hist(errs, bins=hist_bins, alpha=0.75, color=color, density=True)
    ax2.set_xlabel("Prediction Error", fontsize=11 * fontsizefactor, labelpad=0)
    ax2.set_xlim(xlim_hist)
    ax1.axline((0, 0), slope=1.0, color="gray")
    # ax1.grid('.')
    legend = ax1.legend(loc=legend_loc, fontsize=14 * fontsizefactor)
    if limit_y_axis:
        ax1.set_ylim(ylim)
    ax1.set_xlabel(xlabel, fontsize=15 * fontsizefactor)
    # ax1.set_xticks(fontsize=14, rotation=90)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    if xlim != None:
        ax1.set_xlim(xlim)
    if (corner_label == " ") or (corner_label == "a"):
        ax1.set_ylabel(ylabel, fontsize=15 * fontsizefactor)
    else:
        ax1.set_ylabel(ylabel, fontsize=15 * fontsizefactor, color="white")
    ax2.set_yticks([])

    if cv_path_explicit:
        for label, marker in zip(labels, markers):
            df_label = df_explicit[df_explicit["label"] == label]
            y_pred_es = df_label["pred"]
            es_target = df_label["target"]
            mse = mean_squared_error(es_target, y_pred_es)
            mae = mean_absolute_error(es_target, y_pred_es)
            rmse = np.sqrt(mse)
            r2 = r2_score(es_target, y_pred_es)
            ax1.plot(es_target, y_pred_es, marker, color="tab:green", alpha=0.8)
    ax1.text(
        0.01,
        1 - 0.01,
        corner_label,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax1.transAxes,
        fontsize=18 * fontsizefactor,
    )

    plt.savefig(os.path.join(OUTPUTDIRECTORY, f"{filename}.pdf"), bbox_inches="tight")
