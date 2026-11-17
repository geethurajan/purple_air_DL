"""
Model 1 and Model 2 side by side (no re-run needed): one figure per score, two panels on the
same horizontal line with the SAME y-axis, so the two models can be compared directly.

    left panel   Model 1 (hour t only, p = 7):   reg_lin_model, log_trans_lin_model, MLP ensemble
    right panel  Model 2 (hour t + t-1, p = 11): reg_lin_model, log_trans_lin_model,
                                                 MLP without g, MLP with g
Each model type has the same color in both panels and each panel has its own legend. All
scores are test scores (January 2026) on log1p PM2.5, read from the Summary files of the two
model codes.

Input:
    results_model_1_t_only_ensemble/model_1_t_only_ensemble_Summary_<d>km.csv
    results_model_2_t_lag_ensemble/model_2_t_lag_ensemble_Summary_<d>km.csv
Output (in --out-dir, default results_model_1_vs_2):
    model_1_vs_2_side_by_side_Adj_R2_vs_distance.png
    model_1_vs_2_side_by_side_R2_vs_distance.png
    model_1_vs_2_side_by_side_RMSE_vs_distance.png

Usage (from code_full or from inside comparison):
    python plot_model_1_vs_2_side_by_side.py
    python plot_model_1_vs_2_side_by_side.py --model-1-dir <folder> --model-2-dir <folder> --out-dir <folder>
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = "model_1_vs_2_side_by_side"

parser = argparse.ArgumentParser()
parser.add_argument("--model-1-dir", default=os.path.join(HERE, "results_model_1_t_only_ensemble"),
                    help="Folder with model_1_t_only_ensemble_Summary_<d>km.csv")
parser.add_argument("--model-2-dir", default=os.path.join(HERE, "results_model_2_t_lag_ensemble"),
                    help="Folder with model_2_t_lag_ensemble_Summary_<d>km.csv")
parser.add_argument("--out-dir", default=os.path.join(HERE, "results_model_1_vs_2"),
                    help="Folder for the figures")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

DISTANCES = [2.5, 3.5, 5, 7, 9, 11, 13, 15]

# Same color for the same model type in both panels
COLORS = {
    "reg_lin_model": "tab:orange",
    "log_trans_lin_model": "tab:blue",
    "MLP": "tab:green",
    "MLP_no_g": "tab:green",
    "MLP_g": "tab:red",
}

# Panel title, file name prefix, results folder, p, {model key in the Summary file: label}
PANELS = [
    ("Model 1 (hour t only)", "model_1_t_only_ensemble", args.model_1_dir, 7, {
        "log_trans_lin_model": "log_trans_lin_model",
        "reg_lin_model": "reg_lin_model (raw fit, scored on log1p)",
        "MLP": "MLP ensemble (5 seeds, 16-32-16)",
    }),
    ("Model 2 (hour t + hour t-1)", "model_2_t_lag_ensemble", args.model_2_dir, 11, {
        "log_trans_lin_model": "log_trans_lin_model",
        "reg_lin_model": "reg_lin_model (raw fit, scored on log1p)",
        "MLP_no_g": "MLP without g (5 seeds, 16-32-16)",
        "MLP_g": "MLP with g (5 seeds, 16-32-16)",
    }),
]

# Plots: column suffix, y-axis label, title, output file
PLOTS = [
    ("Adj R2", "Adjusted R²", "Adjusted R² vs Distance", f"{OUT_NAME}_Adj_R2_vs_distance.png"),
    ("R2", "R²", "R² vs Distance", f"{OUT_NAME}_R2_vs_distance.png"),
    ("RMSE", "RMSE (log1p PM2.5)", "RMSE vs Distance", f"{OUT_NAME}_RMSE_vs_distance.png"),
]


def read_summaries(folder, name, models):

    files = sorted(glob.glob(os.path.join(folder, f"{name}_Summary_*km.csv")))

    if not files:
        raise SystemExit(f"No {name}_Summary_*km.csv files found in {folder}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    for model in models:
        df[f"{model} Test RMSE"] = np.sqrt(df[f"{model} Test MSE"])

    df = df[df["Distance km"].isin(DISTANCES)]

    return df.sort_values("Distance km").reset_index(drop=True)


data = [read_summaries(folder, name, models) for _, name, folder, _, models in PANELS]

for (title_1, _, folder, _, _), df in zip(PANELS, data):
    print(f"{title_1}: {len(df)} distances from {folder}")


for suffix, y_label, title, plot_file in PLOTS:

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, (panel_title, _, _, p, models), df in zip(axes, PANELS, data):

        for model, label in models.items():
            ax.plot(df["Distance km"], df[f"{model} Test {suffix}"], marker="o",
                    color=COLORS[model], label=f"{label}, p = {p}")

        ax.set_title(f"{panel_title}: {title}\n"
                     f"Train: Jan-Mar 2025, test: Jan 2026 (same sensors, site 1103)")
        ax.set_xlabel("Distance from EPA monitor (km)")
        ax.set_xticks(df["Distance km"])
        ax.set_xticklabels([f"{d:g}" for d in df["Distance km"]])
        ax.grid(True, alpha=0.3)

        # Each panel has its own legend, below it
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    axes[0].set_ylabel(f"{y_label} (test set: January 2026)")

    # Number of test sensors at each distance, just above the x-axis (same y-axis in both panels)
    ymin, ymax = axes[0].get_ylim()

    for ax, df in zip(axes, data):
        for _, row in df.iterrows():
            ax.annotate(f"n={int(row['Test Sensors'])}", (row["Distance km"], ymin), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, plot_file), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {os.path.join(args.out_dir, plot_file)}")
