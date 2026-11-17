"""
Plots for model_1_t_only_ensemble.py (no re-run needed): test score vs distance for the three
Model 1 models, ALL ON THE SAME SCALE (log1p PM2.5). The model code already scores every model
on log1p (reg_lin_model: log1p(max(prediction, 0))), so this script only reads its Summary files.

    log_trans_lin_model   linear regression on log1p
    reg_lin_model         linear regression on raw PM2.5, moved to log1p for scoring
    MLP ensemble          average of the 5 seeds (single-seed min/mean/max are in the table only)

Input (one file per distance):
    results_model_1_t_only_ensemble/model_1_t_only_ensemble_Summary_<d>km.csv
Output (same folder):
    model_1_t_only_ensemble_same_scale_Adj_R2_vs_distance.png
    model_1_t_only_ensemble_same_scale_R2_vs_distance.png
    model_1_t_only_ensemble_same_scale_RMSE_vs_distance.png
    model_1_t_only_ensemble_same_scale_table_all_distances.csv   (train and test scores)

Usage (from code_full or from inside comparison):
    python plot_model_1_t_only_ensemble_same_scale.py
    python plot_model_1_t_only_ensemble_same_scale.py --results-dir <folder>
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt


NAME = "model_1_t_only_ensemble"
OUT_NAME = f"{NAME}_same_scale"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{NAME}"),
    help="Folder with the *_Summary_<d>km.csv files"
)
args = parser.parse_args()

RESULTS_DIR = args.results_dir

DISTANCES = [2.5, 3.5, 5, 7, 9, 11, 13, 15]

# Same color for the same model in every plot file (single-model and side-by-side plots)
COLORS = {
    "log_trans_lin_model": "tab:blue",
    "reg_lin_model": "tab:orange",
    "MLP": "tab:green",
    "MLP_no_g": "tab:green",
    "MLP_g": "tab:red",
}

# Model key in the Summary files -> label on the plot
MODELS = {
    "log_trans_lin_model": "log_trans_lin_model",
    "reg_lin_model": "reg_lin_model (raw fit, scored on log1p)",
    "MLP": "MLP ensemble (5 seeds, 16-32-16)",
}

P = 7

files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{NAME}_Summary_*km.csv")))

if not files:
    raise SystemExit(f"No {NAME}_Summary_*km.csv files found in {RESULTS_DIR}")

print(f"Found {len(files)} summary files in {RESULTS_DIR}")

results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
results = results.sort_values("Distance km").reset_index(drop=True)

# RMSE on log1p, from MSE
for model in MODELS:
    for split in ["Train", "Test"]:
        results[f"{model} {split} RMSE"] = np.sqrt(results[f"{model} {split} MSE"])

for split in ["Train", "Test"]:
    for stat in ["min", "max"]:
        results[f"MLP single-seed {stat} {split} RMSE"] = np.sqrt(results[f"MLP single-seed {stat} {split} MSE"])

table_columns = ["Distance km", "Train Sensors", "Test Sensors", "Train Observations", "Test Observations",
                 "reg_lin_model Test clipped"] + [
    f"{m} {split} {s}" for m in MODELS for split in ["Train", "Test"] for s in ["R2", "Adj R2", "RMSE"]
] + [
    f"MLP single-seed {stat} Test {s}" for stat in ["min", "mean", "max"] for s in ["R2", "Adj R2"]
]

results[table_columns].to_csv(os.path.join(RESULTS_DIR, f"{OUT_NAME}_table_all_distances.csv"), index=False)

print("\nTest scores, January 2026, all models on log1p PM2.5:")
print(results[["Distance km", "Test Sensors"] + [f"{m} Test {s}" for m in MODELS for s in ["R2", "Adj R2", "RMSE"]]]
      .round(4).T.to_string(header=False))

results = results[results["Distance km"].isin(DISTANCES)].reset_index(drop=True)

# Plots: column suffix, y-axis label, title, output file
PLOTS = [
    ("Adj R2", "Adjusted R²", "Adjusted R² vs Distance", f"{OUT_NAME}_Adj_R2_vs_distance.png"),
    ("R2", "R²", "R² vs Distance", f"{OUT_NAME}_R2_vs_distance.png"),
    ("RMSE", "RMSE (log1p PM2.5)", "RMSE vs Distance", f"{OUT_NAME}_RMSE_vs_distance.png"),
]

for suffix, y_label, title, plot_file in PLOTS:

    plt.figure(figsize=(8, 5))

    for model, label in MODELS.items():

        if suffix == "Adj R2":
            label = f"{label}, p = {P}"

        plt.plot(results["Distance km"], results[f"{model} Test {suffix}"], marker="o",
                 color=COLORS[model], label=label)

    # Number of test sensors at each distance, just above the x-axis
    ymin, ymax = plt.ylim()

    for _, row in results.iterrows():
        plt.annotate(f"n={int(row['Test Sensors'])}", (row["Distance km"], ymin), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=8, color="gray")

    plt.xlabel("Distance from EPA monitor (km)")
    plt.ylabel(f"{y_label} (test set: January 2026)")
    plt.title(f"Model 1 (hour t only): {title}, all models on log1p PM2.5\n"
              f"Train: Jan-Mar 2025, test: Jan 2026 (same sensors, site 1103)")

    plt.xticks(results["Distance km"], [f"{d:g}" for d in results["Distance km"]])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.savefig(os.path.join(RESULTS_DIR, plot_file), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {os.path.join(RESULTS_DIR, plot_file)}")

print(f"Saved table to {os.path.join(RESULTS_DIR, f'{OUT_NAME}_table_all_distances.csv')}")
