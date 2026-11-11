"""
SAME-SCALE plots for the results of model_2_jan-mar_train_both_R2_and_adj.py (no re-run needed).

Why: in model_2_jan-mar_train_both_R2_and_adj.py, log_trans_lin_model and the MLP are scored on
the log1p scale, but reg_lin_model on the raw scale, so their R² cannot be compared.
This script puts every model on the SAME scale, log1p PM2.5 (the scale of the MLP and
log_trans_lin_model), using the saved test predictions:
    log_trans_lin_model: log_trans_lin_pred                     (already log1p, unchanged)
    reg_lin_model:       log1p(max(reg_lin_pred, 0))             (raw -> log1p)
    MLP:                 mlp_pred                               (already log1p, unchanged)
and compares each with log1p of the EPA PM2.5 (column log_EPA_PM).

reg_lin_model can predict negative PM2.5 at clean hours (a straight line can go below zero).
PM2.5 cannot be negative, and log1p is undefined at or below -1, so those predictions are set
to 0 before log1p. The number of such predictions is saved ("reg_lin_model Test clipped").

log_trans_lin_model and the MLP get exactly the same numbers as in the original code; only
reg_lin_model changes.

Input (written by model_2_jan-mar_train_both_R2_and_adj.py, one file per distance):
    results_model_2_jan-mar_train_both_R2_and_adj/
        model_2_jan-mar_train_both_R2_and_adj_Test_Predictions_<d>km.csv

Output (in the same folder), test set January 2026, log1p PM2.5:
    model_2_jan-mar_train_both_R2_and_adj_same_scale_Adj_R2_vs_distance.png
    model_2_jan-mar_train_both_R2_and_adj_same_scale_R2_vs_distance.png
    model_2_jan-mar_train_both_R2_and_adj_same_scale_RMSE_vs_distance.png
    model_2_jan-mar_train_both_R2_and_adj_same_scale_table_all_distances.csv
        n, p, R², adjusted R², MSE and RMSE of every model on the log1p scale, the number of
        clipped reg_lin_model predictions, and reg_lin_model's raw-scale R² (this should match
        the original Summary files, as a check)

Only test scores: the original code did not save training predictions.

Run from the training data folder or from inside jan2026_test:
    python jan2026_test/plot_model_2_jan-mar_train_both_R2_and_adj_same_scale.py
    python plot_model_2_jan-mar_train_both_R2_and_adj_same_scale.py
Results in another folder:
    python plot_model_2_jan-mar_train_both_R2_and_adj_same_scale.py --results-dir <folder>
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# Settings
# ============================================================

# Name of the code that wrote the predictions, and the name of this script's outputs
NAME = "model_2_jan-mar_train_both_R2_and_adj"
OUT_NAME = f"{NAME}_same_scale"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{NAME}"),
    help="Folder with the *_Test_Predictions_<d>km.csv files"
)
args = parser.parse_args()

RESULTS_DIR = args.results_dir

# Distances to plot (km)
DISTANCES = [2.5, 3, 5, 7, 9, 11, 13, 15]

# Architecture of the MLP in the original run (for the legend)
ARCH = "16-32-16"

# Model -> (prediction column, prediction is on the raw scale?, p, label on the plot)
MODELS = {
    "log_trans_lin_model": ("log_trans_lin_pred", False, 10, "log_trans_lin_model"),
    "reg_lin_model": ("reg_lin_pred", True, 10, "reg_lin_model"),
    "MLP": ("mlp_pred", False, 11, f"MLP (two-stage, f with t-1 weather) ({ARCH})"),
}

RAW_TARGET = "EPA_PM"        # raw EPA PM2.5, ug/m3
LOG_TARGET = "log_EPA_PM"    # log1p(EPA PM2.5)

# The plots: column suffix, y-axis label, title, output file
PLOTS = [
    ("Adj R2", "Adjusted R²", "Adjusted R² vs Distance", f"{OUT_NAME}_Adj_R2_vs_distance.png"),
    ("R2", "R²", "R² vs Distance", f"{OUT_NAME}_R2_vs_distance.png"),
    ("RMSE", "RMSE (log1p PM2.5)", "RMSE vs Distance", f"{OUT_NAME}_RMSE_vs_distance.png"),
]


# ============================================================
# Scores
# ============================================================

def adjusted_r2(r2, n, p):
    # Same formula as the original code, p = number of predictors
    if n - p - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def distance_of(file):
    # ..._Test_Predictions_2.5km.csv -> 2.5
    match = re.search(r"_([0-9.]+)km\.csv$", os.path.basename(file))
    return float(match.group(1)) if match else None


# ============================================================
# Read the test predictions and score every model on log1p PM2.5
# ============================================================

files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{NAME}_Test_Predictions_*km.csv")))

if not files:
    raise SystemExit(f"No {NAME}_Test_Predictions_*km.csv files found in {RESULTS_DIR}")

print(f"Found {len(files)} prediction files in {RESULTS_DIR}")

rows = []

for file in files:

    d = distance_of(file)
    pred_df = pd.read_csv(file)

    y_log = pred_df[LOG_TARGET].values
    n = len(y_log)

    row = {
        "Distance km": d,
        "Test Sensors": pred_df["sensor_id"].nunique(),
        "Test Observations": n,
    }

    for model, (column, is_raw, p, _) in MODELS.items():

        pred = pred_df[column].values

        if is_raw:
            # Check: the raw-scale R² of the original code (should match its Summary file)
            row[f"{model} Test raw R2 (original scale)"] = r2_score(pred_df[RAW_TARGET].values, pred)

            # PM2.5 cannot be negative: negative predictions are set to 0, then raw -> log1p
            row[f"{model} Test clipped"] = int((pred < 0).sum())
            pred_log = np.log1p(np.clip(pred, 0, None))
        else:
            pred_log = pred

        r2 = r2_score(y_log, pred_log)
        mse = mean_squared_error(y_log, pred_log)

        row[f"{model} Test n"] = n
        row[f"{model} Test p"] = p
        row[f"{model} Test R2"] = r2
        row[f"{model} Test Adj R2"] = adjusted_r2(r2, n, p)
        row[f"{model} Test MSE"] = mse
        row[f"{model} Test RMSE"] = np.sqrt(mse)

    rows.append(row)

results = pd.DataFrame(rows).sort_values("Distance km").reset_index(drop=True)

results.to_csv(os.path.join(RESULTS_DIR, f"{OUT_NAME}_table_all_distances.csv"), index=False)

print("\nTest scores, January 2026, all models on log1p PM2.5:")
show = ["Distance km", "Test Sensors", "Test Observations", "reg_lin_model Test clipped"] + [
    f"{m} Test {s}" for m in MODELS for s in ["R2", "Adj R2", "RMSE"]
]
print(results[show].round(4).T.to_string(header=False))

# Plot the distances in DISTANCES (the CSV above keeps every distance)
results = results[results["Distance km"].isin(DISTANCES)].reset_index(drop=True)


# ============================================================
# Plot test score vs distance (one plot per score)
# ============================================================

for suffix, y_label, title, plot_file in PLOTS:

    plt.figure(figsize=(8, 5))

    for model, (_, _, p, label) in MODELS.items():

        # The adjusted R² plot also shows the p of each model
        if suffix == "Adj R2":
            label = f"{label}, p = {p}"

        plt.plot(results["Distance km"], results[f"{model} Test {suffix}"], marker="o", label=label)

    # Number of test sensors at each distance, just above the x-axis
    ymin, ymax = plt.ylim()

    for _, row in results.iterrows():
        plt.annotate(
            f"n={int(row['Test Sensors'])}",
            (row["Distance km"], ymin),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color="gray"
        )

    plt.xlabel("Distance from EPA monitor (km)")
    plt.ylabel(f"{y_label} (test set: January 2026)")
    plt.title(f"{title}, all models on log1p PM2.5\n"
              f"Train: Jan-Mar 2025, test: Jan 2026 (same sensors, site 1103)")

    plt.xticks(results["Distance km"], [f"{d:g}" for d in results["Distance km"]])
    plt.grid(True, alpha=0.3)

    # Legend below the plot, so it never covers a point
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.savefig(os.path.join(RESULTS_DIR, plot_file), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to {os.path.join(RESULTS_DIR, plot_file)}")

print(f"Saved table to {os.path.join(RESULTS_DIR, f'{OUT_NAME}_table_all_distances.csv')}")
