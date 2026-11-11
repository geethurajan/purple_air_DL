"""
Collect every model_2_jan-mar_train_both_R2_and_adj_Summary_<d>km.csv written by
model_2_jan-mar_train_both_R2_and_adj.py and make two plots of the TEST scores (January 2026)
against distance, one line per model:
    1. adjusted R² vs distance
    2. R² vs distance

Output (in results_model_2_jan-mar_train_both_R2_and_adj, next to this script):
    model_2_jan-mar_train_both_R2_and_adj_Adj_R2_vs_distance.png      plot 1
    model_2_jan-mar_train_both_R2_and_adj_R2_vs_distance.png          plot 2
    model_2_jan-mar_train_both_R2_and_adj_Summary_all_distances.csv   every distance, every column
    model_2_jan-mar_train_both_R2_and_adj_R2_table_all_distances.csv  n, p, R² and adjusted R²

Run from the training data folder or from inside jan2026_test:
    python jan2026_test/plot_model_2_jan-mar_train_both_R2_and_adj.py
    python plot_model_2_jan-mar_train_both_R2_and_adj.py
"""

import os
import glob

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

NAME = "model_2_jan-mar_train_both_R2_and_adj"

# Folder with NAME_Summary_*km.csv, next to this script
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{NAME}")
COMBINED_CSV = f"{NAME}_Summary_all_distances.csv"
TABLE_CSV = f"{NAME}_R2_table_all_distances.csv"

# Distances to plot (km)
DISTANCES = [2.5, 3, 5, 7, 9, 11, 13, 15]

# Model name in the summary columns -> label on the plot
MODELS = {
    "log_trans_lin_model": "log_trans_lin_model",
    "reg_lin_model": "reg_lin_model",
    "MLP": "MLP (two-stage, f with t-1 weather)",
}

# The two plots: column suffix, y-axis label, title, output file
PLOTS = [
    ("Adj R2", "Adjusted R²", "Adjusted R² vs Distance", f"{NAME}_Adj_R2_vs_distance.png"),
    ("R2", "R²", "R² vs Distance", f"{NAME}_R2_vs_distance.png"),
]


# ============================================================
# Read and combine results
# ============================================================

files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{NAME}_Summary_*km.csv")))

if not files:
    raise SystemExit(f"No {NAME}_Summary_*km.csv files found in {RESULTS_DIR}")

print(f"Found {len(files)} result files.")

results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
results = results.sort_values("Distance km").reset_index(drop=True)

results.to_csv(os.path.join(RESULTS_DIR, COMBINED_CSV), index=False)

# Table of n, p, R² and adjusted R² for train and test: one row per distance
# (p = the number of predictors used in the adjusted R² of that model)
table_columns = ["Distance km", "Train Sensors", "Test Sensors", "Train Observations", "Test Observations"]
for model in MODELS:
    for split in ["Train", "Test"]:
        for suffix in ["n", "p", "R2", "Adj R2"]:
            column = f"{model} {split} {suffix}"
            if column in results.columns:
                table_columns.append(column)

results[table_columns].to_csv(os.path.join(RESULTS_DIR, TABLE_CSV), index=False)

test_columns = [f"{m} Test {s}" for m in MODELS for s in ["R2", "Adj R2"]
                if f"{m} Test {s}" in results.columns]
print(results[["Distance km", "Test Sensors", "Test Observations"] + test_columns]
      .round(4).T.to_string(header=False))

ARCH = results["Architecture"].iloc[0]

# Plot the distances in DISTANCES (the combined CSVs above keep every distance)
results = results[results["Distance km"].isin(DISTANCES)].reset_index(drop=True)


# ============================================================
# Plot test score vs distance (one plot per score)
# ============================================================

def p_of(model):
    # Number of predictors p used in this model's adjusted R² (the same at every distance)
    column = f"{model} Test p"
    return int(results[column].iloc[0]) if column in results.columns else None


for suffix, y_label, title, plot_file in PLOTS:

    plt.figure(figsize=(8, 5))

    for model, label in MODELS.items():

        column = f"{model} Test {suffix}"

        if column not in results.columns:
            print(f"Column '{column}' not found, skipping.")
            continue

        if model.startswith("MLP"):
            label = f"{label} ({ARCH})"

        # The adjusted R² plot also shows the p of each model
        if suffix == "Adj R2" and p_of(model) is not None:
            label = f"{label}, p = {p_of(model)}"

        plt.plot(results["Distance km"], results[column], marker="o", label=label)

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
    plt.title(f"{title}\nTrain: Jan-Mar 2025, test: Jan 2026 (same sensors, site 1103)")

    plt.xticks(results["Distance km"], [f"{d:g}" for d in results["Distance km"]])
    plt.grid(True, alpha=0.3)

    # Legend below the plot, so it never covers a point
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.savefig(os.path.join(RESULTS_DIR, plot_file), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to {os.path.join(RESULTS_DIR, plot_file)}")

print(f"Saved combined table to {os.path.join(RESULTS_DIR, COMBINED_CSV)}")
print(f"Saved n, p, R² table to {os.path.join(RESULTS_DIR, TABLE_CSV)}")
