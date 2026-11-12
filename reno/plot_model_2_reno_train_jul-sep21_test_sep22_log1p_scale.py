"""
Collect every model_2_reno_train_jul-sep21_test_sep22_Summary_<d>km.csv written by model_2_reno_train_jul-sep21_test_sep22.py and make two plots of the TEST
scores (September 2022) against distance, one line per model, with ALL THREE MODELS ON THE
log1p SCALE:
    log_trans_lin_model and MLP: log1p, as fitted
    reg_lin_model:               its raw predictions converted with log1p (negative predictions
                                 set to 0 first), columns "reg_lin_model log1p Test ..."
    1. adjusted R² vs distance
    2. R² vs distance

Output (in results_model_2_reno_train_jul-sep21_test_sep22, next to this script):
    model_2_reno_train_jul-sep21_test_sep22_log1p_scale_Adj_R2_vs_distance.png   plot 1
    model_2_reno_train_jul-sep21_test_sep22_log1p_scale_R2_vs_distance.png       plot 2
    model_2_reno_train_jul-sep21_test_sep22_Summary_all_distances.csv   every distance, every column
    model_2_reno_train_jul-sep21_test_sep22_R2_table_all_distances.csv  n, p, R² and adjusted R²

Run from inside the reno folder:
    python3 plot_model_2_reno_train_jul-sep21_test_sep22_log1p_scale.py
Results in another folder:
    python3 plot_model_2_reno_train_jul-sep21_test_sep22_log1p_scale.py --results-dir <folder>
"""

import os
import glob
import argparse

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

NAME = "model_2_reno_train_jul-sep21_test_sep22"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{NAME}"),
    help=f"Folder with the {NAME}_Summary_<d>km.csv files"
)
args = parser.parse_args()

RESULTS_DIR = args.results_dir
COMBINED_CSV = f"{NAME}_Summary_all_distances.csv"
TABLE_CSV = f"{NAME}_R2_table_all_distances.csv"

# Distances to plot (km)
DISTANCES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

MLP_LABEL = "MLP (two-stage, f with t-1 weather)"

# The plots: model -> (column prefix in the summary, label on the plot)
SCALES = {
    "log1p_scale": {
        "title": "all models on log1p PM2.5",
        "models": {
            "log_trans_lin_model": ("log_trans_lin_model", "log_trans_lin_model"),
            "reg_lin_model": ("reg_lin_model log1p", "reg_lin_model"),
            "MLP": ("MLP", MLP_LABEL),
        },
    },
}

# The scores: column suffix, y-axis label, title, file name part
SCORES = [
    ("Adj R2", "Adjusted R²", "Adjusted R² vs Distance", "Adj_R2_vs_distance.png"),
    ("R2", "R²", "R² vs Distance", "R2_vs_distance.png"),
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
table_columns = ["Distance km", "Train Sensors", "Test Sensors", "Test Sensors Also In Train",
                 "Train Observations", "Test Observations"]
for prefix in ["log_trans_lin_model", "reg_lin_model", "reg_lin_model log1p", "MLP"]:
    for split in ["Train", "Test"]:
        for suffix in ["n", "p", "R2", "Adj R2", "clipped"]:
            column = f"{prefix} {split} {suffix}"
            if column in results.columns:
                table_columns.append(column)

results[table_columns].to_csv(os.path.join(RESULTS_DIR, TABLE_CSV), index=False)

test_columns = [c for c in table_columns if " Test " in c and c.endswith(("R2", "clipped"))]
print(results[["Distance km", "Test Sensors", "Test Observations"] + test_columns]
      .round(4).T.to_string(header=False))

ARCH = results["Architecture"].iloc[0]

# Plot the distances in DISTANCES (the combined CSVs above keep every distance)
results = results[results["Distance km"].isin(DISTANCES)].reset_index(drop=True)


# ============================================================
# Plot test score vs distance (one plot per scale and score)
# ============================================================

def p_of(prefix):
    # Number of predictors p used in this model's adjusted R² (the same at every distance)
    column = f"{prefix} Test p"
    return int(results[column].iloc[0]) if column in results.columns else None


for scale_name, scale in SCALES.items():

    for suffix, y_label, title, file_part in SCORES:

        plt.figure(figsize=(8, 5))

        for model, (prefix, label) in scale["models"].items():

            column = f"{prefix} Test {suffix}"

            if column not in results.columns:
                print(f"Column '{column}' not found, skipping.")
                continue

            if model == "MLP":
                label = f"{label} ({ARCH})"

            # The adjusted R² plots also show the p of each model
            if suffix == "Adj R2" and p_of(prefix) is not None:
                label = f"{label}, p = {p_of(prefix)}"

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
        plt.ylabel(f"{y_label} (test set: September 2022)")
        plt.title(f"{title}, {scale['title']}\n"
                  f"Train: Jul-Sep 2021, test: Sep 2022 (same sensors, Reno4 site 32-031-0031)", fontsize=10)

        plt.xticks(results["Distance km"], [f"{d:g}" for d in results["Distance km"]])
        plt.grid(True, alpha=0.3)

        # Legend below the plot, so it never covers a point
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

        plot_file = f"{NAME}_{scale_name}_{file_part}"
        plt.savefig(os.path.join(RESULTS_DIR, plot_file), dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nSaved plot to {os.path.join(RESULTS_DIR, plot_file)}")

print(f"Saved combined table to {os.path.join(RESULTS_DIR, COMBINED_CSV)}")
print(f"Saved n, p, R² table to {os.path.join(RESULTS_DIR, TABLE_CSV)}")
