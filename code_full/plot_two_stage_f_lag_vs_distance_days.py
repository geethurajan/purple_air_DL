"""
Collect every AltDays_Summary_<d>km.csv written by hpc_two_stage_f_lag_by_distance_days.py
and plot the TEST adjusted R² against distance for each model.

Edit the settings below and run:
    python plot_two_stage_f_lag_vs_distance_days.py
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

RESULTS_DIR = "results_two_stage_f_lag_by_distance_days"    # folder with AltDays_Summary_*km.csv
COMBINED_CSV = "AltDays_Summary_all_distances.csv"
PLOT_FILE = "adj_r2_vs_distance_alt_days.png"

# Column in the summary CSVs -> label on the plot
MODELS = {
    "log_trans_lin_model Test Adj R2": "log_trans_lin_model",
    "reg_lin_model Test Adj R2": "reg_lin_model",
    "MLP Test Adj R2": "MLP (two-stage, f with t-1 weather)",
}


# ============================================================
# Read and combine results
# ============================================================

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "AltDays_Summary_*km.csv")))

if not files:
    raise SystemExit(f"No AltDays_Summary_*km.csv files found in {RESULTS_DIR}")

print(f"Found {len(files)} result files.")

results = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)

results = results.sort_values("Distance km").reset_index(drop=True)

results.to_csv(
    os.path.join(RESULTS_DIR, COMBINED_CSV),
    index=False
)

print(results[["Distance km", "Sensors", "Observations"] + list(MODELS)].to_string(index=False))


# ============================================================
# Plot test adjusted R² vs distance
# ============================================================

plt.figure(figsize=(8, 5))

for column, label in MODELS.items():

    if column not in results.columns:
        print(f"Column '{column}' not found, skipping.")
        continue

    plt.plot(
        results["Distance km"],
        results[column],
        marker="o",
        label=label
    )

# Number of sensors at each distance, just above the x-axis
ymin, ymax = plt.ylim()

for _, row in results.iterrows():
    plt.annotate(
        f"n={int(row['Sensors'])}",
        (row["Distance km"], ymin),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        fontsize=8,
        color="gray"
    )

plt.xlabel("Distance from EPA monitor (km)")
plt.ylabel("Adjusted R² (test days, both folds pooled)")
plt.title("Adjusted R² vs Distance\nAlternate-day split (train on every other day, test on the rest)")

plt.xticks(results["Distance km"])
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig(
    os.path.join(RESULTS_DIR, PLOT_FILE),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved plot to {os.path.join(RESULTS_DIR, PLOT_FILE)}")
print(f"Saved combined table to {os.path.join(RESULTS_DIR, COMBINED_CSV)}")
