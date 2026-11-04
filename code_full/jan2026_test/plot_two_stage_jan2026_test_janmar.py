"""
Collect every Jan2026Test_JanMar_Summary_<d>km.csv written by hpc_two_stage_jan2026_test_janmar.py
and plot the TEST adjusted R² (January 2026) against distance for each model.

Run from the training data folder or from inside jan2026_test:
    python jan2026_test/plot_two_stage_jan2026_test_janmar.py
    python plot_two_stage_jan2026_test_janmar.py
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

# Folder with Jan2026Test_JanMar_Summary_*km.csv, next to this script
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_jan2026_test_janmar")
COMBINED_CSV = "Jan2026Test_JanMar_Summary_all_distances.csv"
PLOT_FILE = "adj_r2_vs_distance_jan2026_test_janmar.png"

# Distances to plot (km): the same grid as the other experiments (2.5 to 13 km), plus 15 km
DISTANCES = [2.5, 3, 5, 7, 9, 11, 13, 15]

# Column in the summary CSVs -> label on the plot
MODELS = {
    "log_trans_lin_model Test Adj R2": "log_trans_lin_model",
    "reg_lin_model Test Adj R2": "reg_lin_model",
    "MLP Test Adj R2": "MLP (two-stage)",
}


# ============================================================
# Read and combine results
# ============================================================

files = sorted(glob.glob(os.path.join(RESULTS_DIR, "Jan2026Test_JanMar_Summary_*km.csv")))

if not files:
    raise SystemExit(f"No Jan2026Test_JanMar_Summary_*km.csv files found in {RESULTS_DIR}")

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

print(results[["Distance km", "Train Sensors", "Test Sensors", "Train Observations", "Test Observations"]
              + list(MODELS)].round(4).to_string(index=False))

ARCH = results["Architecture"].iloc[0]

# Plot the distances in DISTANCES (the combined CSV above keeps every distance)
results = results[results["Distance km"].isin(DISTANCES)].reset_index(drop=True)


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
        label=f"{label} ({ARCH})" if column.startswith("MLP") else label
    )

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
plt.ylabel("Adjusted R² (test set: January 2026)")
plt.title("Adjusted R² vs Distance\nTrain: Jan-Mar 2025, test: Jan 2026 (same sensors, site 1103)")

plt.xticks(results["Distance km"], [f"{d:g}" for d in results["Distance km"]])
plt.grid(True, alpha=0.3)

# Legend below the plot, so it never covers a point
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.savefig(
    os.path.join(RESULTS_DIR, PLOT_FILE),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nSaved plot to {os.path.join(RESULTS_DIR, PLOT_FILE)}")
print(f"Saved combined table to {os.path.join(RESULTS_DIR, COMBINED_CSV)}")
