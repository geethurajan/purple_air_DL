"""
Plot the results of hpc_two_stage_jan2026_different_tries.py on one page:

    left:  step 1, the validation adjusted R² (Jan-Feb 2025 validation days) of every MLP version,
           mean ± sd over the seeds, best first; the chosen version is highlighted and the linear
           models' validation scores are drawn as reference lines
    right: step 2, the January 2026 test adjusted R² of the chosen MLP version (one dot per seed,
           bar = mean, line = ± sd) next to the two linear models

Run from the training data folder or from inside jan2026_test:
    python jan2026_test/plot_two_stage_jan2026_different_tries.py
    python plot_two_stage_jan2026_different_tries.py
"""

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed (works on HPC)
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

# Folder with the DifferentTries_*.csv files, next to this script
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_different_tries")
DIST_LABEL = "2.5km"

VALIDATION_FILE = f"DifferentTries_Validation_{DIST_LABEL}.csv"
TEST_FILE = f"DifferentTries_Test_{DIST_LABEL}.csv"
PLOT_FILE = f"different_tries_{DIST_LABEL}.png"


# ============================================================
# Read results
# ============================================================

val = pd.read_csv(os.path.join(RESULTS_DIR, VALIDATION_FILE))
test = pd.read_csv(os.path.join(RESULTS_DIR, TEST_FILE))

val = val.sort_values("Val Adj R2 mean", ascending=False).reset_index(drop=True)

val["label"] = [
    f"{f} | {h} | wd {w:g}"
    for f, h, w in zip(val["f"], val["Hidden layers"], val["Weight decay"])
]

# Step 1 chooses the best mean validation score (the first row)
chosen_label = val.loc[0, "label"]

print(val[["label", "Val Adj R2 mean", "Val Adj R2 sd"]].round(4).to_string(index=False))
print(f"\nChosen: {chosen_label}")


# ============================================================
# Plot
# ============================================================

fig, (ax_val, ax_test) = plt.subplots(
    1, 2,
    figsize=(14, 6.5),
    gridspec_kw={"width_ratios": [1.6, 1]}
)

# ---- Left: step 1, validation scores of every version ----------

y = range(len(val))[::-1]   # best version at the top

ax_val.barh(
    list(y),
    val["Val Adj R2 mean"],
    xerr=val["Val Adj R2 sd"],
    color=["tab:green" if label == chosen_label else "lightgray" for label in val["label"]],
    edgecolor="gray",
    capsize=2
)

ax_val.set_yticks(list(y))
ax_val.set_yticklabels(val["label"], fontsize=8)

# The linear models on the same validation days, as reference lines
for column, label, style in [
    ("log_trans_lin_model Val Adj R2", "log_trans_lin_model", "--"),
    ("reg_lin_model Val Adj R2", "reg_lin_model", ":"),
]:
    if column in val.columns:
        ax_val.axvline(val[column].iloc[0], color="black", linestyle=style, label=label)

ax_val.set_xlabel("Adjusted R² (validation: alternate days of Jan-Feb 2025)")
ax_val.set_title(f"Step 1: choose the MLP version ({DIST_LABEL})\nmean ± sd over seeds, green = chosen")
ax_val.grid(True, axis="x", alpha=0.3)

if ax_val.get_legend_handles_labels()[0]:
    ax_val.legend(loc="lower right", fontsize=8)

# ---- Right: step 2, January 2026 test --------------------------

mlp = test["MLP Test Adj R2"]

models = [
    ("log_trans_lin_model", test["log_trans_lin_model Test Adj R2"].iloc[0], "tab:blue"),
    ("reg_lin_model", test["reg_lin_model Test Adj R2"].iloc[0], "tab:orange"),
    ("MLP (chosen)", mlp.mean(), "tab:green"),
]

for i, (label, value, color) in enumerate(models):
    ax_test.bar(i, value, color=color, alpha=0.8)
    ax_test.annotate(f"{value:.3f}", (i, value), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9)

# One dot per seed for the MLP, and its ± sd
mlp_x = len(models) - 1

ax_test.scatter([mlp_x] * len(mlp), mlp, color="black", s=15, zorder=3, label="MLP, one dot per seed")
ax_test.errorbar(mlp_x, mlp.mean(), yerr=mlp.std(ddof=0), color="black", capsize=4)

ax_test.set_xticks(range(len(models)))
ax_test.set_xticklabels([m[0] for m in models], fontsize=9)
ax_test.set_ylabel("Adjusted R² (test: January 2026)")
ax_test.set_title(f"Step 2: test on January 2026 ({DIST_LABEL})\n"
                  f"MLP: {chosen_label}, mean of {len(mlp)} seeds")
ax_test.grid(True, axis="y", alpha=0.3)
ax_test.legend(loc="lower right", fontsize=8)

fig.suptitle("Different tries for the MLP: train Jan-Feb 2025, test Jan 2026 (site 1103)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))

fig.savefig(
    os.path.join(RESULTS_DIR, PLOT_FILE),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"\nSaved plot to {os.path.join(RESULTS_DIR, PLOT_FILE)}")
