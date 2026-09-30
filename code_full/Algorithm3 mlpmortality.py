# -*- coding: utf-8 -*-
"""
Algorithm3_MLPMortality.py

Wildfire PM2.5 Calibration Project - Algorithm 3
==================================================

Purpose
-------
This script is the MLP analogue of Algorithm 2. Instead of a single
log-linear calibration model, it repeats the following 50 times, with a
freshly (randomly) initialized MLP each time:

    1. Train an MLP calibration model on the synchronized EPA / PurpleAir
       sensor-208493 dataset (identical 7 predictors used throughout this
       project).
    2. Apply that MLP to every retained PurpleAir sensor in the network.
    3. Average calibrated PM2.5 to daily resolution per sensor.
    4. Interpolate a daily citywide PM2.5 surface with a thin plate spline
       and evaluate it at every census tract centroid.
    5. Compute tract-level mortality M_i and its within-estimate variance
       U_i (from the gamma coefficient's 95% CI), using the SAME
       wildfire-vs-baseline DeltaPM definition established in Algorithm 2:
       DeltaPM(tract, day) = PM2.5(tract, day) - baseline_PM2.5(tract),
       where 'day' ranges over the wildfire window (Jan 7-31) and
       baseline_PM2.5 is each tract's mean PM2.5 over the non-fire window
       (Feb 1-28).

After 50 iterations, the final MLP-based mortality estimate per tract is
combined using a multiple-imputation-style rule (per the project
specification):

    Mean estimate = mean_i(M_i)
    Total variance = mean_i(U_i) + Var_i(M_i)
    95% CI = Mean +/- 1.96 * sqrt(Total variance)

MLP architecture
-----------------
This script uses the MINIMALIST single-hidden-layer architecture (matching
the version now used in Algorithm1_ModelComparison.py):

    Input(7) -> Dense(16, relu) -> Dropout(0.2) -> Dense(1)

Random seeds
------------
Per the project specification, Algorithm 3 does NOT fix numpy or
TensorFlow random seeds. Every one of the 50 retrainings begins from a
fresh, independently random weight initialization and a fresh random
train/validation split for early stopping.

All outputs are written beneath ./outputs/ and checkpointed after every
iteration so the run can be resumed (or partially inspected) without
recomputation. This script depends on checkpoint files produced by
Algorithm1_ModelComparison.py and Algorithm2_LogLinearMortality.py; run
those first.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CALIB_DIR = os.path.join(OUTPUT_DIR, "calibration")
MORTALITY_DIR = os.path.join(OUTPUT_DIR, "mortality")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _dir in (OUTPUT_DIR, CALIB_DIR, MORTALITY_DIR, CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

EPA_FILE = os.path.join(DATA_DIR, "LA_Site_1103.csv")
TRACT_FILE = os.path.join(DATA_DIR, "la_city_tracts_final.csv")

SYNCED_CALIBRATION_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "synchronized_calibration_data.csv"
)
SENSOR_DESIGN_MATRIX_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "sensor_design_matrix.csv"
)

STUDY_START = pd.Timestamp("2025-01-01 00:00:00")
STUDY_END = pd.Timestamp("2025-02-28 23:00:00")

# Wildfire-exposure window: mortality is calculated only for these days.
WILDFIRE_START = pd.Timestamp("2025-01-07").date()
WILDFIRE_END = pd.Timestamp("2025-01-31").date()

# Non-fire baseline window: per-tract average PM2.5 here is subtracted from
# each wildfire-period day to form DeltaPM (identical convention to
# Algorithm 2).
BASELINE_START = pd.Timestamp("2025-02-01").date()
BASELINE_END = pd.Timestamp("2025-02-28").date()

MIN_SENSORS_FOR_SPLINE = 4     # minimum sensors required to fit a TPS surface for a day

RBF_KERNEL = "thin_plate_spline"
RBF_SMOOTHING = 1e-3

# Mortality constants (identical to Algorithm 2)
GAMMA_POINT = 0.00094
GAMMA_LOWER = 0.00073
GAMMA_UPPER = 0.00116
BASELINE_MORTALITY_PER_100K_YEAR = 780.0  # Bd

N_MLP_ITERATIONS = 50

# MLP training hyperparameters (minimalist architecture)
MLP_MAX_EPOCHS = 300
MLP_BATCH_SIZE = 32
MLP_PATIENCE = 20
MLP_VALIDATION_SPLIT = 0.15

PREDICTOR_COLUMNS = [
    "log_pa_pm",
    "temperature",
    "humidity",
    "pressure",
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1",
]
TARGET_COLUMN = "log_epa"

# NOTE: No np.random.seed() / tf.random.set_seed() calls in this script.
# Per the project specification, Algorithm 3 must use fresh random
# initialization for every retraining.


# =============================================================================
# Logging
# =============================================================================
def setup_logging() -> logging.Logger:
    """Configure console + file logging for this script."""
    log_filename = os.path.join(
        LOG_DIR, f"algorithm3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("Algorithm3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# =============================================================================
# Loading
# =============================================================================
def load_calibration_dataset(filepath: str) -> pd.DataFrame:
    """Load the synchronized EPA/PurpleAir-208493 dataset checkpointed by
    Algorithm 1.

    Parameters
    ----------
    filepath : str
        Path to the synchronized_calibration_data.csv checkpoint.

    Returns
    -------
    pd.DataFrame
        Synchronized calibration dataset with PREDICTOR_COLUMNS and
        TARGET_COLUMN.

    Raises
    ------
    FileNotFoundError
        If Algorithm 1 has not yet been run.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Calibration dataset checkpoint not found at {filepath}. "
            "Run Algorithm1_ModelComparison.py first."
        )
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    logger.info("Loaded synchronized calibration dataset: %d rows", len(df))
    return df


def load_sensor_design_matrix(filepath: str) -> pd.DataFrame:
    """Load the full PurpleAir network design matrix checkpointed by
    Algorithm 2.

    Parameters
    ----------
    filepath : str
        Path to the sensor_design_matrix.csv checkpoint.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per (sensor, hour), containing
        PREDICTOR_COLUMNS, 'sensor', 'date', 'x_km', 'y_km'.

    Raises
    ------
    FileNotFoundError
        If Algorithm 2 has not yet been run.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Sensor design matrix checkpoint not found at {filepath}. "
            "Run Algorithm2_LogLinearMortality.py first."
        )
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info(
        "Loaded sensor design matrix: %d sensors, %d hourly rows",
        df["sensor"].nunique(), len(df),
    )
    return df


def load_tract_data(filepath: str) -> pd.DataFrame:
    """Load census tract population and centroid coordinates.

    Parameters
    ----------
    filepath : str
        Path to la_city_tracts_final.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['GEOID', 'NAME', 'population', 'x_km', 'y_km', ...].
    """
    tracts = pd.read_csv(filepath, dtype={"GEOID": str})
    required = {"GEOID", "NAME", "population", "x_km", "y_km"}
    missing = required - set(tracts.columns)
    if missing:
        raise ValueError(f"Tract file is missing required columns: {missing}")
    logger.info("Loaded %d census tracts", len(tracts))
    return tracts


# =============================================================================
# MLP model (minimalist architecture)
# =============================================================================
def build_mlp_model(input_dim: int) -> tf.keras.Model:
    """Construct the minimalist MLP architecture.

    Architecture
    ------------
    Input(input_dim) -> Dense(16, relu) -> Dropout(0.2) -> Dense(1)

    Parameters
    ----------
    input_dim : int
        Number of input predictors (7).

    Returns
    -------
    tf.keras.Model
        Compiled Keras model using Adam optimizer and MSE loss.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_mlp_fresh(X_train: np.ndarray, y_train: np.ndarray):
    """Train a freshly initialized MLP with early stopping.

    No random seed is set here or anywhere in this script; each call
    produces an independently random weight initialization, as required
    for Algorithm 3.

    Parameters
    ----------
    X_train : np.ndarray
        Training predictors.
    y_train : np.ndarray
        Training target (log EPA PM2.5).

    Returns
    -------
    tf.keras.Model
        Trained model with best (lowest validation loss) weights restored.
    """
    model = build_mlp_model(X_train.shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=MLP_PATIENCE,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=MLP_VALIDATION_SPLIT,
        epochs=MLP_MAX_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    return model


# =============================================================================
# Calibration application + daily averaging
# =============================================================================
def apply_mlp_calibration_daily_average(design_df: pd.DataFrame, mlp_model) -> pd.DataFrame:
    """Apply a trained MLP calibration model and average to daily PM.

    Parameters
    ----------
    design_df : pd.DataFrame
        Full-network sensor design matrix (PREDICTOR_COLUMNS present).
    mlp_model : tf.keras.Model
        Trained MLP predicting log(EPA-equivalent PM2.5).

    Returns
    -------
    pd.DataFrame
        Columns: ['sensor', 'date', 'x_km', 'y_km', 'calibrated_pm']
        (one row per sensor-day).
    """
    X = design_df[PREDICTOR_COLUMNS].values
    log_calibrated = mlp_model.predict(X, verbose=0).flatten()
    calibrated_pm = np.exp(log_calibrated)

    hourly = design_df[["sensor", "date", "x_km", "y_km"]].copy()
    hourly["calibrated_pm"] = calibrated_pm

    daily = (
        hourly.groupby(["sensor", "date"], as_index=False)
        .agg(x_km=("x_km", "first"), y_km=("y_km", "first"), calibrated_pm=("calibrated_pm", "mean"))
    )
    return daily


# =============================================================================
# Thin plate spline interpolation (identical convention to Algorithm 2)
# =============================================================================
def interpolate_daily_tract_pm(daily_sensor_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Interpolate a thin-plate-spline PM2.5 surface for each day and
    evaluate it at every tract centroid.

    Parameters
    ----------
    daily_sensor_pm : pd.DataFrame
        One row per sensor-day with x_km, y_km, calibrated_pm.
    tracts : pd.DataFrame
        Tract table with GEOID, x_km, y_km.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'].
    """
    tract_xy = tracts[["x_km", "y_km"]].values
    tract_geoid = tracts["GEOID"].values

    records = []
    dates = sorted(daily_sensor_pm["date"].unique())

    for day in dates:
        day_data = daily_sensor_pm.loc[daily_sensor_pm["date"] == day].dropna(subset=["calibrated_pm"])

        if len(day_data) < MIN_SENSORS_FOR_SPLINE:
            continue

        sensor_xy = day_data[["x_km", "y_km"]].values
        sensor_values = day_data["calibrated_pm"].values

        try:
            rbf = RBFInterpolator(
                sensor_xy, sensor_values, kernel=RBF_KERNEL, smoothing=RBF_SMOOTHING,
            )
            tract_values = rbf(tract_xy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spline interpolation failed for %s: %s", day, exc)
            continue

        records.append(pd.DataFrame({"date": day, "GEOID": tract_geoid, "pm25": tract_values}))

    if not records:
        raise RuntimeError("No days produced a valid interpolated PM surface.")

    return pd.concat(records, ignore_index=True)


# =============================================================================
# Mortality calculation (identical DeltaPM convention to Algorithm 2)
# =============================================================================
def compute_tract_baseline_pm(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.Series:
    """Compute each tract's non-fire baseline PM2.5 (mean over Feb 1-28).

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'], covering the full study window.
    tracts : pd.DataFrame
        Tract table with 'GEOID'.

    Returns
    -------
    pd.Series
        Indexed by GEOID: mean PM2.5 over the non-fire baseline window.
    """
    mask = (daily_tract_pm["date"] >= BASELINE_START) & (daily_tract_pm["date"] <= BASELINE_END)
    baseline_df = daily_tract_pm.loc[mask]

    if baseline_df.empty:
        raise RuntimeError(
            f"No interpolated PM2.5 data found in the baseline window "
            f"({BASELINE_START} to {BASELINE_END}); cannot compute DeltaPM."
        )

    baseline = baseline_df.groupby("GEOID")["pm25"].mean()
    network_mean = baseline.mean()

    baseline = baseline.reindex(tracts["GEOID"])
    n_missing = baseline.isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d tract(s) had no baseline-window PM2.5; filling with the "
            "network-wide baseline mean (%.3f).", n_missing, network_mean,
        )
        baseline = baseline.fillna(network_mean)

    return baseline


def compute_delta_pm(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Compute DeltaPM = wildfire-day PM2.5 - tract non-fire baseline PM2.5.

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'], covering the full study window.
    tracts : pd.DataFrame
        Tract table with 'GEOID'.

    Returns
    -------
    pd.DataFrame
        Wildfire-window rows only (Jan 7-31), with an added 'delta_pm' column.
    """
    wildfire_mask = (daily_tract_pm["date"] >= WILDFIRE_START) & (daily_tract_pm["date"] <= WILDFIRE_END)
    wildfire_df = daily_tract_pm.loc[wildfire_mask].copy()

    if wildfire_df.empty:
        raise RuntimeError(
            f"No interpolated PM2.5 data found in the wildfire window "
            f"({WILDFIRE_START} to {WILDFIRE_END}); cannot compute mortality."
        )

    baseline = compute_tract_baseline_pm(daily_tract_pm, tracts)
    wildfire_df = wildfire_df.merge(
        baseline.rename("baseline_pm"), left_on="GEOID", right_index=True, how="left"
    )
    wildfire_df["delta_pm"] = wildfire_df["pm25"] - wildfire_df["baseline_pm"]
    return wildfire_df


def compute_tract_mortality(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame, gamma: float) -> pd.Series:
    """Compute total (summed-over-wildfire-days) excess mortality per tract
    for a single gamma value.

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'], covering the full study window.
    tracts : pd.DataFrame
        Tract table with 'GEOID' and 'population'.
    gamma : float
        Concentration-response coefficient to use.

    Returns
    -------
    pd.Series
        Indexed by GEOID, summed mortality over the wildfire window.
    """
    df = compute_delta_pm(daily_tract_pm, tracts)
    df = df.merge(tracts[["GEOID", "population"]], on="GEOID", how="left")

    rr = np.exp(gamma * df["delta_pm"])
    af = (rr - 1.0) / rr
    daily_mortality = af * BASELINE_MORTALITY_PER_100K_YEAR * df["population"] / 100_000.0 / 365.0

    df["daily_mortality"] = daily_mortality
    tract_totals = df.groupby("GEOID")["daily_mortality"].sum()
    tract_totals = tract_totals.reindex(tracts["GEOID"], fill_value=0.0)
    return tract_totals


def compute_mi_ui(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame):
    """Compute one iteration's point mortality M_i and within-estimate
    variance U_i for every tract.

    U_i = ((M_hi - M_lo) / (2 * 1.96)) ** 2, using the gamma 95% CI bounds.

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Daily interpolated tract PM2.5 for this iteration's MLP.
    tracts : pd.DataFrame
        Tract table.

    Returns
    -------
    (pd.Series, pd.Series)
        (M_i, U_i), both indexed by GEOID.
    """
    m_point = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_POINT)
    m_lo = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_LOWER)
    m_hi = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_UPPER)
    u_i = ((m_hi - m_lo) / (2 * 1.96)) ** 2
    return m_point, u_i


# =============================================================================
# Main MLP iteration loop
# =============================================================================
def run_mlp_iterations(calibration_df: pd.DataFrame, design_df: pd.DataFrame, tracts: pd.DataFrame):
    """Run the full 50-iteration MLP calibration/interpolation/mortality loop.

    Each iteration trains a freshly initialized MLP (no seed fixing),
    applies it to the full sensor network, interpolates a daily PM2.5
    surface, and computes per-tract M_i / U_i.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Synchronized EPA/PurpleAir-208493 dataset.
    design_df : pd.DataFrame
        Full-network sensor design matrix.
    tracts : pd.DataFrame
        Tract table.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        (m_matrix, u_matrix): rows = iteration, columns = GEOID.
    """
    X_all = calibration_df[PREDICTOR_COLUMNS].values
    y_all = calibration_df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    m_rows = []
    u_rows = []

    m_checkpoint = os.path.join(CHECKPOINT_DIR, "mlp_mortality_M_partial.csv")
    u_checkpoint = os.path.join(CHECKPOINT_DIR, "mlp_mortality_U_partial.csv")

    for i in tqdm(range(N_MLP_ITERATIONS), desc="MLP iterations (fresh init)"):
        try:
            # The full synchronized calibration dataset is used for training
            # (matching the "train once on synchronized observations" usage
            # pattern established in Algorithm 1/2); EarlyStopping carves its
            # own validation split internally via MLP_VALIDATION_SPLIT, and
            # no seed is fixed, so this split and the weight initialization
            # are both fresh and random on every iteration.
            mlp_model = train_mlp_fresh(X_all, y_all)

            design_scaled = design_df.copy()
            design_scaled[PREDICTOR_COLUMNS] = scaler.transform(design_df[PREDICTOR_COLUMNS].values)
            daily_sensor_pm = apply_mlp_calibration_daily_average(design_scaled, mlp_model)
            daily_tract_pm = interpolate_daily_tract_pm(daily_sensor_pm, tracts)
            m_i, u_i = compute_mi_ui(daily_tract_pm, tracts)

            m_row = m_i.to_dict()
            m_row["iteration"] = i
            u_row = u_i.to_dict()
            u_row["iteration"] = i

            m_rows.append(m_row)
            u_rows.append(u_row)

        except Exception as exc:  # noqa: BLE001
            logger.exception("MLP iteration %d failed: %s", i, exc)

        # Checkpoint after every iteration.
        pd.DataFrame(m_rows).to_csv(m_checkpoint, index=False)
        pd.DataFrame(u_rows).to_csv(u_checkpoint, index=False)

    m_matrix = pd.DataFrame(m_rows).set_index("iteration")
    u_matrix = pd.DataFrame(u_rows).set_index("iteration")

    logger.info(
        "MLP iteration loop complete: %d / %d iterations succeeded",
        len(m_matrix), N_MLP_ITERATIONS,
    )
    return m_matrix, u_matrix


def combine_mlp_estimates(m_matrix: pd.DataFrame, u_matrix: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Combine per-iteration M_i / U_i into a final MLP-based mortality
    estimate per tract.

        Mean estimate  = mean_i(M_i)
        Total variance = mean_i(U_i) + Var_i(M_i)
        95% CI         = Mean +/- 1.96 * sqrt(Total variance)

    Parameters
    ----------
    m_matrix : pd.DataFrame
        Rows = iteration, columns = GEOID; point mortality per iteration.
    u_matrix : pd.DataFrame
        Rows = iteration, columns = GEOID; within-estimate variance per
        iteration.
    tracts : pd.DataFrame
        Tract table with 'GEOID', 'NAME', 'population'.

    Returns
    -------
    pd.DataFrame
        Columns: ['GEOID', 'NAME', 'population', 'M_mean', 'mean_U',
        'between_iteration_variance', 'total_variance', 'CI_lower',
        'CI_upper'].
    """
    mean_m = m_matrix.mean(axis=0)
    between_variance = m_matrix.var(axis=0, ddof=1)
    mean_u = u_matrix.mean(axis=0)

    total_variance = mean_u + between_variance
    ci_half_width = 1.96 * np.sqrt(total_variance)

    result = tracts[["GEOID", "NAME", "population"]].copy().set_index("GEOID")
    result["M_mean"] = mean_m
    result["mean_U"] = mean_u
    result["between_iteration_variance"] = between_variance
    result["total_variance"] = total_variance
    result["CI_lower"] = result["M_mean"] - ci_half_width
    result["CI_upper"] = result["M_mean"] + ci_half_width
    result = result.reset_index()

    return result


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    """Run the full Algorithm 3 pipeline end to end."""
    logger.info("=" * 70)
    logger.info("ALGORITHM 3: MLP Calibration + Mortality (fresh init x50) - START")
    logger.info("=" * 70)

    calibration_df = load_calibration_dataset(SYNCED_CALIBRATION_CHECKPOINT)
    design_df = load_sensor_design_matrix(SENSOR_DESIGN_MATRIX_CHECKPOINT)
    tracts = load_tract_data(TRACT_FILE)

    m_matrix, u_matrix = run_mlp_iterations(calibration_df, design_df, tracts)

    if m_matrix.empty:
        logger.error("All MLP iterations failed. Aborting.")
        sys.exit(1)

    m_matrix_path = os.path.join(MORTALITY_DIR, "mlp_mortality_M_by_iteration.csv")
    u_matrix_path = os.path.join(MORTALITY_DIR, "mlp_mortality_U_by_iteration.csv")
    m_matrix.to_csv(m_matrix_path)
    u_matrix.to_csv(u_matrix_path)
    logger.info("Saved per-iteration M and U matrices to %s and %s", m_matrix_path, u_matrix_path)

    final_estimate = combine_mlp_estimates(m_matrix, u_matrix, tracts)
    final_path = os.path.join(MORTALITY_DIR, "tract_mortality_mlp.csv")
    final_estimate.to_csv(final_path, index=False)
    logger.info("Saved final MLP tract mortality estimates to %s", final_path)

    summary = {
        "n_successful_iterations": int(len(m_matrix)),
        "n_target_iterations": N_MLP_ITERATIONS,
        "citywide_total_mortality_mean": float(final_estimate["M_mean"].sum()),
        "citywide_total_mortality_CI_lower": float(final_estimate["CI_lower"].sum()),
        "citywide_total_mortality_CI_upper": float(final_estimate["CI_upper"].sum()),
        "mlp_architecture": "Input(7) -> Dense(16, relu) -> Dropout(0.2) -> Dense(1)",
        "seeds_fixed": False,
    }
    summary_path = os.path.join(MORTALITY_DIR, "algorithm3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved run summary to %s", summary_path)
    logger.info("Summary: %s", summary)

    logger.info("=" * 70)
    logger.info("ALGORITHM 3 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
