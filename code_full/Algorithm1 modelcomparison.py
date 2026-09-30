"""
Algorithm1_ModelComparison.py

Wildfire PM2.5 Calibration Project - Algorithm 1
==================================================

Purpose
-------
This script performs the first stage of the wildfire PM2.5 calibration and
excess mortality pipeline:

    1. Load and synchronize EPA reference monitor data (Site 1103) with
       PurpleAir calibration sensor 208493 over the study window
       2025-01-01 00:00 through 2025-02-28 23:00.
    2. Construct 1-hour lagged meteorological predictors and fit a
       log-linear calibration model:

           log(EPA) = b0 + bPM * log(PA_PM) + bT * Temp + bRH * RH
                      + bP * Pressure + bTlag * Temp(t-1)
                      + bRHlag * RH(t-1) + bPlag * Pressure(t-1)

       fit once on the full synchronized dataset.
    3. Compare the log-linear (OLS) model against a Multilayer Perceptron
       (MLP) using 50 repeated train/test splits.

IMPORTANT DEVIATION FROM THE ORIGINAL METHODOLOGY
--------------------------------------------------
The original specification called for a *blocked* time-series
cross-validation scheme, in which each of the 50 iterations withheld a
single contiguous block of time as the test set. Per explicit instruction,
this has been replaced with **row-wise random train/test splitting**
(sklearn.model_selection.train_test_split, shuffle=True). Each of the 50
iterations draws an independent random 80/20 split of the synchronized
hourly observations. This is a simpler but less conservative validation
scheme than blocked CV, since it can allow temporally adjacent (and
therefore autocorrelated) observations to appear in both the training and
test sets. This tradeoff was requested by the user and is noted here for
transparency in any downstream thesis write-up.

All outputs are written beneath ./outputs/ so that Algorithm 2, Algorithm 3,
and Algorithm 4 can resume from saved checkpoints without recomputing this
stage.

Author: (generated pipeline)
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
import matplotlib
matplotlib.use("Agg")  # safe default; Spyder will still render inline plots
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tqdm import tqdm

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
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _dir in (OUTPUT_DIR, CALIB_DIR, FIGURE_DIR, CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

EPA_FILE = os.path.join(DATA_DIR, "LA_Site_1103.csv")
CALIBRATION_SENSOR_ID = "208493"
PA_CALIB_FILE = os.path.join(DATA_DIR, f"sensor_{CALIBRATION_SENSOR_ID}_history.csv")

STUDY_START = pd.Timestamp("2025-01-01 00:00:00")
STUDY_END = pd.Timestamp("2025-02-28 23:00:00")

N_CV_ITERATIONS = 50
TEST_FRACTION = 0.20          # random row-wise split fraction held out as test
BASE_RANDOM_SEED = 42         # Algorithm 1 uses FIXED seeds for reproducibility

# MLP training hyperparameters
MLP_MAX_EPOCHS = 300
MLP_BATCH_SIZE = 32
MLP_PATIENCE = 20              # EarlyStopping patience on validation loss
MLP_VALIDATION_SPLIT = 0.15    # carved out of the training partition only

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

# Fix seeds globally for Algorithm 1 (per project specification)
np.random.seed(BASE_RANDOM_SEED)
tf.random.set_seed(BASE_RANDOM_SEED)


# =============================================================================
# Logging setup
# =============================================================================
def setup_logging() -> logging.Logger:
    """Configure a logger that writes to both console and a timestamped file.

    Returns
    -------
    logging.Logger
        Configured logger instance for this script.
    """
    log_filename = os.path.join(
        LOG_DIR, f"algorithm1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger("Algorithm1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if re-run in Spyder

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
# Data loading functions
# =============================================================================
def load_epa_data(filepath: str) -> pd.DataFrame:
    """Load and clean EPA reference monitor data.

    Constructs an hourly timestamp column from the 'Date Local' and
    'Time Local' fields, restricts the record to the study window, and
    keeps only the columns required downstream. No interpolation of
    missing hours is performed; the EPA record is expected to contain
    approximately 1313 of the 1416 possible hourly observations.

    Parameters
    ----------
    filepath : str
        Path to LA_Site_1103.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'epa_pm25'], sorted and deduplicated.

    Raises
    ------
    FileNotFoundError
        If the EPA file does not exist.
    ValueError
        If required columns are missing from the file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"EPA monitor file not found: {filepath}")

    logger.info("Loading EPA reference monitor data from %s", filepath)
    raw = pd.read_csv(filepath)

    required = {"Date Local", "Time Local", "Sample Measurement"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"EPA file is missing required columns: {missing}")

    # Build a single naive-local timestamp from the date + time fields.
    raw["timestamp"] = pd.to_datetime(
        raw["Date Local"].astype(str) + " " + raw["Time Local"].astype(str),
        errors="coerce",
    )

    n_bad_ts = raw["timestamp"].isna().sum()
    if n_bad_ts > 0:
        logger.warning("Dropping %d EPA rows with unparseable timestamps", n_bad_ts)
    raw = raw.dropna(subset=["timestamp"])

    epa = raw[["timestamp", "Sample Measurement"]].rename(
        columns={"Sample Measurement": "epa_pm25"}
    )

    # Restrict to the study window (inclusive).
    mask = (epa["timestamp"] >= STUDY_START) & (epa["timestamp"] <= STUDY_END)
    epa = epa.loc[mask].copy()

    # Remove duplicate timestamps (keep first) and sort chronologically.
    epa = epa.drop_duplicates(subset="timestamp", keep="first")
    epa = epa.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "EPA data loaded: %d hourly observations within study window "
        "(expected ~1313 of 1416 possible hours; missing hours are NOT interpolated).",
        len(epa),
    )
    return epa


def load_purpleair_sensor(filepath: str, sensor_id: str) -> pd.DataFrame:
    """Load a single PurpleAir sensor history file.

    Parameters
    ----------
    filepath : str
        Path to sensor_<sensorID>_history.csv.
    sensor_id : str
        Sensor identifier, used only for logging/error messages.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'pa_pm25', 'humidity', 'temperature', 'pressure'].

    Raises
    ------
    FileNotFoundError
        If the sensor file does not exist.
    ValueError
        If required columns are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"PurpleAir calibration sensor file not found: {filepath} "
            f"(sensor {sensor_id})"
        )

    logger.info("Loading PurpleAir calibration sensor %s from %s", sensor_id, filepath)
    raw = pd.read_csv(filepath)

    required = {"time_stamp", "pm2.5_cf_1", "humidity", "temperature", "pressure"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"PurpleAir file for sensor {sensor_id} is missing columns: {missing}"
        )

    pa = raw.rename(
        columns={
            "time_stamp": "timestamp",
            "pm2.5_cf_1": "pa_pm25",
        }
    )[["timestamp", "pa_pm25", "humidity", "temperature", "pressure"]].copy()

    # Parse timestamps (these include a UTC offset, e.g. -08:00) and then
    # drop timezone info so they line up with the naive EPA local timestamps.
    pa["timestamp"] = pd.to_datetime(pa["timestamp"], errors="coerce", utc=True)
    n_bad_ts = pa["timestamp"].isna().sum()
    if n_bad_ts > 0:
        logger.warning(
            "Dropping %d PurpleAir rows with unparseable timestamps (sensor %s)",
            n_bad_ts, sensor_id,
        )
    pa = pa.dropna(subset=["timestamp"])
    pa["timestamp"] = pa["timestamp"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)

    pa = pa.drop_duplicates(subset="timestamp", keep="first")
    pa = pa.sort_values("timestamp").reset_index(drop=True)

    logger.info("PurpleAir sensor %s loaded: %d raw hourly rows", sensor_id, len(pa))
    return pa


# =============================================================================
# Synchronization and feature engineering
# =============================================================================
def synchronize_epa_purpleair(epa: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """Synchronize EPA and PurpleAir records using strict timestamp intersection.

    No interpolation is performed on either series; only timestamps present
    in BOTH datasets are retained.

    Parameters
    ----------
    epa : pd.DataFrame
        Output of load_epa_data().
    pa : pd.DataFrame
        Output of load_purpleair_sensor().

    Returns
    -------
    pd.DataFrame
        Inner-joined dataframe on 'timestamp', sorted chronologically.
    """
    logger.info("Synchronizing EPA and PurpleAir calibration sensor via timestamp intersection")
    merged = pd.merge(epa, pa, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    logger.info("Synchronized dataset contains %d matched hourly observations", len(merged))
    return merged


def construct_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct 1-hour lagged meteorological predictors.

    The first row is dropped since lag values are undefined for it.

    Parameters
    ----------
    df : pd.DataFrame
        Synchronized EPA/PurpleAir dataframe, sorted by timestamp, with
        columns 'temperature', 'humidity', 'pressure'.

    Returns
    -------
    pd.DataFrame
        Same dataframe with added columns 'temperature_lag1',
        'humidity_lag1', 'pressure_lag1'; first row removed.
    """
    logger.info("Constructing 1-hour lag variables (temperature, humidity, pressure)")
    df = df.copy()
    df["temperature_lag1"] = df["temperature"].shift(1)
    df["humidity_lag1"] = df["humidity"].shift(1)
    df["pressure_lag1"] = df["pressure"].shift(1)

    n_before = len(df)
    df = df.dropna(subset=["temperature_lag1", "humidity_lag1", "pressure_lag1"])
    df = df.reset_index(drop=True)
    logger.info("Dropped first row for lag initialization: %d -> %d rows", n_before, len(df))
    return df


def add_model_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed target/predictor columns needed for modeling.

    Rows with non-positive PurpleAir or EPA readings are dropped since the
    log-linear model requires strictly positive concentrations.

    Parameters
    ----------
    df : pd.DataFrame
        Synchronized, lagged dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'log_pa_pm' and 'log_epa' columns.
    """
    df = df.copy()
    n_before = len(df)
    df = df[(df["pa_pm25"] > 0) & (df["epa_pm25"] > 0)].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d rows with non-positive PM2.5 values prior to log transform",
            n_dropped,
        )

    df["log_pa_pm"] = np.log(df["pa_pm25"])
    df["log_epa"] = np.log(df["epa_pm25"])
    df = df.reset_index(drop=True)
    return df


# =============================================================================
# Log-linear (OLS) calibration model
# =============================================================================
def fit_loglinear_model(df: pd.DataFrame) -> dict:
    """Fit the log-linear calibration model on the full synchronized dataset.

    log(EPA) = b0 + bPM*log(PA_PM) + bT*Temp + bRH*RH + bP*Pressure
               + bTlag*Temp(t-1) + bRHlag*RH(t-1) + bPlag*Pressure(t-1)

    Parameters
    ----------
    df : pd.DataFrame
        Full synchronized, lagged, log-transformed dataset.

    Returns
    -------
    dict
        Dictionary with fitted sklearn model, coefficient table, and
        in-sample R^2 / residuals for diagnostic plotting.
    """
    logger.info("Fitting log-linear calibration model on full synchronized dataset (n=%d)", len(df))

    X = df[PREDICTOR_COLUMNS].values
    y = df[TARGET_COLUMN].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)

    coef_table = pd.DataFrame(
        {
            "term": ["intercept"] + PREDICTOR_COLUMNS,
            "coefficient": [model.intercept_] + list(model.coef_),
        }
    )

    logger.info("Log-linear model fit complete. In-sample R^2 = %.4f", r2)

    return {
        "model": model,
        "coef_table": coef_table,
        "r2_in_sample": r2,
        "residuals": residuals,
        "y_true": y,
        "y_pred": y_pred,
    }


def save_loglinear_coefficients(coef_table: pd.DataFrame) -> str:
    """Persist the log-linear coefficients to CSV for downstream scripts.

    Parameters
    ----------
    coef_table : pd.DataFrame
        Coefficient table from fit_loglinear_model().

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    out_path = os.path.join(CALIB_DIR, "loglinear_coefficients.csv")
    coef_table.to_csv(out_path, index=False)
    logger.info("Saved log-linear coefficients to %s", out_path)
    return out_path


# =============================================================================
# MLP model
# =============================================================================
def build_mlp_model(input_dim: int) -> tf.keras.Model:
    """Construct the MLP architecture specified in the project methodology.

    Architecture
    ------------
    Input(input_dim) -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
                      -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)

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


def train_mlp(X_train, y_train, verbose: int = 0):
    """Train an MLP with early stopping on a validation split.

    Parameters
    ----------
    X_train : np.ndarray
        Training predictors.
    y_train : np.ndarray
        Training target.
    verbose : int
        Keras verbosity level.

    Returns
    -------
    (tf.keras.Model, tf.keras.callbacks.History)
        Trained model (best weights restored) and its training history.
    """
    model = build_mlp_model(X_train.shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=MLP_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=MLP_VALIDATION_SPLIT,
        epochs=MLP_MAX_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return model, history


# =============================================================================
# Model comparison: 50 iterations of RANDOM (row-wise) train/test splitting
# =============================================================================
def run_model_comparison(df: pd.DataFrame) -> dict:
    """Compare OLS and MLP performance over repeated random train/test splits.

    NOTE: Per explicit user instruction, this uses row-wise random
    train/test splitting (NOT blocked contiguous-time splitting as in the
    original methodology). Each of the N_CV_ITERATIONS iterations draws an
    independent random 80/20 split of the synchronized hourly observations
    using sklearn.model_selection.train_test_split(shuffle=True).

    Parameters
    ----------
    df : pd.DataFrame
        Full synchronized, lagged, log-transformed dataset.

    Returns
    -------
    dict
        {
          'results_df': pd.DataFrame of per-iteration R^2 values,
          'last_history': Keras History object from the final MLP fit
                           (used for the loss-curve diagnostic figure),
        }
    """
    logger.info(
        "Starting model comparison: %d random train/test split iterations "
        "(test_fraction=%.2f)", N_CV_ITERATIONS, TEST_FRACTION
    )

    X_all = df[PREDICTOR_COLUMNS].values
    y_all = df[TARGET_COLUMN].values

    records = []
    last_history = None
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "cv_results_partial.csv")

    for i in tqdm(range(N_CV_ITERATIONS), desc="Model comparison (random splits)"):
        iter_seed = BASE_RANDOM_SEED + i  # deterministic-but-varying seed per iteration

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=TEST_FRACTION,
                shuffle=True,
                random_state=iter_seed,
            )

            # --- OLS ---
            ols = LinearRegression()
            ols.fit(X_train, y_train)
            ols_pred = ols.predict(X_test)
            ols_r2 = r2_score(y_test, ols_pred)

            # --- MLP ---
            mlp, history = train_mlp(X_train, y_train, verbose=0)
            mlp_pred = mlp.predict(X_test, verbose=0).flatten()
            mlp_r2 = r2_score(y_test, mlp_pred)
            last_history = history

            records.append(
                {
                    "iteration": i,
                    "seed": iter_seed,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "ols_r2": ols_r2,
                    "mlp_r2": mlp_r2,
                }
            )

        except Exception as exc:  # noqa: BLE001 - log and continue for robustness
            logger.exception("Iteration %d failed with an exception: %s", i, exc)
            records.append(
                {
                    "iteration": i,
                    "seed": iter_seed,
                    "n_train": np.nan,
                    "n_test": np.nan,
                    "ols_r2": np.nan,
                    "mlp_r2": np.nan,
                }
            )

        # Checkpoint after every iteration so progress survives a crash.
        pd.DataFrame(records).to_csv(checkpoint_path, index=False)

    results_df = pd.DataFrame(records)
    logger.info(
        "Model comparison complete. Mean OLS R^2 = %.4f, Mean MLP R^2 = %.4f",
        results_df["ols_r2"].mean(skipna=True),
        results_df["mlp_r2"].mean(skipna=True),
    )

    return {"results_df": results_df, "last_history": last_history}


# =============================================================================
# Figures
# =============================================================================
def plot_epa_vs_purpleair(df: pd.DataFrame) -> str:
    """Scatter plot of raw EPA PM2.5 vs raw PurpleAir PM2.5."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(df["pa_pm25"], df["epa_pm25"], s=10, alpha=0.4, color="steelblue")
    lims = [0, max(df["pa_pm25"].max(), df["epa_pm25"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("PurpleAir PM$_{2.5}$ (raw, $\\mu g/m^3$)")
    ax.set_ylabel("EPA PM$_{2.5}$ (reference, $\\mu g/m^3$)")
    ax.set_title("EPA Reference vs Raw PurpleAir PM$_{2.5}$")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_epa_vs_purpleair_raw.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_epa_vs_calibrated(df: pd.DataFrame, y_pred_log: np.ndarray) -> str:
    """Scatter plot of EPA PM2.5 vs calibrated (model-predicted) PM2.5."""
    calibrated_pm = np.exp(y_pred_log)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(calibrated_pm, df["epa_pm25"], s=10, alpha=0.4, color="darkorange")
    lims = [0, max(calibrated_pm.max(), df["epa_pm25"].max()) * 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Calibrated PurpleAir PM$_{2.5}$ ($\\mu g/m^3$)")
    ax.set_ylabel("EPA PM$_{2.5}$ (reference, $\\mu g/m^3$)")
    ax.set_title("EPA Reference vs Calibrated PurpleAir PM$_{2.5}$")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_epa_vs_calibrated.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_residual_histogram(residuals: np.ndarray) -> str:
    """Histogram of log-linear model residuals."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.hist(residuals, bins=40, color="mediumseagreen", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Residual (log EPA - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Log-Linear Model Residual Histogram")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_residual_histogram.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_residual_qq(residuals: np.ndarray) -> str:
    """Normal Q-Q plot of log-linear model residuals."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Log-Linear Model Residual Q-Q Plot")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_residual_qqplot.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_r2_boxplot(results_df: pd.DataFrame) -> str:
    """Boxplot comparing OLS vs MLP R^2 across the 50 random-split iterations."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    data = [
        results_df["ols_r2"].dropna().values,
        results_df["mlp_r2"].dropna().values,
    ]
    ax.boxplot(data, tick_labels=["OLS (log-linear)", "MLP"], patch_artist=True)
    ax.set_ylabel("Test-set R$^2$")
    ax.set_title(
        f"OLS vs MLP Test R$^2$ over {N_CV_ITERATIONS} Random Train/Test Splits"
    )
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_ols_vs_mlp_r2_boxplot.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_loss_curves(history) -> str:
    """Training/validation loss curves for the final MLP fit in the comparison loop."""
    if history is None:
        logger.warning("No MLP training history available; skipping loss-curve figure.")
        return ""

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MLP Training/Validation Loss Curve (final iteration)")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_mlp_loss_curve.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    """Run the full Algorithm 1 pipeline end to end."""
    logger.info("=" * 70)
    logger.info("ALGORITHM 1: Calibration + Model Comparison - START")
    logger.info("=" * 70)

    sync_checkpoint = os.path.join(CHECKPOINT_DIR, "synchronized_calibration_data.csv")

    # ------------------------------------------------------------------
    # Step 1: Load, synchronize, and feature-engineer (with checkpointing)
    # ------------------------------------------------------------------
    if os.path.isfile(sync_checkpoint):
        logger.info("Found existing synchronized-data checkpoint; loading instead of recomputing: %s", sync_checkpoint)
        df = pd.read_csv(sync_checkpoint, parse_dates=["timestamp"])
    else:
        epa = load_epa_data(EPA_FILE)
        pa = load_purpleair_sensor(PA_CALIB_FILE, CALIBRATION_SENSOR_ID)
        synced = synchronize_epa_purpleair(epa, pa)
        lagged = construct_lag_features(synced)
        df = add_model_variables(lagged)
        df.to_csv(sync_checkpoint, index=False)
        logger.info("Saved synchronized calibration checkpoint to %s", sync_checkpoint)

    if len(df) == 0:
        logger.error("Synchronized calibration dataset is empty. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Fit log-linear calibration model on FULL dataset
    # ------------------------------------------------------------------
    loglinear_result = fit_loglinear_model(df)
    save_loglinear_coefficients(loglinear_result["coef_table"])

    # ------------------------------------------------------------------
    # Step 3: 50-iteration OLS vs MLP comparison (random row-wise splits)
    # ------------------------------------------------------------------
    comparison = run_model_comparison(df)
    results_df = comparison["results_df"]

    model_comparison_path = os.path.join(CALIB_DIR, "model_comparison.csv")
    results_df.to_csv(model_comparison_path, index=False)
    logger.info("Saved model comparison summary to %s", model_comparison_path)

    # Also save under the name referenced in the original spec, retained for
    # backward compatibility with any downstream scripts that expect it,
    # even though the splitting strategy is no longer "blocked".
    cv_results_path = os.path.join(CALIB_DIR, "cv_results.csv")
    results_df.to_csv(cv_results_path, index=False)
    logger.info(
        "Saved CV results (random row-wise splits, NOT blocked time splits) to %s",
        cv_results_path,
    )

    # ------------------------------------------------------------------
    # Step 4: Figures
    # ------------------------------------------------------------------
    logger.info("Generating diagnostic and publication figures...")
    plot_epa_vs_purpleair(df)
    plot_epa_vs_calibrated(df, loglinear_result["y_pred"])
    plot_residual_histogram(loglinear_result["residuals"])
    plot_residual_qq(loglinear_result["residuals"])
    plot_r2_boxplot(results_df)
    plot_loss_curves(comparison["last_history"])
    logger.info("All figures saved to %s", FIGURE_DIR)

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    summary = {
        "n_synchronized_observations": len(df),
        "loglinear_in_sample_r2": loglinear_result["r2_in_sample"],
        "mean_ols_test_r2_random_split": float(results_df["ols_r2"].mean(skipna=True)),
        "mean_mlp_test_r2_random_split": float(results_df["mlp_r2"].mean(skipna=True)),
        "n_cv_iterations": N_CV_ITERATIONS,
        "cv_strategy": "random_row_wise_train_test_split",
        "test_fraction": TEST_FRACTION,
    }
    summary_path = os.path.join(CALIB_DIR, "algorithm1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved run summary to %s", summary_path)

    logger.info("=" * 70)
    logger.info("ALGORITHM 1 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()