# -*- coding: utf-8 -*-
"""
Algorithm2_LogLinearMortality.py

Wildfire PM2.5 Calibration Project - Algorithm 2
==================================================

Purpose
-------
This script takes the log-linear calibration model fit in Algorithm 1 and:

    1. Applies it to every PurpleAir sensor in the network (after discarding
       sensors with fewer than 120 hourly observations).
    2. Averages calibrated PM2.5 to a daily resolution for each retained
       sensor.
    3. Interpolates a daily citywide PM2.5 surface using a thin plate spline
       (scipy.interpolate.RBFInterpolator) and evaluates it at every census
       tract centroid.
    4. Computes tract-level excess mortality attributable to the estimated
       PM2.5, using the Krewski/ACS-style log-linear relative-risk model,
       summing daily mortality across the full study window (never using
       an averaged event-level PM2.5, per the project specification).
    5. Quantifies uncertainty two ways:
         - "within-estimate" variance U from the gamma coefficient's 95% CI,
         - "bootstrap" variance B from 50 replicate refits of the
           calibration model (resampling the calibration dataset,
           refitting OLS, and re-running the full
           calibration -> interpolation -> mortality pipeline),
       combined as total variance T = U + B, with 95% CI = M +/- 1.96*sqrt(T).
    6. Computes a spatially uniform EPA-only mortality estimate for
       comparison, using the EPA monitor's own daily average PM2.5 applied
       identically to every tract (differing only by tract population).

Key modeling assumption
------------------------
The project specification defines RR = exp(gamma * DeltaPM) but does not
specify an explicit non-wildfire "baseline" PM2.5 to subtract. Consistent
with the wildfire-attributable framing of this project, DeltaPM is taken to
be the daily *calibrated* (or EPA) PM2.5 concentration itself, interpreted
as the wildfire-period excess PM2.5. If a distinct non-wildfire baseline
period/concentration is intended, adjust `compute_delta_pm()` below to
subtract it before the mortality calculation.

All outputs are written beneath ./outputs/ and checkpointed at each major
stage so Algorithm 3 and Algorithm 4 can resume without recomputation.
"""

# =============================================================================
# Imports
# =============================================================================
import os
import re
import sys
import glob
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from tqdm import tqdm

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
SENSOR_COORDS_FILE = os.path.join(DATA_DIR, "Sensors_with_coordinates.csv")
TRACT_FILE = os.path.join(DATA_DIR, "la_city_tracts_final.csv")
LOGLINEAR_COEF_FILE = os.path.join(CALIB_DIR, "loglinear_coefficients.csv")
SYNCED_CALIBRATION_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR, "synchronized_calibration_data.csv"
)

STUDY_START = pd.Timestamp("2025-01-01 00:00:00")
STUDY_END = pd.Timestamp("2025-02-28 23:00:00")

# Wildfire-exposure window: mortality is calculated only for these days.
WILDFIRE_START = pd.Timestamp("2025-01-07").date()
WILDFIRE_END = pd.Timestamp("2025-01-31").date()

# Non-fire baseline window: per-tract average PM2.5 here is subtracted from
# each wildfire-period day to form DeltaPM.
BASELINE_START = pd.Timestamp("2025-02-01").date()
BASELINE_END = pd.Timestamp("2025-02-28").date()

MIN_SENSOR_OBS = 120           # discard sensors with fewer hourly observations
MIN_SENSORS_FOR_SPLINE = 4     # minimum sensors required to fit a TPS surface for a day

RBF_KERNEL = "thin_plate_spline"
RBF_SMOOTHING = 1e-3

# Mortality constants
GAMMA_POINT = 0.00094
GAMMA_LOWER = 0.00073
GAMMA_UPPER = 0.00116
BASELINE_MORTALITY_PER_100K_YEAR = 780.0  # Bd

N_BOOTSTRAP = 50
BOOTSTRAP_SEED = 42

PREDICTOR_COLUMNS = [
    "log_pa_pm",
    "temperature",
    "humidity",
    "pressure",
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1",
]


# =============================================================================
# Logging
# =============================================================================
def setup_logging() -> logging.Logger:
    """Configure console + file logging for this script."""
    log_filename = os.path.join(
        LOG_DIR, f"algorithm2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("Algorithm2")
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
# Geometry helpers
# =============================================================================
def latlon_to_xy_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Convert lat/lon to a local flat-earth (x_km, y_km) projection.

    Uses the same convention described for the tract file: coordinates are
    expressed in kilometers relative to the EPA monitor location, with
    standard degree-to-km scale factors (110.574 km/deg latitude,
    111.320*cos(lat0) km/deg longitude).

    Parameters
    ----------
    lat, lon : array-like
        Latitude / longitude in decimal degrees.
    lat0, lon0 : float
        Reference (EPA monitor) latitude / longitude in decimal degrees.

    Returns
    -------
    (np.ndarray, np.ndarray)
        x_km, y_km arrays.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    y_km = (lat - lat0) * 110.574
    x_km = (lon - lon0) * 111.320 * np.cos(np.radians(lat0))
    return x_km, y_km


def get_epa_monitor_coordinates(filepath: str):
    """Extract the EPA monitor's (lat, lon) from the EPA site file.

    Parameters
    ----------
    filepath : str
        Path to LA_Site_1103.csv.

    Returns
    -------
    (float, float)
        (latitude, longitude) of the EPA monitor.
    """
    raw = pd.read_csv(filepath, usecols=["Latitude", "Longitude"])
    raw = raw.dropna()
    if raw.empty:
        raise ValueError("Could not determine EPA monitor coordinates from file.")
    lat0 = float(raw["Latitude"].iloc[0])
    lon0 = float(raw["Longitude"].iloc[0])
    logger.info("EPA monitor coordinates: lat=%.6f, lon=%.6f", lat0, lon0)
    return lat0, lon0


# =============================================================================
# Loading: coefficients, coordinates, tracts
# =============================================================================
def load_loglinear_coefficients(filepath: str) -> np.ndarray:
    """Load the fitted log-linear coefficients from Algorithm 1.

    Parameters
    ----------
    filepath : str
        Path to loglinear_coefficients.csv.

    Returns
    -------
    np.ndarray
        Coefficient vector ordered as [intercept] + PREDICTOR_COLUMNS.

    Raises
    ------
    FileNotFoundError
        If Algorithm 1 has not yet been run.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Log-linear coefficients not found at {filepath}. "
            "Run Algorithm1_ModelComparison.py first."
        )
    coef_table = pd.read_csv(filepath)
    expected_terms = ["intercept"] + PREDICTOR_COLUMNS
    coef_table = coef_table.set_index("term").loc[expected_terms]
    coef_vector = coef_table["coefficient"].values.astype(float)
    logger.info("Loaded log-linear coefficients: %s", dict(zip(expected_terms, coef_vector)))
    return coef_vector


def load_sensor_coordinates(filepath: str, lat0: float, lon0: float) -> pd.DataFrame:
    """Load PurpleAir sensor coordinates and project to x_km/y_km.

    Parameters
    ----------
    filepath : str
        Path to Sensors_with_coordinates.csv.
    lat0, lon0 : float
        EPA monitor reference coordinates.

    Returns
    -------
    pd.DataFrame
        Columns: ['sensor', 'lat', 'lon', 'x_km', 'y_km'], indexed by
        sensor id as string.
    """
    coords = pd.read_csv(filepath)
    coords["sensor"] = coords["sensor"].astype(str)
    x_km, y_km = latlon_to_xy_km(coords["lat"], coords["lon"], lat0, lon0)
    coords["x_km"] = x_km
    coords["y_km"] = y_km
    coords = coords.set_index("sensor", drop=False)
    logger.info("Loaded coordinates for %d PurpleAir sensors", len(coords))
    return coords


def load_tract_data(filepath: str) -> pd.DataFrame:
    """Load census tract population and centroid coordinates.

    Parameters
    ----------
    filepath : str
        Path to la_city_tracts_final.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['GEOID', 'NAME', 'population', 'centroid_lat',
        'centroid_lon', 'x_km', 'y_km'].
    """
    tracts = pd.read_csv(filepath, dtype={"GEOID": str})
    required = {"GEOID", "NAME", "population", "x_km", "y_km"}
    missing = required - set(tracts.columns)
    if missing:
        raise ValueError(f"Tract file is missing required columns: {missing}")
    logger.info("Loaded %d census tracts", len(tracts))
    return tracts


# =============================================================================
# PurpleAir network: load, filter, lag features
# =============================================================================
def discover_sensor_files(data_dir: str) -> dict:
    """Find all sensor_<id>_history.csv files in the data directory.

    Parameters
    ----------
    data_dir : str
        Directory containing PurpleAir sensor history files.

    Returns
    -------
    dict
        Mapping of sensor_id (str) -> filepath.
    """
    pattern = os.path.join(data_dir, "sensor_*_history.csv")
    files = glob.glob(pattern)
    sensor_map = {}
    regex = re.compile(r"sensor_(.+)_history\.csv$")
    for f in files:
        match = regex.search(os.path.basename(f))
        if match:
            sensor_map[match.group(1)] = f
    logger.info("Discovered %d PurpleAir sensor history files", len(sensor_map))
    return sensor_map


def load_single_sensor(filepath: str) -> pd.DataFrame:
    """Load and lightly clean a single PurpleAir sensor file.

    Parameters
    ----------
    filepath : str
        Path to sensor_<id>_history.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['timestamp', 'pa_pm25', 'humidity', 'temperature', 'pressure'],
        restricted to the study window.
    """
    raw = pd.read_csv(filepath)
    required = {"time_stamp", "pm2.5_cf_1", "humidity", "temperature", "pressure"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{filepath} is missing columns: {missing}")

    df = raw.rename(columns={"time_stamp": "timestamp", "pm2.5_cf_1": "pa_pm25"})[
        ["timestamp", "pa_pm25", "humidity", "temperature", "pressure"]
    ].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)

    df = df.drop_duplicates(subset="timestamp", keep="first")
    df = df.sort_values("timestamp")

    mask = (df["timestamp"] >= STUDY_START) & (df["timestamp"] <= STUDY_END)
    df = df.loc[mask].reset_index(drop=True)
    return df


def build_sensor_design_matrix(sensor_map: dict, coords: pd.DataFrame) -> pd.DataFrame:
    """Load every sensor, filter by minimum observation count, and build lag
    features + model predictors for all retained sensors.

    Sensors with fewer than MIN_SENSOR_OBS hourly observations (within the
    study window) are discarded entirely. For retained sensors, missing
    hours are left missing (never interpolated); the 1-hour lag features
    are computed only across each sensor's own available (non-imputed)
    hourly sequence, and rows lacking a valid lag are dropped for that
    sensor, mirroring the treatment in Algorithm 1.

    Parameters
    ----------
    sensor_map : dict
        Mapping of sensor_id -> filepath, from discover_sensor_files().
    coords : pd.DataFrame
        Sensor coordinate table indexed by sensor id, with x_km / y_km.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per (sensor, hour), containing
        all PREDICTOR_COLUMNS plus 'sensor', 'timestamp', 'date',
        'x_km', 'y_km'.
    """
    logger.info("Loading and filtering %d candidate PurpleAir sensors...", len(sensor_map))

    all_rows = []
    n_discarded = 0
    n_missing_coords = 0

    for sensor_id, filepath in tqdm(sensor_map.items(), desc="Loading PurpleAir sensors"):
        try:
            df = load_single_sensor(filepath)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping sensor %s due to load error: %s", sensor_id, exc)
            continue

        if len(df) < MIN_SENSOR_OBS:
            n_discarded += 1
            continue

        if sensor_id not in coords.index:
            n_missing_coords += 1
            continue

        # Lag features (per-sensor; first row dropped)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["temperature_lag1"] = df["temperature"].shift(1)
        df["humidity_lag1"] = df["humidity"].shift(1)
        df["pressure_lag1"] = df["pressure"].shift(1)
        df = df.dropna(subset=["temperature_lag1", "humidity_lag1", "pressure_lag1"])

        # Log transform (drop non-positive PM readings)
        df = df[df["pa_pm25"] > 0].copy()
        if df.empty:
            continue
        df["log_pa_pm"] = np.log(df["pa_pm25"])

        df["sensor"] = sensor_id
        df["date"] = df["timestamp"].dt.date
        df["x_km"] = coords.loc[sensor_id, "x_km"]
        df["y_km"] = coords.loc[sensor_id, "y_km"]

        all_rows.append(
            df[["sensor", "timestamp", "date", "x_km", "y_km"] + PREDICTOR_COLUMNS]
        )

    if n_discarded > 0:
        logger.info(
            "Discarded %d sensors with fewer than %d hourly observations",
            n_discarded, MIN_SENSOR_OBS,
        )
    if n_missing_coords > 0:
        logger.warning(
            "Discarded %d sensors with no matching entry in Sensors_with_coordinates.csv",
            n_missing_coords,
        )

    if not all_rows:
        raise RuntimeError("No PurpleAir sensors passed filtering. Cannot proceed.")

    design_df = pd.concat(all_rows, ignore_index=True)
    n_retained = design_df["sensor"].nunique()
    logger.info(
        "Retained %d PurpleAir sensors, %d total hourly observations",
        n_retained, len(design_df),
    )
    return design_df


# =============================================================================
# Calibration application + daily averaging
# =============================================================================
def apply_calibration_daily_average(design_df: pd.DataFrame, coef_vector: np.ndarray) -> pd.DataFrame:
    """Apply the log-linear calibration coefficients and average to daily PM.

    Parameters
    ----------
    design_df : pd.DataFrame
        Output of build_sensor_design_matrix(), containing PREDICTOR_COLUMNS.
    coef_vector : np.ndarray
        [intercept] + coefficients for PREDICTOR_COLUMNS, in that order.

    Returns
    -------
    pd.DataFrame
        Columns: ['sensor', 'date', 'x_km', 'y_km', 'calibrated_pm']
        (one row per sensor-day; calibrated_pm is the mean of the hourly
        calibrated concentrations for that sensor-day).
    """
    X = design_df[PREDICTOR_COLUMNS].values
    intercept = coef_vector[0]
    betas = coef_vector[1:]

    log_calibrated = intercept + X @ betas
    calibrated_pm = np.exp(log_calibrated)

    hourly = design_df[["sensor", "date", "x_km", "y_km"]].copy()
    hourly["calibrated_pm"] = calibrated_pm

    daily = (
        hourly.groupby(["sensor", "date"], as_index=False)
        .agg(x_km=("x_km", "first"), y_km=("y_km", "first"), calibrated_pm=("calibrated_pm", "mean"))
    )
    return daily


# =============================================================================
# Thin plate spline interpolation
# =============================================================================
def interpolate_daily_tract_pm(daily_sensor_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Interpolate a thin-plate-spline PM2.5 surface for each day and
    evaluate it at every tract centroid.

    Parameters
    ----------
    daily_sensor_pm : pd.DataFrame
        Output of apply_calibration_daily_average(): one row per
        sensor-day with x_km, y_km, calibrated_pm.
    tracts : pd.DataFrame
        Tract table with GEOID, x_km, y_km.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'] — daily interpolated PM2.5 at
        every tract centroid.
    """
    tract_xy = tracts[["x_km", "y_km"]].values
    tract_geoid = tracts["GEOID"].values

    records = []
    dates = sorted(daily_sensor_pm["date"].unique())
    n_skipped = 0

    for day in tqdm(dates, desc="Thin plate spline interpolation (daily)"):
        day_data = daily_sensor_pm.loc[daily_sensor_pm["date"] == day]
        day_data = day_data.dropna(subset=["calibrated_pm"])

        if len(day_data) < MIN_SENSORS_FOR_SPLINE:
            n_skipped += 1
            continue

        sensor_xy = day_data[["x_km", "y_km"]].values
        sensor_values = day_data["calibrated_pm"].values

        try:
            rbf = RBFInterpolator(
                sensor_xy,
                sensor_values,
                kernel=RBF_KERNEL,
                smoothing=RBF_SMOOTHING,
            )
            tract_values = rbf(tract_xy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spline interpolation failed for %s: %s", day, exc)
            n_skipped += 1
            continue

        records.append(
            pd.DataFrame({"date": day, "GEOID": tract_geoid, "pm25": tract_values})
        )

    if n_skipped > 0:
        logger.warning(
            "Skipped %d days due to insufficient sensor coverage (<%d sensors) "
            "or interpolation failure", n_skipped, MIN_SENSORS_FOR_SPLINE,
        )

    if not records:
        raise RuntimeError("No days produced a valid interpolated PM surface.")

    result = pd.concat(records, ignore_index=True)
    logger.info(
        "Interpolated PM2.5 for %d day(s) across %d tracts",
        result["date"].nunique(), result["GEOID"].nunique(),
    )
    return result


# =============================================================================
# Mortality calculation
# =============================================================================
def compute_tract_baseline_pm(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.Series:
    """Compute each tract's non-fire baseline PM2.5 (mean over BASELINE
    window, Feb 1-28).

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
        Tracts with no baseline-window data fall back to the network-wide
        baseline mean (logged as a warning), since a missing baseline would
        otherwise silently zero out that tract's DeltaPM.
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
    """Define the wildfire-attributable PM2.5 increment (DeltaPM).

    DeltaPM(tract, day) = PM2.5(tract, day) - baseline_PM2.5(tract)

    where 'day' ranges only over the wildfire-exposure window
    (WILDFIRE_START to WILDFIRE_END, i.e. Jan 7-31) and baseline_PM2.5 is
    each tract's mean PM2.5 over the non-fire window (BASELINE_START to
    BASELINE_END, i.e. Feb 1-28).

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'], covering the full study window
        (both the wildfire and baseline periods must be present).
    tracts : pd.DataFrame
        Tract table with 'GEOID'.

    Returns
    -------
    pd.DataFrame
        Wildfire-window rows only, with an added 'delta_pm' column.
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


def compute_tract_mortality(
    daily_tract_pm: pd.DataFrame,
    tracts: pd.DataFrame,
    gamma: float,
) -> pd.Series:
    """Compute total (summed-over-wildfire-days) excess mortality per tract
    for a single gamma value.

    M_day = AF * Bd * Population / 100000 / 365
    AF = (RR - 1) / RR,  RR = exp(gamma * DeltaPM)

    DeltaPM is PM2.5 on each wildfire-window day (Jan 7-31) minus that
    tract's non-fire baseline mean PM2.5 (Feb 1-28). Daily mortality is
    computed separately for each wildfire day and then summed (never
    computed from an averaged PM2.5), per the project specification.

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

    # Ensure every tract appears, even if it had no interpolated days.
    tract_totals = tract_totals.reindex(tracts["GEOID"], fill_value=0.0)
    return tract_totals


def compute_mortality_with_ci(daily_tract_pm: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Compute point/lower/upper mortality estimates and the resulting
    within-estimate variance U for every tract.

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Daily interpolated tract PM2.5.
    tracts : pd.DataFrame
        Tract table.

    Returns
    -------
    pd.DataFrame
        Columns: ['GEOID', 'NAME', 'population', 'M_point', 'M_lower_gamma',
        'M_upper_gamma', 'U'].
    """
    m_point = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_POINT)
    m_lo = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_LOWER)
    m_hi = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_UPPER)

    u_variance = ((m_hi - m_lo) / (2 * 1.96)) ** 2

    result = tracts[["GEOID", "NAME", "population"]].copy()
    result = result.set_index("GEOID")
    result["M_point"] = m_point
    result["M_lower_gamma"] = m_lo
    result["M_upper_gamma"] = m_hi
    result["U"] = u_variance
    result = result.reset_index()
    return result


# =============================================================================
# Bootstrap uncertainty
# =============================================================================
def refit_loglinear_bootstrap(calibration_df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample the calibration dataset with replacement and refit OLS.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        The synchronized EPA/PurpleAir-208493 dataset (with PREDICTOR_COLUMNS
        and 'log_epa') used to originally fit the log-linear model.
    rng : np.random.Generator
        Random number generator controlling the resampling.

    Returns
    -------
    np.ndarray
        Refit coefficient vector [intercept] + betas.
    """
    n = len(calibration_df)
    idx = rng.integers(0, n, size=n)
    resampled = calibration_df.iloc[idx]

    X = resampled[PREDICTOR_COLUMNS].values
    y = resampled["log_epa"].values

    X_design = np.column_stack([np.ones(len(X)), X])
    coef_vector, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return coef_vector


def run_bootstrap(
    calibration_df: pd.DataFrame,
    design_df: pd.DataFrame,
    tracts: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full bootstrap pipeline: resample -> refit -> calibrate ->
    interpolate -> mortality, repeated N_BOOTSTRAP times.

    Point-estimate gamma (GAMMA_POINT) is used for every bootstrap
    replicate; the resulting spread across replicates estimates the
    calibration-model uncertainty component (B) of total mortality
    variance.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Synchronized EPA/PurpleAir-208493 calibration dataset.
    design_df : pd.DataFrame
        Full-network sensor design matrix (fixed across bootstrap replicates;
        only the coefficients change).
    tracts : pd.DataFrame
        Tract table.

    Returns
    -------
    pd.DataFrame
        Rows = bootstrap replicate, columns = tract GEOID; values = total
        mortality for that tract in that replicate.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "bootstrap_mortality_partial.csv")

    replicate_rows = []

    for b in tqdm(range(N_BOOTSTRAP), desc="Bootstrap replicates"):
        try:
            coef_vector = refit_loglinear_bootstrap(calibration_df, rng)
            daily_sensor_pm = apply_calibration_daily_average(design_df, coef_vector)
            daily_tract_pm = interpolate_daily_tract_pm(daily_sensor_pm, tracts)
            m_point = compute_tract_mortality(daily_tract_pm, tracts, GAMMA_POINT)
            row = m_point.to_dict()
            row["replicate"] = b
            replicate_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Bootstrap replicate %d failed: %s", b, exc)

        # Checkpoint after every replicate.
        pd.DataFrame(replicate_rows).to_csv(checkpoint_path, index=False)

    bootstrap_df = pd.DataFrame(replicate_rows).set_index("replicate")
    logger.info("Bootstrap complete: %d successful replicates", len(bootstrap_df))
    return bootstrap_df


# =============================================================================
# EPA-only comparison
# =============================================================================
def load_full_epa_series(filepath: str) -> pd.DataFrame:
    """Load the EPA record and compute a daily average PM2.5 series over
    the full study window (no interpolation of missing hours).

    Parameters
    ----------
    filepath : str
        Path to LA_Site_1103.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['date', 'epa_pm25_daily_mean'].
    """
    raw = pd.read_csv(filepath)
    raw["timestamp"] = pd.to_datetime(
        raw["Date Local"].astype(str) + " " + raw["Time Local"].astype(str), errors="coerce"
    )
    raw = raw.dropna(subset=["timestamp"])
    mask = (raw["timestamp"] >= STUDY_START) & (raw["timestamp"] <= STUDY_END)
    raw = raw.loc[mask]

    raw["date"] = raw["timestamp"].dt.date
    daily = raw.groupby("date", as_index=False)["Sample Measurement"].mean()
    daily = daily.rename(columns={"Sample Measurement": "epa_pm25_daily_mean"})
    return daily


def compute_epa_only_mortality(epa_daily: pd.DataFrame, tracts: pd.DataFrame) -> pd.DataFrame:
    """Compute the spatially uniform EPA-only mortality estimate.

    The same daily EPA PM2.5 value is applied to every tract; totals differ
    only through each tract's population.

    Parameters
    ----------
    epa_daily : pd.DataFrame
        Output of load_full_epa_series().
    tracts : pd.DataFrame
        Tract table.

    Returns
    -------
    pd.DataFrame
        Columns: ['GEOID', 'NAME', 'population', 'M_point', 'M_lower_gamma',
        'M_upper_gamma', 'U'].
    """
    # Build a synthetic "daily_tract_pm" table: every tract gets the same
    # EPA PM2.5 value on each day.
    n_tracts = len(tracts)
    frames = []
    for _, row in epa_daily.iterrows():
        frames.append(
            pd.DataFrame(
                {
                    "date": row["date"],
                    "GEOID": tracts["GEOID"].values,
                    "pm25": row["epa_pm25_daily_mean"],
                }
            )
        )
    synthetic = pd.concat(frames, ignore_index=True)
    return compute_mortality_with_ci(synthetic, tracts)


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    """Run the full Algorithm 2 pipeline end to end."""
    logger.info("=" * 70)
    logger.info("ALGORITHM 2: Log-Linear Calibration + Mortality - START")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: Load prerequisites from Algorithm 1
    # ------------------------------------------------------------------
    coef_vector = load_loglinear_coefficients(LOGLINEAR_COEF_FILE)

    if not os.path.isfile(SYNCED_CALIBRATION_CHECKPOINT):
        raise FileNotFoundError(
            f"Missing calibration dataset checkpoint: {SYNCED_CALIBRATION_CHECKPOINT}. "
            "Run Algorithm1_ModelComparison.py first."
        )
    calibration_df = pd.read_csv(SYNCED_CALIBRATION_CHECKPOINT, parse_dates=["timestamp"])

    lat0, lon0 = get_epa_monitor_coordinates(EPA_FILE)
    coords = load_sensor_coordinates(SENSOR_COORDS_FILE, lat0, lon0)
    tracts = load_tract_data(TRACT_FILE)

    # ------------------------------------------------------------------
    # Step 1: Build (or load checkpointed) sensor design matrix
    # ------------------------------------------------------------------
    design_checkpoint = os.path.join(CHECKPOINT_DIR, "sensor_design_matrix.csv")
    if os.path.isfile(design_checkpoint):
        logger.info("Loading cached sensor design matrix from %s", design_checkpoint)
        design_df = pd.read_csv(design_checkpoint, parse_dates=["timestamp"])
        design_df["date"] = pd.to_datetime(design_df["date"]).dt.date
    else:
        sensor_map = discover_sensor_files(DATA_DIR)
        design_df = build_sensor_design_matrix(sensor_map, coords)
        design_df.to_csv(design_checkpoint, index=False)
        logger.info("Saved sensor design matrix checkpoint to %s", design_checkpoint)

    # ------------------------------------------------------------------
    # Step 2: Apply calibration + daily average (point estimate coefficients)
    # ------------------------------------------------------------------
    daily_sensor_pm = apply_calibration_daily_average(design_df, coef_vector)
    daily_sensor_path = os.path.join(MORTALITY_DIR, "daily_calibrated_pm_by_sensor.csv")
    daily_sensor_pm.to_csv(daily_sensor_path, index=False)
    logger.info("Saved daily calibrated PM by sensor to %s", daily_sensor_path)

    # ------------------------------------------------------------------
    # Step 3: Thin plate spline interpolation to tract centroids
    # ------------------------------------------------------------------
    daily_tract_pm = interpolate_daily_tract_pm(daily_sensor_pm, tracts)
    daily_tract_path = os.path.join(MORTALITY_DIR, "daily_tract_pm_loglinear.csv")
    daily_tract_pm.to_csv(daily_tract_path, index=False)
    logger.info("Saved daily tract-level interpolated PM to %s", daily_tract_path)

    # ------------------------------------------------------------------
    # Step 4: Point/lower/upper gamma mortality (within-estimate variance U)
    # ------------------------------------------------------------------
    mortality_summary = compute_mortality_with_ci(daily_tract_pm, tracts)

    # ------------------------------------------------------------------
    # Step 5: Bootstrap variance B
    # ------------------------------------------------------------------
    bootstrap_df = run_bootstrap(calibration_df, design_df, tracts)
    bootstrap_path = os.path.join(MORTALITY_DIR, "bootstrap_mortality_replicates.csv")
    bootstrap_df.to_csv(bootstrap_path)
    logger.info("Saved bootstrap replicate mortality table to %s", bootstrap_path)

    b_variance = bootstrap_df.var(axis=0, ddof=1)
    mortality_summary = mortality_summary.set_index("GEOID")
    mortality_summary["B"] = b_variance.reindex(mortality_summary.index)
    mortality_summary["T"] = mortality_summary["U"] + mortality_summary["B"]
    mortality_summary["CI_half_width"] = 1.96 * np.sqrt(mortality_summary["T"])
    mortality_summary["CI_lower"] = mortality_summary["M_point"] - mortality_summary["CI_half_width"]
    mortality_summary["CI_upper"] = mortality_summary["M_point"] + mortality_summary["CI_half_width"]
    mortality_summary = mortality_summary.reset_index()

    mortality_path = os.path.join(MORTALITY_DIR, "tract_mortality_loglinear.csv")
    mortality_summary.to_csv(mortality_path, index=False)
    logger.info("Saved final log-linear tract mortality estimates to %s", mortality_path)

    # ------------------------------------------------------------------
    # Step 6: EPA-only comparison (spatially uniform estimate)
    # ------------------------------------------------------------------
    epa_daily = load_full_epa_series(EPA_FILE)
    epa_mortality = compute_epa_only_mortality(epa_daily, tracts)
    epa_mortality_path = os.path.join(MORTALITY_DIR, "epa_only_mortality.csv")
    epa_mortality.to_csv(epa_mortality_path, index=False)
    logger.info("Saved EPA-only (spatially uniform) mortality estimate to %s", epa_mortality_path)

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    summary = {
        "n_sensors_retained": int(design_df["sensor"].nunique()),
        "n_days_interpolated": int(daily_tract_pm["date"].nunique()),
        "citywide_total_mortality_point": float(mortality_summary["M_point"].sum()),
        "citywide_total_mortality_CI_lower": float(mortality_summary["CI_lower"].sum()),
        "citywide_total_mortality_CI_upper": float(mortality_summary["CI_upper"].sum()),
        "citywide_total_mortality_epa_only": float(epa_mortality["M_point"].sum()),
        "n_bootstrap_replicates": int(len(bootstrap_df)),
    }
    summary_path = os.path.join(MORTALITY_DIR, "algorithm2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved run summary to %s", summary_path)
    logger.info("Summary: %s", summary)

    logger.info("=" * 70)
    logger.info("ALGORITHM 2 COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
