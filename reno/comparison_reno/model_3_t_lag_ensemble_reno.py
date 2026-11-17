"""
MODEL 3: hour-t PurpleAir measurements + only the hour-(t-1) WEATHER + location.
MLP with g and without g. PM2.5(t-1) enters only the MLP with g, through g.

Models as in code_full/jan2026_test/model_2_no_g_jan-mar_train_both_R2_and_adj.py (same inputs)
with ONE change: both MLPs are ensembles of --n-seeds seeds (default 5) instead of one seed.
As in Models 1 and 2, all four models are also scored on log1p inside this code.

Same data, sensors, radii, train/test split, quality control and rows as Models 1 and 2:
train on the PurpleAir sensors around the Reno4 EPA monitor (Reno, Nevada; site 32-031-0031,
POC 2) in July-September 2021, test on the SAME sensors and the same monitor in September 2022.
Both periods had heavy wildfire smoke (training: Dixie, Tamarack and Caldor fires; test:
Mosquito Fire). Every model is trained on the 2021 data only; the 2022 EPA values are used
only for scoring.

RENO VERSION of code_full/comparison/model_3_t_lag_ensemble.py (LA): the same models, inputs, MLP settings,
seeds, scoring and rows. Only these differ (taken from
reno/model_2_reno_train_jul-sep21_test_sep22_ensemble.py):
  - data: Reno4 EPA files and the Reno sensor folders (train_2021, test_2022)
  - periods: train July 1 - September 30, 2021; test September 1-30, 2022 (Pacific time)
  - pressure valid range 800-1100 hPa (Reno is ~1,370 m above sea level: ~850-880 hPa)
  - the faulty sensors 70519 and 115819 are excluded (EXCLUDED_SENSORS)
  - same sensors: only sensors with rows in BOTH periods are used
  - radii 2-10 km (every Reno sensor is within 9.91 km of Reno4)

Notation (t = current hour, t-1 = the hour before at the same sensor):
    y(t)             EPA PM2.5 at Reno4
    x(t)             PurpleAir PM2.5 (pm2.5_cf_1) at the sensor
    T(t), H(t), P(t) PurpleAir temperature, humidity, pressure
    lat, lon, d      sensor location and its distance (km) to the EPA monitor
t-1 values are used only when the previous reading is exactly one hour earlier.

Models:

    a. reg_lin_model (p = 10): linear regression (OLS) on the raw scale
        y(t) = b0 + b1 x(t) + b2 T(t) + b3 H(t) + b4 P(t)
                  + b5 T(t-1) + b6 H(t-1) + b7 P(t-1) + b8 lat + b9 lon + b10 d
       Scored on log1p:  log1p( max(y_hat(t), 0) )

    b. log_trans_lin_model (p = 10): linear regression (OLS) on the log1p scale
        log1p y(t) = c0 + c1 log1p x(t) + c2 T(t) + c3 H(t) + c4 P(t)
                     + c5 T(t-1) + c6 H(t-1) + c7 P(t-1) + c8 lat + c9 lon + c10 d

    c. MLP_no_g ensemble (p = 10): one MLP f per seed, the same 10 inputs as b
        log1p y_hat_s(t) = f_s( log1p x(t), T(t), H(t), P(t), T(t-1), H(t-1), P(t-1), lat, lon, d )

    d. MLP_g ensemble (p = 11): two-stage
        g_s: log1p x(t), T(t), H(t), P(t), lat, lon, d -> log1p y(t)
             trained first, on ALL training hours (it needs only that hour's readings)
        g_hat_s(t-1) = g_s( log1p x(t-1), T(t-1), H(t-1), P(t-1), lat, lon, d )
        log1p y_hat_s(t) = f_s( log1p x(t), T(t), H(t), P(t),
                                T(t-1), H(t-1), P(t-1), g_hat_s(t-1), lat, lon, d )
       Plain two-stage (no cross-fitting): the g trained on all training hours gives g_hat(t-1)
       for f's training rows; the same g and f are used for the test. Each seed trains its own
       g_s and f_s. g_hat(t-1) is the only route by which PM2.5(t-1) enters any model.

    Both ensembles: seeds --seed .. --seed + --n-seeds - 1 (default 42..46), prediction =
    row-by-row average of the seeds' predictions on log1p. Every MLP (g and f): hidden layers
    --hidden (default 16,32,16, ReLU), --epochs (default 150), Adam lr 0.001, batch 64, MSE
    loss, inputs standardized with the training mean and sd. The seed is set before each MLP
    model, so c and d do not depend on the order they run in.

ALL FOUR MODELS ARE SCORED ON THE SAME SCALE: R², adjusted R² and MSE against log1p y(t), on
train and on test. reg_lin_model's raw-scale scores are also saved, for reference only.

Rows: the hours that have their hour-t AND hour-(t-1) readings (the same rows as Models 1 and 2).

Folders (defaults are for running from inside reno/comparison_reno):
    --train-dir  ../train_2021   Reno4_Site_0031_JulSep2021.csv, Sensors_with_coordinates.csv,
                                 sensor_*_history.csv (July-September 2021)
    --test-dir   ../test_2022    Reno4_Site_0031_Sep2022.csv, Sensors_with_coordinates.csv,
                                 sensor_*_history.csv (September 2022)

Output files (folder results_model_3_t_lag_ensemble_reno, <d> = distance),
NAME = model_3_t_lag_ensemble_reno:
    NAME_Summary_<d>km.csv              n, p, R², adjusted R², MSE of every model, train and test,
                                        all on log1p (+ reg_lin raw scale, + single-seed min/mean/max)
    NAME_Sensors_<d>km.csv              sensors used in each period and their rows
    NAME_Test_Predictions_<d>km.csv     September 2022 predictions (reg_lin_pred (raw),
                                        reg_lin_pred_log1p, log_trans_lin_pred, mlp_no_g_pred,
                                        mlp_g_pred = ensemble averages, *_seed<s> = each seed)
    NAME_reg_lin_model_<d>km.txt        OLS coefficient table
    NAME_log_trans_lin_model_<d>km.txt  OLS coefficient table
Run all distances with run_model_3_t_lag_ensemble_reno.sh, then plot them with
plot_model_3_t_lag_ensemble_reno_same_scale.py.

Usage:
    python model_3_t_lag_ensemble_reno.py --distance 5
    python model_3_t_lag_ensemble_reno.py --distance 5 --n-seeds 5 --epochs 150 --hidden 16,32,16
"""

import os
import random
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

import statsmodels.api as sm


# ============================================================
# Arguments
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

parser.add_argument("--distance", type=float, required=True,
                    help="Radius in km around the EPA monitor")
parser.add_argument("--epochs", type=int, default=150,
                    help="MLP training epochs (fixed, no search), for g and f")
parser.add_argument("--hidden", default="16,32,16",
                    help="Neurons in each hidden layer of every MLP (g and f)")
parser.add_argument("--seed", type=int, default=42,
                    help="First MLP seed; the ensembles use seed, seed+1, ..., seed+n_seeds-1")
parser.add_argument("--n-seeds", type=int, default=5,
                    help="Number of MLPs in each ensemble; their predictions are averaged")
parser.add_argument("--train-dir", default=os.path.join(HERE, "..", "train_2021"),
                    help="Folder with Reno4_Site_0031_JulSep2021.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (July-September 2021 training data)")
parser.add_argument("--test-dir", default=os.path.join(HERE, "..", "test_2022"),
                    help="Folder with Reno4_Site_0031_Sep2022.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (September 2022 test data)")
parser.add_argument("--out-dir", default=os.path.join(HERE, "results_model_3_t_lag_ensemble_reno"),
                    help="Folder for output files")

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs
HIDDEN = [int(n) for n in args.hidden.split(",")]

# Seeds of the MLP ensembles, e.g. 42, 43, 44, 45, 46
SEEDS = [args.seed + i for i in range(args.n_seeds)]

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

# e.g. [16, 32, 16] -> "16-32-16"
ARCH_LABEL = "-".join(str(n) for n in HIDDEN)

ROWS_LABEL = "hours with t and t-1 readings (same rows as Models 1 and 2)"

# Every output file starts with the name of this code
NAME = "model_3_t_lag_ensemble_reno"

os.makedirs(args.out_dir, exist_ok=True)

# Plausible ranges of the PurpleAir readings; a row with any value outside them is dropped
VALID_RANGES = {
    "pm2.5_cf_1": (0, 1000),     # ug/m3
    "temperature": (-40, 150),   # deg F
    "humidity": (0, 100),        # %
    "pressure": (800, 1100),     # hPa (Reno is ~1,370 m above sea level: ~850-880 hPa)
}

# A sensor with a larger share of its hours outside VALID_RANGES is not used for that period
MAX_INVALID_SHARE = 0.1

# Faulty sensors, not used: their log PM2.5 correlated poorly with the Reno4 EPA monitor during
# the 2021 training period (r = 0.36 and 0.53; the other sensors 0.79-0.95)
EXCLUDED_SENSORS = {"70519", "115819"}


# ============================================================
# Device and threads
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Respect the number of cores the scheduler gave us
n_threads = (
    os.environ.get("NSLOTS")
    or os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("OMP_NUM_THREADS")
)
if n_threads:
    torch.set_num_threads(int(n_threads))

print(f"MODEL 3 (hour t + hour t-1 weather only) | Radius: {DISTANCE_KM} km | Epochs: {EPOCHS} | Hidden layers: {ARCH_LABEL} "
      f"| MLP seeds: {SEEDS}")
print("Train: Jul-Sep 2021 | Test: Sep 2022 (Reno4, site 32-031-0031)")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Files
# ============================================================

TRAIN_EPA_FILE = os.path.join(args.train_dir, "Reno4_Site_0031_JulSep2021.csv")
TRAIN_COORD_FILE = os.path.join(args.train_dir, "Sensors_with_coordinates.csv")

TEST_EPA_FILE = os.path.join(args.test_dir, "Reno4_Site_0031_Sep2022.csv")
TEST_COORD_FILE = os.path.join(args.test_dir, "Sensors_with_coordinates.csv")

# Training and test periods, in Pacific time (Reno and LA share it; start included, end not)
TRAIN_START, TRAIN_END = "2021-07-01", "2021-10-01"   # July-September 2021
TEST_START, TEST_END = "2022-09-01", "2022-10-01"     # September 2022


# ============================================================
# Haversine distance
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    return R * 2 * np.arcsin(np.sqrt(a))


# ============================================================
# Predictor columns
# ============================================================

LOCATION_COLUMNS = ["sensor_lat", "sensor_lon", "distance_km"]

# Hour-t and hour-(t-1) readings, log1p scale for PM2.5
T_COLUMNS = ["log_PA_PM", "temperature", "humidity", "pressure"]
LAG_COLUMNS = ["log_PA_PM_lag1", "temperature_lag1", "humidity_lag1", "pressure_lag1"]
WEATHER_LAG_COLUMNS = LAG_COLUMNS[1:]

TARGET_COLUMN = "log_EPA_PM"
RAW_TARGET_COLUMN = "EPA_PM"

# a. reg_lin_model: PM2.5 on the raw scale, only the weather at t-1 (no PM2.5(t-1))
RAW_FEATURE_COLUMNS = (
    ["pm2.5_cf_1", "temperature", "humidity", "pressure"]
    + WEATHER_LAG_COLUMNS
    + LOCATION_COLUMNS
)

# b. log_trans_lin_model and c. MLP_no_g: PM2.5 on log1p, only the weather at t-1
FEATURE_COLUMNS = T_COLUMNS + WEATHER_LAG_COLUMNS + LOCATION_COLUMNS

# d. MLP_g: g's inputs at hour t, and the same inputs one hour earlier (same order)
G_COLUMNS = T_COLUMNS + LOCATION_COLUMNS
G_LAG_COLUMNS = LAG_COLUMNS + LOCATION_COLUMNS

# d. MLP_g, f's inputs: the inputs of c plus g_hat(t-1)
F_G_COLUMNS = T_COLUMNS + WEATHER_LAG_COLUMNS + ["g_hat_lag1"] + LOCATION_COLUMNS

P = len(FEATURE_COLUMNS)    # 10 predictors in a, b and c
P_G = len(F_G_COLUMNS)      # 11 predictors in d
assert len(RAW_FEATURE_COLUMNS) == P == 10 and P_G == 11

MODEL_COLUMNS = list(dict.fromkeys(
    ["sensor_id", "timestamp", TARGET_COLUMN, RAW_TARGET_COLUMN]
    + RAW_FEATURE_COLUMNS
    + FEATURE_COLUMNS
    + ["pm2.5_cf_1_lag1"]
    + G_LAG_COLUMNS
))


# ============================================================
# Readers
# ============================================================

def la_midnight(day):
    # Midnight at the start of a day in Los Angeles time, as a UTC timestamp
    return pd.Timestamp(day, tz="America/Los_Angeles").tz_convert("UTC")


def read_epa(file, start, end):

    epa = pd.read_csv(file)

    # EPA's "Time Local" has no daylight saving; the GMT columns give the exact hour all year
    epa["timestamp"] = pd.to_datetime(epa["Date GMT"] + " " + epa["Time GMT"]).dt.tz_localize("UTC")

    epa = epa[(epa["timestamp"] >= la_midnight(start)) & (epa["timestamp"] < la_midnight(end))]

    epa = epa.rename(columns={"Sample Measurement": "EPA_PM"})
    epa = epa[["timestamp", "Latitude", "Longitude", "EPA_PM"]].dropna()

    # PM2.5 cannot be negative (instrument noise at very clean hours); log1p would make them extreme
    too_low = epa["EPA_PM"] < 0
    print(f"{file}: dropped {too_low.sum()} hours with negative EPA PM2.5")

    return epa[~too_low]


def monitor_location(epa, file):

    locations = epa[["Latitude", "Longitude"]].drop_duplicates()

    if len(locations) != 1:
        raise SystemExit(f"{file} has {len(locations)} monitor locations, expected 1.")

    return locations.iloc[0]["Latitude"], locations.iloc[0]["Longitude"]


def read_coords(file):

    coords = pd.read_csv(file)
    coords = coords.rename(columns={"sensor": "sensor_id", "lat": "sensor_lat", "lon": "sensor_lon"})
    coords["sensor_id"] = coords["sensor_id"].astype(str)
    coords = coords.dropna(subset=["sensor_lat", "sensor_lon"])

    return coords[["sensor_id", "sensor_lat", "sensor_lon"]].drop_duplicates("sensor_id")


def sensor_files(sensor_id, folders):

    # This sensor's history files in the folders (one per period), possibly none
    paths = [os.path.join(folder, f"sensor_{sensor_id}_history.csv") for folder in folders]

    return [path for path in paths if os.path.exists(path)]


def read_file(file):

    # One sensor history file after quality control, or None
    try:
        df = pd.read_csv(file)
    except Exception:
        print("Could not read:", file)
        return None

    required_columns = ["time_stamp", "pm2.5_cf_1", "temperature", "humidity", "pressure"]

    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns in {file}")
        return None

    df = df[required_columns].copy()

    if df.empty:
        print(f"No rows in {file}")
        return None

    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quality control: drop rows from a broken sensor (missing values are dropped later anyway)
    valid = pd.Series(True, index=df.index)

    for col, (low, high) in VALID_RANGES.items():
        valid &= df[col].between(low, high) | df[col].isna()

    if not valid.all():
        print(f"{file}: dropped {(~valid).sum()} of {len(df)} rows with readings outside the valid ranges")

    # A sensor that is broken most of the time in this file is not used for this file's period
    if (~valid).mean() > MAX_INVALID_SHARE:
        print(f"{file}: {(~valid).mean():.0%} of rows outside the valid ranges, file not used")
        return None

    df = df[valid].copy()

    if df.empty:
        return None

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)

    return df


def read_sensor(files, sensor_id):

    # All usable files of one sensor joined in time order, then the t-1 readings
    parts = [df for df in (read_file(file) for file in files) if df is not None]

    if not parts:
        return None

    df = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("time_stamp")
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )

    df["sensor_id"] = sensor_id

    # t-1 readings, only when the previous reading is exactly one hour earlier
    # (so Aug 31 23:00 is the t-1 hour of Sep 1 00:00)
    one_hour_before = df["time_stamp"].diff() == pd.Timedelta(hours=1)

    for col in ["pm2.5_cf_1", "temperature", "humidity", "pressure"]:
        df[f"{col}_lag1"] = df[col].shift(1).where(one_hour_before)

    return df


# ============================================================
# Read EPA data
# ============================================================

print("Loading EPA data...")

epa_train = read_epa(TRAIN_EPA_FILE, TRAIN_START, TRAIN_END)
epa_test = read_epa(TEST_EPA_FILE, TEST_START, TEST_END)

TRAIN_SITE = monitor_location(epa_train, TRAIN_EPA_FILE)
TEST_SITE = monitor_location(epa_test, TEST_EPA_FILE)

print(f"Training EPA observations (Jul-Sep 2021): {len(epa_train)} | location {TRAIN_SITE}")
print(f"Test EPA observations (Sep 2022)        : {len(epa_test)} | location {TEST_SITE}")

moved_km = haversine(TRAIN_SITE[0], TRAIN_SITE[1], TEST_SITE[0], TEST_SITE[1])

if moved_km > 0.1:
    raise SystemExit(f"The training and test EPA files are {moved_km:.2f} km apart; expected the same monitor.")


# ============================================================
# Choose the sensors within the radius
# ============================================================

def sensors_in_radius(coords_file, folders):

    coords = read_coords(coords_file)

    coords["km_to_site"] = haversine(
        TRAIN_SITE[0], TRAIN_SITE[1], coords["sensor_lat"].values, coords["sensor_lon"].values
    )

    coords["files"] = [sensor_files(s, folders) for s in coords["sensor_id"]]

    # Faulty sensors are not used
    coords = coords[~coords["sensor_id"].isin(EXCLUDED_SENSORS)]

    return coords[
        (coords["km_to_site"] <= DISTANCE_KM) & (coords["files"].map(len) > 0)
    ].reset_index(drop=True)


print(f"Excluded faulty sensors: {sorted(EXCLUDED_SENSORS, key=int)}")

train_sensors = sensors_in_radius(TRAIN_COORD_FILE, [args.train_dir])
test_sensors = sensors_in_radius(TEST_COORD_FILE, [args.test_dir])

print(f"\nTraining sensors (Jul-Sep 2021) within {DISTANCE_KM} km of Reno4: {len(train_sensors)}")
print(f"Test sensor files (Sep 2022) within {DISTANCE_KM} km of Reno4    : {len(test_sensors)}")

if test_sensors.empty:
    raise SystemExit(f"No test sensor files within {DISTANCE_KM} km of Reno4. Nothing to test.")

if train_sensors.empty:
    raise SystemExit(f"No training sensor files within {DISTANCE_KM} km of Reno4. Nothing to train.")


# ============================================================
# Read PurpleAir sensors and merge them with the EPA monitor
# ============================================================

def load_period(sensors, epa, name):

    print(f"\nLoading {name} PurpleAir sensors...")

    all_sensor_data = []

    for sensor_id, files in zip(sensors["sensor_id"], sensors["files"]):
        df = read_sensor(files, sensor_id)
        if df is not None:
            all_sensor_data.append(df)

    if not all_sensor_data:
        raise SystemExit(f"No readable {name} sensor files within {DISTANCE_KM} km.")

    purpleair = pd.concat(all_sensor_data, ignore_index=True)

    print(f"{name} PurpleAir observations:", len(purpleair))

    purpleair = purpleair.merge(
        sensors[["sensor_id", "sensor_lat", "sensor_lon"]], on="sensor_id", how="left"
    )

    purpleair = purpleair.rename(columns={"time_stamp": "timestamp"})

    merged = purpleair.merge(epa, on="timestamp", how="inner")

    print(f"{name} merged observations:", len(merged))

    # Distance from the sensor to the EPA monitor
    merged["distance_km"] = haversine(
        merged["Latitude"], merged["Longitude"], merged["sensor_lat"], merged["sensor_lon"]
    )

    # Log transforms
    merged["log_PA_PM"] = np.log1p(merged["pm2.5_cf_1"])
    merged["log_PA_PM_lag1"] = np.log1p(merged["pm2.5_cf_1_lag1"])
    merged["log_EPA_PM"] = np.log1p(merged["EPA_PM"])

    merged = merged[MODEL_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Every row needs its hour-t values. The t-1 values may be missing here: g learns from every
    # hour, and the rows used by all four models are chosen below.
    merged = merged.dropna(subset=[TARGET_COLUMN, RAW_TARGET_COLUMN, "pm2.5_cf_1"] + G_COLUMNS)

    return merged[merged["distance_km"] <= DISTANCE_KM].reset_index(drop=True)


train_all = load_period(train_sensors, epa_train, "Training (Jul-Sep 2021)")
test_all = load_period(test_sensors, epa_test, "Test (Sep 2022)")


# ============================================================
# Same sensors in training and test
# ============================================================

# A sensor can lose a period in quality control (a broken file) or have no hours that match the
# EPA monitor. Only sensors with rows in BOTH periods are used, so training and test always use
# the same sensors (as in reno/model_2_reno_train_jul-sep21_test_sep22_ensemble.py).
common_sensors = set(train_all["sensor_id"]) & set(test_all["sensor_id"])

only_train = sorted(set(train_all["sensor_id"]) - common_sensors, key=int)
only_test = sorted(set(test_all["sensor_id"]) - common_sensors, key=int)

if only_train or only_test:
    print(f"\nNot used (data in one period only): training only {only_train}, test only {only_test}")

if not common_sensors:
    raise SystemExit(f"No sensor within {DISTANCE_KM} km has data in both periods.")

train_all = train_all[train_all["sensor_id"].isin(common_sensors)].reset_index(drop=True)
test_all = test_all[test_all["sensor_id"].isin(common_sensors)].reset_index(drop=True)

print(f"Sensors used in both periods within {DISTANCE_KM} km: {len(common_sensors)} "
      f"{sorted(common_sensors, key=int)}")


# ============================================================
# Training and test rows
# ============================================================

# All four models are trained and tested on the hours that also have the t-1 readings
lag_needed = ["pm2.5_cf_1_lag1"] + LAG_COLUMNS

train_df = train_all[train_all[lag_needed].notna().all(axis=1)].reset_index(drop=True)
test_df = test_all[test_all[lag_needed].notna().all(axis=1)].reset_index(drop=True)

# MLP_g's g learns from every training hour (it only needs that hour's readings)
g_train_df = train_all

if len(train_df) == 0 or len(test_df) == 0:
    raise SystemExit(
        f"Empty data within {DISTANCE_KM} km (train {len(train_df)}, test {len(test_df)} observations)."
    )

n_train_sensors = train_df["sensor_id"].nunique()
n_test_sensors = test_df["sensor_id"].nunique()
n_test_sensors_in_train = len(set(test_df["sensor_id"]) & set(train_df["sensor_id"]))


def local_range(d):
    t = d["timestamp"].dt.tz_convert("America/Los_Angeles")
    return f"{t.min():%Y-%m-%d %H:%M} -> {t.max():%Y-%m-%d %H:%M}"


print("\n================ Dataset Summary ================\n")
print(DIST_LABEL, "|", ROWS_LABEL)
print(f"Train (Jul-Sep 2021) : {len(train_df)} obs | {n_train_sensors} sensors | {local_range(train_df)} (Pacific time)")
print(f"Test  (Sep 2022)     : {len(test_df)} obs | {n_test_sensors} sensors | {local_range(test_df)} (Pacific time)")
print(f"Test sensors also in training: {n_test_sensors_in_train} of {n_test_sensors}")
print(f"MLP_g's g train      : {len(g_train_df)} obs (every training hour)")
print(f"Mean PM2.5   : train {train_df[RAW_TARGET_COLUMN].mean():.2f} | test {test_df[RAW_TARGET_COLUMN].mean():.2f}")
print(f"Mean distance: train {train_df['distance_km'].mean():.2f} km | test {test_df['distance_km'].mean():.2f} km")
print()


# ============================================================
# Scores
# ============================================================

def adjusted_r2(r2, n, p):

    if n - p - 1 <= 0:
        return np.nan

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def scores(y, pred, p):

    r2 = r2_score(y, pred)
    n = len(y)

    return {"n": n, "p": p, "r2": r2, "adj_r2": adjusted_r2(r2, n, p), "mse": mean_squared_error(y, pred)}


def print_scores(name, s):
    print(f"{name:<14} R²: {s['r2']:.4f} | Adjusted R² (p={s['p']}): {s['adj_r2']:.4f} "
          f"| MSE: {s['mse']:.6f} | n={s['n']}")


def raw_to_log1p(pred):
    # PM2.5 cannot be negative: negative raw predictions are set to 0, then raw -> log1p
    return np.log1p(np.clip(pred, 0, None))


# ============================================================
# Linear regression: fit on train, predict train and test
# ============================================================

def run_ols(train_df, test_df, columns, target):

    model = sm.OLS(train_df[target], sm.add_constant(train_df[columns], has_constant="add")).fit()

    return {
        "model": model,
        "train_pred": np.asarray(model.predict(sm.add_constant(train_df[columns], has_constant="add"))),
        "test_pred": np.asarray(model.predict(sm.add_constant(test_df[columns], has_constant="add")))
    }


# ============================================================
# MLP
# ============================================================

def create_model(input_dim):

    # One Linear + ReLU per entry of HIDDEN, then a Linear output layer
    layers = []
    size_in = input_dim

    for size_out in HIDDEN:
        layers += [nn.Linear(size_in, size_out), nn.ReLU()]
        size_in = size_out

    layers.append(nn.Linear(size_in, 1))

    return nn.Sequential(*layers).to(DEVICE)


def train_model(model, X_train, y_train, epochs, batch_size=64, lr=0.001):

    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(X_tensor)

    model.train()

    for epoch in range(epochs):

        perm = torch.randperm(n)

        for i in range(0, n, batch_size):

            idx = perm[i:i + batch_size]

            optimizer.zero_grad()
            loss = criterion(model(X_tensor[idx]), y_tensor[idx])
            loss.backward()
            optimizer.step()

    return model


def predict(model, X):

    model.eval()

    with torch.no_grad():
        pred = model(torch.FloatTensor(X).to(DEVICE))

    return pred.cpu().numpy().flatten()


def values(d, columns):
    return d[columns].values.astype(np.float32)


def fit_mlp(X, y):

    # Inputs standardized with the training mean and sd, then one MLP trained
    scaler = StandardScaler().fit(X)

    model = create_model(X.shape[1])
    train_model(model, scaler.transform(X).astype(np.float32), y, epochs=EPOCHS)

    return scaler, model


def mlp_predict(scaler, model, X):
    return predict(model, scaler.transform(X).astype(np.float32))


def run_mlp_no_g(train_df, test_df):

    # c. f: hour-t readings + hour-(t-1) weather + location -> log1p EPA PM2.5 at hour t
    X_train = values(train_df, FEATURE_COLUMNS)

    scaler, model = fit_mlp(X_train, values(train_df, [TARGET_COLUMN]).flatten())

    return {
        "train_pred": mlp_predict(scaler, model, X_train),
        "test_pred": mlp_predict(scaler, model, values(test_df, FEATURE_COLUMNS))
    }


def run_mlp_g(g_train_df, train_df, test_df):

    # d. g: hour-t readings + location -> log1p EPA PM2.5 at the same hour (every training hour)
    g_scaler, g_model = fit_mlp(
        values(g_train_df, G_COLUMNS),
        values(g_train_df, [TARGET_COLUMN]).flatten()
    )

    def f_inputs(d):
        # g applied to the t-1 readings; g_hat(t-1) is added to the inputs of c
        g_hat = mlp_predict(g_scaler, g_model, values(d, G_LAG_COLUMNS))
        return np.column_stack([
            values(d, T_COLUMNS), values(d, WEATHER_LAG_COLUMNS), g_hat, values(d, LOCATION_COLUMNS)
        ]).astype(np.float32)

    # f: plain two-stage, the g trained on all training hours gives g_hat(t-1) for f's rows
    X_f_train = f_inputs(train_df)

    f_scaler, f_model = fit_mlp(X_f_train, values(train_df, [TARGET_COLUMN]).flatten())

    return {
        "train_pred": mlp_predict(f_scaler, f_model, X_f_train),
        "test_pred": mlp_predict(f_scaler, f_model, f_inputs(test_df))
    }


def run_ensemble(name, run_one, p):

    # The same MLP trained once per seed; the prediction is the row-by-row average on log1p
    runs = {}

    for seed in SEEDS:
        set_seed(seed)
        runs[seed] = run_one()

        results[f"{name} seed {seed} Train"] = scores(train_df[TARGET_COLUMN], runs[seed]["train_pred"], p)
        results[f"{name} seed {seed} Test"] = scores(test_df[TARGET_COLUMN], runs[seed]["test_pred"], p)
        print_scores(f"seed {seed} test", results[f"{name} seed {seed} Test"])

    ensemble = {
        "train_pred": np.mean([run["train_pred"] for run in runs.values()], axis=0),
        "test_pred": np.mean([run["test_pred"] for run in runs.values()], axis=0),
        "runs": runs
    }

    results[f"{name} Train"] = scores(train_df[TARGET_COLUMN], ensemble["train_pred"], p)
    results[f"{name} Test"] = scores(test_df[TARGET_COLUMN], ensemble["test_pred"], p)

    print("ensemble (average of the seeds' predictions):")
    print_scores("train", results[f"{name} Train"])
    print_scores("test", results[f"{name} Test"])

    return ensemble


# ============================================================
# Experiment: train on Jul-Sep 2021, test on Sep 2022
# ============================================================

print("=======================================================")
print(f"MODEL 3, hour t + hour t-1 weather only ({DIST_LABEL})")
print("Train: Jul-Sep 2021 | Test: Sep 2022 (same sensors, Reno4)")
print("All models scored on log1p PM2.5")
print("=======================================================")

results = {}

########################################################
# a. reg_lin_model: fitted on raw PM2.5, scored on log1p
########################################################

reg_lin = run_ols(train_df, test_df, RAW_FEATURE_COLUMNS, RAW_TARGET_COLUMN)

# Same scale as the others: raw prediction -> max(., 0) -> log1p
results["reg_lin_model Train"] = scores(train_df[TARGET_COLUMN], raw_to_log1p(reg_lin["train_pred"]), P)
results["reg_lin_model Test"] = scores(test_df[TARGET_COLUMN], raw_to_log1p(reg_lin["test_pred"]), P)

# For reference only: its scores on the raw scale it was fitted on
results["reg_lin_model raw-scale Train"] = scores(train_df[RAW_TARGET_COLUMN], reg_lin["train_pred"], P)
results["reg_lin_model raw-scale Test"] = scores(test_df[RAW_TARGET_COLUMN], reg_lin["test_pred"], P)

reg_lin_clipped = {
    "Train": int((reg_lin["train_pred"] < 0).sum()),
    "Test": int((reg_lin["test_pred"] < 0).sum())
}

print("\n==========================================")
print(f"a. reg_lin_model: linear regression on raw PM2.5, scored on log1p ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["reg_lin_model Train"])
print_scores("test", results["reg_lin_model Test"])
print(f"Negative predictions set to 0 before log1p: train {reg_lin_clipped['Train']}, test {reg_lin_clipped['Test']}")
print_scores("raw train", results["reg_lin_model raw-scale Train"])
print_scores("raw test", results["reg_lin_model raw-scale Test"])

with open(os.path.join(args.out_dir, f"{NAME}_reg_lin_model_{DIST_LABEL}.txt"), "w") as f:
    f.write(str(reg_lin["model"].summary()))

########################################################
# b. log_trans_lin_model
########################################################

log_trans_lin = run_ols(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

results["log_trans_lin_model Train"] = scores(train_df[TARGET_COLUMN], log_trans_lin["train_pred"], P)
results["log_trans_lin_model Test"] = scores(test_df[TARGET_COLUMN], log_trans_lin["test_pred"], P)

print("\n==========================================")
print(f"b. log_trans_lin_model: linear regression on log1p PM2.5 ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["log_trans_lin_model Train"])
print_scores("test", results["log_trans_lin_model Test"])

with open(os.path.join(args.out_dir, f"{NAME}_log_trans_lin_model_{DIST_LABEL}.txt"), "w") as f:
    f.write(str(log_trans_lin["model"].summary()))

########################################################
# c. MLP_no_g ensemble
########################################################

print("\n==========================================")
print(f"c. MLP_no_g ensemble of {len(SEEDS)}: no PM2.5(t-1), t-1 weather only "
      f"({DIST_LABEL}, {EPOCHS} epochs, hidden layers {ARCH_LABEL})")
print("==========================================")

mlp_no_g = run_ensemble("MLP_no_g", lambda: run_mlp_no_g(train_df, test_df), P)

########################################################
# d. MLP_g ensemble (two-stage)
########################################################

print("\n==========================================")
print(f"d. MLP_g ensemble of {len(SEEDS)}: t-1 weather + g_hat(t-1), plain two-stage "
      f"({DIST_LABEL}, {EPOCHS} epochs, hidden layers {ARCH_LABEL})")
print("==========================================")

mlp_g = run_ensemble("MLP_g", lambda: run_mlp_g(g_train_df, train_df, test_df), P_G)


# ============================================================
# Save results
# ============================================================

summary = {
    "Distance km": DISTANCE_KM,
    "Rows": ROWS_LABEL,
    "Train Sensors": n_train_sensors,
    "Test Sensors": n_test_sensors,
    "Test Sensors Also In Train": n_test_sensors_in_train,
    "Train Observations": len(train_df),
    "Test Observations": len(test_df),
    "MLP_g g Train Observations": len(g_train_df),
    "Train period": "2021-07-01 to 2021-09-30",
    "Test period": "2022-09-01 to 2022-09-30",
    "Site": "32-031-0031",
    "Score scale": "log1p PM2.5 (all models)",
    "Epochs": EPOCHS,
    "Architecture": ARCH_LABEL,
    "Two-stage Training": "plain (no cross-fitting)",
    "MLP Ensemble Seeds": " ".join(str(seed) for seed in SEEDS),
    "Max Invalid Share": MAX_INVALID_SHARE,
    "reg_lin_model Train clipped": reg_lin_clipped["Train"],
    "reg_lin_model Test clipped": reg_lin_clipped["Test"],
}

for name, s in results.items():
    summary[f"{name} n"] = s["n"]
    summary[f"{name} p"] = s["p"]
    summary[f"{name} R2"] = s["r2"]
    summary[f"{name} Adj R2"] = s["adj_r2"]
    summary[f"{name} MSE"] = s["mse"]

# Spread of the single MLPs (each seed alone): min, mean and max
for model in ["MLP_no_g", "MLP_g"]:
    for split in ["Train", "Test"]:
        for key, label in [("r2", "R2"), ("adj_r2", "Adj R2"), ("mse", "MSE")]:
            vals = [results[f"{model} seed {seed} {split}"][key] for seed in SEEDS]
            summary[f"{model} single-seed min {split} {label}"] = np.min(vals)
            summary[f"{model} single-seed mean {split} {label}"] = np.mean(vals)
            summary[f"{model} single-seed max {split} {label}"] = np.max(vals)

summary_df = pd.DataFrame([summary])

print("\n")
print(summary_df.T.to_string(header=False))

summary_df.to_csv(os.path.join(args.out_dir, f"{NAME}_Summary_{DIST_LABEL}.csv"), index=False)

# Which sensors were used in each period
sensor_table = pd.concat([
    train_sensors.assign(set="train Jul-Sep 2021"),
    test_sensors.assign(set="test Sep 2022")
], ignore_index=True).drop(columns=["files"])

rows_used = {
    "train Jul-Sep 2021": train_df["sensor_id"].value_counts(),
    "test Sep 2022": test_df["sensor_id"].value_counts()
}

sensor_table["rows used"] = [
    int(rows_used[s].get(sensor, 0)) for sensor, s in zip(sensor_table["sensor_id"], sensor_table["set"])
]

sensor_table.to_csv(os.path.join(args.out_dir, f"{NAME}_Sensors_{DIST_LABEL}.csv"), index=False)

# Test predictions (reg_lin_pred is raw; every other *_pred column is log1p)
predictions = test_df[["sensor_id", "timestamp", "distance_km", RAW_TARGET_COLUMN, TARGET_COLUMN]].copy()
predictions["reg_lin_pred"] = reg_lin["test_pred"]
predictions["reg_lin_pred_log1p"] = raw_to_log1p(reg_lin["test_pred"])
predictions["log_trans_lin_pred"] = log_trans_lin["test_pred"]
predictions["mlp_no_g_pred"] = mlp_no_g["test_pred"]
predictions["mlp_g_pred"] = mlp_g["test_pred"]

for seed in SEEDS:
    predictions[f"mlp_no_g_pred_seed{seed}"] = mlp_no_g["runs"][seed]["test_pred"]
    predictions[f"mlp_g_pred_seed{seed}"] = mlp_g["runs"][seed]["test_pred"]

predictions.to_csv(os.path.join(args.out_dir, f"{NAME}_Test_Predictions_{DIST_LABEL}.csv"), index=False)

print(f"\nSaved results to {args.out_dir}")
