"""
MODEL 1: only the hour-t PurpleAir measurements + location, no t-1 values, no g.

Train on the PurpleAir sensors around the downtown LA EPA monitor (site 06-037-1103) in
January-March 2025, test on the SAME sensors and the same monitor in January 2026.
Every model is trained on the 2025 data only; the 2026 EPA values are used only for scoring.

Notation (t = current hour):
    y(t)             EPA PM2.5 at site 1103
    x(t)             PurpleAir PM2.5 (pm2.5_cf_1) at the sensor
    T(t), H(t), P(t) PurpleAir temperature, humidity, pressure
    lat, lon, d      sensor location and its distance (km) to the EPA monitor

Models (p = 7 predictors each):

    1. log_trans_lin_model: linear regression (OLS) on the log1p scale
        log1p y(t) = b0 + b1 log1p x(t) + b2 T(t) + b3 H(t) + b4 P(t) + b5 lat + b6 lon + b7 d

    2. reg_lin_model: linear regression (OLS) on the raw scale
        y(t) = c0 + c1 x(t) + c2 T(t) + c3 H(t) + c4 P(t) + c5 lat + c6 lon + c7 d
       For the same-scale comparison its prediction is moved to log1p:
        log1p( max(y_hat(t), 0) )      (PM2.5 cannot be negative; log1p is undefined at <= -1)

    3. MLP ensemble: one MLP f (no g), trained --n-seeds times (default 5: seeds 42..46)
        log1p y_hat_s(t) = f_s( log1p x(t), T(t), H(t), P(t), lat, lon, d )
        log1p y_hat(t)   = average of log1p y_hat_s(t) over the seeds
       Every seed uses the same data and settings (hidden layers --hidden, default 16,32,16;
       --epochs, default 150; Adam lr 0.001; batch 64; MSE loss; standardized inputs); only
       the random start and the batch order differ. Each seed's own scores are also saved.

ALL THREE MODELS ARE SCORED ON THE SAME SCALE: R², adjusted R² and MSE against log1p y(t), on
train and on test. reg_lin_model's raw-scale scores are also saved, for reference only.

Rows: by default only the hours that ALSO have the t-1 readings of the same sensor (previous
reading exactly one hour earlier). The t-1 values are NOT used by any model here; this only
makes Model 1 use exactly the same rows as Model 2 (the t-1 code), so the two can be compared.
With --all-hours, every hour with its hour-t readings is used instead.

Quality control (training and test alike, decided without the test EPA values), the same as
jan2026_test/model_2_no_g_jan-mar_train_both_R2_and_adj.py:
  - PurpleAir readings outside VALID_RANGES are dropped (broken sensors).
  - A sensor file with more than MAX_INVALID_SHARE of its hours outside VALID_RANGES is not used.
  - EPA hours with negative PM2.5 are dropped.
EPA hours are matched to PurpleAir with "Date GMT" / "Time GMT" (exact all year).

Folders (defaults are for running from inside comparison, which sits inside code_full):
    --train-dir        ..                  LA_Site_1103.csv, Sensors_with_coordinates.csv,
                                           sensor_*_history.csv (Jan-Feb 2025)
    --train-extra-dir  ../mar2025_train    sensor_*_history.csv (March 2025)
    --test-dir         ../jan2026_test     LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv,
                                           sensor_*_history.csv (January 2026)

Output files (folder results_model_1_t_only_ensemble, <d> = distance), NAME = model_1_t_only_ensemble:
    NAME_Summary_<d>km.csv              n, p, R², adjusted R², MSE of every model, train and test,
                                        all on log1p (+ reg_lin raw scale, + single-seed min/mean/max)
    NAME_Sensors_<d>km.csv              sensors used in each period and their rows
    NAME_Test_Predictions_<d>km.csv     January 2026 predictions (log_trans_lin_pred, reg_lin_pred
                                        (raw), reg_lin_pred_log1p, mlp_pred = ensemble average,
                                        mlp_pred_seed<s> = each seed)
    NAME_log_trans_lin_model_<d>km.txt  OLS coefficient table
    NAME_reg_lin_model_<d>km.txt        OLS coefficient table
Run all distances with run_model_1_t_only_ensemble.sh, then plot them with
plot_model_1_t_only_ensemble_same_scale.py.

Usage:
    python model_1_t_only_ensemble.py --distance 5
    python model_1_t_only_ensemble.py --distance 5 --n-seeds 5 --epochs 150 --hidden 16,32,16
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
                    help="MLP training epochs (fixed, no search)")
parser.add_argument("--hidden", default="16,32,16",
                    help="Neurons in each hidden layer of the MLP")
parser.add_argument("--seed", type=int, default=42,
                    help="First MLP seed; the ensemble uses seed, seed+1, ..., seed+n_seeds-1")
parser.add_argument("--n-seeds", type=int, default=5,
                    help="Number of MLPs in the ensemble; their predictions are averaged")
parser.add_argument("--all-hours", action="store_true",
                    help="Use every hour, not only the hours that also have t-1 readings")
parser.add_argument("--train-dir", default=os.path.join(HERE, ".."),
                    help="Folder with LA_Site_1103.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (Jan-Feb 2025 training data)")
parser.add_argument("--train-extra-dir", default=os.path.join(HERE, "..", "mar2025_train"),
                    help="Folder with sensor_*_history.csv (March 2025 training data)")
parser.add_argument("--test-dir", default=os.path.join(HERE, "..", "jan2026_test"),
                    help="Folder with LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (January 2026 test data)")
parser.add_argument("--out-dir", default=os.path.join(HERE, "results_model_1_t_only_ensemble"),
                    help="Folder for output files")

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs
HIDDEN = [int(n) for n in args.hidden.split(",")]

# Seeds of the MLP ensemble, e.g. 42, 43, 44, 45, 46
SEEDS = [args.seed + i for i in range(args.n_seeds)]

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

# e.g. [16, 32, 16] -> "16-32-16"
ARCH_LABEL = "-".join(str(n) for n in HIDDEN)

ROWS_LABEL = "all hours" if args.all_hours else "hours with t-1 readings (same rows as Model 2)"

# Every output file starts with the name of this code
NAME = "model_1_t_only_ensemble"

os.makedirs(args.out_dir, exist_ok=True)

# Plausible ranges of the PurpleAir readings; a row with any value outside them is dropped
VALID_RANGES = {
    "pm2.5_cf_1": (0, 1000),     # ug/m3
    "temperature": (-40, 150),   # deg F
    "humidity": (0, 100),        # %
    "pressure": (900, 1100),     # hPa
}

# A sensor with a larger share of its hours outside VALID_RANGES is not used for that period
MAX_INVALID_SHARE = 0.1


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

print(f"MODEL 1 (hour t only) | Radius: {DISTANCE_KM} km | Epochs: {EPOCHS} | Hidden layers: {ARCH_LABEL} "
      f"| MLP seeds: {SEEDS} | Rows: {ROWS_LABEL}")
print("Train: Jan-Mar 2025 | Test: Jan 2026 (site 1103, downtown LA)")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Files
# ============================================================

TRAIN_EPA_FILE = os.path.join(args.train_dir, "LA_Site_1103.csv")
TRAIN_COORD_FILE = os.path.join(args.train_dir, "Sensors_with_coordinates.csv")

TEST_EPA_FILE = os.path.join(args.test_dir, "LA_Site_1103_Jan2026.csv")
TEST_COORD_FILE = os.path.join(args.test_dir, "Sensors_with_coordinates.csv")

# Training and test periods, in LA time (start included, end not)
TRAIN_START, TRAIN_END = "2025-01-01", "2025-04-01"   # January-March 2025
TEST_START, TEST_END = "2026-01-01", "2026-02-01"     # January 2026


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
# Predictor columns (hour t only, 7 each)
# ============================================================

# log_trans_lin_model and the MLP: log1p x(t), T(t), H(t), P(t), lat, lon, d
FEATURE_COLUMNS = [
    "log_PA_PM",
    "temperature",
    "humidity",
    "pressure",
    "sensor_lat",
    "sensor_lon",
    "distance_km"
]

TARGET_COLUMN = "log_EPA_PM"

# reg_lin_model: x(t) on the raw scale, the rest unchanged
RAW_FEATURE_COLUMNS = ["pm2.5_cf_1"] + FEATURE_COLUMNS[1:]
RAW_TARGET_COLUMN = "EPA_PM"

P = len(FEATURE_COLUMNS)   # 7 predictors in every model

# t-1 readings: NOT model inputs, only used to pick the same rows as Model 2
LAG_COLUMNS = ["pm2.5_cf_1_lag1", "temperature_lag1", "humidity_lag1", "pressure_lag1"]

MODEL_COLUMNS = list(dict.fromkeys(
    ["sensor_id", "timestamp", TARGET_COLUMN, RAW_TARGET_COLUMN]
    + RAW_FEATURE_COLUMNS
    + FEATURE_COLUMNS
    + LAG_COLUMNS
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

    # t-1 readings, only when the previous reading is exactly one hour earlier.
    # Not model inputs here: only used to choose the same rows as Model 2.
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

print(f"Training EPA observations (Jan-Mar 2025): {len(epa_train)} | location {TRAIN_SITE}")
print(f"Test EPA observations (Jan 2026)        : {len(epa_test)} | location {TEST_SITE}")

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

    return coords[
        (coords["km_to_site"] <= DISTANCE_KM) & (coords["files"].map(len) > 0)
    ].reset_index(drop=True)


train_sensors = sensors_in_radius(TRAIN_COORD_FILE, [args.train_dir, args.train_extra_dir])
test_sensors = sensors_in_radius(TEST_COORD_FILE, [args.test_dir])

print(f"\nTraining sensors (Jan-Mar 2025) within {DISTANCE_KM} km of site 1103: {len(train_sensors)}")
print(f"Test sensor files (Jan 2026) within {DISTANCE_KM} km of site 1103    : {len(test_sensors)}")

if test_sensors.empty:
    raise SystemExit(f"No test sensor files within {DISTANCE_KM} km of site 1103. Nothing to test.")

if train_sensors.empty:
    raise SystemExit(f"No training sensor files within {DISTANCE_KM} km of site 1103. Nothing to train.")


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
    merged["log_EPA_PM"] = np.log1p(merged["EPA_PM"])

    merged = merged[MODEL_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Every row needs its hour-t values
    merged = merged.dropna(subset=[TARGET_COLUMN, RAW_TARGET_COLUMN] + RAW_FEATURE_COLUMNS + FEATURE_COLUMNS)

    return merged[merged["distance_km"] <= DISTANCE_KM].reset_index(drop=True)


train_all = load_period(train_sensors, epa_train, "Training (2025)")
test_all = load_period(test_sensors, epa_test, "Test (Jan 2026)")


# ============================================================
# Training and test rows
# ============================================================

if args.all_hours:
    train_df = train_all
    test_df = test_all
else:
    # Only the hours that also have the t-1 readings: the same rows as Model 2
    train_df = train_all[train_all[LAG_COLUMNS].notna().all(axis=1)].reset_index(drop=True)
    test_df = test_all[test_all[LAG_COLUMNS].notna().all(axis=1)].reset_index(drop=True)

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
print(f"Train (Jan-Mar 2025) : {len(train_df)} obs | {n_train_sensors} sensors | {local_range(train_df)} (LA time)")
print(f"Test  (Jan 2026)     : {len(test_df)} obs | {n_test_sensors} sensors | {local_range(test_df)} (LA time)")
print(f"Test sensors also in training: {n_test_sensors_in_train} of {n_test_sensors}")
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


def run_mlp(train_df, test_df):

    # One MLP f: hour-t readings + location -> log1p EPA PM2.5 at hour t
    X_train = train_df[FEATURE_COLUMNS].values.astype(np.float32)
    X_test = test_df[FEATURE_COLUMNS].values.astype(np.float32)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float32)

    scaler = StandardScaler().fit(X_train)

    model = create_model(X_train.shape[1])
    train_model(model, scaler.transform(X_train).astype(np.float32), y_train, epochs=EPOCHS)

    return {
        "train_pred": predict(model, scaler.transform(X_train).astype(np.float32)),
        "test_pred": predict(model, scaler.transform(X_test).astype(np.float32))
    }


# ============================================================
# Experiment: train on Jan-Mar 2025, test on Jan 2026
# ============================================================

print("=======================================================")
print(f"MODEL 1, hour t only ({DIST_LABEL})")
print("Train: Jan-Mar 2025 | Test: Jan 2026 (same sensors, site 1103)")
print("All models scored on log1p PM2.5")
print("=======================================================")

results = {}

########################################################
# 1. log_trans_lin_model
########################################################

log_trans_lin = run_ols(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

results["log_trans_lin_model Train"] = scores(train_df[TARGET_COLUMN], log_trans_lin["train_pred"], P)
results["log_trans_lin_model Test"] = scores(test_df[TARGET_COLUMN], log_trans_lin["test_pred"], P)

print("\n==========================================")
print(f"1. log_trans_lin_model: linear regression on log1p PM2.5 ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["log_trans_lin_model Train"])
print_scores("test", results["log_trans_lin_model Test"])

with open(os.path.join(args.out_dir, f"{NAME}_log_trans_lin_model_{DIST_LABEL}.txt"), "w") as f:
    f.write(str(log_trans_lin["model"].summary()))

########################################################
# 2. reg_lin_model: fitted on raw PM2.5, scored on log1p
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
print(f"2. reg_lin_model: linear regression on raw PM2.5, scored on log1p ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["reg_lin_model Train"])
print_scores("test", results["reg_lin_model Test"])
print(f"Negative predictions set to 0 before log1p: train {reg_lin_clipped['Train']}, test {reg_lin_clipped['Test']}")
print_scores("raw train", results["reg_lin_model raw-scale Train"])
print_scores("raw test", results["reg_lin_model raw-scale Test"])

with open(os.path.join(args.out_dir, f"{NAME}_reg_lin_model_{DIST_LABEL}.txt"), "w") as f:
    f.write(str(reg_lin["model"].summary()))

########################################################
# 3. MLP ensemble: the same MLP trained once per seed, predictions averaged
########################################################

print("\n==========================================")
print(f"3. MLP ensemble of {len(SEEDS)} ({DIST_LABEL}, {EPOCHS} epochs, hidden layers {ARCH_LABEL})")
print("==========================================")

mlp_runs = {}

for seed in SEEDS:
    set_seed(seed)
    mlp_runs[seed] = run_mlp(train_df, test_df)

    results[f"MLP seed {seed} Train"] = scores(train_df[TARGET_COLUMN], mlp_runs[seed]["train_pred"], P)
    results[f"MLP seed {seed} Test"] = scores(test_df[TARGET_COLUMN], mlp_runs[seed]["test_pred"], P)
    print_scores(f"seed {seed} test", results[f"MLP seed {seed} Test"])

# Ensemble prediction = row-by-row average of the seeds' predictions (on log1p)
mlp = {
    "train_pred": np.mean([run["train_pred"] for run in mlp_runs.values()], axis=0),
    "test_pred": np.mean([run["test_pred"] for run in mlp_runs.values()], axis=0)
}

results["MLP Train"] = scores(train_df[TARGET_COLUMN], mlp["train_pred"], P)
results["MLP Test"] = scores(test_df[TARGET_COLUMN], mlp["test_pred"], P)

print("ensemble (average of the seeds' predictions):")
print_scores("train", results["MLP Train"])
print_scores("test", results["MLP Test"])


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
    "Train period": "2025-01-01 to 2025-03-31",
    "Test period": "2026-01-01 to 2026-01-31",
    "Site": "06-037-1103",
    "Score scale": "log1p PM2.5 (all models)",
    "Epochs": EPOCHS,
    "Architecture": ARCH_LABEL,
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
for split in ["Train", "Test"]:
    for key, label in [("r2", "R2"), ("adj_r2", "Adj R2"), ("mse", "MSE")]:
        values = [results[f"MLP seed {seed} {split}"][key] for seed in SEEDS]
        summary[f"MLP single-seed min {split} {label}"] = np.min(values)
        summary[f"MLP single-seed mean {split} {label}"] = np.mean(values)
        summary[f"MLP single-seed max {split} {label}"] = np.max(values)

summary_df = pd.DataFrame([summary])

print("\n")
print(summary_df.T.to_string(header=False))

summary_df.to_csv(os.path.join(args.out_dir, f"{NAME}_Summary_{DIST_LABEL}.csv"), index=False)

# Which sensors were used in each period
sensor_table = pd.concat([
    train_sensors.assign(set="train Jan-Mar 2025"),
    test_sensors.assign(set="test Jan 2026")
], ignore_index=True).drop(columns=["files"])

rows_used = {
    "train Jan-Mar 2025": train_df["sensor_id"].value_counts(),
    "test Jan 2026": test_df["sensor_id"].value_counts()
}

sensor_table["rows used"] = [
    int(rows_used[s].get(sensor, 0)) for sensor, s in zip(sensor_table["sensor_id"], sensor_table["set"])
]

sensor_table.to_csv(os.path.join(args.out_dir, f"{NAME}_Sensors_{DIST_LABEL}.csv"), index=False)

# Test predictions (reg_lin_pred is raw; every other *_pred column is log1p)
predictions = test_df[["sensor_id", "timestamp", "distance_km", RAW_TARGET_COLUMN, TARGET_COLUMN]].copy()
predictions["log_trans_lin_pred"] = log_trans_lin["test_pred"]
predictions["reg_lin_pred"] = reg_lin["test_pred"]
predictions["reg_lin_pred_log1p"] = raw_to_log1p(reg_lin["test_pred"])
predictions["mlp_pred"] = mlp["test_pred"]

for seed, run in mlp_runs.items():
    predictions[f"mlp_pred_seed{seed}"] = run["test_pred"]

predictions.to_csv(os.path.join(args.out_dir, f"{NAME}_Test_Predictions_{DIST_LABEL}.csv"), index=False)

print(f"\nSaved results to {args.out_dir}")
