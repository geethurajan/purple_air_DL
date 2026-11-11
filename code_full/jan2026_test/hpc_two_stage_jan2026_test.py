"""
Test on a later year for ONE radius: train on the PurpleAir sensors around the downtown LA EPA
monitor (site 06-037-1103) in January-February 2025, test on the SAME sensors and the same
monitor in January 2026.

    training: sensors within --distance km of site 1103, January 1 - February 28, 2025, compared
              with the site 1103 EPA values. Every hour is used (A and B days), nothing is held out.
    test:     the same sensors (those that still report), January 1-31, 2026, compared with the
              site 1103 EPA values of January 2026.
Every model is trained on the 2025 data only; the 2026 EPA values are used only for scoring.
All three models are trained and tested on the same rows.

Models (t = current hour, t-1 = the hour before at the same sensor), the same as
hpc_two_stage_by_distance_days_AB.py:

    log_trans_lin_model: linear regression on log1p PM2.5
        log EPA PM2.5(t) ~ log PA PM2.5(t), temp(t), hum(t), press(t),
                           temp(t-1), hum(t-1), press(t-1), distance, lat, lon

    reg_lin_model: the same linear regression without the log transform

    MLP: two-stage model, f and g are MLPs trained for a fixed number of epochs (default 150)
        log EPA PM2.5(t) = f( log PA PM2.5(t), PA press(t), PA temp(t), PA hum(t),
                              lat, lon, distance, g_hat(t-1) )
        g_hat(t-1)       = g( log PA PM2.5(t-1), PA press(t-1), PA temp(t-1), PA hum(t-1),
                              lat, lon, distance )
        g estimates log EPA PM2.5 from the PurpleAir readings of the same hour. It is trained
        first (target = real log EPA PM2.5), then applied to the t-1 readings; its estimate
        is one input of f.
        Hidden layers of g and f: --hidden, default 16,32,16. This architecture had the highest
        validation adjusted R² in the layer search on the 2025 data
        (hpc_two_stage_by_distance_days_AB_layers_mixed.py); it is fixed before the 2026 test.

t-1 values are used only when the previous reading is exactly one hour earlier.

Quality control (training and test alike, decided without the test EPA values):
  - PurpleAir readings outside VALID_RANGES come from broken sensors (e.g. temperature -228 F,
    pressure 260 hPa, PM2.5 stuck near 2500) and are dropped before the t-1 values are built.
  - A sensor with more than MAX_INVALID_SHARE of its hours outside VALID_RANGES is broken for
    the whole period and is not used at all: its remaining hours are not reliable either
    (in January 2026, sensors 96837, 38399 and 86185 read about 2500 for 73-88% of the hours).
  - EPA hours with negative PM2.5 are dropped: PM2.5 cannot be negative; these are instrument
    noise at very clean hours, and log1p turns them into extreme values (-0.9 -> -2.3).

Two-stage training with cross-fitting: f learns how much to trust g_hat(t-1). If g_hat(t-1) on
f's training rows came from the final g, which was trained on those same hours, it would look
far more accurate than g is on new data (R² 0.74 on the training hours vs 0.38 in January 2026
at 5 km), and f would rely on it too much. So the training days are split into
--crossfit-folds groups (consecutive days take turns), and g_hat(t-1) for f's training rows
comes from a g trained without that group. The final g (all training days) is used for the test.

Folders (run from inside jan2026_test, which sits inside the training data folder):
    --train-dir ..  LA_Site_1103.csv, Sensors_with_coordinates.csv, sensor_*_history.csv (2025)
    --test-dir  .   LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv, sensor_*_history.csv (2026)
Test sensor files are read from --test-dir only (the files in --train-dir are the 2025 ones).

Usage:
    python hpc_two_stage_jan2026_test.py --distance 7
    python hpc_two_stage_jan2026_test.py --distance 7 --train-dir .. --test-dir . --out-dir results
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

parser = argparse.ArgumentParser()

parser.add_argument("--distance", type=float, required=True,
                    help="Radius in km around the EPA monitor")
parser.add_argument("--epochs", type=int, default=150,
                    help="MLP training epochs (fixed, no search)")
parser.add_argument("--hidden", default="16,32,16",
                    help="Neurons in each hidden layer of g and f")
parser.add_argument("--crossfit-folds", type=int, default=1,
                    help="Day groups for cross-fitting g_hat(t-1) on f's training rows "
                         "(1 = no cross-fitting, the old behavior)")
parser.add_argument("--train-dir", default="..",
                    help="Folder with LA_Site_1103.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (Jan-Feb 2025 training data)")
parser.add_argument("--test-dir", default=".",
                    help="Folder with LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (January 2026 test data)")
parser.add_argument("--out-dir", default="results_jan2026_test",
                    help="Folder for output files")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs
HIDDEN = [int(n) for n in args.hidden.split(",")]
CROSSFIT_FOLDS = args.crossfit_folds

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

# e.g. [16, 32, 16] -> "16-32-16"
ARCH_LABEL = "-".join(str(n) for n in HIDDEN)

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
# Random seed
# ============================================================

SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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

print(f"Radius: {DISTANCE_KM} km | Epochs: {EPOCHS} | Hidden layers: {ARCH_LABEL} "
      f"| Train: Jan-Feb 2025 | Test: Jan 2026 (site 1103, downtown LA)")


# ============================================================
# Files
# ============================================================

TRAIN_EPA_FILE = os.path.join(args.train_dir, "LA_Site_1103.csv")
TRAIN_COORD_FILE = os.path.join(args.train_dir, "Sensors_with_coordinates.csv")

TEST_EPA_FILE = os.path.join(args.test_dir, "LA_Site_1103_Jan2026.csv")
TEST_COORD_FILE = os.path.join(args.test_dir, "Sensors_with_coordinates.csv")


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

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ============================================================
# Predictor columns
# ============================================================

# log_trans_lin_model: the same inputs as the original script
FEATURE_COLUMNS = [
    "log_PA_PM",
    "temperature",
    "humidity",
    "pressure",
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1",
    "distance_km",
    "sensor_lat",
    "sensor_lon"
]

TARGET_COLUMN = "log_EPA_PM"

# reg_lin_model: raw (no log transform) versions
RAW_FEATURE_COLUMNS = ["pm2.5_cf_1"] + FEATURE_COLUMNS[1:]
RAW_TARGET_COLUMN = "EPA_PM"

# MLP g: inputs at hour t (lat, lon and distance do not change over time)
G_COLUMNS = [
    "log_PA_PM",
    "temperature",
    "humidity",
    "pressure",
    "distance_km",
    "sensor_lat",
    "sensor_lon"
]

# The same inputs one hour earlier, in the same order as G_COLUMNS
G_LAG_COLUMNS = [
    "log_PA_PM_lag1",
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1",
    "distance_km",
    "sensor_lat",
    "sensor_lon"
]

# MLP f: the hour-t inputs plus g's estimate for t-1
F_COLUMNS = G_COLUMNS + ["g_hat_lag1"]

MODEL_COLUMNS = list(dict.fromkeys(
    ["sensor_id", "timestamp", TARGET_COLUMN, RAW_TARGET_COLUMN]
    + RAW_FEATURE_COLUMNS
    + FEATURE_COLUMNS
    + G_LAG_COLUMNS
))


# ============================================================
# Readers
# ============================================================

def read_epa(file):

    epa = pd.read_csv(file)

    epa["datetime"] = pd.to_datetime(
        epa["Date Local"] + " " + epa["Time Local"]
    )

    epa = epa.rename(
        columns={
            "Sample Measurement": "EPA_PM"
        }
    )

    epa = epa[
        [
            "datetime",
            "Latitude",
            "Longitude",
            "EPA_PM"
        ]
    ]

    epa = epa.dropna()

    # PM2.5 cannot be negative: the monitor reports small negative values at very clean hours
    # (noise below its detection limit), and log1p turns them into extreme values
    # (-0.9 -> -2.3, -inf at -1). Those hours are dropped.
    too_low = epa["EPA_PM"] < 0

    print(f"{file}: dropped {too_low.sum()} hours with negative EPA PM2.5")

    epa = epa[~too_low]

    epa["timestamp"] = (
        epa["datetime"]
        .dt.tz_localize("America/Los_Angeles", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    return epa.drop(columns=["datetime"])


def monitor_location(epa, file):

    locations = epa[["Latitude", "Longitude"]].drop_duplicates()

    if len(locations) != 1:
        raise SystemExit(f"{file} has {len(locations)} monitor locations, expected 1.")

    return locations.iloc[0]["Latitude"], locations.iloc[0]["Longitude"]


def read_coords(file):

    coords = pd.read_csv(file)

    coords = coords.rename(
        columns={
            "sensor": "sensor_id",
            "lat": "sensor_lat",
            "lon": "sensor_lon"
        }
    )

    coords["sensor_id"] = coords["sensor_id"].astype(str)

    coords = coords.dropna(subset=["sensor_lat", "sensor_lon"])

    return coords[["sensor_id", "sensor_lat", "sensor_lon"]].drop_duplicates("sensor_id")


def sensor_file(sensor_id, folder):

    # This sensor's history file in the folder, or None
    path = os.path.join(folder, f"sensor_{sensor_id}_history.csv")

    return path if os.path.exists(path) else None


def read_sensor(file, sensor_id):

    try:
        df = pd.read_csv(file)
    except Exception:
        print("Could not read:", file)
        return None

    required_columns = [
        "time_stamp",
        "pm2.5_cf_1",
        "temperature",
        "humidity",
        "pressure"
    ]

    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns in {file}")
        return None

    df = df[required_columns].copy()

    # Skip header-only files (they turn numeric columns into object dtype on concat)
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

    # A sensor that is broken most of the time is not used at all: its other hours are not reliable
    if (~valid).mean() > MAX_INVALID_SHARE:
        print(f"{file}: {(~valid).mean():.0%} of rows outside the valid ranges, sensor not used")
        return None

    df = df[valid].copy()

    if df.empty:
        return None

    df["sensor_id"] = sensor_id

    df["time_stamp"] = pd.to_datetime(
        df["time_stamp"],
        utc=True
    )

    df = df.sort_values("time_stamp")

    # t-1 values, only when the previous reading is exactly one hour earlier
    one_hour_before = df["time_stamp"].diff() == pd.Timedelta(hours=1)

    for col in required_columns[1:]:
        df[f"{col}_lag1"] = df[col].shift(1).where(one_hour_before)

    return df


# ============================================================
# Read EPA data
# ============================================================

print("Loading EPA data...")

epa_train = read_epa(TRAIN_EPA_FILE)
epa_test = read_epa(TEST_EPA_FILE)

TRAIN_SITE = monitor_location(epa_train, TRAIN_EPA_FILE)
TEST_SITE = monitor_location(epa_test, TEST_EPA_FILE)

print(f"Training EPA observations (Jan-Feb 2025): {len(epa_train)} | location {TRAIN_SITE}")
print(f"Test EPA observations (Jan 2026)        : {len(epa_test)} | location {TEST_SITE}")

# Both files must be the same monitor
moved_km = haversine(TRAIN_SITE[0], TRAIN_SITE[1], TEST_SITE[0], TEST_SITE[1])

if moved_km > 0.1:
    raise SystemExit(f"The 2025 and 2026 EPA files are {moved_km:.2f} km apart; expected the same monitor.")


# ============================================================
# Choose the sensors within the radius
# ============================================================

def sensors_in_radius(coords_file, folder):

    coords = read_coords(coords_file)

    coords["km_to_site"] = haversine(
        TRAIN_SITE[0], TRAIN_SITE[1], coords["sensor_lat"].values, coords["sensor_lon"].values
    )

    coords["file"] = [sensor_file(s, folder) for s in coords["sensor_id"]]

    return coords[
        (coords["km_to_site"] <= DISTANCE_KM) & coords["file"].notna()
    ].reset_index(drop=True)


train_sensors = sensors_in_radius(TRAIN_COORD_FILE, args.train_dir)
test_sensors = sensors_in_radius(TEST_COORD_FILE, args.test_dir)

print(f"\nTraining sensor files (2025) within {DISTANCE_KM} km of site 1103: {len(train_sensors)}")
print(f"Test sensor files (2026) within {DISTANCE_KM} km of site 1103    : {len(test_sensors)}")

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

    for sensor_id, file in zip(sensors["sensor_id"], sensors["file"]):
        df = read_sensor(file, sensor_id)
        if df is not None:
            all_sensor_data.append(df)

    if not all_sensor_data:
        raise SystemExit(f"No readable {name} sensor files within {DISTANCE_KM} km.")

    purpleair = pd.concat(
        all_sensor_data,
        ignore_index=True
    )

    print(f"{name} PurpleAir observations:", len(purpleair))

    # Attach coordinates
    purpleair = purpleair.merge(
        sensors[["sensor_id", "sensor_lat", "sensor_lon"]],
        on="sensor_id",
        how="left"
    )

    purpleair = purpleair.rename(columns={"time_stamp": "timestamp"})

    # Merge EPA with PurpleAir
    merged = purpleair.merge(
        epa,
        on="timestamp",
        how="inner"
    )

    print(f"{name} merged observations:", len(merged))

    # Distance from the sensor to the EPA monitor
    merged["distance_km"] = haversine(
        merged["Latitude"],
        merged["Longitude"],
        merged["sensor_lat"],
        merged["sensor_lon"]
    )

    # Log transforms
    merged["log_PA_PM"] = np.log1p(merged["pm2.5_cf_1"])
    merged["log_PA_PM_lag1"] = np.log1p(merged["pm2.5_cf_1_lag1"])
    merged["log_EPA_PM"] = np.log1p(merged["EPA_PM"])

    # A log1p of -1 would be -inf; treat it as missing
    merged = merged[MODEL_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Every row needs its hour-t values. The t-1 values may be missing:
    # the MLP's g does not need them, and the rows used by all three models must have them.
    merged = merged.dropna(
        subset=[TARGET_COLUMN, RAW_TARGET_COLUMN, "pm2.5_cf_1"] + G_COLUMNS
    )

    return merged[merged["distance_km"] <= DISTANCE_KM].reset_index(drop=True)


train_all = load_period(train_sensors, epa_train, "Training (2025)")
test_all = load_period(test_sensors, epa_test, "Test (2026)")


# ============================================================
# Training and test rows
# ============================================================

train_has_lag = train_all[G_LAG_COLUMNS].notna().all(axis=1)
test_has_lag = test_all[G_LAG_COLUMNS].notna().all(axis=1)

# All three models are trained and tested on these rows (they need the t-1 readings)
train_df = train_all[train_has_lag].reset_index(drop=True)
test_df = test_all[test_has_lag].reset_index(drop=True)

# The MLP's g learns from every training hour (it only needs that hour's readings)
g_train_df = train_all

if len(train_df) == 0 or len(test_df) == 0:
    raise SystemExit(
        f"Empty data within {DISTANCE_KM} km "
        f"(train {len(train_df)}, test {len(test_df)} observations)."
    )

n_train_sensors = train_df["sensor_id"].nunique()
n_test_sensors = test_df["sensor_id"].nunique()

# Test sensors that were also in training (expected: all of them, in a different year)
n_test_sensors_in_train = len(set(test_df["sensor_id"]) & set(train_df["sensor_id"]))


def local_range(d):
    t = d["timestamp"].dt.tz_convert("America/Los_Angeles")
    return f"{t.min():%Y-%m-%d %H:%M} -> {t.max():%Y-%m-%d %H:%M}"


print("\n================ Dataset Summary ================\n")
print(DIST_LABEL)
print(f"Train (2025) : {len(train_df)} obs | {n_train_sensors} sensors | {local_range(train_df)} (LA time)")
print(f"Test  (2026) : {len(test_df)} obs | {n_test_sensors} sensors | {local_range(test_df)} (LA time)")
print(f"Test sensors also in training: {n_test_sensors_in_train} of {n_test_sensors}")
print(f"MLP g train  : {len(g_train_df)} obs")
print(f"Mean PM2.5   : train {train_df[RAW_TARGET_COLUMN].mean():.2f} | test {test_df[RAW_TARGET_COLUMN].mean():.2f}")
print(f"Mean distance: train {train_df['distance_km'].mean():.2f} km | test {test_df['distance_km'].mean():.2f} km")
print()


# ============================================================
# Build MLP
# ============================================================

def create_model(input_dim):

    # One Linear + ReLU per entry of HIDDEN, then a Linear output layer
    layers = []
    size_in = input_dim

    for size_out in HIDDEN:
        layers += [nn.Linear(size_in, size_out), nn.ReLU()]
        size_in = size_out

    layers.append(nn.Linear(size_in, 1))

    model = nn.Sequential(*layers)

    return model.to(DEVICE)


# ============================================================
# Train model
# ============================================================

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

            X_batch = X_tensor[idx]
            y_batch = y_tensor[idx]

            optimizer.zero_grad()

            pred = model(X_batch)

            loss = criterion(pred, y_batch)

            loss.backward()

            optimizer.step()

    return model


# ============================================================
# Predict / evaluate
# ============================================================

def predict(model, X):

    model.eval()

    X_tensor = torch.FloatTensor(X).to(DEVICE)

    with torch.no_grad():
        pred = model(X_tensor)

    return pred.cpu().numpy().flatten()


def adjusted_r2(r2, n, p):

    # Same formula as the original script, p = number of predictors
    if n - p - 1 <= 0:
        return np.nan

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def scores(y, pred, p):

    r2 = r2_score(y, pred)

    return {
        "r2": r2,
        "adj_r2": adjusted_r2(r2, len(y), p),
        "mse": mean_squared_error(y, pred)
    }


def print_scores(name, s):
    print(f"{name:<14} R²: {s['r2']:.4f} | Adjusted R²: {s['adj_r2']:.4f} | MSE: {s['mse']:.6f}")


# ============================================================
# Linear regression: fit on train, predict train and test
# ============================================================

def run_ols(train_df, test_df, columns, target):

    model = sm.OLS(
        train_df[target],
        sm.add_constant(train_df[columns], has_constant="add")
    ).fit()

    return {
        "model": model,
        "train_pred": np.asarray(model.predict(sm.add_constant(train_df[columns], has_constant="add"))),
        "test_pred": np.asarray(model.predict(sm.add_constant(test_df[columns], has_constant="add")))
    }


# ============================================================
# MLP two-stage model: g and f are MLPs (fixed epochs)
# ============================================================

def fit_mlp(X, y):

    scaler = StandardScaler().fit(X)

    model = create_model(X.shape[1])

    train_model(
        model,
        scaler.transform(X).astype(np.float32),
        y,
        epochs=EPOCHS
    )

    return scaler, model


def mlp_predict(scaler, model, X):
    return predict(model, scaler.transform(X).astype(np.float32))


def day_fold(timestamps):
    # Cross-fitting group of each hour: consecutive days (LA time) take turns, 2025-01-01 is group 0
    day = (
        timestamps
        .dt.tz_convert("America/Los_Angeles")
        .dt.tz_localize(None)
        .dt.normalize()
        - pd.Timestamp("2025-01-01")
    ).dt.days

    return (day % CROSSFIT_FOLDS).values


def run_mlp_two_stage(g_train_df, train_df, test_df):

    def values(d, columns):
        return d[columns].values.astype(np.float32)

    # g: hour readings -> log EPA PM2.5 at the same hour
    g_scaler, g_model = fit_mlp(
        values(g_train_df, G_COLUMNS),
        values(g_train_df, [TARGET_COLUMN]).flatten()
    )

    def f_inputs(d):
        g_hat = mlp_predict(g_scaler, g_model, values(d, G_LAG_COLUMNS))   # g applied to the t-1 readings
        return np.column_stack([values(d, G_COLUMNS), g_hat]).astype(np.float32)

    # f: hour-t readings + g_hat(t-1) -> log EPA PM2.5 at hour t.
    # Cross-fitting: g_hat(t-1) on f's training rows comes from a g that did not see the day of
    # hour t-1, so it is as accurate as g will be on new data (see the top of this file).
    if CROSSFIT_FOLDS > 1:
        g_fold = day_fold(g_train_df["timestamp"])
        f_fold = day_fold(train_df["timestamp"] - pd.Timedelta(hours=1))

        g_hat_train = np.zeros(len(train_df), dtype=np.float32)

        for k in range(CROSSFIT_FOLDS):
            g_k = g_train_df[g_fold != k]

            scaler_k, model_k = fit_mlp(
                values(g_k, G_COLUMNS),
                values(g_k, [TARGET_COLUMN]).flatten()
            )

            rows = f_fold == k
            g_hat_train[rows] = mlp_predict(scaler_k, model_k, values(train_df[rows], G_LAG_COLUMNS))

        X_f_train = np.column_stack([values(train_df, G_COLUMNS), g_hat_train]).astype(np.float32)
    else:
        X_f_train = f_inputs(train_df)

    f_scaler, f_model = fit_mlp(
        X_f_train,
        values(train_df, [TARGET_COLUMN]).flatten()
    )

    return {
        "g_test_pred": mlp_predict(g_scaler, g_model, values(test_df, G_COLUMNS)),
        "train_pred": mlp_predict(f_scaler, f_model, X_f_train),
        "test_pred": mlp_predict(f_scaler, f_model, f_inputs(test_df))
    }


# ============================================================
# Experiment: train on Jan-Feb 2025, test on Jan 2026
# ============================================================

print("=======================================================")
print(f"LATER-YEAR TEST ({DIST_LABEL})")
print("Train: Jan-Feb 2025 | Test: Jan 2026 (same sensors, site 1103)")
print("=======================================================")

results = {}

########################################################
# log_trans_lin_model
########################################################

log_trans_lin = run_ols(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

results["log_trans_lin_model Train"] = scores(
    train_df[TARGET_COLUMN], log_trans_lin["train_pred"], len(FEATURE_COLUMNS)
)
results["log_trans_lin_model Test"] = scores(
    test_df[TARGET_COLUMN], log_trans_lin["test_pred"], len(FEATURE_COLUMNS)
)

print("\n==========================================")
print(f"log_trans_lin_model: linear regression on log1p PM2.5 ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["log_trans_lin_model Train"])
print_scores("test", results["log_trans_lin_model Test"])

log_trans_lin_summary_text = str(log_trans_lin["model"].summary())
print("\nCoefficients:")
print(log_trans_lin_summary_text)

with open(os.path.join(args.out_dir, f"log_trans_lin_model_summary_{DIST_LABEL}.txt"), "w") as f:
    f.write(log_trans_lin_summary_text)

########################################################
# reg_lin_model: linear regression without log transform (raw PM2.5)
########################################################

reg_lin = run_ols(train_df, test_df, RAW_FEATURE_COLUMNS, RAW_TARGET_COLUMN)

results["reg_lin_model Train"] = scores(
    train_df[RAW_TARGET_COLUMN], reg_lin["train_pred"], len(RAW_FEATURE_COLUMNS)
)
results["reg_lin_model Test"] = scores(
    test_df[RAW_TARGET_COLUMN], reg_lin["test_pred"], len(RAW_FEATURE_COLUMNS)
)

print("\n==========================================")
print(f"reg_lin_model: linear regression, no log transform ({DIST_LABEL})")
print("==========================================")
print_scores("train", results["reg_lin_model Train"])
print_scores("test", results["reg_lin_model Test"])

reg_lin_summary_text = str(reg_lin["model"].summary())
print("\nCoefficients:")
print(reg_lin_summary_text)

with open(os.path.join(args.out_dir, f"reg_lin_model_summary_{DIST_LABEL}.txt"), "w") as f:
    f.write(reg_lin_summary_text)

########################################################
# MLP two-stage model (fixed epochs)
########################################################

mlp = run_mlp_two_stage(g_train_df, train_df, test_df)

results["MLP Train"] = scores(
    train_df[TARGET_COLUMN], mlp["train_pred"], len(F_COLUMNS)
)
results["MLP Test"] = scores(
    test_df[TARGET_COLUMN], mlp["test_pred"], len(F_COLUMNS)
)
# g alone on the same test hours (no t-1 information), to see what the t-1 term adds
results["MLP g only Test"] = scores(
    test_df[TARGET_COLUMN], mlp["g_test_pred"], len(G_COLUMNS)
)

print("\n==========================================")
print(f"MLP two-stage model ({DIST_LABEL}, {EPOCHS} epochs, hidden layers {ARCH_LABEL}, "
      f"cross-fitting folds {CROSSFIT_FOLDS})")
print("==========================================")
print_scores("train", results["MLP Train"])
print_scores("test", results["MLP Test"])
print_scores("g only test", results["MLP g only Test"])


# ============================================================
# Save results
# ============================================================

summary = {
    "Distance km": DISTANCE_KM,
    "Train Sensors": n_train_sensors,
    "Test Sensors": n_test_sensors,
    "Test Sensors Also In Train": n_test_sensors_in_train,
    "Observations": len(train_df) + len(test_df),
    "Train Observations": len(train_df),
    "Test Observations": len(test_df),
    "Train period": "2025-01-01 to 2025-02-28",
    "Test period": "2026-01-01 to 2026-01-31",
    "Site": "06-037-1103",
    "Epochs": EPOCHS,
    "Architecture": ARCH_LABEL,
    "Crossfit Folds": CROSSFIT_FOLDS,
    "Max Invalid Share": MAX_INVALID_SHARE
}

for name, s in results.items():
    summary[f"{name} R2"] = s["r2"]
    summary[f"{name} Adj R2"] = s["adj_r2"]
    summary[f"{name} MSE"] = s["mse"]

summary_df = pd.DataFrame([summary])

print("\n")
print(summary_df.T.to_string(header=False))

summary_df.to_csv(
    os.path.join(args.out_dir, f"Jan2026Test_Summary_{DIST_LABEL}.csv"),
    index=False
)

# Which sensors were used in each period
sensor_table = pd.concat([
    train_sensors.assign(set="train 2025"),
    test_sensors.assign(set="test 2026")
], ignore_index=True).drop(columns=["file"])

rows_used = {
    "train 2025": train_df["sensor_id"].value_counts(),
    "test 2026": test_df["sensor_id"].value_counts()
}

sensor_table["rows used"] = [
    int(rows_used[s].get(sensor, 0))
    for sensor, s in zip(sensor_table["sensor_id"], sensor_table["set"])
]

sensor_table.to_csv(
    os.path.join(args.out_dir, f"Jan2026Test_Sensors_{DIST_LABEL}.csv"),
    index=False
)

# Test predictions, for plots of predicted vs EPA over time
# (reg_lin_model is on the raw scale, the others on the log1p scale)
predictions = test_df[["sensor_id", "timestamp", "distance_km", RAW_TARGET_COLUMN, TARGET_COLUMN]].copy()
predictions["log_trans_lin_pred"] = log_trans_lin["test_pred"]
predictions["reg_lin_pred"] = reg_lin["test_pred"]
predictions["mlp_pred"] = mlp["test_pred"]
predictions["mlp_g_only_pred"] = mlp["g_test_pred"]

predictions.to_csv(
    os.path.join(args.out_dir, f"Jan2026Test_Test_Predictions_{DIST_LABEL}.csv"),
    index=False
)

print(f"\nSaved results to {args.out_dir}")
