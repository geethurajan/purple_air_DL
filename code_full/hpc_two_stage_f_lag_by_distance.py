"""
Time split (train on one month, test on the next) for ONE radius:
every PurpleAir sensor within --distance km of the EPA monitor.

Same as hpc_two_stage_by_distance.py, except that the MLP's f also gets the
PurpleAir temperature, humidity and pressure of hour t-1.

Split: hours before --split-date (Los Angeles time) are training, the rest are test.
The default 2025-02-01 trains on January 2025 and tests on February 2025.
Every model is trained on the training hours only; test EPA values are used only for scoring.
All three models are trained and tested on the same rows.

Models (t = current hour, t-1 = the hour before at the same sensor):

    log_trans_lin_model: linear regression on log1p PM2.5, same inputs as the original script
        log EPA PM2.5(t) ~ log PA PM2.5(t), temp(t), hum(t), press(t),
                           temp(t-1), hum(t-1), press(t-1), distance, lat, lon

    reg_lin_model: the same linear regression without the log transform

    MLP: two-stage model, f and g are MLPs trained for a fixed number of epochs (default 150)
        log EPA PM2.5(t) = f( log PA PM2.5(t), PA press(t), PA temp(t), PA hum(t),
                              PA press(t-1), PA temp(t-1), PA hum(t-1),
                              lat, lon, distance, g_hat(t-1) )
        g_hat(t-1)       = g( log PA PM2.5(t-1), PA press(t-1), PA temp(t-1), PA hum(t-1),
                              lat, lon, distance )
        g estimates log EPA PM2.5 from the PurpleAir readings of the same hour. It is trained
        first (target = real log EPA PM2.5), then applied to the t-1 readings; its estimate
        is one input of f.

t-1 values are used only when the previous reading is exactly one hour earlier.

Usage:
    python hpc_two_stage_f_lag_by_distance.py --distance 3
    python hpc_two_stage_f_lag_by_distance.py --distance 3 --split-date 2025-02-01 --data-dir /path/to/code_full --out-dir results
"""

import os
import glob
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
parser.add_argument("--split-date", default="2025-02-01",
                    help="First day of the test period (Los Angeles time); "
                         "earlier hours are training")
parser.add_argument("--data-dir", default=".",
                    help="Folder with LA_Site_1103.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv")
parser.add_argument("--out-dir", default="results_two_stage_f_lag_by_distance",
                    help="Folder for output files")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

# Start of the test period in UTC (all timestamps below are UTC)
SPLIT_TIME = (
    pd.Timestamp(args.split_date)
    .tz_localize("America/Los_Angeles")
    .tz_convert("UTC")
)

os.makedirs(args.out_dir, exist_ok=True)


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

print(f"Radius: {DISTANCE_KM} km | Epochs: {EPOCHS} | Test starts: {args.split_date} (LA time)")


# ============================================================
# Files
# ============================================================

EPA_FILE = os.path.join(args.data_dir, "LA_Site_1103.csv")
COORD_FILE = os.path.join(args.data_dir, "Sensors_with_coordinates.csv")

SENSOR_FILES = sorted(glob.glob(os.path.join(args.data_dir, "sensor_*_history.csv")))

print(f"Found {len(SENSOR_FILES)} PurpleAir files.")


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
# Read EPA data
# ============================================================

print("Loading EPA data...")

epa = pd.read_csv(EPA_FILE)

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

print("EPA observations:", len(epa))


# ============================================================
# Read PurpleAir coordinate table
# ============================================================

coords = pd.read_csv(COORD_FILE)

coords = coords.rename(
    columns={
        "sensor": "sensor_id",
        "lat": "sensor_lat",
        "lon": "sensor_lon"
    }
)

coords["sensor_id"] = coords["sensor_id"].astype(str)

print("Coordinate table:", len(coords))


# ============================================================
# Keep only sensors within the radius (before loading them)
# ============================================================

# Distance from each sensor to the nearest EPA monitor location in the file.
epa_locations = epa[["Latitude", "Longitude"]].drop_duplicates().values

coords = coords.dropna(subset=["sensor_lat", "sensor_lon"]).copy()

coords["min_epa_distance"] = np.min(
    [
        haversine(lat, lon, coords["sensor_lat"].values, coords["sensor_lon"].values)
        for lat, lon in epa_locations
    ],
    axis=0
)

sensors_in_radius = set(
    coords.loc[coords["min_epa_distance"] <= DISTANCE_KM, "sensor_id"]
)

coords = coords.drop(columns=["min_epa_distance"])

print(f"Sensors within {DISTANCE_KM} km (from coordinate table):", len(sensors_in_radius))


def sensor_id_from_file(file):
    sensor_id = os.path.basename(file)
    sensor_id = sensor_id.replace("sensor_", "")
    sensor_id = sensor_id.replace("_history.csv", "")
    return sensor_id


SENSOR_FILES = [f for f in SENSOR_FILES if sensor_id_from_file(f) in sensors_in_radius]

print(f"Sensor files to load: {len(SENSOR_FILES)}")

if not SENSOR_FILES:
    raise SystemExit(f"No sensor files within {DISTANCE_KM} km. Nothing to run.")


# ============================================================
# Read PurpleAir sensors in the radius
# ============================================================

all_sensor_data = []

print("\nLoading PurpleAir sensors...")

for file in SENSOR_FILES:

    sensor_id = sensor_id_from_file(file)

    try:
        df = pd.read_csv(file)
    except Exception:
        print("Could not read:", file)
        continue

    required_columns = [
        "time_stamp",
        "pm2.5_cf_1",
        "temperature",
        "humidity",
        "pressure"
    ]

    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns in {file}")
        continue

    df = df[required_columns].copy()

    # Skip header-only files (they turn numeric columns into object dtype on concat)
    if df.empty:
        print(f"No rows in {file}")
        continue

    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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

    all_sensor_data.append(df)

if not all_sensor_data:
    raise SystemExit(f"No readable sensor files within {DISTANCE_KM} km.")

purpleair = pd.concat(
    all_sensor_data,
    ignore_index=True
)

print("\nTotal PurpleAir observations:", len(purpleair))


# ============================================================
# Attach coordinates
# ============================================================

purpleair["sensor_id"] = purpleair["sensor_id"].astype(str)

purpleair = purpleair.merge(
    coords,
    on="sensor_id",
    how="left"
)

missing = purpleair["sensor_lat"].isna().sum()

print("Rows missing coordinates:", missing)

purpleair = purpleair.dropna(
    subset=["sensor_lat", "sensor_lon"]
)

purpleair = purpleair.rename(columns={"time_stamp": "timestamp"})


# ============================================================
# Prepare EPA timestamps
# ============================================================

epa["datetime"] = pd.to_datetime(
    epa["datetime"]
)

if epa["datetime"].dt.tz is None:
    epa["timestamp"] = (
        epa["datetime"]
        .dt.tz_localize("America/Los_Angeles", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
else:
    epa["timestamp"] = (
        epa["datetime"]
        .dt.tz_convert("UTC")
    )

epa = epa.drop(columns=["datetime"])


# ============================================================
# Merge EPA with PurpleAir
# ============================================================

print("\nMerging EPA and PurpleAir observations...")

merged = purpleair.merge(
    epa,
    on="timestamp",
    how="inner"
)

print("Merged observations:", len(merged))


# ============================================================
# Compute distance from EPA monitor
# ============================================================

merged["distance_km"] = haversine(
    merged["Latitude"],
    merged["Longitude"],
    merged["sensor_lat"],
    merged["sensor_lon"]
)


# ============================================================
# Log transforms
# ============================================================

merged["log_PA_PM"] = np.log1p(
    merged["pm2.5_cf_1"]
)

merged["log_PA_PM_lag1"] = np.log1p(
    merged["pm2.5_cf_1_lag1"]
)

merged["log_EPA_PM"] = np.log1p(
    merged["EPA_PM"]
)


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

# PurpleAir weather readings one hour earlier, also given to f
F_LAG_COLUMNS = [
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1"
]

# MLP f: the hour-t inputs, the t-1 weather readings and g's estimate for t-1
F_COLUMNS = G_COLUMNS + F_LAG_COLUMNS + ["g_hat_lag1"]


# ============================================================
# Remove incomplete rows
# ============================================================

model_columns = list(dict.fromkeys(
    ["sensor_id", "timestamp", TARGET_COLUMN, RAW_TARGET_COLUMN]
    + RAW_FEATURE_COLUMNS
    + FEATURE_COLUMNS
    + G_LAG_COLUMNS
))

merged = merged[model_columns]

# Every row needs its hour-t values. The t-1 values may be missing:
# the MLP's g does not need them, and the split below keeps only rows that have them.
merged = merged.dropna(
    subset=[TARGET_COLUMN, RAW_TARGET_COLUMN, "pm2.5_cf_1"] + G_COLUMNS
).reset_index(drop=True)


# ============================================================
# Build the dataset for this radius
# ============================================================

df = (
    merged[
        merged["distance_km"] <= DISTANCE_KM
    ]
    .copy()
    .reset_index(drop=True)
)

if len(df) == 0:
    raise SystemExit(f"No merged observations within {DISTANCE_KM} km.")

n_sensors = df["sensor_id"].nunique()


# ============================================================
# Train / test split by time
# ============================================================

is_train = df["timestamp"] < SPLIT_TIME
has_lag = df[G_LAG_COLUMNS].notna().all(axis=1)

# All three models are trained and tested on these rows (they need the t-1 readings)
train_df = df[is_train & has_lag].reset_index(drop=True)
test_df = df[~is_train & has_lag].reset_index(drop=True)

# The MLP's g learns from every training hour (it only needs that hour's readings)
g_train_df = df[is_train].reset_index(drop=True)

if len(train_df) == 0 or len(test_df) == 0:
    raise SystemExit(
        f"Empty split within {DISTANCE_KM} km "
        f"(train {len(train_df)}, test {len(test_df)} observations)."
    )


def local_range(d):
    t = d["timestamp"].dt.tz_convert("America/Los_Angeles")
    return f"{t.min():%Y-%m-%d %H:%M} -> {t.max():%Y-%m-%d %H:%M}"


print("\n================ Dataset Summary ================\n")
print(DIST_LABEL)
print(f"Sensors          : {n_sensors}")
print(f"Train            : {len(train_df)} obs | {train_df['sensor_id'].nunique()} sensors"
      f" | {local_range(train_df)} (LA time)")
print(f"Test             : {len(test_df)} obs | {test_df['sensor_id'].nunique()} sensors"
      f" | {local_range(test_df)} (LA time)")
print(f"MLP g train      : {len(g_train_df)} obs")
print(f"Mean PM2.5 train : {train_df[RAW_TARGET_COLUMN].mean():.2f}")
print(f"Mean PM2.5 test  : {test_df[RAW_TARGET_COLUMN].mean():.2f}")
print(f"Mean Distance    : {df['distance_km'].mean():.2f} km")
print()


# ============================================================
# Build MLP
# ============================================================

def create_model(input_dim):

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

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
        return np.column_stack([values(d, G_COLUMNS + F_LAG_COLUMNS), g_hat]).astype(np.float32)

    # f: hour-t readings + t-1 temp, hum, press + g_hat(t-1) -> log EPA PM2.5 at hour t
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
# Experiment: train on one period, test on the next
# ============================================================

print("=======================================================")
print(f"TIME SPLIT EXPERIMENT ({DIST_LABEL})")
print(f"Train before {args.split_date}, test from {args.split_date} (LA time)")
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
print(f"MLP two-stage model ({DIST_LABEL}, {EPOCHS} epochs)")
print("==========================================")
print_scores("train", results["MLP Train"])
print_scores("test", results["MLP Test"])
print_scores("g only test", results["MLP g only Test"])


# ============================================================
# Save results
# ============================================================

summary = {
    "Distance km": DISTANCE_KM,
    "Sensors": n_sensors,
    "Observations": len(train_df) + len(test_df),
    "Train Observations": len(train_df),
    "Test Observations": len(test_df),
    "Split date": args.split_date,
    "Epochs": EPOCHS
}

for name, s in results.items():
    summary[f"{name} R2"] = s["r2"]
    summary[f"{name} Adj R2"] = s["adj_r2"]
    summary[f"{name} MSE"] = s["mse"]

summary_df = pd.DataFrame([summary])

print("\n")
print(summary_df.T.to_string(header=False))

summary_df.to_csv(
    os.path.join(args.out_dir, f"TimeSplit_Summary_{DIST_LABEL}.csv"),
    index=False
)

# Test-period predictions, for plots of predicted vs EPA over time
# (reg_lin_model is on the raw scale, the others on the log1p scale)
predictions = test_df[["sensor_id", "timestamp", "distance_km", RAW_TARGET_COLUMN, TARGET_COLUMN]].copy()
predictions["log_trans_lin_pred"] = log_trans_lin["test_pred"]
predictions["reg_lin_pred"] = reg_lin["test_pred"]
predictions["mlp_pred"] = mlp["test_pred"]
predictions["mlp_g_only_pred"] = mlp["g_test_pred"]

predictions.to_csv(
    os.path.join(args.out_dir, f"TimeSplit_Test_Predictions_{DIST_LABEL}.csv"),
    index=False
)

print(f"\nSaved results to {args.out_dir}")
