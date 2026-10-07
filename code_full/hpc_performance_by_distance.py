"""
Single-distance version of final_presentation2.py.

Runs Experiment 1 (train and test on the full dataset) for ONE radius:
every PurpleAir sensor within --distance km of the EPA monitor.
The MLP is trained for a fixed number of epochs (default 150), no epoch search.

Usage:
    python final_presentation_distance.py --distance 3
    python final_presentation_distance.py --distance 3 --data-dir /path/to/geethu_code_full --out-dir results
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
parser.add_argument("--data-dir", default=".",
                    help="Folder with LA_Site_1103.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv")
parser.add_argument("--out-dir", default="results_by_distance",
                    help="Folder for output files")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

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

print(f"Radius: {DISTANCE_KM} km | Epochs: {EPOCHS}")


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

    df["temperature_lag1"] = df["temperature"].shift(1)
    df["humidity_lag1"] = df["humidity"].shift(1)
    df["pressure_lag1"] = df["pressure"].shift(1)

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

merged["log_EPA_PM"] = np.log1p(
    merged["EPA_PM"]
)


# ============================================================
# Remove incomplete rows
# ============================================================

model_columns = [
    "sensor_id",
    "timestamp",
    "log_PA_PM",
    "log_EPA_PM",
    "pm2.5_cf_1",
    "EPA_PM",
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

merged = merged[model_columns]

merged = merged.dropna().reset_index(drop=True)


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

n_obs = len(df)
n_sensors = df["sensor_id"].nunique()

print("\n================ Dataset Summary ================\n")
print(DIST_LABEL)
print(f"Observations : {n_obs}")
print(f"Sensors      : {n_sensors}")
print(f"Mean PM2.5   : {np.expm1(df['log_EPA_PM']).mean():.2f}")
print(f"Mean Distance: {df['distance_km'].mean():.2f} km")
print()


# ============================================================
# Predictor columns
# ============================================================

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

# Raw (no log transform) versions for the plain linear regression
RAW_FEATURE_COLUMNS = ["pm2.5_cf_1"] + FEATURE_COLUMNS[1:]
RAW_TARGET_COLUMN = "EPA_PM"


def get_X_y(df):

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    return X, y


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


def evaluate(model, X, y):

    pred = predict(model, X)

    mse = mean_squared_error(y, pred)

    r2 = r2_score(y, pred)

    return mse, r2, pred


# ============================================================
# log_trans_lin_model: Log-Linear Regression (full data for train and test)
# ============================================================

def run_log_trans_lin_model(train_df):

    X_train = sm.add_constant(train_df[FEATURE_COLUMNS], has_constant="add")
    y_train = train_df[TARGET_COLUMN]

    model = sm.OLS(y_train, X_train).fit()

    pred = model.predict(X_train)

    return {
        "model": model,
        "predictions": pred,
        "mse": mean_squared_error(y_train, pred),
        "r2": r2_score(y_train, pred),
        "adj_r2": model.rsquared_adj
    }


# ============================================================
# Experiment 1: Train and Test on Entire Dataset
# ============================================================

print("=======================================================")
print(f"EXPERIMENT 1 ({DIST_LABEL})")
print("Train and Test on Entire Dataset")
print("=======================================================")

########################################################
# log_trans_lin_model
########################################################

log_trans_lin_results = run_log_trans_lin_model(df)

print("\n==========================================")
print(f"log_trans_lin_model: linear regression on log1p PM2.5 ({DIST_LABEL})")
print("==========================================")
print(f"R²: {log_trans_lin_results['r2']:.4f}")
print(f"Adjusted R²: {log_trans_lin_results['adj_r2']:.4f}")
print(f"MSE: {log_trans_lin_results['mse']:.6f}")

log_trans_lin_summary_text = str(log_trans_lin_results["model"].summary())
print("\nCoefficients:")
print(log_trans_lin_summary_text)

with open(os.path.join(args.out_dir, f"log_trans_lin_model_summary_{DIST_LABEL}.txt"), "w") as f:
    f.write(log_trans_lin_summary_text)

########################################################
# reg_lin_model: linear regression without log transform (raw PM2.5)
########################################################

X_raw = sm.add_constant(df[RAW_FEATURE_COLUMNS], has_constant="add")
y_raw = df[RAW_TARGET_COLUMN]

reg_lin_model = sm.OLS(y_raw, X_raw).fit()

raw_pred = reg_lin_model.predict(X_raw)

reg_lin_results = {
    "mse": mean_squared_error(y_raw, raw_pred),
    "r2": reg_lin_model.rsquared,
    "adj_r2": reg_lin_model.rsquared_adj
}

print("\n==========================================")
print(f"reg_lin_model: linear regression, no log transform ({DIST_LABEL})")
print("==========================================")
print(f"R²: {reg_lin_results['r2']:.4f}")
print(f"Adjusted R²: {reg_lin_results['adj_r2']:.4f}")
print(f"MSE: {reg_lin_results['mse']:.6f}")

reg_lin_summary_text = str(reg_lin_model.summary())
print("\nCoefficients:")
print(reg_lin_summary_text)

with open(os.path.join(args.out_dir, f"reg_lin_model_summary_{DIST_LABEL}.txt"), "w") as f:
    f.write(reg_lin_summary_text)

########################################################
# MLP (fixed epochs)
########################################################

X, y = get_X_y(df)

X_scaled = StandardScaler().fit_transform(X)

model = create_model(X_scaled.shape[1])

train_model(
    model,
    X_scaled,
    y,
    epochs=EPOCHS
)

mse, r2, predictions = evaluate(
    model,
    X_scaled,
    y
)

# Adjusted R² (same formula as the original script, p = number of predictors)
n = len(y)
p = len(FEATURE_COLUMNS)

mlp_adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nMLP Results")
print("------------------------")
print(f"Epochs : {EPOCHS}")
print(f"Training MSE : {mse:.6f}")
print(f"Training R²  : {r2:.6f}")
print(f"Training Adjusted R²: {mlp_adj_r2:.6f}")


# ============================================================
# Save results
# ============================================================

summary_df = pd.DataFrame([{
    "Distance km": DISTANCE_KM,
    "Observations": n_obs,
    "Sensors": n_sensors,
    "Epochs": EPOCHS,
    "log_trans_lin_model R2": log_trans_lin_results["r2"],
    "log_trans_lin_model Adj R2": log_trans_lin_results["adj_r2"],
    "log_trans_lin_model MSE": log_trans_lin_results["mse"],
    "reg_lin_model R2": reg_lin_results["r2"],
    "reg_lin_model Adj R2": reg_lin_results["adj_r2"],
    "reg_lin_model MSE": reg_lin_results["mse"],
    "MLP R2": r2,
    "MLP Adj R2": mlp_adj_r2,
    "MLP MSE": mse
}])

print("\n")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    os.path.join(args.out_dir, f"Experiment1_Summary_{DIST_LABEL}.csv"),
    index=False
)

print(f"\nSaved results to {args.out_dir}")
