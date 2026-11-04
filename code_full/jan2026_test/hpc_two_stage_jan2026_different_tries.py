"""
Different tries for ONE radius (default 2.5 km): which MLP version to use, chosen on 2025 data
only, then tested once on January 2026.

The final split is the same as in hpc_two_stage_jan2026_test.py: train on January-February 2025,
test on January 2026, same sensors and the same EPA monitor (site 06-037-1103), with the same
quality control and the same two-stage training (cross-fitting).

Step 1, choose the MLP version (January-February 2025 only; January 2026 is not used):
    the Jan-Feb 2025 days alternate between fit days (2025-01-01, 01-03, ...) and validation
    days (01-02, 01-04, ...). Every version below is trained on the fit days and scored on the
    validation days (adjusted R², log scale), once per seed in --seeds. The version with the
    highest mean validation adjusted R² is chosen.
        f:             Model 1 (8 inputs) or Model 2 (f_lag: also temp, hum, press at t-1, 11 inputs)
        hidden layers: 8-8, 16-16, 16-32-16, 32-32, 64-32 (the same for g and f)
        weight decay:  0 or 0.001 (L2 penalty on the weights, a standard fix for small data)
    Validation hours whose t-1 hour falls on a fit day are not scored (g has seen that hour).

Step 2, test the chosen version once: train it on all Jan-Feb 2025 days and score it on
January 2026, once per seed. The two linear models are scored on the same rows.

Only the chosen version is scored on January 2026, so the January 2026 score stays a fair test.

Models (t = current hour, t-1 = the hour before at the same sensor):

    log_trans_lin_model: log EPA PM2.5(t) ~ log PA PM2.5(t), temp(t), hum(t), press(t),
                                            temp(t-1), hum(t-1), press(t-1), distance, lat, lon
    reg_lin_model:       the same without the log transform
    MLP:  g_hat(t-1)       = g( log PA PM2.5(t-1), temp(t-1), hum(t-1), press(t-1), distance, lat, lon )
          Model 1: log EPA PM2.5(t) = f( log PA PM2.5(t), temp(t), hum(t), press(t), distance, lat, lon,
                                         g_hat(t-1) )
          Model 2: log EPA PM2.5(t) = f( the Model 1 inputs + temp(t-1), hum(t-1), press(t-1) )

Folders (run from inside jan2026_test, which sits inside the training data folder):
    --train-dir ..  LA_Site_1103.csv, Sensors_with_coordinates.csv, sensor_*_history.csv (2025)
    --test-dir  .   LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv, sensor_*_history.csv (2026)

Usage:
    python hpc_two_stage_jan2026_different_tries.py
    python hpc_two_stage_jan2026_different_tries.py --distance 2.5 --train-dir .. --test-dir . --out-dir results
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

parser.add_argument("--distance", type=float, default=2.5,
                    help="Radius in km around the EPA monitor")
parser.add_argument("--epochs", type=int, default=150,
                    help="MLP training epochs (fixed, no search)")
parser.add_argument("--seeds", default="42,1,2,3,4",
                    help="Random seeds; every version is trained once per seed")
parser.add_argument("--crossfit-folds", type=int, default=2,
                    help="Day groups for cross-fitting g_hat(t-1) on f's training rows")
parser.add_argument("--train-dir", default="..",
                    help="Folder with LA_Site_1103.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (Jan-Feb 2025 training data)")
parser.add_argument("--test-dir", default=".",
                    help="Folder with LA_Site_1103_Jan2026.csv, Sensors_with_coordinates.csv "
                         "and sensor_*_history.csv (January 2026 test data)")
parser.add_argument("--out-dir", default="results_different_tries",
                    help="Folder for output files")

args = parser.parse_args()

DISTANCE_KM = args.distance
EPOCHS = args.epochs
SEEDS = [int(s) for s in args.seeds.split(",")]
CROSSFIT_FOLDS = args.crossfit_folds

# Short label for file names, e.g. 3.0 -> "3km", 2.5 -> "2.5km"
DIST_LABEL = f"{DISTANCE_KM:g}km"

os.makedirs(args.out_dir, exist_ok=True)

# The versions tried in step 1
F_VERSIONS = ["Model 1", "Model 2"]
HIDDEN_LAYERS = [[8, 8], [16, 16], [16, 32, 16], [32, 32], [64, 32]]
WEIGHT_DECAYS = [0.0, 0.001]

# Plausible ranges of the PurpleAir readings; a row with any value outside them is dropped
VALID_RANGES = {
    "pm2.5_cf_1": (0, 1000),     # ug/m3
    "temperature": (-40, 150),   # deg F
    "humidity": (0, 100),        # %
    "pressure": (900, 1100),     # hPa
}

# A sensor with a larger share of its hours outside VALID_RANGES is not used for that period
MAX_INVALID_SHARE = 0.1

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

print(f"Radius: {DISTANCE_KM} km | Epochs: {EPOCHS} | Seeds: {SEEDS} "
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

# Model 2 (f_lag) also gives f the PurpleAir weather readings one hour earlier
F_LAG_COLUMNS = [
    "temperature_lag1",
    "humidity_lag1",
    "pressure_lag1"
]


def f_columns(version):
    # f's inputs besides g_hat(t-1)
    return G_COLUMNS + (F_LAG_COLUMNS if version == "Model 2" else [])


MODEL_COLUMNS = list(dict.fromkeys(
    ["sensor_id", "timestamp", "distance_km", TARGET_COLUMN, RAW_TARGET_COLUMN]
    + RAW_FEATURE_COLUMNS
    + FEATURE_COLUMNS
    + G_LAG_COLUMNS
))


# ============================================================
# Readers (the same as hpc_two_stage_jan2026_test.py)
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
# Read EPA data and choose the sensors within the radius
# ============================================================

print("Loading EPA data...")

epa_train = read_epa(TRAIN_EPA_FILE)
epa_test = read_epa(TEST_EPA_FILE)

SITE = monitor_location(epa_train, TRAIN_EPA_FILE)
TEST_SITE = monitor_location(epa_test, TEST_EPA_FILE)

if haversine(SITE[0], SITE[1], TEST_SITE[0], TEST_SITE[1]) > 0.1:
    raise SystemExit("The 2025 and 2026 EPA files are not the same monitor.")


def sensors_in_radius(coords_file, folder):

    coords = read_coords(coords_file)

    coords["km_to_site"] = haversine(
        SITE[0], SITE[1], coords["sensor_lat"].values, coords["sensor_lon"].values
    )

    coords["file"] = [sensor_file(s, folder) for s in coords["sensor_id"]]

    return coords[
        (coords["km_to_site"] <= DISTANCE_KM) & coords["file"].notna()
    ].reset_index(drop=True)


train_sensors = sensors_in_radius(TRAIN_COORD_FILE, args.train_dir)
test_sensors = sensors_in_radius(TEST_COORD_FILE, args.test_dir)

if train_sensors.empty or test_sensors.empty:
    raise SystemExit(f"No training or test sensor files within {DISTANCE_KM} km.")


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

    purpleair = pd.concat(all_sensor_data, ignore_index=True)

    purpleair = purpleair.merge(
        sensors[["sensor_id", "sensor_lat", "sensor_lon"]],
        on="sensor_id",
        how="left"
    )

    purpleair = purpleair.rename(columns={"time_stamp": "timestamp"})

    merged = purpleair.merge(epa, on="timestamp", how="inner")

    merged["distance_km"] = haversine(
        merged["Latitude"],
        merged["Longitude"],
        merged["sensor_lat"],
        merged["sensor_lon"]
    )

    merged["log_PA_PM"] = np.log1p(merged["pm2.5_cf_1"])
    merged["log_PA_PM_lag1"] = np.log1p(merged["pm2.5_cf_1_lag1"])
    merged["log_EPA_PM"] = np.log1p(merged["EPA_PM"])

    # A log1p of -1 would be -inf; treat it as missing
    merged = merged[MODEL_COLUMNS].replace([np.inf, -np.inf], np.nan)

    merged = merged.dropna(
        subset=[TARGET_COLUMN, RAW_TARGET_COLUMN, "pm2.5_cf_1"] + G_COLUMNS
    )

    return merged[merged["distance_km"] <= DISTANCE_KM].reset_index(drop=True)


train_all = load_period(train_sensors, epa_train, "Training (2025)")
test_all = load_period(test_sensors, epa_test, "Test (2026)")


def with_lag(d):
    # Rows that have the t-1 readings (needed by the linear models and by f)
    return d[d[G_LAG_COLUMNS].notna().all(axis=1)].reset_index(drop=True)


def day_number(timestamps):
    # Calendar day in Los Angeles time, counted from 2025-01-01
    return (
        timestamps
        .dt.tz_convert("America/Los_Angeles")
        .dt.tz_localize(None)
        .dt.normalize()
        - pd.Timestamp("2025-01-01")
    ).dt.days


# Final split: all of Jan-Feb 2025 for training, January 2026 for the test
train_df = with_lag(train_all)
test_df = with_lag(test_all)

# Step 1 split, inside Jan-Feb 2025: fit days and validation days alternate
is_fit_day = day_number(train_all["timestamp"]) % 2 == 0

g_fit_df = train_all[is_fit_day].reset_index(drop=True)
fit_df = with_lag(train_all[is_fit_day])
val_df = with_lag(train_all[~is_fit_day])

# Score only validation hours whose t-1 hour is also on the validation day (g never saw it)
val_df = val_df[
    day_number(val_df["timestamp"] - pd.Timedelta(hours=1)) == day_number(val_df["timestamp"])
].reset_index(drop=True)

print("\n================ Dataset Summary ================\n")
print(DIST_LABEL)
print(f"Train (Jan-Feb 2025)      : {len(train_df)} obs | {train_df['sensor_id'].nunique()} sensors")
print(f"Test  (Jan 2026)          : {len(test_df)} obs | {test_df['sensor_id'].nunique()} sensors")
print(f"Step 1 fit / validation   : {len(fit_df)} / {len(val_df)} obs (alternate days of Jan-Feb 2025)")
print()


# ============================================================
# MLP
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(input_dim, hidden):

    # One Linear + ReLU per hidden layer, then a Linear output layer
    layers = []
    size_in = input_dim

    for size_out in hidden:
        layers += [nn.Linear(size_in, size_out), nn.ReLU()]
        size_in = size_out

    layers.append(nn.Linear(size_in, 1))

    return nn.Sequential(*layers).to(DEVICE)


def train_model(model, X_train, y_train, epochs, weight_decay, batch_size=64, lr=0.001):

    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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


def fit_mlp(X, y, hidden, weight_decay):

    scaler = StandardScaler().fit(X)

    model = create_model(X.shape[1], hidden)

    train_model(model, scaler.transform(X).astype(np.float32), y, EPOCHS, weight_decay)

    return scaler, model


def mlp_predict(scaler, model, X):
    return predict(model, scaler.transform(X).astype(np.float32))


def values(d, columns):
    return d[columns].values.astype(np.float32)


def target(d):
    return d[TARGET_COLUMN].values.astype(np.float32)


def run_mlp_two_stage(g_train_df, train_df, eval_df, version, hidden, weight_decay):

    # g: hour readings -> log EPA PM2.5 at the same hour (trained on every training hour)
    g_scaler, g_model = fit_mlp(values(g_train_df, G_COLUMNS), target(g_train_df), hidden, weight_decay)

    # Cross-fitting: g_hat(t-1) on f's training rows comes from a g that did not see the day of
    # hour t-1. The days g trains on take turns between the groups; a day g does not train on at
    # all can use any group's g (group 0).
    g_days = sorted(set(day_number(g_train_df["timestamp"])))
    group = {day: i % CROSSFIT_FOLDS for i, day in enumerate(g_days)}

    g_group = day_number(g_train_df["timestamp"]).map(group).values
    f_group = day_number(train_df["timestamp"] - pd.Timedelta(hours=1)).map(group).fillna(0).astype(int).values

    g_hat_train = np.zeros(len(train_df), dtype=np.float32)

    for k in range(CROSSFIT_FOLDS):
        g_k = g_train_df[g_group != k]

        scaler_k, model_k = fit_mlp(values(g_k, G_COLUMNS), target(g_k), hidden, weight_decay)

        rows = f_group == k

        if rows.any():
            g_hat_train[rows] = mlp_predict(scaler_k, model_k, values(train_df[rows], G_LAG_COLUMNS))

    # f: f's inputs + g_hat(t-1) -> log EPA PM2.5 at hour t
    columns = f_columns(version)

    X_f_train = np.column_stack([values(train_df, columns), g_hat_train]).astype(np.float32)

    f_scaler, f_model = fit_mlp(X_f_train, target(train_df), hidden, weight_decay)

    g_hat_eval = mlp_predict(g_scaler, g_model, values(eval_df, G_LAG_COLUMNS))
    X_f_eval = np.column_stack([values(eval_df, columns), g_hat_eval]).astype(np.float32)

    return mlp_predict(f_scaler, f_model, X_f_eval), len(columns) + 1


# ============================================================
# Scores and linear models
# ============================================================

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


def run_ols(train_df, eval_df, columns, target_column):

    model = sm.OLS(
        train_df[target_column],
        sm.add_constant(train_df[columns], has_constant="add")
    ).fit()

    return np.asarray(model.predict(sm.add_constant(eval_df[columns], has_constant="add")))


def linear_scores(train_df, eval_df):

    return {
        "log_trans_lin_model": scores(
            eval_df[TARGET_COLUMN],
            run_ols(train_df, eval_df, FEATURE_COLUMNS, TARGET_COLUMN),
            len(FEATURE_COLUMNS)
        )["adj_r2"],
        "reg_lin_model": scores(
            eval_df[RAW_TARGET_COLUMN],
            run_ols(train_df, eval_df, RAW_FEATURE_COLUMNS, RAW_TARGET_COLUMN),
            len(RAW_FEATURE_COLUMNS)
        )["adj_r2"],
    }


# ============================================================
# Step 1: choose the MLP version on the 2025 validation days
# ============================================================

print("=======================================================")
print(f"STEP 1 ({DIST_LABEL}): choose the MLP version on Jan-Feb 2025 validation days")
print("=======================================================")

linear_val = linear_scores(fit_df, val_df)

print(f"For reference, on the same validation days: log_trans_lin_model {linear_val['log_trans_lin_model']:.4f}"
      f" | reg_lin_model {linear_val['reg_lin_model']:.4f}\n")

tries = []

for version in F_VERSIONS:
    for hidden in HIDDEN_LAYERS:
        for weight_decay in WEIGHT_DECAYS:

            arch = "-".join(str(n) for n in hidden)
            per_seed = []

            for seed in SEEDS:
                set_seed(seed)
                pred, p = run_mlp_two_stage(g_fit_df, fit_df, val_df, version, hidden, weight_decay)
                per_seed.append(scores(val_df[TARGET_COLUMN], pred, p)["adj_r2"])

            tries.append({
                "Distance km": DISTANCE_KM,
                "f": version,
                "Hidden layers": arch,
                "Weight decay": weight_decay,
                "Val Adj R2 mean": np.mean(per_seed),
                "Val Adj R2 sd": np.std(per_seed),
                "Val Adj R2 min": np.min(per_seed),
                "Val Adj R2 max": np.max(per_seed),
                "log_trans_lin_model Val Adj R2": linear_val["log_trans_lin_model"],
                "reg_lin_model Val Adj R2": linear_val["reg_lin_model"],
            })

            print(f"{version} | {arch:>8} | weight decay {weight_decay:<5g} | "
                  f"validation adj R² {np.mean(per_seed):.4f} ± {np.std(per_seed):.4f}")

tries = pd.DataFrame(tries).sort_values("Val Adj R2 mean", ascending=False).reset_index(drop=True)

tries.to_csv(
    os.path.join(args.out_dir, f"DifferentTries_Validation_{DIST_LABEL}.csv"),
    index=False
)

chosen = tries.iloc[0]

print("\nAll versions, best validation score first:")
print(tries.round(4).to_string(index=False))

print(f"\nChosen: {chosen['f']} | hidden layers {chosen['Hidden layers']} | weight decay {chosen['Weight decay']:g}")


# ============================================================
# Step 2: test the chosen version once on January 2026
# ============================================================

print("\n=======================================================")
print(f"STEP 2 ({DIST_LABEL}): train the chosen version on all of Jan-Feb 2025, test on Jan 2026")
print("=======================================================")

hidden = [int(n) for n in chosen["Hidden layers"].split("-")]

linear_test = linear_scores(train_df, test_df)

test_rows = []

for seed in SEEDS:
    set_seed(seed)
    pred, p = run_mlp_two_stage(train_all, train_df, test_df, chosen["f"], hidden, chosen["Weight decay"])
    s = scores(test_df[TARGET_COLUMN], pred, p)

    test_rows.append({
        "Distance km": DISTANCE_KM,
        "Seed": seed,
        "f": chosen["f"],
        "Hidden layers": chosen["Hidden layers"],
        "Weight decay": chosen["Weight decay"],
        "MLP Test Adj R2": s["adj_r2"],
        "MLP Test MSE": s["mse"],
        "log_trans_lin_model Test Adj R2": linear_test["log_trans_lin_model"],
        "reg_lin_model Test Adj R2": linear_test["reg_lin_model"],
    })

    print(f"seed {seed:>3}: MLP test adj R² {s['adj_r2']:.4f}")

test_rows = pd.DataFrame(test_rows)

test_rows.to_csv(
    os.path.join(args.out_dir, f"DifferentTries_Test_{DIST_LABEL}.csv"),
    index=False
)

print(f"\nMLP (chosen version), mean over {len(SEEDS)} seeds: "
      f"{test_rows['MLP Test Adj R2'].mean():.4f} ± {test_rows['MLP Test Adj R2'].std(ddof=0):.4f}")
print(f"log_trans_lin_model: {linear_test['log_trans_lin_model']:.4f} | "
      f"reg_lin_model: {linear_test['reg_lin_model']:.4f}")

print(f"\nSaved results to {args.out_dir}")
