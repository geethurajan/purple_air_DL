import os
import glob
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import statsmodels.api as sm


# ============================================================
# Random seed
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
# Files
# ============================================================

EPA_FILE = "LA_Site_1103.csv"
COORD_FILE = "Sensors_with_coordinates.csv"

SENSOR_FILES = sorted(glob.glob("sensor_*_history.csv"))

# all results go in this folder, so nothing from final_presentation2.py
# is overwritten
OUTPUT_DIR = "results_more_distances"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# Read every PurpleAir sensor
# ============================================================

all_sensor_data = []

print("\nLoading PurpleAir sensors...")

for file in tqdm(SENSOR_FILES):

    sensor_id = os.path.basename(file)
    sensor_id = sensor_id.replace("sensor_", "")
    sensor_id = sensor_id.replace("_history.csv", "")


    try:
        df = pd.read_csv(file)
    except:
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

print(
    "Distance range:",
    round(merged["distance_km"].min(), 3),
    "-",
    round(merged["distance_km"].max(), 3),
    "km"
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

print("\nFinal modeling observations:", len(merged))

# ============================================================
# Build datasets
# ============================================================

print("\nBuilding datasets...")

sensor_distances = (
    merged.groupby("sensor_id")["distance_km"]
    .first()
    .sort_values()
)

closest_sensor = sensor_distances.index[0]
closest_distance = sensor_distances.iloc[0]

print(f"\nClosest sensor: {closest_sensor}")
print(f"Distance: {closest_distance:.3f} km")

# ------------------------------------------------------------
# Distance groups: closest sensor, then every sensor within each
# distance below, then all sensors. Add or remove distances here.
# More distances = longer run (each one goes through both experiments).
# ------------------------------------------------------------

DISTANCES_KM = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

datasets = {}

datasets["closest"] = (
    merged[
        merged["sensor_id"] == closest_sensor
    ]
    .copy()
    .reset_index(drop=True)
)

for d in DISTANCES_KM:
    datasets[f"{d}km"] = (
        merged[
            merged["distance_km"] <= d
        ]
        .copy()
        .reset_index(drop=True)
    )

datasets["all"] = merged.copy().reset_index(drop=True)

# distance used on the x-axis of the adjusted R² plot
dataset_distance = {"closest": closest_distance, "all": sensor_distances.iloc[-1]}
for d in DISTANCES_KM:
    dataset_distance[f"{d}km"] = d


# ============================================================
# Save the sensors in each distance group
# ============================================================

sensor_rows = []

for name, df in datasets.items():

    per_sensor = (
        df.groupby("sensor_id")
        .agg(
            distance_km=("distance_km", "first"),
            n_rows=("distance_km", "size"),
            sensor_lat=("sensor_lat", "first"),
            sensor_lon=("sensor_lon", "first"),
        )
        .reset_index()
        .sort_values("distance_km")
    )

    per_sensor.insert(0, "group", name)

    sensor_rows.append(per_sensor)

pd.concat(sensor_rows, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "Sensor_Groups_by_Distance.csv"),
    index=False
)

# every sensor used, once, with its distance from the EPA monitor
(
    sensor_distances
    .round(3)
    .rename("distance_km")
    .reset_index()
    .to_csv(os.path.join(OUTPUT_DIR, "Sensor_Distances.csv"), index=False)
)


# ============================================================
# Dataset summaries
# ============================================================

print("\n================ Dataset Summary ================\n")

for name, df in datasets.items():

    n_obs = len(df)
    n_sensors = df["sensor_id"].nunique()

    print(f"{name}")
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


# ============================================================
# Helper function
# ============================================================

def get_X_y(df):

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    return X, y


print("Datasets successfully created.")

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
    y_tensor = torch.FloatTensor(y_train.reshape(-1,1)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(X_tensor)

    model.train()

    for epoch in range(epochs):

        perm = torch.randperm(n)

        for i in range(0, n, batch_size):

            idx = perm[i:i+batch_size]

            X_batch = X_tensor[idx]
            y_batch = y_tensor[idx]

            optimizer.zero_grad()

            pred = model(X_batch)

            loss = criterion(pred, y_batch)

            loss.backward()

            optimizer.step()

    return model


# ============================================================
# Predict
# ============================================================

def predict(model, X):

    model.eval()

    X_tensor = torch.FloatTensor(X).to(DEVICE)

    with torch.no_grad():
        pred = model(X_tensor)

    return pred.cpu().numpy().flatten()


# ============================================================
# Evaluation metrics
# ============================================================

def evaluate(model, X, y):

    pred = predict(model, X)

    mse = mean_squared_error(y, pred)

    r2 = r2_score(y, pred)

    return mse, r2, pred


# ============================================================
# Standardize data
# ============================================================

def standardize_data(X_train, X_test=None):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is None:
        return X_train_scaled, scaler

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

# ============================================================
# Find Best Epoch
# ============================================================

def find_best_epoch(df, dataset_name):

    print(f"\nFinding best epoch for {dataset_name} dataset...")

    X, y = get_X_y(df)

    epochs_to_test = list(range(5, 205, 5))

    average_mse = []

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for epoch in tqdm(epochs_to_test):

        fold_mse = []

        for train_idx, test_idx in kf.split(X):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            X_train, X_test, scaler = standardize_data(
                X_train,
                X_test
            )

            model = create_model(X_train.shape[1])

            train_model(
                model,
                X_train,
                y_train,
                epochs=epoch
            )

            mse, r2, pred = evaluate(
                model,
                X_test,
                y_test
            )

            fold_mse.append(mse)

        average_mse.append(np.mean(fold_mse))

    best_epoch = epochs_to_test[np.argmin(average_mse)]

    print(f"Best epoch = {best_epoch}")

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs_to_test,
        average_mse,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Average Test MSE")
    plt.title(f"{dataset_name}: Test MSE vs Epoch")

    plt.grid(True)

    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{dataset_name}_epoch_search.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return best_epoch

# ============================================================
# Log-Linear Regression
# ============================================================

def run_linear_model(train_df, test_df=None):

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_train = sm.add_constant(X_train)

    model = sm.OLS(y_train, X_train).fit()

    if test_df is None:

        pred = model.predict(X_train)

        mse = mean_squared_error(y_train, pred)
        r2 = r2_score(y_train, pred)

        return {
            "model": model,
            "predictions": pred,
            "mse": mse,
            "r2": r2,
            "adj_r2": model.rsquared_adj
        }

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    X_test = sm.add_constant(X_test)

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    n = len(y_test)
    p = len(FEATURE_COLUMNS)

    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

    return {
        "model": model,
        "predictions": pred,
        "mse": mse,
        "r2": r2,
        "adj_r2": adj_r2
    }


# ============================================================
# Print Linear Model Results
# ============================================================

def print_linear_results(results, dataset_name):

    print("\n==========================================")
    print(f"Linear Model ({dataset_name})")
    print("==========================================")

    print(f"R²: {results['r2']:.4f}")
    print(f"Adjusted R²: {results['adj_r2']:.4f}")
    print(f"MSE: {results['mse']:.6f}")

    print("\nCoefficients:")

    print(results["model"].summary())
    
# ============================================================
# Experiment 1
# ============================================================

print("\n")
print("=======================================================")
print("EXPERIMENT 1")
print("Train and Test on Entire Dataset")
print("=======================================================")

experiment1_results = {}

for dataset_name, df in datasets.items():

    print(f"\nProcessing {dataset_name} dataset...")

    ########################################################
    # Find best epoch
    ########################################################

    best_epoch = find_best_epoch(df, dataset_name)

    ########################################################
    # Linear Model
    ########################################################

    linear_results = run_linear_model(df)

    print_linear_results(
        linear_results,
        dataset_name
    )

    ########################################################
    # MLP
    ########################################################

    X, y = get_X_y(df)

    X_scaled, scaler = standardize_data(X)

    model = create_model(X_scaled.shape[1])

    train_model(
        model,
        X_scaled,
        y,
        epochs=best_epoch
    )

    mse, r2, predictions = evaluate(
        model,
        X_scaled,
        y
    )

    print("\nMLP Results")
    print("------------------------")
    print(f"Epochs : {best_epoch}")
    print(f"Training MSE : {mse:.6f}")
    print(f"Training R²  : {r2:.6f}")

    experiment1_results[dataset_name] = {
        "best_epoch": best_epoch,
        "linear_r2": linear_results["r2"],
        "linear_adj_r2": linear_results["adj_r2"],
        "linear_mse": linear_results["mse"],
        "mlp_r2": r2,
        "mlp_mse": mse
    }

print("\n")
print("=======================================================")
print("Experiment 1 Summary")
print("=======================================================")

summary_rows = []

for dataset_name in experiment1_results:

    result = experiment1_results[dataset_name]

    summary_rows.append({
        "Dataset": dataset_name,
        "Best Epoch": result["best_epoch"],
        "Linear R2": result["linear_r2"],
        "Linear Adj R2": result["linear_adj_r2"],
        "Linear MSE": result["linear_mse"],
        "MLP R2": result["mlp_r2"],
        "MLP MSE": result["mlp_mse"]
    })

summary_df = pd.DataFrame(summary_rows)

print(summary_df)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "Experiment1_Summary.csv"),
    index=False
)

# ============================================================
# Experiment 2 (Full Epoch Search Every Replication)
# ============================================================

print("\n")
print("=======================================================")
print("EXPERIMENT 2")
print("50 Random Train/Test Splits")
print("Epoch Search Repeated Every Replication")
print("=======================================================")

experiment2_results = {}

N_REPS = 50

for dataset_name, df in datasets.items():

    print(f"\nRunning {dataset_name} dataset...")

    linear_r2 = []
    linear_adj_r2 = []
    linear_mse = []

    mlp_r2 = []
    mlp_mse = []
    chosen_epochs = []

    for rep in tqdm(range(N_REPS)):

        ####################################################
        # Train/Test Split
        ####################################################

        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=SEED + rep,
            shuffle=True
        )


        ####################################################
        # Epoch Search ONLY on Training Data
        ####################################################

        best_epoch = find_best_epoch(
            train_df,
            f"{dataset_name}_rep_{rep}"
        )

        chosen_epochs.append(best_epoch)


        ####################################################
        # Linear Model
        ####################################################

        linear_results = run_linear_model(
            train_df,
            test_df
        )

        linear_r2.append(
            linear_results["r2"]
        )

        linear_adj_r2.append(
            linear_results["adj_r2"]
        )

        linear_mse.append(
            linear_results["mse"]
        )


        ####################################################
        # MLP
        ####################################################

        X_train, y_train = get_X_y(train_df)

        X_test, y_test = get_X_y(test_df)


        X_train, X_test, scaler = standardize_data(
            X_train,
            X_test
        )


        model = create_model(
            X_train.shape[1]
        )


        train_model(
            model,
            X_train,
            y_train,
            epochs=best_epoch
        )


        mse, r2, predictions = evaluate(
            model,
            X_test,
            y_test
        )


        mlp_r2.append(r2)

        mlp_mse.append(mse)


    ########################################################
    # Store Dataset Results
    ########################################################

    experiment2_results[dataset_name] = {

        "linear_mean_r2":
            np.mean(linear_r2),

        "linear_sd_r2":
            np.std(linear_r2),

        "linear_mean_adj_r2":
            np.mean(linear_adj_r2),

        "linear_sd_adj_r2":
            np.std(linear_adj_r2),

        "linear_mean_mse":
            np.mean(linear_mse),

        "linear_sd_mse":
            np.std(linear_mse),


        "mlp_mean_r2":
            np.mean(mlp_r2),

        "mlp_sd_r2":
            np.std(mlp_r2),

        "mlp_mean_mse":
            np.mean(mlp_mse),

        "mlp_sd_mse":
            np.std(mlp_mse),

        "average_best_epoch":
            np.mean(chosen_epochs)
    }


print("\n")
print("=======================================================")
print("Experiment 2 Summary")
print("=======================================================")


experiment2_summary = pd.DataFrame(
    experiment2_results
).T


print(experiment2_summary)


experiment2_summary.to_csv(
    os.path.join(OUTPUT_DIR, "Experiment2_Summary_Full_Epoch_Search.csv")
)


# ============================================================
# Plot OLS adjusted R² and MLP R² vs distance
# ============================================================

def plot_r2_vs_distance(names, ols_adj_r2, mlp_r2, title, filename):

    x = [dataset_distance[n] for n in names]

    plt.figure(figsize=(9, 5))

    plt.plot(x, ols_adj_r2, marker="o", linewidth=2, color="#2a6fdb",
             label="OLS (log-linear) adjusted R²")

    plt.plot(x, mlp_r2, marker="s", linewidth=2, color="#e07b00",
             label="MLP R²")

    # dataset names above the higher of the two points
    for n, xi, a, b in zip(names, x, ols_adj_r2, mlp_r2):
        plt.annotate(
            n,
            (xi, max(a, b)),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            color="#444444"
        )

    plt.xlabel("Distance from EPA monitor (km)")
    plt.ylabel("R²")
    plt.title(title)

    plt.legend()
    plt.grid(True, color="#dddddd")

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


names = list(datasets.keys())

plot_r2_vs_distance(
    names,
    [experiment1_results[n]["linear_adj_r2"] for n in names],
    [experiment1_results[n]["mlp_r2"] for n in names],
    "Experiment 1 (train and test on entire dataset): R² vs distance",
    "Experiment1_R2_vs_Distance.png"
)

plot_r2_vs_distance(
    names,
    [experiment2_results[n]["linear_mean_adj_r2"] for n in names],
    [experiment2_results[n]["mlp_mean_r2"] for n in names],
    "Experiment 2 (mean over 50 train/test splits): R² vs distance",
    "Experiment2_R2_vs_Distance.png"
)

print(f"\nAll results saved in {os.path.abspath(OUTPUT_DIR)}")
