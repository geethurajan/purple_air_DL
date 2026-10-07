"""
Save which PurpleAir sensors are in each distance group (closest, 3, 4, 5, ... km, all)
used by final_presentation2.py.

This reproduces the data pipeline of final_presentation2.py up to the
"Build datasets" step (same files, column checks, coordinate merge,
UTC timestamp merge with EPA, dropna on model_columns, distance filters),
but trains no models, so it runs quickly.

Run it from the same folder as final_presentation2.py:
    python save_sensor_groups_more_distances.py

Output: Sensor_Groups_more_distances.csv with one row per (group, sensor):
    group, sensor_id, distance_km, n_rows, sensor_lat, sensor_lon
n_rows is the number of rows that sensor contributes to that group's dataset.
"""

import os
import glob

import numpy as np
import pandas as pd


# ============================================================
# Files (same as final_presentation2.py)
# ============================================================

# read the data files from the folder this script is in,
# whichever folder it is run from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

EPA_FILE = "LA_Site_1103.csv"
COORD_FILE = "Sensors_with_coordinates.csv"
OUTPUT_FILE = "Sensor_Groups_more_distances.csv"

SENSOR_FILES = sorted(glob.glob("sensor_*_history.csv"))

print(f"Found {len(SENSOR_FILES)} PurpleAir files.")


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

epa = pd.read_csv(EPA_FILE)

epa["datetime"] = pd.to_datetime(
    epa["Date Local"] + " " + epa["Time Local"]
)

epa = epa.rename(columns={"Sample Measurement": "EPA_PM"})

epa = epa[["datetime", "Latitude", "Longitude", "EPA_PM"]]

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


# ============================================================
# Read every PurpleAir sensor
# ============================================================

required_columns = [
    "time_stamp",
    "pm2.5_cf_1",
    "temperature",
    "humidity",
    "pressure"
]

all_sensor_data = []
skipped = []

for file in SENSOR_FILES:

    sensor_id = os.path.basename(file)
    sensor_id = sensor_id.replace("sensor_", "")
    sensor_id = sensor_id.replace("_history.csv", "")

    try:
        df = pd.read_csv(file)
    except Exception:
        print("Could not read:", file)
        skipped.append(file)
        continue

    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns in {file}")
        skipped.append(file)
        continue

    df = df[required_columns].copy()

    df["sensor_id"] = sensor_id

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True)

    df = df.sort_values("time_stamp")

    # lag columns matter here because the dropna below removes their NaNs
    df["temperature_lag1"] = df["temperature"].shift(1)
    df["humidity_lag1"] = df["humidity"].shift(1)
    df["pressure_lag1"] = df["pressure"].shift(1)

    all_sensor_data.append(df)

purpleair = pd.concat(all_sensor_data, ignore_index=True)

print("Sensors loaded:", purpleair["sensor_id"].nunique())


# ============================================================
# Attach coordinates
# ============================================================

purpleair["sensor_id"] = purpleair["sensor_id"].astype(str)

purpleair = purpleair.merge(coords, on="sensor_id", how="left")

no_coords = sorted(
    purpleair.loc[purpleair["sensor_lat"].isna(), "sensor_id"].unique()
)

purpleair = purpleair.dropna(subset=["sensor_lat", "sensor_lon"])

purpleair = purpleair.rename(columns={"time_stamp": "timestamp"})

print("Sensors with coordinates:", purpleair["sensor_id"].nunique())


# ============================================================
# Prepare EPA timestamps
# ============================================================

epa["datetime"] = pd.to_datetime(epa["datetime"])

if epa["datetime"].dt.tz is None:
    epa["timestamp"] = (
        epa["datetime"]
        .dt.tz_localize("America/Los_Angeles", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
else:
    epa["timestamp"] = epa["datetime"].dt.tz_convert("UTC")

epa = epa.drop(columns=["datetime"])


# ============================================================
# Merge EPA with PurpleAir, distance, log transforms, dropna
# ============================================================

merged = purpleair.merge(epa, on="timestamp", how="inner")

merged["distance_km"] = haversine(
    merged["Latitude"],
    merged["Longitude"],
    merged["sensor_lat"],
    merged["sensor_lon"]
)

# log1p turns pm values below -1 into NaN, which the dropna below removes,
# so it is kept to match final_presentation2.py exactly
merged["log_PA_PM"] = np.log1p(merged["pm2.5_cf_1"])
merged["log_EPA_PM"] = np.log1p(merged["EPA_PM"])

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

print("Final modeling observations:", len(merged))


# ============================================================
# Build datasets (same filters as final_presentation2.py)
# ============================================================

sensor_distances = (
    merged.groupby("sensor_id")["distance_km"]
    .first()
    .sort_values()
)

closest_sensor = sensor_distances.index[0]

# distance groups: closest sensor, then every sensor within each distance
# below, then all sensors (same list as distance_experiments.py).
# Add or remove distances here.
DISTANCES_KM = [3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

datasets = {"closest": merged[merged["sensor_id"] == closest_sensor]}

for d in DISTANCES_KM:
    datasets[f"{d}km"] = merged[merged["distance_km"] <= d]

datasets["all"] = merged


# ============================================================
# Save sensor list per group
# ============================================================

rows = []

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
        .sort_values(["distance_km", "sensor_id"])
    )

    per_sensor.insert(0, "group", name)

    rows.append(per_sensor)

groups = pd.concat(rows, ignore_index=True)

groups["distance_km"] = groups["distance_km"].round(3)

groups.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# Summary
# ============================================================

print("\n================ Sensor Groups ================\n")

for name, df in datasets.items():
    print(
        f"{name:8s} sensors: {df['sensor_id'].nunique():4d}   "
        f"rows: {len(df)}"
    )

print(f"\nClosest sensor: {closest_sensor} "
      f"({sensor_distances.iloc[0]:.3f} km)")

loaded = set(purpleair["sensor_id"]) | set(no_coords)
dropped_in_merge = sorted(loaded - set(no_coords) - set(merged["sensor_id"]))

print(f"\nSensor files skipped (unreadable or missing columns): {len(skipped)}")
print(f"Sensors not in {COORD_FILE}: {len(no_coords)}")
print(f"Sensors with no complete rows after EPA merge: {len(dropped_in_merge)}")

print(f"\nSaved {OUTPUT_FILE}")
