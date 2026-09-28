"""
End-to-end pipeline for the Los Angeles application of Algorithms 1-3,
using real data:

    LA_Site_1103.csv            -- EPA PM2.5 monitor (hourly, local time)
    la_city_tracts_final.csv    -- census tract centroids, population, x_km/y_km
    City_Boundary.geojson       -- LA city boundary (optional filtering)
    sensor_{code}_history.csv   -- ~260 PurpleAir sensors (one file each)

======================= THINGS YOU NEED TO SET/VERIFY =======================

1. PURPLEAIR_SENSORS_DIR below -- point this at the folder containing your
   ~260 sensor_{code}_history.csv files.

   CONFIRMED SCHEMA (from sensor_82071_history.csv): columns are
   `time_stamp, pm2.5_cf_1, humidity, temperature, pressure`, with timestamps
   as ISO strings carrying an embedded UTC offset (e.g.
   "2025-01-01 00:00:00-08:00") and temperature in Fahrenheit. The loader in
   real_data_loaders.py already handles this schema correctly out of the box.
   Note: the source paper used the "PM2.5 ATM" field specifically; these
   files provide "CF=1" instead (both are standard PurpleAir outputs from
   the same raw counts, using different internal correction curves) -- this
   is a flagged, minor deviation, not a bug. If any of your other sensor
   files use a different schema, real_data_loaders.py's column-detection
   lists will fall back gracefully, but double check with the CHECK block
   below if something looks off.

2. Sensor lat/lon -- confirmed ABSENT from the per-sensor history file (as
   expected -- it's static metadata, not a time series field). Set
   SENSOR_METADATA_PATH to a CSV with columns [sensor_id, lat, lon] (usually
   exportable from PurpleAir's sensor list/map). sensor_id values must match
   the {code} in each filename.

3. EVENT_START / EVENT_END / BASELINE_START / BASELINE_END -- set these to
   your actual wildfire event window and a pre-fire baseline period, matching
   the date range covered by LA_Site_1103.csv.

===============================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import GAMMA_POINT, GAMMA_LO, GAMMA_HI, GBD_BASELINE_MORTALITY_PER_100K_PER_YEAR
from real_data_loaders import (
    load_epa_data, get_epa_site_location, load_tract_data, load_city_boundary,
    load_all_purpleair_sensors, resample_to_hourly, find_closest_sensor, project_to_km,
)
from mortality_utils import compute_tract_baseline_pm25, compute_gbd_baseline_daily_mortality, interpolate_to_tracts
from algorithm1_model_comparison import run_algorithm1, summarize_r2
from algorithm2_loglinear import run_algorithm2
from algorithm3_mlp import run_algorithm3

# ------------------------------------------------------------------------
# --- SETTINGS: edit these for your run -----------------------------------
# ------------------------------------------------------------------------
EPA_PATH = "LA_Site_1103.csv"
TRACTS_PATH = "la_city_tracts_final.csv"
BOUNDARY_PATH = "City_Boundary.geojson"

PURPLEAIR_SENSORS_DIR = "mock_sensors"          # <-- point at your real directory
SENSOR_METADATA_PATH = "mock_sensors/sensor_metadata.csv"  # <-- or None if lat/lon are in the files themselves

# Wildfire event window and a pre-fire baseline window, both within the
# range covered by your EPA file. EDIT to match your actual event.
BASELINE_START, BASELINE_END = "2025-07-01", "2025-07-31"
EVENT_START, EVENT_END = "2025-08-01", "2025-08-20"


def check_one_sensor_file(directory: str):
    """
    Run this FIRST against one real sensor_*_history.csv to sanity-check
    that column auto-detection worked before running the full pipeline.
    """
    from real_data_loaders import discover_purpleair_files, load_purpleair_sensor_file

    files = discover_purpleair_files(directory)
    if not files:
        print(f"No sensor_*_history.csv files found in {directory}.")
        return
    sensor_id, path = files[0]
    print(f"Checking {path} (sensor_id={sensor_id}) ...")
    df = load_purpleair_sensor_file(path, sensor_id)
    print(df.head())
    print(df.dtypes)
    print(f"lat/lon present in file: {'lat' in df.columns}")


def main():
    # ---- 1. Load EPA monitor data -----------------------------------------
    epa_df = load_epa_data(EPA_PATH)
    epa_lat, epa_lon = get_epa_site_location(epa_df)
    print(f"EPA site: {epa_lat}, {epa_lon} | {len(epa_df)} hourly readings "
          f"from {epa_df['timestamp'].min()} to {epa_df['timestamp'].max()}")

    # ---- 2. Load census tracts + optional city boundary --------------------
    tract_centroids, population = load_tract_data(TRACTS_PATH)
    print(f"{len(tract_centroids)} census tracts loaded.")

    boundary = load_city_boundary(BOUNDARY_PATH)

    # ---- 3. Load the full PurpleAir sensor network --------------------------
    all_sensors = load_all_purpleair_sensors(
        PURPLEAIR_SENSORS_DIR, metadata_path=SENSOR_METADATA_PATH,
    )
    all_sensors = resample_to_hourly(all_sensors)
    print(f"{all_sensors['sensor_id'].nunique()} PurpleAir sensors loaded, "
          f"{len(all_sensors)} hourly sensor-readings total.")

    # ---- 4. Identify the sensor closest to the EPA monitor for calibration --
    closest_sensor_id = find_closest_sensor(all_sensors, epa_lat, epa_lon)
    print(f"Closest PurpleAir sensor to EPA monitor: {closest_sensor_id}")
    purpleair_calibration_df = all_sensors[all_sensors["sensor_id"] == closest_sensor_id].copy()

    # ---- 5. Algorithm 1: calibration model comparison -----------------------
    print("\n" + "=" * 70)
    print("ALGORITHM 1: Sensor Calibration Model Comparison")
    print("=" * 70)
    algo1_results = run_algorithm1(purpleair_calibration_df, epa_df, n_splits=10)
    print(summarize_r2(algo1_results["r2_results"]))

    # ---- 6. Baseline (pre-fire) PM2.5 per tract, using the log-linear ------
    #         calibration applied to ALL sensors over the baseline window
    baseline_sensors = all_sensors[
        (all_sensors["date"] >= BASELINE_START) & (all_sensors["date"] <= BASELINE_END)
    ].copy()
    baseline_calibrated = baseline_sensors.copy()
    from calibration_models import predict_log_linear
    from data_utils import add_lagged_meteorology
    from config import ALL_FEATURES

    baseline_lagged = add_lagged_meteorology(baseline_calibrated, sensor_id_col="sensor_id")
    baseline_lagged["adjusted_pm25"] = predict_log_linear(
        algo1_results["full_data_log_linear_fit"], baseline_lagged, features=ALL_FEATURES
    )
    baseline_pm25_by_tract_day = interpolate_to_tracts(
        baseline_lagged, tract_centroids, value_col="adjusted_pm25", day_col="date"
    )
    tract_baseline_pm25 = compute_tract_baseline_pm25(baseline_pm25_by_tract_day)
    print(f"\nComputed baseline (pre-fire) PM2.5 for {len(tract_baseline_pm25)} tracts.")

    # ---- 7. Baseline daily mortality (Bd), GBD 2017 rate --------------------
    baseline_daily_mortality = compute_gbd_baseline_daily_mortality(
        population, yearly_rate_per_100k=GBD_BASELINE_MORTALITY_PER_100K_PER_YEAR
    )

    # ---- 8. EPA-only daily series over the event window, for comparison -----
    epa_daily_df = (
        epa_df.assign(date=epa_df["timestamp"].dt.date.astype(str))
        .groupby("date", as_index=False)["epa_pm25"].mean()
    )
    epa_daily_df = epa_daily_df[(epa_daily_df["date"] >= EVENT_START) & (epa_daily_df["date"] <= EVENT_END)]

    # ---- 9. Algorithm 2: log-linear mortality estimate ----------------------
    print("\n" + "=" * 70)
    print("ALGORITHM 2: Excess Mortality via Log-Linear Calibration")
    print("=" * 70)
    event_sensors = all_sensors[
        (all_sensors["date"] >= EVENT_START) & (all_sensors["date"] <= EVENT_END)
    ].copy()

    algo2_results = run_algorithm2(
        full_data_log_linear_fit=algo1_results["full_data_log_linear_fit"],
        synchronized_calibration_data=algo1_results["synchronized_calibration_data"],
        all_sensor_readings_df=event_sensors,
        tract_centroids=tract_centroids,
        baseline_pm25=tract_baseline_pm25,          # per-tract, matching the paper
        gamma_point=GAMMA_POINT, gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
        baseline_daily_mortality=baseline_daily_mortality,
        population=population,
        epa_daily_df=epa_daily_df,
        m=10,
    )
    print(algo2_results["point_estimate"].sort_values("M", ascending=False).head(10))
    print(f"\nCity-wide total (log-linear): {algo2_results['point_estimate']['M'].sum():.1f} "
          f"[{algo2_results['point_estimate']['CI_lower'].sum():.1f}, "
          f"{algo2_results['point_estimate']['CI_upper'].sum():.1f}]")
    print(f"City-wide total (EPA-only uniform): {algo2_results['uniform_epa_only']['cumulative_M'].sum():.1f}")

    # ---- 10. Algorithm 3: MLP mortality estimate -----------------------------
    print("\n" + "=" * 70)
    print("ALGORITHM 3: Excess Mortality via MLP Calibration")
    print("=" * 70)
    algo3_results = run_algorithm3(
        full_training_data=algo1_results["synchronized_calibration_data"],
        all_sensor_readings_df=event_sensors,
        tract_centroids=tract_centroids,
        baseline_pm25=tract_baseline_pm25,
        gamma_point=GAMMA_POINT, gamma_lo=GAMMA_LO, gamma_hi=GAMMA_HI,
        baseline_daily_mortality=baseline_daily_mortality,
        population=population,
        m=5,  # MLP retraining is expensive; raise to 50 for a final run
    )
    print(algo3_results["ensemble_point_estimate"].sort_values("M_mean", ascending=False).head(10))
    print(f"\nCity-wide total (MLP ensemble): {algo3_results['ensemble_point_estimate']['M_mean'].sum():.1f}")


if __name__ == "__main__":
    main()