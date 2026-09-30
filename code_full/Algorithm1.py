"""
ALGORITHM 1 -- Sensor Calibration Model Comparison (Log-Linear vs. MLP)

This is a self-contained script covering everything Algorithm 1 needs:
    1. Load the EPA monitor, census tracts, and the full PurpleAir sensor
       network (with lat/lon attached from Sensors_with_coordinates.csv).
    2. Resample every sensor to hourly and keep only sensors with at least
       MIN_SENSOR_HOURS of data (default 120; edit in pipeline_config.py).
    3. Identify the sensor closest to the EPA monitor and use it to fit the
       full-data log-linear calibration model (the point estimate carried
       forward to Algorithm 2).
    4. Run the 50-iteration (configurable) paired R^2 comparison between the
       log-linear model and an MLP on leakage-resistant time-block splits.

Every output Algorithm 2 will need is saved to CHECKPOINTS_DIR as a plain
CSV, so you can run Algorithm 2 later (in a separate session, without
re-running any of this) just by reading these files back in.

Run:  python algorithm1.py

All settings (file paths, MIN_SENSOR_HOURS, replicate counts) live in
pipeline_config.py -- edit that file, not this one.

===============================================================================
OUTPUTS WRITTEN TO checkpoints/ (all needed by Algorithm 2):
===============================================================================
    epa_data.csv                      [timestamp, epa_pm25, latitude, longitude]
    tract_centroids.csv               [tract_id, lat, lon, x_km, y_km]
    population.csv                    [tract_id, population]
    sensor_hour_counts.csv            [sensor_id, n_hours, kept] -- audit trail
                                        of which sensors passed the 120-hour filter
    all_sensors_filtered.csv          hourly readings for every sensor that
                                        passed the filter -- Algorithm 2 applies
                                        the calibration to this
    closest_sensor_id.txt             which sensor was used for calibration
    log_linear_params.csv             full-data log-linear coefficients
                                        (Algorithm 2's calibration point estimate)
    r2_comparison.csv                 paired R^2, log-linear vs. MLP, per split
    synchronized_calibration_data.csv the EPA/PurpleAir paired data Algorithm 2
                                        bootstrap-resamples from
===============================================================================
"""

import pandas as pd

from pipeline_config import (
    EPA_PATH, TRACTS_PATH, PURPLEAIR_SENSORS_DIR, SENSOR_METADATA_PATH,
    METADATA_SENSOR_ID_COL, MIN_SENSOR_HOURS, N_TIME_BLOCK_SPLITS, ckpt,
)
from real_data_loaders import (
    load_epa_data, load_tract_data, load_all_purpleair_sensors,
    resample_to_hourly, filter_sensors_by_min_hours,
    get_epa_site_location, find_closest_sensor,
)
from calibration_models import save_log_linear_params
from algorithm1_model_comparison import run_algorithm1, summarize_r2


def main():
    # ---- Step 1: load EPA monitor ------------------------------------------
    print("Loading EPA monitor data...")
    epa_df = load_epa_data(EPA_PATH)
    epa_df.to_csv(ckpt("epa_data.csv"), index=False)
    epa_lat, epa_lon = get_epa_site_location(epa_df)
    print(f"  -> {len(epa_df)} hourly readings, "
          f"{epa_df['timestamp'].min()} to {epa_df['timestamp'].max()}")
    print(f"  -> EPA site location: {epa_lat}, {epa_lon}")

    # ---- Step 2: load census tracts ----------------------------------------
    print("\nLoading census tract data...")
    tract_centroids, population = load_tract_data(TRACTS_PATH)
    tract_centroids.to_csv(ckpt("tract_centroids.csv"), index=False)
    population.to_csv(ckpt("population.csv"), index=False)
    print(f"  -> {len(tract_centroids)} tracts")

    # ---- Step 3: load the full PurpleAir sensor network ---------------------
    print(f"\nLoading PurpleAir sensors from '{PURPLEAIR_SENSORS_DIR}' "
          f"(this can take a while for ~260 files)...")
    all_sensors = load_all_purpleair_sensors(
        PURPLEAIR_SENSORS_DIR,
        metadata_path=SENSOR_METADATA_PATH,
        metadata_sensor_id_col=METADATA_SENSOR_ID_COL,
    )
    n_raw_sensors = all_sensors["sensor_id"].nunique()
    print(f"  -> {n_raw_sensors} sensors loaded, {len(all_sensors)} raw readings total")

    print("\nResampling every sensor to hourly...")
    all_sensors = resample_to_hourly(all_sensors)

    # ---- Step 4: filter sensors by minimum data coverage ---------------------
    print(f"\nFiltering to sensors with >= {MIN_SENSOR_HOURS} hours of data...")
    all_sensors, hour_counts = filter_sensors_by_min_hours(all_sensors, min_hours=MIN_SENSOR_HOURS)
    hour_counts.to_csv(ckpt("sensor_hour_counts.csv"), index=False)
    all_sensors.to_csv(ckpt("all_sensors_filtered.csv"), index=False)
    print(f"  -> {all_sensors['sensor_id'].nunique()} of {n_raw_sensors} sensors kept "
          f"({len(all_sensors)} hourly readings).")

    # ---- Step 5: identify the calibration sensor (closest to EPA monitor) ----
    closest_sensor_id = find_closest_sensor(all_sensors, epa_lat, epa_lon)
    print(f"\nClosest PurpleAir sensor to EPA monitor: {closest_sensor_id}")
    with open(ckpt("closest_sensor_id.txt"), "w") as f:
        f.write(str(closest_sensor_id))

    purpleair_calibration_df = all_sensors[all_sensors["sensor_id"] == closest_sensor_id].copy()
    print(f"  -> {len(purpleair_calibration_df)} hourly readings from this sensor.")

    # ---- Steps 1-10 of Algorithm 1 proper: calibration + model comparison ----
    print(f"\nRunning Algorithm 1 ({N_TIME_BLOCK_SPLITS} time-block splits; "
          f"this may take a while, since each split trains an MLP)...")
    algo1_results = run_algorithm1(purpleair_calibration_df, epa_df, n_splits=N_TIME_BLOCK_SPLITS)

    print("\nR^2 comparison summary (log-linear vs. MLP):")
    print(summarize_r2(algo1_results["r2_results"]))

    save_log_linear_params(algo1_results["full_data_log_linear_fit"], ckpt("log_linear_params.csv"))
    algo1_results["r2_results"].to_csv(ckpt("r2_comparison.csv"), index=False)
    algo1_results["synchronized_calibration_data"].to_csv(ckpt("synchronized_calibration_data.csv"), index=False)

    print(f"\nALGORITHM 1 COMPLETE. All checkpoints written to '{ckpt('')}'.")
    print("Everything Algorithm 2 needs is now saved -- you can run it in a separate session.")


if __name__ == "__main__":
    main()