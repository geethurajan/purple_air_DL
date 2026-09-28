# Notes: some sensors were bought during the fires or after
# 137 sensors total
# potential outliers: some sensors placed indoors when labeled outdoors
# train station produces heavy pollution and many in small area with train
# each sensor numbered 1 - 137, with IDs and GPS listed in Sensors.csv

import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from math import radians, cos, sin, asin, sqrt

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_HOURS            = 1416  # Jan 1 – Mar 1, 2025
TARGET                 = 'ln_pm25_epa'
EXCLUDE_SENSORS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 120, 119, 118, 59, 72, 121, 74, 122, 117, 13, 73, 66, 116, 25, 123, 124, 125, 115, 18, 12, 7, 110, 75, 6, 109, 111, 113, 11, 76, 112, 108, 10, 114, 107, 77, 17, 78, 85, 104, 86, 5, 106, 105, 61, 67, 60, 103, 63, 102, 20, 62, 9, 64, 87, 81, 135, 89, 69, 80, 92, 93, 71, 79, 68, 90, 82, 134, 91, 88, 70, 16, 8, 136, 130, 100, 131, 133, 84, 132, 98, 15, 129, 94, 137, 99, 128, 95, 83, 127, 96, 97, 126]    # exclude potential outliers

# =============================================================================
# 1. HAVERSINE DISTANCE
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * asin(sqrt(a))

# =============================================================================
# 2. LOAD SENSOR METADATA
# =============================================================================

sensors_df = pd.read_csv(os.path.join(BASE_DIR, 'Sensors.csv'))
sensors_df.columns = sensors_df.columns.str.strip().str.lower().str.replace(' ', '_')

# =============================================================================
# 3. LOAD EPA DATA
# =============================================================================

epa_df = pd.read_csv(os.path.join(BASE_DIR, 'LA_Site_1103.csv'))
epa_df.columns = epa_df.columns.str.strip()

epa_df['timestamp'] = pd.to_datetime(
    epa_df['Date Local'].astype(str) + ' ' + epa_df['Time Local'].astype(str)
)

EPA_LAT = epa_df['Latitude'].iloc[0]
EPA_LON = epa_df['Longitude'].iloc[0]

epa_df = epa_df[['timestamp', 'Sample Measurement']].rename(
    columns={'Sample Measurement': 'pm25_epa'}
)
epa_df = epa_df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

# =============================================================================
# 4. COMPUTE DISTANCES
# =============================================================================

sensors_df['distance_km'] = sensors_df.apply(
    lambda row: haversine(EPA_LAT, EPA_LON, row['latitude'], row['longitude']),
    axis=1
)

# =============================================================================
# 5. LOAD PURPLEAIR FILES
# =============================================================================

pa_records           = []
completeness_records = []

for _, meta_row in sensors_df.iterrows():
    sensor_num = meta_row['sensor_num']
    sensor_id  = meta_row['sensor_id']
    lat        = meta_row['latitude']
    lon        = meta_row['longitude']
    dist_km    = meta_row['distance_km']
    if int(sensor_num) in EXCLUDE_SENSORS:
        completeness_records.append({
            'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
            'distance_km': dist_km, 'rows_found': 0,
            'completeness_pct': 0.0, 'status': 'EXCLUDED'
        })
        continue
    folder = os.path.join(BASE_DIR, f'PurpleAir Download Sensor {int(sensor_num)}')

    if not os.path.isdir(folder):
        completeness_records.append({
            'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
            'distance_km': dist_km, 'rows_found': 0,
            'completeness_pct': 0.0, 'status': 'FOLDER MISSING'
        })
        continue

    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        completeness_records.append({
            'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
            'distance_km': dist_km, 'rows_found': 0,
            'completeness_pct': 0.0, 'status': 'CSV MISSING'
        })
        continue

    try:
        pa = pd.read_csv(csv_files[0])
    except Exception as e:
        completeness_records.append({
            'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
            'distance_km': dist_km, 'rows_found': 0,
            'completeness_pct': 0.0, 'status': f'READ ERROR: {e}'
        })
        continue

    if pa.empty:
        completeness_records.append({
            'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
            'distance_km': dist_km, 'rows_found': 0,
            'completeness_pct': 0.0, 'status': 'EMPTY FILE'
        })
        continue

    pa.columns    = pa.columns.str.strip()
    pa['timestamp'] = pd.to_datetime(pa['time_stamp']).dt.tz_localize(None)
    pa = pa.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    rows_found = len(pa)
    completeness_records.append({
        'sensor_num': int(sensor_num), 'sensor_id': sensor_id,
        'distance_km': dist_km, 'rows_found': rows_found,
        'completeness_pct': round(rows_found / TOTAL_HOURS * 100, 1),
        'status': 'OK' if rows_found == TOTAL_HOURS else 'PARTIAL'
    })

    pa['sensor_num']  = int(sensor_num)
    pa['sensor_id']   = sensor_id
    pa['latitude']    = lat
    pa['longitude']   = lon
    pa['distance_km'] = dist_km
    pa_records.append(pa)

# =============================================================================
# 6. SENSOR DISTANCE REPORT (saved to CSV)
# =============================================================================

completeness_df = pd.DataFrame(completeness_records).sort_values('distance_km').reset_index(drop=True)
completeness_df.to_csv(os.path.join(BASE_DIR, 'sensors_by_distance.csv'), index=False)
print("Sensor report saved to sensors_by_distance.csv")

# =============================================================================
# 7. ASSEMBLE PANEL
# =============================================================================

if not pa_records:
    raise RuntimeError("No PurpleAir data loaded. Check folders and threshold settings.")

panel = pd.concat(pa_records, ignore_index=True)
panel = panel.rename(columns={'pm2.5_atm': 'pm25_pa_atm', 'pm2.5_cf_1': 'pm25_pa_cf1'})

keep = ['timestamp', 'sensor_num', 'latitude', 'longitude', 'distance_km',
        'temperature', 'pressure', 'humidity', 'pm25_pa_atm', 'pm25_pa_cf1']
panel = panel[[c for c in keep if c in panel.columns]].copy()
panel = panel.merge(epa_df, on='timestamp', how='inner')

# =============================================================================
# 8. FEATURE ENGINEERING
# =============================================================================

panel = panel.sort_values(['sensor_num', 'timestamp']).reset_index(drop=True)

# Lag features — only valid if previous row is exactly 1 hour earlier
for col in ['temperature', 'pressure', 'humidity']:
    shifted_vals  = panel.groupby('sensor_num')[col].shift(1)
    shifted_times = panel.groupby('sensor_num')['timestamp'].shift(1)
    is_one_hour   = (panel['timestamp'] - shifted_times) == pd.Timedelta(hours=1)
    panel[f'{col}_lag1'] = shifted_vals.where(is_one_hour, other=np.nan)

# Log transforms
panel['ln_pm25_pa_atm'] = np.where(panel['pm25_pa_atm'] > 0, np.log(panel['pm25_pa_atm']), np.nan)
panel['ln_pm25_pa_cf1'] = np.where(panel['pm25_pa_cf1'] > 0, np.log(panel['pm25_pa_cf1']), np.nan)
panel['ln_pm25_epa']    = np.where(panel['pm25_epa']    > 0, np.log(panel['pm25_epa']),    np.nan)

# =============================================================================
# 9. OLS MODELS
# =============================================================================

FEATURE_SETS = {
    'Full Model — pm2.5_atm': [
        'ln_pm25_pa_atm', 'temperature', 'pressure', 'humidity',
        'temperature_lag1', 'pressure_lag1', 'humidity_lag1',
        'latitude', 'longitude', 'distance_km'
    ],
    'Full Model — pm2.5_cf_1': [
        'ln_pm25_pa_cf1', 'temperature', 'pressure', 'humidity',
        'temperature_lag1', 'pressure_lag1', 'humidity_lag1',
        'latitude', 'longitude', 'distance_km'
    ],
    'Reduced Model — pm2.5_atm': [
        'ln_pm25_pa_atm', 'temperature', 'pressure', 'humidity', 'distance_km'
    ],
    'Reduced Model — pm2.5_cf_1': [
        'ln_pm25_pa_cf1', 'temperature', 'pressure', 'humidity', 'distance_km'
    ],
}


results_summary = []

for model_name, features in FEATURE_SETS.items():
    subset  = panel.dropna(subset=features + [TARGET]).reset_index(drop=True)
    X       = sm.add_constant(subset[features])
    y       = subset[TARGET]
    model   = sm.OLS(y, X).fit()

    print(f"\n{'='*60}")
    print(f"{model_name}  (n = {len(subset):,})")
    print(f"  R²: {model.rsquared:.4f}    Adj. R²: {model.rsquared_adj:.4f}")
    print(f"{'='*60}")
    print(f"  {'Variable':<22} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}")
    print(f"  {'-'*60}")
    for var in model.params.index:
        print(f"  {var:<22} {model.params[var]:>10.4f} "
              f"{model.bse[var]:>10.4f} "
              f"{model.tvalues[var]:>8.3f} "
              f"{model.pvalues[var]:>8.3f}")

    results_summary.append({
        'model'      : model_name,
        'n'          : len(subset),
        'r2'         : round(model.rsquared, 4),
        'adj_r2'     : round(model.rsquared_adj, 4),
    })

# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'Model':<35} {'n':>7} {'R²':>8} {'Adj R²':>8}")
print(f"  {'-'*60}")
for r in results_summary:
    print(f"  {r['model']:<35} {r['n']:>7,} {r['r2']:>8.4f} {r['adj_r2']:>8.4f}")
print(f"{'='*60}")

# =============================================================================
# 11. OUTLIER SENSOR DETECTION
# =============================================================================

NEIGHBOR_RADIUS_KM = 0.8   # sensors within this distance are considered neighbors
Z_THRESHOLD        = 2.0   # standard deviations away to flag as outlier

sensor_stats = []

for sensor_num in panel['sensor_num'].unique():
    s = panel[panel['sensor_num'] == sensor_num].copy()
    avg_atm = s['pm25_pa_atm'].mean()
    avg_cf1 = s['pm25_pa_cf1'].mean()
    lat     = s['latitude'].iloc[0]
    lon     = s['longitude'].iloc[0]
    dist_km = s['distance_km'].iloc[0]

    # Per-sensor R² using full atm model features
    features = FEATURE_SETS['Full Model — pm2.5_atm']
    s_clean  = s.dropna(subset=features + [TARGET])
    if len(s_clean) > len(features) + 1:
        X_s   = sm.add_constant(s_clean[features], has_constant='add')
        y_s   = s_clean[TARGET]
        fit_s = sm.OLS(y_s, X_s).fit()
        r2_s  = fit_s.rsquared
    else:
        r2_s  = np.nan

    sensor_stats.append({
        'sensor_num'  : sensor_num,
        'latitude'    : lat,
        'longitude'   : lon,
        'distance_km' : dist_km,
        'avg_pm25_atm': avg_atm,
        'avg_pm25_cf1': avg_cf1,
        'r2'          : r2_s,
    })

sensor_stats = pd.DataFrame(sensor_stats).sort_values('sensor_num').reset_index(drop=True)

# --- Compare each sensor to neighbors within NEIGHBOR_RADIUS_KM ---
pm25_diff = []
for i, row in sensor_stats.iterrows():
    dists = sensor_stats.apply(
        lambda r: haversine(row['latitude'], row['longitude'], r['latitude'], r['longitude']),
        axis=1
    )
    neighbor_idx = dists[(dists > 0) & (dists <= NEIGHBOR_RADIUS_KM)].index
    if len(neighbor_idx) == 0:
        pm25_diff.append({'diff_atm': np.nan, 'diff_cf1': np.nan})
    else:
        neighbor_avg_atm = sensor_stats.loc[neighbor_idx, 'avg_pm25_atm'].mean()
        neighbor_avg_cf1 = sensor_stats.loc[neighbor_idx, 'avg_pm25_cf1'].mean()
        pm25_diff.append({
            'diff_atm': row['avg_pm25_atm'] - neighbor_avg_atm,
            'diff_cf1': row['avg_pm25_cf1'] - neighbor_avg_cf1,
        })

pm25_diff_df = pd.DataFrame(pm25_diff)
sensor_stats['diff_from_neighbors_atm'] = pm25_diff_df['diff_atm'].values
sensor_stats['diff_from_neighbors_cf1'] = pm25_diff_df['diff_cf1'].values

# --- Flag outliers: PM2.5 unusually different from neighbors ---
diff_mean_atm = sensor_stats['diff_from_neighbors_atm'].mean()
diff_std_atm  = sensor_stats['diff_from_neighbors_atm'].std()
diff_mean_cf1 = sensor_stats['diff_from_neighbors_cf1'].mean()
diff_std_cf1  = sensor_stats['diff_from_neighbors_cf1'].std()

sensor_stats['z_diff_atm'] = (sensor_stats['diff_from_neighbors_atm'] - diff_mean_atm) / diff_std_atm
sensor_stats['z_diff_cf1'] = (sensor_stats['diff_from_neighbors_cf1'] - diff_mean_cf1) / diff_std_cf1

pm25_outliers = sensor_stats[
    (sensor_stats['z_diff_atm'].abs() > Z_THRESHOLD) |
    (sensor_stats['z_diff_cf1'].abs() > Z_THRESHOLD)
].sort_values('z_diff_atm', key=abs, ascending=False).reset_index(drop=True)

# --- Flag outliers: R² much lower than other sensors ---
r2_mean = sensor_stats['r2'].mean()
r2_std  = sensor_stats['r2'].std()
sensor_stats['z_r2'] = (sensor_stats['r2'] - r2_mean) / r2_std

r2_outliers = sensor_stats[
    sensor_stats['z_r2'] < -Z_THRESHOLD
].sort_values('z_r2').reset_index(drop=True)

# --- Print results ---
print(f"\n{'='*65}")
print(f"OUTLIER DETECTION  (radius = {NEIGHBOR_RADIUS_KM} km, z-threshold = {Z_THRESHOLD})")
print(f"{'='*65}")

print(f"\nSensors with PM2.5 unusually different from neighbors within {NEIGHBOR_RADIUS_KM} km:")
if pm25_outliers.empty:
    print("  None flagged.")
else:
    print(f"  {'Sensor':>8} {'Dist(km)':>10} {'Avg ATM':>9} {'Diff ATM':>10} {'Z(ATM)':>8} {'Avg CF1':>9} {'Diff CF1':>10} {'Z(CF1)':>8}")
    print(f"  {'-'*75}")
    for _, r in pm25_outliers.iterrows():
        print(f"  {int(r['sensor_num']):>8} {r['distance_km']:>10.3f} "
              f"{r['avg_pm25_atm']:>9.2f} {r['diff_from_neighbors_atm']:>10.2f} {r['z_diff_atm']:>8.2f} "
              f"{r['avg_pm25_cf1']:>9.2f} {r['diff_from_neighbors_cf1']:>10.2f} {r['z_diff_cf1']:>8.2f}")

print(f"\nSensors with R² much lower than other sensors:")
if r2_outliers.empty:
    print("  None flagged.")
else:
    print(f"  {'Sensor':>8} {'Dist(km)':>10} {'R²':>8} {'Z(R²)':>8}")
    print(f"  {'-'*38}")
    for _, r in r2_outliers.iterrows():
        print(f"  {int(r['sensor_num']):>8} {r['distance_km']:>10.3f} "
              f"{r['r2']:>8.4f} {r['z_r2']:>8.2f}")

print(f"\n{'='*65}")

# =============================================================================
# 12. NEURAL NETWORK (Standard MLP)
# =============================================================================

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

NN_FEATURES = [
    'pm25_pa_cf1',
    'temperature', 'pressure', 'humidity',
    'temperature_lag1', 'pressure_lag1', 'humidity_lag1',
    'latitude', 'longitude', 'distance_km'
]

NN_TARGET = 'pm25_epa'

subset_nn = panel.dropna(subset=NN_FEATURES + [NN_TARGET]).reset_index(drop=True)
subset_nn = subset_nn[subset_nn[NN_TARGET] > 0].reset_index(drop=True)
subset_nn = subset_nn[subset_nn['pm25_pa_cf1'] > 0].reset_index(drop=True)

X_nn = subset_nn[NN_FEATURES]
y_nn = subset_nn[NN_TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42
)

scaler_nn      = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled  = scaler_nn.transform(X_test)

mlp_model = MLPRegressor(
    hidden_layer_sizes = (64, 32, 16),
    activation         = 'relu',
    solver             = 'adam',
    max_iter           = 500,
    random_state       = 42,
)

mlp_model.fit(X_train_scaled, y_train)

y_pred_mlp = mlp_model.predict(X_train_scaled) #changed X_test to X_train

n_mlp      = len(y_train)         #used test
p_mlp      = len(NN_FEATURES)
r2_mlp     = r2_score(y_train, y_pred_mlp)           #used test
adj_r2_mlp = 1 - (1 - r2_mlp) * (n_mlp - 1) / (n_mlp - p_mlp - 1)

print(f"\n{'='*60}")
print(f"NEURAL NETWORK — Standard MLP")
print(f"(n_test = {n_mlp:,})")
print(f"{'='*60}")
print(f"  R²:        {r2_mlp:.4f}")
print(f"  Adj. R²:   {adj_r2_mlp:.4f}")
print(f"{'='*60}")
print("Done.")