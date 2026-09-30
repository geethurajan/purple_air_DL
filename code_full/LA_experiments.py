# =============================================================================
# LA PM2.5 EXPERIMENTS
# Three experiments comparing OLS log-linear models vs MLP neural network
#
# Experiment 3 (runs first): find best epoch via avg test MSE over 50 reps
# Experiment 1: in-sample R² — train and evaluate on full dataset
# Experiment 2: avg out-of-sample test R² over 50 replications
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_HOURS = 1416  # Jan 1 – Mar 1, 2025
TARGET      = 'ln_pm25_epa'
NN_TARGET   = 'pm25_epa'

EXCLUDE_SENSORS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                   54, 55, 56, 57, 58, 73, 66, 116, 25, 123, 124, 125, 115, 18, 12, 7,
                   110, 75, 6, 109, 111, 113, 11, 76, 112, 108, 10, 114, 107,
                   77, 17, 78, 85, 104, 86, 5, 106, 105, 61, 67, 60, 103, 63,
                   102, 20, 62, 9, 64, 87, 81, 135, 89, 69, 80, 92, 93, 71,
                   79, 68, 90, 82, 134, 91, 88, 70, 16, 8, 136, 130, 100,
                   131, 133, 84, 132, 98, 15, 129, 94, 137, 99, 128, 95, 83,
                   127, 96, 97, 126]

# Experiment settings
N_REPLICATIONS = 50
TEST_SIZE      = 0.2
EPOCH_VALUES   = [50, 100, 200, 300, 500, 750]

# OLS feature sets
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

# MLP feature set (raw scale, not log)
NN_FEATURES = [
    'pm25_pa_cf1',
    'temperature', 'pressure', 'humidity',
    'temperature_lag1', 'pressure_lag1', 'humidity_lag1',
    'latitude', 'longitude', 'distance_km'
]

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

pa_records = []

for _, meta_row in sensors_df.iterrows():
    sensor_num = meta_row['sensor_num']
    sensor_id  = meta_row['sensor_id']
    lat        = meta_row['latitude']
    lon        = meta_row['longitude']
    dist_km    = meta_row['distance_km']

    if int(sensor_num) in EXCLUDE_SENSORS:
        continue

    folder = os.path.join(BASE_DIR, f'PurpleAir Download Sensor {int(sensor_num)}')
    if not os.path.isdir(folder):
        continue

    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        continue

    try:
        pa = pd.read_csv(csv_files[0])
    except Exception:
        continue

    if pa.empty:
        continue

    pa.columns      = pa.columns.str.strip()
    pa['timestamp'] = pd.to_datetime(pa['time_stamp']).dt.tz_localize(None)
    pa = pa.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    pa['sensor_num']  = int(sensor_num)
    pa['sensor_id']   = sensor_id
    pa['latitude']    = lat
    pa['longitude']   = lon
    pa['distance_km'] = dist_km
    pa_records.append(pa)

print(f"Loaded {len(pa_records)} sensors.")

# =============================================================================
# 6. ASSEMBLE PANEL
# =============================================================================

if not pa_records:
    raise RuntimeError("No PurpleAir data loaded. Check folders and EXCLUDE_SENSORS.")

panel = pd.concat(pa_records, ignore_index=True)
panel = panel.rename(columns={'pm2.5_atm': 'pm25_pa_atm', 'pm2.5_cf_1': 'pm25_pa_cf1'})

keep = ['timestamp', 'sensor_num', 'latitude', 'longitude', 'distance_km',
        'temperature', 'pressure', 'humidity', 'pm25_pa_atm', 'pm25_pa_cf1']
panel = panel[[c for c in keep if c in panel.columns]].copy()
panel = panel.merge(epa_df, on='timestamp', how='inner')

# =============================================================================
# 7. FEATURE ENGINEERING
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
# 8. PREPARE CLEAN SUBSETS
# =============================================================================

# One clean subset per OLS feature set
ols_subsets = {}
for model_name, features in FEATURE_SETS.items():
    ols_subsets[model_name] = panel.dropna(
        subset=features + [TARGET]
    ).reset_index(drop=True)

# Clean subset for MLP (raw scale)
subset_nn = panel.dropna(subset=NN_FEATURES + [NN_TARGET]).reset_index(drop=True)
subset_nn = subset_nn[subset_nn[NN_TARGET]     > 0].reset_index(drop=True)
subset_nn = subset_nn[subset_nn['pm25_pa_cf1'] > 0].reset_index(drop=True)

X_nn_all = subset_nn[NN_FEATURES].values
y_nn_all = subset_nn[NN_TARGET].values

print(f"Panel assembled: {len(panel):,} rows.")
print(f"MLP subset: {len(subset_nn):,} rows.")

# =============================================================================
# EXPERIMENT 3: Average test MSE vs epoch over 50 replications (run first)
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT 3 — MLP Test MSE vs Epoch (50 replications)")
print("="*60)

epoch_avg_mse = {}

for epochs in EPOCH_VALUES:
    mse_list = []
    for rep in range(N_REPLICATIONS):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_nn_all, y_nn_all,
            test_size=TEST_SIZE,
            random_state=rep
        )
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        mlp = MLPRegressor(
            hidden_layer_sizes = (64, 32, 16),
            activation         = 'relu',
            solver             = 'adam',
            max_iter           = epochs,
            random_state       = 42,
        )
        mlp.fit(X_tr_sc, y_tr)
        y_pred = mlp.predict(X_te_sc)
        mse_list.append(mean_squared_error(y_te, y_pred))

    avg_mse = np.mean(mse_list)
    epoch_avg_mse[epochs] = avg_mse
    print(f"  Epochs = {epochs:>4}   Avg Test MSE = {avg_mse:.4f}")

# Plot
plt.figure(figsize=(7, 4))
plt.plot(list(epoch_avg_mse.keys()), list(epoch_avg_mse.values()),
         marker='o', linewidth=2)
plt.xlabel('Epochs (max_iter)')
plt.ylabel('Average Test MSE')
plt.title('MLP: Average Test MSE vs Epoch (50 replications)')
plt.xticks(EPOCH_VALUES)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'experiment3_mse_vs_epochs.png'), dpi=150)
plt.show()
print("Plot saved to experiment3_mse_vs_epochs.png")

BEST_EPOCH = min(epoch_avg_mse, key=epoch_avg_mse.get)
print(f"\n  Best epoch: {BEST_EPOCH}  (Avg Test MSE = {epoch_avg_mse[BEST_EPOCH]:.4f})")

# =============================================================================
# EXPERIMENT 1: In-sample R² — fit and evaluate on full dataset
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT 1 — In-Sample R² (full dataset, no split)")
print("="*60)

exp1_results = {}

print(f"\n  {'Model':<35} {'n':>8} {'R²':>8} {'Adj R²':>8}")
print(f"  {'-'*62}")

for model_name, features in FEATURE_SETS.items():
    subset = ols_subsets[model_name]
    X      = sm.add_constant(subset[features])
    y      = subset[TARGET]
    fit    = sm.OLS(y, X).fit()
    exp1_results[model_name] = {
        'r2': fit.rsquared, 'adj_r2': fit.rsquared_adj, 'n': len(subset)
    }
    print(f"  {model_name:<35} {len(subset):>8,} {fit.rsquared:>8.4f} {fit.rsquared_adj:>8.4f}")

# MLP on full data
scaler_e1   = StandardScaler()
X_nn_sc_all = scaler_e1.fit_transform(X_nn_all)
mlp_e1 = MLPRegressor(
    hidden_layer_sizes = (64, 32, 16),
    activation         = 'relu',
    solver             = 'adam',
    max_iter           = BEST_EPOCH,
    random_state       = 42,
)
mlp_e1.fit(X_nn_sc_all, y_nn_all)
y_pred_e1 = mlp_e1.predict(X_nn_sc_all)
r2_e1     = r2_score(y_nn_all, y_pred_e1)
n_e1      = len(y_nn_all)
p_e1      = len(NN_FEATURES)
adj_r2_e1 = 1 - (1 - r2_e1) * (n_e1 - 1) / (n_e1 - p_e1 - 1)

exp1_results['MLP'] = {'r2': r2_e1, 'adj_r2': adj_r2_e1, 'n': n_e1}
mlp_label = f"MLP (epochs={BEST_EPOCH})"
print(f"  {mlp_label:<35} {n_e1:>8,} {r2_e1:>8.4f} {adj_r2_e1:>8.4f}")
print(f"{'='*60}")

# =============================================================================
# EXPERIMENT 2: Average test R² over 50 replications
# =============================================================================

print("\n" + "="*60)
print("EXPERIMENT 2 — Avg Test R² (50 replications, 80/20 split)")
print("="*60)

ols_r2_accum = {name: [] for name in FEATURE_SETS}
mlp_r2_list  = []

for rep in range(N_REPLICATIONS):

    # OLS
    for model_name, features in FEATURE_SETS.items():
        subset = ols_subsets[model_name]
        X_all  = subset[features].values
        y_all  = subset[TARGET].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all,
            test_size=TEST_SIZE,
            random_state=rep
        )
        X_tr_c = sm.add_constant(pd.DataFrame(X_tr, columns=features))
        X_te_c = sm.add_constant(pd.DataFrame(X_te, columns=features))
        fit    = sm.OLS(y_tr, X_tr_c).fit()
        y_pred = fit.predict(X_te_c)
        ols_r2_accum[model_name].append(r2_score(y_te, y_pred))

    # MLP
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nn_all, y_nn_all,
        test_size=TEST_SIZE,
        random_state=rep
    )
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    mlp = MLPRegressor(
        hidden_layer_sizes = (64, 32, 16),
        activation         = 'relu',
        solver             = 'adam',
        max_iter           = BEST_EPOCH,
        random_state       = 42,
    )
    mlp.fit(X_tr_sc, y_tr)
    y_pred = mlp.predict(X_te_sc)
    mlp_r2_list.append(r2_score(y_te, y_pred))

print(f"\n  {'Model':<35} {'Avg Test R²':>12} {'Std R²':>10}")
print(f"  {'-'*60}")
for model_name in FEATURE_SETS:
    vals = ols_r2_accum[model_name]
    print(f"  {model_name:<35} {np.mean(vals):>12.4f} {np.std(vals):>10.4f}")

mlp_label = f"MLP (epochs={BEST_EPOCH})"
print(f"  {mlp_label:<35} {np.mean(mlp_r2_list):>12.4f} {np.std(mlp_r2_list):>10.4f}")
print(f"{'='*60}")

# =============================================================================
# FINAL COMPARISON: Experiments 1 and 2 side by side
# =============================================================================

print("\n" + "="*75)
print("FINAL COMPARISON — In-Sample R² (Exp 1) vs Avg Test R² (Exp 2)")
print("="*75)
print(f"  {'Model':<35} {'Exp1 R²':>10} {'Exp2 Avg R²':>12} {'Drop':>8}")
print(f"  {'-'*68}")

for name in FEATURE_SETS:
    r2_in  = exp1_results[name]['r2']
    r2_out = np.mean(ols_r2_accum[name])
    print(f"  {name:<35} {r2_in:>10.4f} {r2_out:>12.4f} {r2_in - r2_out:>8.4f}")

r2_in  = exp1_results['MLP']['r2']
r2_out = np.mean(mlp_r2_list)
label  = f"MLP (epochs={BEST_EPOCH})"
print(f"  {label:<35} {r2_in:>10.4f} {r2_out:>12.4f} {r2_in - r2_out:>8.4f}")

print("="*75)
print("\nDone.")
