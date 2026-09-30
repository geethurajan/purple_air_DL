import os
import glob
from math import radians, sin, cos, sqrt, asin

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_SENSORS = []

MODEL_CFG = dict(
    hidden=64,
    r_dim=32,
    lr=1e-3,
    weight_decay=1e-5,
    max_epochs=400,
    batch_episodes=16,        # gradient step every N episodes
    min_context_frac=0.3,     # random context size, as a fraction of that hour's sensors
    max_context_frac=0.9,
    min_context=5,            # skip hours with fewer than this many reporting sensors
    val_frac=0.15,
    patience=40,
    random_seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

torch.manual_seed(MODEL_CFG["random_seed"])
np.random.seed(MODEL_CFG["random_seed"])

# =============================================================================
# 1. HAVERSINE / LOCAL KM COORDINATES
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def latlon_to_local_km(lat, lon, ref_lat, ref_lon):
    """Flat-earth x/y in km relative to a fixed reference point. Fine over ~10 km."""
    R = 6371.0
    lat_rad, ref_lat_rad = np.radians(lat), np.radians(ref_lat)
    dlat = np.radians(np.asarray(lat) - ref_lat)
    dlon = np.radians(np.asarray(lon) - ref_lon)
    x = dlon * np.cos((lat_rad + ref_lat_rad) / 2) * R
    y = dlat * R
    return x, y


# =============================================================================
# 2. LOAD SENSOR METADATA + EPA PM2.5 (EPA used only for scoring, at the end)
# =============================================================================

sensors_df = pd.read_csv(os.path.join(BASE_DIR, 'Sensors.csv'))
sensors_df.columns = sensors_df.columns.str.strip().str.lower().str.replace(' ', '_')

_epa_raw = pd.read_csv(os.path.join(BASE_DIR, 'LA_Site_1103.csv'))
_epa_raw.columns = _epa_raw.columns.str.strip()
EPA_LAT = _epa_raw['Latitude'].iloc[0]
EPA_LON = _epa_raw['Longitude'].iloc[0]

epa_df = _epa_raw.copy()
epa_df['timestamp'] = pd.to_datetime(epa_df['Date Local'].astype(str) + ' ' + epa_df['Time Local'].astype(str))
epa_df = epa_df[['timestamp', 'Sample Measurement']].rename(columns={'Sample Measurement': 'pm25_epa'})
epa_df = epa_df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
del _epa_raw

# Restrict EPA data to the study window -- otherwise "scored against N EPA hours
# out of M available" counts months of EPA data outside the period we have any
# PurpleAir coverage for at all.
STUDY_START = pd.Timestamp('2025-01-01')
STUDY_END = pd.Timestamp('2025-03-01')
epa_df = epa_df[(epa_df['timestamp'] >= STUDY_START) & (epa_df['timestamp'] < STUDY_END)].reset_index(drop=True)

# =============================================================================
# 3. LOCAL KM COORDINATES FOR EVERY SENSOR (relative to the EPA site, just as
#    a fixed, convenient origin for storage -- the model itself only ever
#    sees relative displacements between context and query, computed later)
# =============================================================================

sensors_df['x_km'], sensors_df['y_km'] = latlon_to_local_km(
    sensors_df['latitude'], sensors_df['longitude'], EPA_LAT, EPA_LON
)

# =============================================================================
# 4. LOAD PURPLEAIR FILES (location + PM2.5 only)
# =============================================================================

pa_records = []
for _, meta_row in sensors_df.iterrows():
    sensor_num = meta_row['sensor_num']
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

    pa.columns = pa.columns.str.strip()
    pa['timestamp'] = pd.to_datetime(pa['time_stamp']).dt.tz_localize(None)
    pa = pa.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    pa['sensor_num'] = int(sensor_num)
    pa['x_km'] = meta_row['x_km']
    pa['y_km'] = meta_row['y_km']
    pa_records.append(pa)

print(f"Loaded {len(pa_records)} sensors.")
if not pa_records:
    raise RuntimeError("No PurpleAir data loaded. Check folders and EXCLUDE_SENSORS.")

# =============================================================================
# 5. ASSEMBLE PANEL (location + PM2.5 only; EPA merged in for scoring later)
# =============================================================================

panel = pd.concat(pa_records, ignore_index=True)
panel = panel.rename(columns={'pm2.5_atm': 'pm25_pa_atm', 'pm2.5_cf_1': 'pm25_pa_cf1'})

keep = ['timestamp', 'sensor_num', 'x_km', 'y_km', 'pm25_pa_cf1']
panel = panel[[c for c in keep if c in panel.columns]].copy()
panel['pm25_pa_cf1'] = panel['pm25_pa_cf1'].clip(lower=0)
panel = panel.dropna(subset=['x_km', 'y_km', 'pm25_pa_cf1']).reset_index(drop=True)

print(f"Panel assembled: {len(panel):,} rows across {panel['timestamp'].nunique():,} hours.")

# =============================================================================
# 6. NORMALIZATION STATISTICS (fit on the PurpleAir network only)
# =============================================================================

loc_mean = panel[['x_km', 'y_km']].values.mean(axis=0)
loc_std = panel[['x_km', 'y_km']].values.std(axis=0) + 1e-6
pm25_log_mean = np.log1p(panel['pm25_pa_cf1'].values).mean()
pm25_log_std = np.log1p(panel['pm25_pa_cf1'].values).std() + 1e-6


def norm_loc(xy):
    return (xy - loc_mean) / loc_std


def norm_pm25_log(p_log):
    return (p_log - pm25_log_mean) / pm25_log_std


def denorm_pm25_log(p_log_norm):
    return p_log_norm * pm25_log_std + pm25_log_mean


# =============================================================================
# 7. BUILD PER-HOUR EPISODES (context = every sensor reporting that hour)
# =============================================================================

def build_episodes(panel_df, min_context):
    episodes = []
    for ts, group in panel_df.groupby('timestamp'):
        if len(group) < min_context + 1:
            continue
        xy = norm_loc(group[['x_km', 'y_km']].values.astype(np.float32))
        pm25_log = norm_pm25_log(np.log1p(group['pm25_pa_cf1'].values.astype(np.float32)))
        episodes.append(dict(timestamp=ts, xy=xy, pm25_log=pm25_log))
    return episodes


episodes = build_episodes(panel, MODEL_CFG["min_context"])
print(f"Built {len(episodes)} usable hourly episodes.")

# =============================================================================
# 8. MODEL
# =============================================================================

class SetEncoder(nn.Module):
    def __init__(self, in_dim, hidden, r_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, r_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, r_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(r_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),  # mean, log_var (normalized log1p-PM2.5 space)
        )

    def forward(self, r):
        out = self.net(r)
        mu, log_var = out[..., 0], out[..., 1]
        log_var = torch.clamp(log_var, min=-6.0, max=6.0)
        return mu, log_var


class SpatialNeuralProcess(nn.Module):
    """Context: (xy, pm25_log) pairs for one hour. Query: one or more target
    locations for that same hour. Displacement is computed target-relative,
    so the trained model can be queried anywhere, at any hour that has
    reporting context sensors -- not just at one fixed point."""

    def __init__(self, hidden, r_dim):
        super().__init__()
        enc_in_dim = 2 + 1  # (dx,dy) + pm25_log
        self.encoder = SetEncoder(enc_in_dim, hidden, r_dim)
        self.decoder = Decoder(r_dim, hidden)

    def forward(self, xy_c, pm25_c, xy_t):
        # xy_c: (Nc,2)  pm25_c: (Nc,)   xy_t: (Nt,2)
        Nc, Nt = xy_c.shape[0], xy_t.shape[0]
        rel = xy_c.unsqueeze(0) - xy_t.unsqueeze(1)             # (Nt,Nc,2)
        pm25_c_exp = pm25_c.view(1, Nc, 1).expand(Nt, Nc, 1)
        enc_in = torch.cat([rel, pm25_c_exp], dim=-1).reshape(Nt * Nc, -1)
        r_i = self.encoder(enc_in).view(Nt, Nc, -1)
        r = r_i.mean(dim=1)                                      # (Nt, r_dim)
        return self.decoder(r)


def gaussian_nll(mu, log_var, y):
    var = torch.exp(log_var)
    return 0.5 * (np.log(2 * np.pi) + log_var + (y - mu) ** 2 / var)


# =============================================================================
# 9. TRAINING LOOP
# =============================================================================

def split_context_target(n_total, cfg):
    frac = np.random.uniform(cfg["min_context_frac"], cfg["max_context_frac"])
    n_ctx = max(cfg["min_context"], int(frac * n_total))
    n_ctx = min(n_ctx, n_total - 1)
    perm = np.random.permutation(n_total)
    return perm[:n_ctx], perm[n_ctx:]


def episode_loss(model, ep, device, cfg, fixed_frac=None):
    n_total = ep['xy'].shape[0]
    if fixed_frac is None:
        ctx_idx, tgt_idx = split_context_target(n_total, cfg)
    else:
        n_ctx = max(cfg["min_context"], int(fixed_frac * n_total))
        n_ctx = min(n_ctx, n_total - 1)
        perm = np.random.permutation(n_total)
        ctx_idx, tgt_idx = perm[:n_ctx], perm[n_ctx:]

    xy = torch.tensor(ep['xy'], dtype=torch.float32, device=device)
    pm25_log = torch.tensor(ep['pm25_log'], dtype=torch.float32, device=device)

    mu, log_var = model(xy[ctx_idx], pm25_log[ctx_idx], xy[tgt_idx])
    return gaussian_nll(mu, log_var, pm25_log[tgt_idx]).mean()


def train_cnp(episodes, cfg):
    device = cfg["device"]
    model = SpatialNeuralProcess(hidden=cfg["hidden"], r_dim=cfg["r_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    n = len(episodes)
    idx = np.random.permutation(n)
    n_val = int(n * cfg["val_frac"])
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    best_val, best_state, patience_left = np.inf, None, cfg["patience"]

    for epoch in range(cfg["max_epochs"]):
        model.train()
        np.random.shuffle(train_idx)
        optimizer.zero_grad()
        for step, i in enumerate(train_idx):
            loss = episode_loss(model, episodes[i], device, cfg)
            (loss / cfg["batch_episodes"]).backward()
            if (step + 1) % cfg["batch_episodes"] == 0:
                optimizer.step()
                optimizer.zero_grad()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in val_idx:
                val_losses.append(episode_loss(model, episodes[i], device, cfg, fixed_frac=0.7).item())
        val_loss = float(np.mean(val_losses)) if val_losses else np.inf

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}, best val NLL {best_val:.4f}")
                break

        if epoch % 5 == 0:
            print(f"epoch {epoch:3d}  val NLL {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


model = train_cnp(episodes, MODEL_CFG)
torch.save(model.state_dict(), os.path.join(BASE_DIR, "spatial_np_state_dict.pt"))

# =============================================================================
# 10. EVALUATE AT THE EPA SITE -- the only place EPA data is used
# =============================================================================
# For every hour the EPA has a reading AND we have a built episode (i.e.
# enough PurpleAir sensors reported that hour), query the trained model at
# the EPA's location using that hour's reporting sensors as context, and
# compare against the real EPA reading. EPA never touched training.

device = MODEL_CFG["device"]
model.eval()

episodes_by_time = {ep['timestamp']: ep for ep in episodes}
epa_target_xy_norm = norm_loc(np.array([0.0, 0.0], dtype=np.float32))  # EPA site is the coordinate origin

pred_means, epa_actuals, eval_timestamps = [], [], []
with torch.no_grad():
    for _, row in epa_df.iterrows():
        ts = row['timestamp']
        if ts not in episodes_by_time:
            continue
        ep = episodes_by_time[ts]

        xy = torch.tensor(ep['xy'], dtype=torch.float32, device=device)
        pm25_log = torch.tensor(ep['pm25_log'], dtype=torch.float32, device=device)
        target_xy = torch.tensor(epa_target_xy_norm, dtype=torch.float32, device=device).unsqueeze(0)

        mu, log_var = model(xy, pm25_log, target_xy)
        mu_log = denorm_pm25_log(mu.item())

        pred_means.append(np.expm1(mu_log))
        epa_actuals.append(row['pm25_epa'])
        eval_timestamps.append(ts)

pred_means = np.array(pred_means)
epa_actuals = np.array(epa_actuals)
print(f"\nScored against {len(epa_actuals)} EPA hours (out of {len(epa_df)} available).")

# =============================================================================
# 10b. DIAGNOSTIC: WHEN ARE THE MISSING EPA HOURS? (random gaps vs. clustered
# during the fire period, which would matter a lot more for the evaluation)
# =============================================================================

full_range = pd.date_range(STUDY_START, STUDY_END, freq='h')
full_range = full_range[full_range < STUDY_END]

epa_present_ts = set(epa_df['timestamp'])
episode_ts = set(episodes_by_time.keys())

missing_epa = [ts for ts in full_range if ts not in epa_present_ts]
missing_epa_with_pa_context = [ts for ts in missing_epa if ts in episode_ts]

print(f"\nEPA missing hours in study window: {len(missing_epa)} / {len(full_range)}")
print(f"Of those, PurpleAir had enough context sensors reporting anyway: "
      f"{len(missing_epa_with_pa_context)}")

if missing_epa:
    missing_df = pd.DataFrame({'timestamp': missing_epa})
    missing_df['date'] = missing_df['timestamp'].dt.date
    missing_df['hour'] = missing_df['timestamp'].dt.hour

    daily_missing = missing_df.groupby('date').size()
    hourly_missing = missing_df.groupby('hour').size()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].bar([str(d) for d in daily_missing.index], daily_missing.values, color='indianred')
    axes[0].set_ylabel('Missing EPA hours')
    axes[0].set_title(f'Missing EPA hours per day, {STUDY_START.date()} to {STUDY_END.date()}')
    axes[0].tick_params(axis='x', rotation=90, labelsize=6)

    axes[1].bar(hourly_missing.index, hourly_missing.values, color='steelblue')
    axes[1].set_xlabel('Hour of day (0-23)')
    axes[1].set_ylabel('Missing count')
    axes[1].set_title('Missing EPA hours by hour-of-day (checks for routine QA-cycle clustering)')
    axes[1].set_xticks(range(0, 24))

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'epa_missing_diagnostic.png'), dpi=150)
    print("Saved missing-data diagnostic to epa_missing_diagnostic.png")
else:
    print("No missing EPA hours in the study window.")

# =============================================================================
# 11. METRICS (same definitions as Shen, Crippa & Castruccio 2021, Table 1)
# =============================================================================

def paper_metrics(epa_actual, epa_pred):
    epa_actual = np.asarray(epa_actual, dtype=float)
    epa_pred = np.asarray(epa_pred, dtype=float)
    mean_actual, mean_pred = epa_actual.mean(), epa_pred.mean()

    mae = np.mean(np.abs(epa_pred - epa_actual))
    if mean_pred >= mean_actual:
        nmbf = np.sum((epa_actual / np.sum(epa_actual)) * ((epa_pred - epa_actual) / epa_actual))
        nmaef = np.sum(np.abs(epa_pred - epa_actual)) / np.sum(epa_actual)
    else:
        nmbf = np.sum((epa_pred / np.sum(epa_pred)) * ((mean_pred - epa_actual) / epa_pred))
        nmaef = np.sum(np.abs(epa_pred - epa_actual)) / np.sum(epa_pred)

    ss_res = np.sum((epa_pred - epa_actual) ** 2)
    ss_tot = np.sum((epa_actual - mean_actual) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ioa_denom = np.sum((np.abs(epa_pred - mean_actual) + np.abs(epa_actual - mean_actual)) ** 2)
    ioa = 1 - ss_res / ioa_denom if ioa_denom > 0 else np.nan

    return dict(R2=r2, MAE=mae, NMBF=nmbf, NMAEF=nmaef, IOA=ioa, n=len(epa_actual))


metrics = paper_metrics(epa_actuals, pred_means)
print("\n=== Spatial Neural Process vs EPA (evaluation only, never trained on EPA) ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

plt.figure(figsize=(6, 6))
plt.scatter(epa_actuals, pred_means, alpha=0.4, s=15)
lims = [0, max(epa_actuals.max(), pred_means.max()) * 1.05]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("EPA PM2.5 (ug/m3)")
plt.ylabel("Neural Process predicted PM2.5 (ug/m3)")
plt.title("Neural Process vs EPA reference PM2.5")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "np_predicted_vs_epa.png"), dpi=150)
print("\nSaved scatter plot to np_predicted_vs_epa.png")

# =============================================================================
# 12. CALIBRATION: ln(EPA) ~ ln(NP pred) + temp + pressure + humidity
#     Met covariates come from the PurpleAir sensor nearest the EPA site.
# =============================================================================

import statsmodels.api as sm

sensors_df['dist_to_epa_km'] = np.sqrt(sensors_df['x_km'] ** 2 + sensors_df['y_km'] ** 2)
nearest_sensor_num = int(sensors_df.loc[sensors_df['dist_to_epa_km'].idxmin(), 'sensor_num'])
nearest_csv = glob.glob(os.path.join(BASE_DIR, f'PurpleAir Download Sensor {nearest_sensor_num}', '*.csv'))[0]

met_df = pd.read_csv(nearest_csv)
met_df.columns = met_df.columns.str.strip()
met_df['timestamp'] = pd.to_datetime(met_df['time_stamp']).dt.tz_localize(None)
met_df = met_df[['timestamp', 'temperature', 'pressure', 'humidity']].drop_duplicates('timestamp')

calib_df = pd.DataFrame({'timestamp': eval_timestamps, 'pm25_epa': epa_actuals, 'pm25_np_pred': pred_means})
calib_df = calib_df.merge(met_df, on='timestamp', how='left')
calib_df = calib_df[(calib_df['pm25_epa'] > 0) & (calib_df['pm25_np_pred'] > 0)]
calib_df = calib_df.dropna(subset=['temperature', 'pressure', 'humidity'])
calib_df['ln_epa'] = np.log(calib_df['pm25_epa'])
calib_df['ln_np_pred'] = np.log(calib_df['pm25_np_pred'])

feature_cols = ['ln_np_pred', 'temperature', 'pressure', 'humidity']
X = sm.add_constant(calib_df[feature_cols].values)
y = calib_df['ln_epa'].values

# ---- fit our own calibration on this data ----
fitted = sm.OLS(y, X).fit()
print(f"\nFitted calibration (n={len(calib_df)}) -- R^2: {fitted.rsquared:.4f}   "
      f"Adj. R^2: {fitted.rsquared_adj:.4f}")

# ---- compare against a fixed, externally-specified coefficient set ----
B0, B1, B2, B3, B4 = -31.758545792479087, 0.58881352, -0.00639275, -0.01259735, 0.03414593
ln_epa_fixed_pred = (B0 + B1 * calib_df['ln_np_pred'] + B2 * calib_df['temperature']
                      + B3 * calib_df['humidity'] + B4 * calib_df['pressure'])
ss_res = np.sum((y - ln_epa_fixed_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2_fixed = 1 - ss_res / ss_tot
print(f"Fixed-coefficient model (n={len(calib_df)}) -- R^2: {r2_fixed:.4f}")