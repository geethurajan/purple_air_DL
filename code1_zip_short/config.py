"""
Shared configuration / constants for the wildfire PM2.5 excess-mortality pipeline.

Edit these in one place rather than hunting through every module.
"""

# ---- Replication counts -----------------------------------------------
N_TIME_BLOCK_SPLITS = 50   # Algorithm 1: number of time-block train/test splits
N_BOOTSTRAP = 50           # Algorithm 2: number of bootstrap resamples
N_MLP_RETRAIN = 50         # Algorithm 3: number of MLP retraining replicates

# ---- Feature columns ----------------------------------------------------
# "Current hour" PurpleAir features
CURRENT_FEATURES = ["pa_pm25", "pa_temp", "pa_humidity", "pa_pressure"]

# Lagged (t-1) meteorological features constructed from the current features
LAGGED_FEATURES = ["pa_temp_lag1", "pa_humidity_lag1", "pa_pressure_lag1"]

ALL_FEATURES = CURRENT_FEATURES + LAGGED_FEATURES  # 7 predictors total

TARGET_COL = "epa_pm25"        # EPA reference monitor reading
TIMESTAMP_COL = "timestamp"

# ---- MLP architecture (Algorithm 1, step 7 / Algorithm 3) --------------
MLP_HIDDEN_LAYERS = [128, 64, 32]
MLP_DROPOUT_RATE = 0.3
MLP_EPOCHS = 200
MLP_BATCH_SIZE = 32
MLP_EARLY_STOPPING_PATIENCE = 10

# ---- Confidence interval z-value ---------------------------------------
Z_95 = 1.96

# ---- Thin plate spline smoothing penalty (paper Eq. 6) -------------------
# The source paper fixes this at 1e-3 for BOTH the pre-fire and during-fire
# periods "as this choice was found to be a good compromise between
# flexibility of the model and smooth spatial patterns." Fit/predict must
# happen in a planar (km) coordinate system, not raw lat/lon degrees.
SPATIAL_SMOOTHING_LAMBDA = 1e-3

# ---- LA projection reference point ---------------------------------------
# Inferred directly from la_city_tracts_final.csv's x_km/y_km columns
# (which correlate ~1.0 with a simple equirectangular projection centered
# almost exactly on the LA_Site_1103 EPA monitor at 34.06659 N, -118.22688 W).
# Reuse the SAME reference point + scale factors for sensors and the EPA
# site so every spatial coordinate lands in one consistent system.
LA_PROJECTION_REF_LAT = 34.06659
LA_PROJECTION_REF_LON = -118.22688
LA_KM_PER_DEG_LAT = 111.1949268618148
LA_KM_PER_DEG_LON = 92.03388147528278

# ---- Exposure-response function (Atkinson et al. 2014, North America) ----
# gamma: excess mortality per unit (ug/m^3) increase in PM2.5, all-age/
# all-cause mortality, as used in the source paper.
GAMMA_POINT = 0.00094
GAMMA_LO = 0.00073
GAMMA_HI = 0.00116

# ---- Baseline mortality (GBD 2017, non-communicable disease, US) ---------
# 780 excess deaths per 100,000 people per year, as used in the source paper.
GBD_BASELINE_MORTALITY_PER_100K_PER_YEAR = 780