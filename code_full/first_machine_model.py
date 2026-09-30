import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# 1. LOAD FILES
# ==========================================================

# Change these to point to YOUR data files
epa_file = "Site_4002.csv"
purpleair_file = "sensor_data.csv"

epa = pd.read_csv(epa_file)

pa = pd.read_csv(purpleair_file)

# ==========================================================
# 2. CREATE DATETIME
# ==========================================================

epa["datetime"] = pd.to_datetime(
    epa["Date GMT"] + " " + epa["Time GMT"]
)

pa["datetime"] = pd.to_datetime(
    pa["time_stamp"],
    utc=True
).dt.tz_localize(None)

# ==========================================================
# 3. SELECT COLUMNS
# ==========================================================

epa = epa[
    [
        "datetime",
        "Sample Measurement"
    ]
]

epa = epa.rename(
    columns={
        "Sample Measurement": "EPA_PM25"
    }
)

pa = pa[
    [
        "datetime",

        "pm2.5_cf_1",

        "humidity",

        "temperature",

        "pressure"
    ]
]

# ==========================================================
# 4. MERGE DATA
# ==========================================================

merged = pd.merge(
    epa,
    pa,
    on="datetime",
    how="inner"
)

merged = merged.dropna()

print("Merged rows:", len(merged))

# ==========================================================
# ==========================================================
# METHOD 1:
# LOG-LINEAR REGRESSION
#
# log(EPA_t)
# =
# beta0
# + beta_PM * log(PA_PM_t)
# + beta_T * T_t
# + beta_RH * RH_t
# + epsilon_t
# ==========================================================
# ==========================================================

print("\n===================================")
print("LOG-LINEAR REGRESSION")
print("===================================")

eps = 1e-6

# Remove nonpositive values
model_data = merged[
    (merged["EPA_PM25"] > 0) &
    (merged["pm2.5_cf_1"] > 0)
].copy()

# Log transforms
model_data["log_EPA"] = np.log(
    model_data["EPA_PM25"] + eps
)

model_data["log_PA_PM"] = np.log(
    model_data["pm2.5_cf_1"] + eps
)

# ----------------------------------------------------------
# FEATURES
# ----------------------------------------------------------

X_linear = model_data[
    [
        "log_PA_PM",
        "temperature",
        "humidity",
        "pressure"
    ]
]

y_linear = model_data["log_EPA"]

# ----------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear,
    y_linear,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# SCALE
# ----------------------------------------------------------

scaler_linear = StandardScaler()

# lookup later

X_train_lin_scaled = scaler_linear.fit_transform(
    X_train_lin
)

X_test_lin_scaled = scaler_linear.transform(
    X_test_lin
)

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(
    X_train_lin_scaled,
    y_train_lin
)

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

y_pred_log = linear_model.predict(
    X_test_lin_scaled
)

# Convert back to EPA scale
y_test_linear = np.exp(y_test_lin)

y_pred_linear = np.exp(y_pred_log)

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

rmse_linear = np.sqrt(
    mean_squared_error(
        y_test_linear,
        y_pred_linear
    )
)

r2_linear = r2_score(
    y_test_linear,
    y_pred_linear
)

n_linear = len(y_test_linear)

p_linear = X_linear.shape[1]

adjusted_r2_linear = (
    1 -
    (1 - r2_linear) *
    (n_linear - 1) /
    (n_linear - p_linear - 1)
)

print("RMSE:", rmse_linear)

print("R²:", r2_linear)

print("Adjusted R²:", adjusted_r2_linear)

# ----------------------------------------------------------
# COEFFICIENTS
# ----------------------------------------------------------

coef_df = pd.DataFrame({

    "Variable": [

        "Intercept",

        "log_PA_PM",

        "Temperature",

        "Humidity",

        "Pressure"
    ],

    "Coefficient": [

        linear_model.intercept_,

        *linear_model.coef_
    ]
})

print("\nRegression Coefficients")

print(coef_df)
