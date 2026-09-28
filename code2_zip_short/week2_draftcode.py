import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
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

# ==========================================================
# ==========================================================
# METHOD 2:
# STANDARD MLP
# ==========================================================
# ==========================================================

print("\n===================================")
print("STANDARD MLP")
print("===================================")

feature_cols = [

    "pm2.5_cf_1",

    "humidity",

    "temperature",

    "pressure"
]

X = merged[feature_cols]

y = merged["EPA_PM25"]

# ----------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------------------------
# SCALE
# ----------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# MODEL
# ----------------------------------------------------------

mlp_model = MLPRegressor(

    hidden_layer_sizes=(64,32,16),

    activation="relu",

    solver="adam",

    max_iter=500,

    random_state=42
)

# ----------------------------------------------------------
# TRAIN
# ----------------------------------------------------------

mlp_model.fit(
    X_train_scaled,
    y_train
)

# ----------------------------------------------------------
# PREDICT
# ----------------------------------------------------------

y_pred_mlp = mlp_model.predict(
    X_test_scaled
)

# ----------------------------------------------------------
# METRICS
# ----------------------------------------------------------

rmse_mlp = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_mlp
    )
)

r2_mlp = r2_score(
    y_test,
    y_pred_mlp
)

n_mlp = len(y_test)

p_mlp = X.shape[1]

adjusted_r2_mlp = (
    1 -
    (1 - r2_mlp) *
    (n_mlp - 1) /
    (n_mlp - p_mlp - 1)
)

print("RMSE:", rmse_mlp)

print("R²:", r2_mlp)

print("Adjusted R²:", adjusted_r2_mlp)

# ==========================================================
# ==========================================================
# METHOD 3:
# FULLY CONNECTED DEEP LEARNING (log scale)
#
# log(EPA_t)
# =
# f( log(PA_PM_t), T_t, RH_t, P_t )
# where f is a deep FC network
# ==========================================================
# ==========================================================

print("\n===================================")
print("FC DEEP LEARNING (log scale)")
print("===================================")

deep_data = merged[
    (merged["EPA_PM25"] > 0) &
    (merged["pm2.5_cf_1"] > 0)
].copy()

deep_data["log_EPA"] = np.log(deep_data["EPA_PM25"] + eps)

deep_data["log_PA_PM"] = np.log(deep_data["pm2.5_cf_1"] + eps)

X_deep = deep_data[
    [
        "log_PA_PM",
        "temperature",
        "humidity",
        "pressure"
    ]
]

y_deep = deep_data["log_EPA"]

X_train_deep, X_test_deep, y_train_deep, y_test_deep = train_test_split(
    X_deep,
    y_deep,
    test_size=0.2,
    random_state=42
)

scaler_deep = StandardScaler()

X_train_deep_scaled = scaler_deep.fit_transform(X_train_deep)

X_test_deep_scaled = scaler_deep.transform(X_test_deep)

deep_model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

deep_model.fit(X_train_deep_scaled, y_train_deep)

y_pred_log_deep = deep_model.predict(X_test_deep_scaled)

# Back to original scale
y_test_deep_orig = np.exp(y_test_deep)
y_pred_deep_orig = np.exp(y_pred_log_deep)

# Metrics on log scale
rmse_deep_log = np.sqrt(mean_squared_error(y_test_deep, y_pred_log_deep))
r2_deep_log = r2_score(y_test_deep, y_pred_log_deep)
n_deep = len(y_test_deep)
p_deep = X_deep.shape[1]
adj_r2_deep_log = 1 - (1 - r2_deep_log) * (n_deep - 1) / (n_deep - p_deep - 1)

# Metrics on original scale
rmse_deep_orig = np.sqrt(mean_squared_error(y_test_deep_orig, y_pred_deep_orig))
r2_deep_orig = r2_score(y_test_deep_orig, y_pred_deep_orig)
adj_r2_deep_orig = 1 - (1 - r2_deep_orig) * (n_deep - 1) / (n_deep - p_deep - 1)

print("--- Log scale ---")
print("RMSE:", rmse_deep_log)
print("R²:", r2_deep_log)
print("Adjusted R²:", adj_r2_deep_log)
print("\n--- Original scale (exp back-transformed) ---")
print("RMSE:", rmse_deep_orig)
print("R²:", r2_deep_orig)
print("Adjusted R²:", adj_r2_deep_orig)

# ==========================================================
# FINAL COMPARISON TABLE
# ==========================================================

results = pd.DataFrame({

    "Method": [
        "Log-Linear Regression",
        "Standard MLP",
        "FC Deep Learning (log)"
    ],

    "RMSE": [
        rmse_linear,
        rmse_mlp,
        rmse_deep_orig
    ],

    "R2": [
        r2_linear,
        r2_mlp,
        r2_deep_orig
    ],

    "Adjusted_R2": [
        adjusted_r2_linear,
        adjusted_r2_mlp,
        adj_r2_deep_orig
    ]
})

print("\n===================================")
print("FINAL COMPARISON")
print("===================================")

print(results)

# ==========================================================
# PLOT TRUE VS PREDICTED (all three methods)
# ==========================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

datasets = [
    (y_test_linear,    y_pred_linear,    "Log-Linear Regression",    axes[0]),
    (y_test,           y_pred_mlp,       "Standard MLP",             axes[1]),
    (y_test_deep_orig, y_pred_deep_orig, "FC Deep Learning (log)",   axes[2]),
]

for y_true, y_pred, title, ax in datasets:
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.5)
    ax.set_xlabel("True EPA PM2.5")
    ax.set_ylabel("Predicted EPA PM2.5")
    ax.set_title(title)

plt.suptitle("EPA PM2.5 Prediction Comparison — True vs Predicted", fontsize=13)

plt.tight_layout()

plt.savefig("mlp_vs_log_lin.png", dpi=150)

plt.show()
