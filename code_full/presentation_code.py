import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

epa_file = "Site_4002.csv"
purpleair_file = "sensor_data.csv"

epa = pd.read_csv(epa_file)
pa = pd.read_csv(purpleair_file)

epa["datetime"] = pd.to_datetime(
    epa["Date Local"] + " " + epa["Time Local"]
)

start_date = pd.Timestamp("2025-07-01 00:00:00")
end_date = pd.Timestamp("2025-07-07 23:00:00")

epa = epa[
    (epa["datetime"] >= start_date) &
    (epa["datetime"] <= end_date)
].copy()

pa["datetime"] = (
    pd.to_datetime(pa["time_stamp"])
      .dt.tz_localize(None)
)

epa = epa[
    [
        "datetime",
        "Sample Measurement"
    ]
].rename(
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

print("\nEPA Date Range")
print(epa["datetime"].min())
print(epa["datetime"].max())

print("\nPurpleAir Date Range")
print(pa["datetime"].min())
print(pa["datetime"].max())

merged = pd.merge(
    epa,
    pa,
    on="datetime",
    how="inner"
)

epa_only = epa[
    ~epa["datetime"].isin(pa["datetime"])
]

print("\nEPA timestamps with NO PurpleAir match:")
print(epa_only[["datetime"]])

print("\nCount:", len(epa_only))

pa_only = pa[
    ~pa["datetime"].isin(epa["datetime"])
]

print("\nPurpleAir timestamps with NO EPA match:")
print(pa_only[["datetime"]])

print("\nCount:", len(pa_only))

merged = merged.dropna()

print("\nMerged rows:", len(merged))

if len(merged) == 0:
    raise ValueError(
        "No matching timestamps found between EPA and PurpleAir data."
    )

model_data = merged[
    (merged["EPA_PM25"] > 0) &
    (merged["pm2.5_cf_1"] > 0)
].copy()

eps = 1e-6

model_data["log_EPA"] = np.log(
    model_data["EPA_PM25"] + eps
)

model_data["log_PA"] = np.log(
    model_data["pm2.5_cf_1"] + eps
)

print("\n===================================")
print("STANDARD LINEAR REGRESSION")
print("===================================")

X1 = model_data[
    [
        "pm2.5_cf_1",
        "temperature",
        "humidity",
        "pressure"
    ]
]

y1 = model_data["EPA_PM25"]

linear_model = LinearRegression()

linear_model.fit(X1, y1)

pred1 = linear_model.predict(X1)

r2_1 = r2_score(y1, pred1)

n1 = len(y1)
p1 = X1.shape[1]

adj_r2_1 = (
    1 -
    (1 - r2_1) *
    (n1 - 1) /
    (n1 - p1 - 1)
)

print("Intercept:", linear_model.intercept_)
print("Coefficients:", linear_model.coef_)
print("R²:", r2_1)
print("Adjusted R²:", adj_r2_1)

print("\n===================================")
print("CLASSICAL LOG-LINEAR")
print("===================================")

X2 = model_data[
    [
        "log_PA",
        "temperature",
        "humidity",
        "pressure"
    ]
]

y2 = model_data["log_EPA"]

log_model = LinearRegression()

log_model.fit(X2, y2)

pred2_log = log_model.predict(X2)

r2_2 = r2_score(y2, pred2_log)

n2 = len(y2)
p2 = X2.shape[1]

adj_r2_2 = (
    1 -
    (1 - r2_2) *
    (n2 - 1) /
    (n2 - p2 - 1)
)

pred2 = np.exp(pred2_log)

print("Intercept:", log_model.intercept_)
print("Coefficients:", log_model.coef_)
print("R²:", r2_2)
print("Adjusted R²:", adj_r2_2)

print("\n===================================")
print("ML LOG-LINEAR")
print("===================================")

X_ml = model_data[
    [
        "log_PA",
        "temperature",
        "humidity",
        "pressure"
    ]
]

y_ml = model_data["log_EPA"]

X_train, X_test, y_train, y_test = train_test_split(
    X_ml,
    y_ml,
    test_size=0.50,
    random_state=42
)

ml_model = LinearRegression()

ml_model.fit(
    X_train,
    y_train
)

y_pred_log = ml_model.predict(
    X_test
)

y_test_actual = np.exp(y_test)

y_pred_actual = np.exp(
    y_pred_log
)

r2_ml = r2_score(
    y_test_actual,
    y_pred_actual
)

n_ml = len(y_test_actual)
p_ml = X_ml.shape[1]

adj_r2_ml = (
    1 -
    (1-r2_ml)*(n_ml-1)/(n_ml-p_ml-1)
)

print("Intercept:", ml_model.intercept_)
print("Coefficients:", ml_model.coef_)
print("R²:", r2_ml)
print("Adjusted R²:", adj_r2_ml)

plot_df = model_data.sort_values("datetime").copy()

plot_df["pred_linear"] = linear_model.predict(
    plot_df[["pm2.5_cf_1", "temperature", "humidity", "pressure"]]
)

plot_df["pred_log"] = np.exp(
    log_model.predict(
        plot_df[["log_PA", "temperature", "humidity", "pressure"]]
    )
)

plot_df["pred_ml"] = np.exp(
    ml_model.predict(
        plot_df[["log_PA", "temperature", "humidity", "pressure"]]
    )
)

plt.figure(figsize=(15, 6))

plt.plot(
    plot_df["datetime"],
    plot_df["EPA_PM25"],
    label="EPA (Observed)",
    linewidth=2
)

plt.plot(
    plot_df["datetime"],
    plot_df["pred_linear"],
    label="Linear Model",
    alpha=0.8
)

plt.plot(
    plot_df["datetime"],
    plot_df["pred_log"],
    label="Log-Linear Model",
    alpha=0.8
)

plt.plot(
    plot_df["datetime"],
    plot_df["pred_ml"],
    label="ML Log-Linear Model",
    alpha=0.8
)

plt.xlabel("Time")
plt.ylabel("PM2.5 (µg/m³)")
plt.title("EPA vs PurpleAir Calibration Models (Raw Time Series)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()