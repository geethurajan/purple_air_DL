import csv
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

epa_file = "Site_4002.csv"

start_date = datetime(2025, 7, 1, 0, 0, 0)
end_date   = datetime(2025, 7, 7, 23, 59, 59)

measurements = []

with open(epa_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        date_str = row["Date Local"].strip()
        time_str = row["Time Local"].strip()

        # Parse datetime (HH:MM:SS)
        dt = datetime.strptime(date_str + " " + time_str, "%m/%d/%Y %H:%M:%S")

        # Filter by date range
        if start_date <= dt <= end_date:
            val = row["Sample Measurement"].strip()
            if val != "":
                measurements.append(float(val))

measurements = np.array(measurements)

sensor_file = "sensor_data.csv"

humidity = []
temperature = []
pressure = []
pm25_cf1 = []

with open(sensor_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        humidity.append(float(row["humidity"]))
        temperature.append(float(row["temperature"]))
        pressure.append(float(row["pressure"]))
        pm25_cf1.append(float(row["pm2.5_cf_1"]))

humidity = np.array(humidity)
temperature = np.array(temperature)
pressure = np.array(pressure)
pm25_cf1 = np.array(pm25_cf1)

y = measurements

X = np.column_stack([pm25_cf1, humidity, pressure, temperature])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

R2 = model.score(X, y)

n = len(y)
p = X.shape[1]

adj_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R^2:", R2)
print("Adjusted R^2:", adj_R2)

residuals = y - y_pred

plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)

plt.title("Q-Q Plot of Regression Residuals")
plt.grid(True)

plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.show()

X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

fitted = model_sm.fittedvalues

standardized_residuals = (
    model_sm.get_influence()
            .resid_studentized_internal
)

sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))

plt.figure(figsize=(8,6))
plt.scatter(fitted, sqrt_abs_std_resid, alpha=0.6)

plt.xlabel("Fitted Values")
plt.ylabel("√|Standardized Residuals|")
plt.title("Scale-Location Plot")
plt.grid(True)
plt.show()

influence = model_sm.get_influence()

leverage = influence.hat_matrix_diag
standardized_residuals = influence.resid_studentized_internal
cooks = influence.cooks_distance[0]

plt.figure(figsize=(8,6))

plt.scatter(
    leverage,
    standardized_residuals,
    s=1000 * cooks,
    alpha=0.5
)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Leverage")
plt.ylabel("Standardized Residuals")
plt.title("Residuals vs Leverage (bubble size = Cook's Distance)")
plt.grid(True)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

variables = {
    "Purple Air PM2.5 (pm2.5_cf_1)": pm25_cf1,
    "Humidity (%)": humidity,
    "Pressure (hPa)": pressure,
    "Temperature (°F)": temperature
}

for i, (name, x_val) in enumerate(variables.items()):
    ax = axes[i]
    ax.scatter(x_val, y, alpha=0.6, color='royalblue', edgecolors='none', label='Observations')

    slope, intercept = np.polyfit(x_val, y, 1)
    x_line = np.linspace(x_val.min(), x_val.max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, color='crimson', linestyle='--', linewidth=2, 
            label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

    ax.set_xlabel(name, fontsize=11)
    ax.set_ylabel("EPA PM2.5 Measurement", fontsize=11)
    ax.set_title(f"EPA PM2.5 vs {name}", fontsize=12, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("y_vs_individual_variables.png", dpi=300)