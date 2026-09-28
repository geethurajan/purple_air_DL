import csv
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

epa_file = "Site_4002.csv"
sensor_file = "164317 2025-07-01 2025-07-08 60-Minute Average.csv"

start_date = datetime(2025, 7, 1, 0, 0, 0)
end_date   = datetime(2025, 7, 7, 23, 59, 59)

epa_data = {}

with open(epa_file, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dt = datetime.strptime(
                row["Date GMT"].strip() + " " +
                row["Time GMT"].strip(),
                "%m/%d/%Y %H:%M:%S"
            )
            if not (start_date <= dt <= end_date):
                continue
            value = row["Sample Measurement"].strip()
            if value != "":
                epa_data[dt] = float(value)
        except Exception:
            continue

sensor_data = {}

with open(sensor_file, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            dt = datetime.strptime(
                row["time_stamp"].strip(),
                "%Y-%m-%dT%H:%M:%SZ"
            )
            sensor_data[dt] = {
                "pm25_cf1": float(row["pm2.5_cf_1"]),
                "humidity": float(row["humidity"]),
                "pressure": float(row["pressure"]),
                "temperature": float(row["temperature"])
            }
        except Exception:
            continue

common_times = sorted(
    set(epa_data.keys()).intersection(sensor_data.keys())
)

print("\nEPA observations:", len(epa_data))
print("Sensor observations:", len(sensor_data))
print("Matched observations:", len(common_times))

if len(common_times) == 0:
    print("\nFirst few EPA timestamps:")
    for t in sorted(list(epa_data.keys()))[:5]:
        print(t)
    print("\nFirst few Sensor timestamps:")
    for t in sorted(list(sensor_data.keys()))[:5]:
        print(t)
    raise ValueError(
        "No matching timestamps found. Check timestamp formats and time zones."
    )

X = []
y = []
for dt in common_times:
    s = sensor_data[dt]
    X.append([
        s["pm25_cf1"],
        s["humidity"],
        s["pressure"],
        s["temperature"]
    ])
    y.append(epa_data[dt])

X = np.array(X)
y = np.array(y)

print("\nRegression dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

R2 = model.score(X, y)

n = len(y)
p = X.shape[1]

if n > p + 1:
    adj_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
else:
    adj_R2 = float("nan")

print("\n================ RESULTS ================\n")
print("Intercept:")
print(model.intercept_)
print("\nCoefficients:")
print("pm25_cf_1   =", model.coef_[0])
print("humidity    =", model.coef_[1])
print("pressure    =", model.coef_[2])
print("temperature =", model.coef_[3])
print("\nR² =", R2)
print("Adjusted R² =", adj_R2)
print("\nNumber of matched observations =", n)

print("\nDate range of matched data:")
print("First match:", common_times[0])
print("Last match :", common_times[-1])
print("Total matches:", len(common_times))
print("\nAll matched observations:")
print("-" * 100)

for dt in common_times:

    s = sensor_data[dt]

    print(
        dt,
        "EPA =", epa_data[dt],
        "PM25_CF1 =", s["pm25_cf1"],
        "Humidity =", s["humidity"],
        "Pressure =", s["pressure"],
        "Temperature =", s["temperature"]
    )