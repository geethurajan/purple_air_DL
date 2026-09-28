import pandas as pd
import requests
import math
import time

# ==========================
# Configuration
# ==========================
API_KEY = "8F17EA16-5541-11F1-B596-4201AC1DC123"
INPUT_CSV = "Sensors + coordinates - Sheet1.csv"
OUTPUT_CSV = "Sensors_with_coordinates.csv"

BATCH_SIZE = 100

df = pd.read_csv(INPUT_CSV)
df["sensor"] = df["sensor"].astype(int)

# Create/overwrite lat/lon columns
df["lat"] = pd.NA
df["lon"] = pd.NA

url = "https://api.purpleair.com/v1/sensors"
headers = {"X-API-Key": API_KEY}

# Lookup table: sensor_index -> (lat, lon)
coords = {}

sensor_ids = df["sensor"].tolist()

for i in range(0, len(sensor_ids), BATCH_SIZE):
    batch = sensor_ids[i:i+BATCH_SIZE]

    params = {
        "show_only": ",".join(map(str, batch)),
        "fields": "latitude,longitude"
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()

    result = r.json()

    fields = result["fields"]
    idx_sensor = fields.index("sensor_index")
    idx_lat = fields.index("latitude")
    idx_lon = fields.index("longitude")

    for row in result["data"]:
        sensor = row[idx_sensor]
        lat = row[idx_lat]
        lon = row[idx_lon]
        coords[sensor] = (lat, lon)

    print(f"Processed {min(i+BATCH_SIZE, len(sensor_ids))}/{len(sensor_ids)} sensors")

    # Small pause to be polite to the API
    time.sleep(0.25)

# Fill dataframe
df["lat"] = df["sensor"].map(lambda s: coords.get(s, (None, None))[0])
df["lon"] = df["sensor"].map(lambda s: coords.get(s, (None, None))[1])

# Save updated CSV
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone! Saved to {OUTPUT_CSV}")