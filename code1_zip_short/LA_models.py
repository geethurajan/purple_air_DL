import pandas as pd

sensor_info = pd.read_csv("Sensors - Sheet1.csv")

sensor_data = {}
all_data = []

for _, row in sensor_info.iterrows():

    sensor_num = int(row["Sensor Num"])
    sensor_id = int(row["Sensor ID"])
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    filepath = (
        f"PurpleAir Download Sensor {sensor_num}/"
        f"{sensor_id} 2025-01-01 2025-03-01 60-Minute Average.csv"
    )

    try:
        df = pd.read_csv(filepath)

        # Add sensor information
        df["Sensor Num"] = sensor_num
        df["Sensor ID"] = sensor_id
        df["Latitude"] = latitude
        df["Longitude"] = longitude

        sensor_data[sensor_id] = df
        all_data.append(df)

    except FileNotFoundError:
        print(f"Missing file for Sensor {sensor_num}")

combined_df = pd.concat(all_data, ignore_index=True)

print(combined_df.head())
print(f"Loaded {len(sensor_data)} sensors.")

print(combined_df.columns)