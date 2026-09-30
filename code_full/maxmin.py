import pandas as pd

# Load the data
df = pd.read_csv("sensor_94397_history.csv")

# Find min and max of pm2.5_cf_!
min_value = df["humidity"].min()
max_value = df["humidity"].max()

print(f"Minimum pm2.5_cf_1: {min_value}")
print(f"Maximum pm2.5_cf_1: {max_value}")
