"""
Keep only the hours of an hourly MAIA CSV that fall in a UTC date range.

Usage:
  python filter_utc.py maia_2025Q1_hourly.csv --start 2025-01-01 --end 2025-03-31 --out maia_2025Q1_hourly_utc.csv
(start and end days are both included, 00:00 to 23:00 UTC)
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
ap.add_argument("--end", required=True, help="last UTC day, YYYY-MM-DD")
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.csv, low_memory=False)
day = df["hour_UTC"].str[:10]
kept = df[(day >= args.start) & (day <= args.end)]
kept.to_csv(args.out, index=False)

print(f"Kept {len(kept)} of {len(df)} rows ({kept['hour_UTC'].min()} to {kept['hour_UTC'].max()} UTC)")
if "daily_pm_mass_concentration" in kept:
    print(f"Rows without a daily value: {kept['daily_pm_mass_concentration'].isna().sum()}")
