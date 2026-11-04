"""
Download hourly PurpleAir history for January 2026 for the SAME sensors that were used for
training (Jan-Feb 2025, around the downtown LA EPA monitor 06-037-1103), so the models trained
on Jan-Feb 2025 can be tested on January 2026.

The sensor list comes from the training folder (code_full/Sensors_with_coordinates.csv); no new
sensors are searched for. By default every sensor that has data in the training folder
(Jan-Feb 2025) is downloaded, at any distance (192 sensors). --radius limits the list to the
sensors within that many km of site 1103. Nothing is written to the training folder.

Same fields and format as code_full/download_data.py:
    time_stamp (LA time), pm2.5_cf_1, humidity, temperature, pressure   (hourly averages)

The API key is read from the PURPLEAIR_API_KEY environment variable. Before running, in the
same terminal:
    PowerShell:  $env:PURPLEAIR_API_KEY = "your-key"
    bash:        export PURPLEAIR_API_KEY="your-key"

Then:
    python download_data_jan2026.py --list-only        # only count the sensors and estimate points (no API request)
    python download_data_jan2026.py                    # download all sensors that have 2025 data
    python download_data_jan2026.py --radius 13        # only those within 13 km of site 1103
    python download_data_jan2026.py --all-listed       # also sensors in the list that had no 2025 data

Files that were already downloaded are skipped, so the script can be re-run after an
interruption without spending API points twice.
"""

import argparse
import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("PURPLEAIR_API_KEY")

# Save everything in the same directory this script lives in (no subfolder).
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Training folder: sensor list and the Jan-Feb 2025 sensor files (read only)
TRAINING_DATA_DIR = r"C:\Users\geeth\OneDrive\Desktop\git_repository\purple_air_DL\code_full"

# Downtown LA EPA monitor 06-037-1103 (the training site)
SITE_LAT, SITE_LON = 34.06659, -118.22688

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
START_LOCAL = datetime(2026, 1, 1, 0, 0, 0, tzinfo=LOCAL_TZ)
END_LOCAL = datetime(2026, 2, 1, 0, 0, 0, tzinfo=LOCAL_TZ)

FIELDS = ["pm2.5_cf_1", "humidity", "temperature", "pressure"]
AVERAGE_MINUTES = 60  # hourly averages

# Rough cost seen in the earlier downloads (about 23,000 points per sensor for 2 months)
POINTS_PER_SENSOR_MONTH = 11_500

BASE_URL = "https://api.purpleair.com/v1"
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 10
REQUEST_DELAY_SECONDS = 1.0  # politeness delay between sensor requests

parser = argparse.ArgumentParser()
parser.add_argument("--radius", type=float, default=None,
                    help="Keep only training sensors within this many km of site 1103 (default: all)")
parser.add_argument("--all-listed", action="store_true",
                    help="Also download sensors in the training list that had no data in Jan-Feb 2025")
parser.add_argument("--list-only", action="store_true",
                    help="Only count the sensors and estimate the points, no API requests")
ARGS = parser.parse_args()


# ---------------------------------------------------------------------------
def api_get(url, params, headers):
    """GET with retry/backoff on rate limiting or transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  Rate limited (429). Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
        elif resp.status_code in (500, 502, 503, 504):
            wait = RETRY_BACKOFF_SECONDS
            print(f"  Server error {resp.status_code}. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
        else:
            # Non-retryable error (e.g. 401 bad key, 402 out of points, 400 bad params)
            print(f"  Request failed: {resp.status_code} {resp.text[:300]}")
            resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def had_training_data(sensor):
    """True if the Jan-Feb 2025 file of this sensor has at least one row."""
    path = os.path.join(TRAINING_DATA_DIR, f"sensor_{sensor}_history.csv")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) > 1


def training_sensors():
    """Training sensors (within --radius km of site 1103 if given), nearest first."""
    s = pd.read_csv(os.path.join(TRAINING_DATA_DIR, "Sensors_with_coordinates.csv"))
    s["distance_km"] = haversine(SITE_LAT, SITE_LON, s["lat"], s["lon"])

    if ARGS.radius is not None:
        s = s[s["distance_km"] <= ARGS.radius]

    s = s.copy()
    s["had_2025_data"] = s["sensor"].map(had_training_data)

    if not ARGS.all_listed:
        s = s[s["had_2025_data"]]

    return s.sort_values("distance_km").reset_index(drop=True)


def get_sensor_history(api_key, sensor_index, start_ts, end_ts):
    """Return a DataFrame of hourly history for one sensor, local-time timestamps."""
    url = f"{BASE_URL}/sensors/{sensor_index}/history"
    params = {
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "average": AVERAGE_MINUTES,
        "fields": ",".join(FIELDS),
    }
    headers = {"X-API-Key": api_key}
    data = api_get(url, params, headers)

    cols = data["fields"]  # first column is always time_stamp (unix epoch, UTC)
    rows = data["data"]

    if not rows:
        return pd.DataFrame(columns=["time_stamp"] + FIELDS)

    df = pd.DataFrame(rows, columns=cols)

    # Convert unix epoch (UTC) -> Pacific local time, formatted with offset
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], unit="s", utc=True)
    df["time_stamp"] = df["time_stamp"].dt.tz_convert(LOCAL_TZ)

    # Same column order as the training sensor files
    ordered_cols = ["time_stamp"] + [c for c in FIELDS if c in df.columns]
    df = df[ordered_cols].sort_values("time_stamp").reset_index(drop=True)

    return df


def main():
    sensors_df = training_sensors()

    where = "at any distance" if ARGS.radius is None else f"within {ARGS.radius:g} km of site 1103"
    print(f"{len(sensors_df)} training sensors {where}"
          f"{'' if ARGS.all_listed else ' that had data in Jan-Feb 2025'}.")

    # Same format as code_full/Sensors_with_coordinates.csv, which the experiment scripts read
    sensors_df[["sensor", "lat", "lon"]].to_csv(
        os.path.join(OUTPUT_DIR, "Sensors_with_coordinates.csv"), index=False
    )
    sensors_df.to_csv(os.path.join(OUTPUT_DIR, "_sensor_list.csv"), index=False)

    already = sum(
        os.path.exists(os.path.join(OUTPUT_DIR, f"sensor_{s}_history.csv"))
        for s in sensors_df["sensor"]
    )
    to_download = len(sensors_df) - already

    if ARGS.list_only:
        print(f"List only: {already} already downloaded here, {to_download} to download.")
        print(f"Estimated cost: about {to_download * POINTS_PER_SENSOR_MONTH:,} points "
              f"(~{POINTS_PER_SENSOR_MONTH:,} per sensor for one month; sensors with no data cost much less).")
        for r in [2.5, 3, 5, 7, 9, 11, 13, 20, 30, 41]:
            if ARGS.radius is None or r <= ARGS.radius:
                print(f"  within {r:g} km: {(sensors_df['distance_km'] <= r).sum()}")
        return

    if not API_KEY:
        raise SystemExit(
            "PURPLEAIR_API_KEY is not set. In this terminal run\n"
            '    PowerShell:  $env:PURPLEAIR_API_KEY = "your-key"\n'
            '    bash:        export PURPLEAIR_API_KEY="your-key"\n'
            "and start the script again."
        )

    start_ts = int(START_LOCAL.timestamp())
    end_ts = int(END_LOCAL.timestamp())

    succeeded, skipped, empty, failed = [], [], [], []

    for i, row in sensors_df.iterrows():
        sensor_index = int(row["sensor"])
        out_path = os.path.join(OUTPUT_DIR, f"sensor_{sensor_index}_history.csv")

        print(f"[{i+1}/{len(sensors_df)}] sensor_index={sensor_index}, {row['distance_km']:.1f} km")

        if os.path.exists(out_path):
            print("  Already downloaded, skipping.")
            skipped.append(sensor_index)
            continue

        try:
            hist_df = get_sensor_history(API_KEY, sensor_index, start_ts, end_ts)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((sensor_index, str(e)))

            # 402 = out of API points: every next request would fail too
            if "402" in str(e):
                print("\nOut of PurpleAir points (402). Stopping; add points and run the script again "
                      "to continue (finished sensors are skipped).")
                break

            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        if hist_df.empty:
            print("  No data returned (empty).")
            empty.append(sensor_index)
        else:
            hist_df.to_csv(out_path, index=False)
            print(f"  Saved {len(hist_df)} rows -> {out_path}")
            succeeded.append(sensor_index)

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\n=== Summary ===")
    print(f"Sensors in list:         {len(sensors_df)}")
    print(f"Downloaded successfully: {len(succeeded)}")
    print(f"Already downloaded:      {len(skipped)}")
    print(f"Empty (no data):         {len(empty)}")
    print(f"Failed:                  {len(failed)}")
    if failed:
        print("\nFailed sensors:")
        for idx, err in failed:
            print(f"  {idx}: {err}")


if __name__ == "__main__":
    main()
