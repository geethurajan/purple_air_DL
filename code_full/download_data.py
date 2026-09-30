import requests
import pandas as pd
import time
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from shapely.geometry import shape, Point

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
API_KEY = "8F17EA16-5541-11F1-B596-4201AC1DC123"

# Save everything in the same directory this script lives in (no subfolder).
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the LA city boundary GeoJSON (from LA GeoHub, same file used to
# build la_city_tracts_final.csv). Correctly excludes enclaved cities
# (Beverly Hills, Culver City, West Hollywood, etc.) via polygon holes.
LA_CITY_GEOJSON_PATH = "City_Boundary.geojson"  # place this file alongside the script

# Rough bounding box around the City of LA + a small buffer. Loose on purpose --
# filter down using your city polygon afterward. (SW corner, NE corner)
BBOX_SW_LAT, BBOX_SW_LNG = 33.70, -118.68
BBOX_NE_LAT, BBOX_NE_LNG = 34.34, -118.15

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
START_LOCAL = datetime(2025, 1, 1, 0, 0, 0, tzinfo=LOCAL_TZ)
END_LOCAL = datetime(2025, 3, 1, 0, 0, 0, tzinfo=LOCAL_TZ)

FIELDS = ["pm2.5_cf_1", "humidity", "temperature", "pressure"]
AVERAGE_MINUTES = 60  # hourly averages

BASE_URL = "https://api.purpleair.com/v1"
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 10
REQUEST_DELAY_SECONDS = 1.0  # politeness delay between sensor requests


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
            # Non-retryable error (e.g. 401 bad key, 400 bad params)
            print(f"  Request failed: {resp.status_code} {resp.text[:300]}")
            resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def load_la_city_polygon(geojson_path):
    """
    Load the LA city boundary GeoJSON and return a single (Multi)Polygon
    geometry, correctly handling holes for enclaved cities (Beverly Hills,
    Culver City, West Hollywood, etc.) -- shapely respects interior rings
    (holes) automatically as long as the GeoJSON encodes them correctly.
    """
    with open(geojson_path, "r") as f:
        gj = json.load(f)

    geoms = []
    if gj.get("type") == "FeatureCollection":
        for feat in gj["features"]:
            geoms.append(shape(feat["geometry"]))
    elif gj.get("type") == "Feature":
        geoms.append(shape(gj["geometry"]))
    else:
        # Bare geometry (Polygon / MultiPolygon), not wrapped in Feature(s)
        geoms.append(shape(gj))

    if len(geoms) == 1:
        return geoms[0]
    # If the boundary is split across multiple features, union them
    from shapely.ops import unary_union
    return unary_union(geoms)


def filter_sensors_to_city(sensors_df, city_polygon):
    """Keep only sensors whose (lon, lat) point falls inside the city polygon."""
    inside_mask = sensors_df.apply(
        lambda row: city_polygon.contains(Point(row["longitude"], row["latitude"])),
        axis=1,
    )
    return sensors_df.loc[inside_mask].reset_index(drop=True)


def get_sensors_in_bbox(api_key):
    """Return list of sensor_index values within the bounding box."""
    url = f"{BASE_URL}/sensors"
    params = {
        "fields": "sensor_index,name,latitude,longitude",
        "location_type": 0,  # outside sensors only
        "nwlng": BBOX_SW_LNG,
        "nwlat": BBOX_NE_LAT,
        "selng": BBOX_NE_LNG,
        "selat": BBOX_SW_LAT,
    }
    headers = {"X-API-Key": api_key}
    data = api_get(url, params, headers)

    cols = data["fields"]
    rows = data["data"]
    df = pd.DataFrame(rows, columns=cols)
    return df


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

    # Reorder columns to match your existing sensor CSV convention
    ordered_cols = ["time_stamp"] + [c for c in FIELDS if c in df.columns]
    df = df[ordered_cols].sort_values("time_stamp").reset_index(drop=True)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_ts = int(START_LOCAL.timestamp())
    end_ts = int(END_LOCAL.timestamp())

    print("Fetching sensor list in bounding box...")
    sensors_df = get_sensors_in_bbox(API_KEY)
    print(f"Found {len(sensors_df)} sensors in bounding box.\n")

    print(f"Loading LA city boundary polygon from {LA_CITY_GEOJSON_PATH} ...")
    city_polygon = load_la_city_polygon(LA_CITY_GEOJSON_PATH)

    sensors_df = filter_sensors_to_city(sensors_df, city_polygon)
    print(f"{len(sensors_df)} sensors fall inside LA city limits "
          f"(enclaves like Beverly Hills / Culver City / West Hollywood correctly excluded).\n")

    sensors_df.to_csv(os.path.join(OUTPUT_DIR, "_sensor_list.csv"), index=False)

    succeeded, empty, failed = [], [], []

    for i, row in sensors_df.iterrows():
        sensor_index = row["sensor_index"]
        out_path = os.path.join(OUTPUT_DIR, f"sensor_{sensor_index}_history.csv")

        print(f"[{i+1}/{len(sensors_df)}] sensor_index={sensor_index} ({row.get('name', '')})")

        try:
            hist_df = get_sensor_history(API_KEY, sensor_index, start_ts, end_ts)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((sensor_index, str(e)))
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
    print(f"Total sensors found:     {len(sensors_df)}")
    print(f"Downloaded successfully: {len(succeeded)}")
    print(f"Empty (no data):         {len(empty)}")
    print(f"Failed:                  {len(failed)}")
    if failed:
        print("\nFailed sensors:")
        for idx, err in failed:
            print(f"  {idx}: {err}")


if __name__ == "__main__":
    main()