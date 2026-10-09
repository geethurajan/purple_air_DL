"""
Convert MAIA Surface Monitor .nc files into one CSV.

Works for both MAIA products:
  MAIA_ANC_SURFACEMONITOR_PM_2.5_SPECIES_...nc  (black carbon, sulfate, nitrate, EC, OC, dust)
  MAIA_ANC_SURFACEMONITOR_PM_TOTAL_...nc        (total PM2.5 and PM10)

Output: one row per day x pollutant x monitor, with the daily value (ug/m3),
the monitor's name, ID, latitude/longitude, data source, and QC fields.

Usage (in the VS Code terminal, from the folder holding this script):
  python maia_to_csv_hourly.py FOLDER_WITH_NC_FILES
  python maia_to_csv_hourly.py FOLDER_WITH_NC_FILES --out my_output.csv
  python maia_to_csv_hourly.py FOLDER_WITH_NC_FILES --raw     # also save the sub-daily raw readings
  python maia_to_csv_hourly.py FOLDER_WITH_NC_FILES --hourly  # also save hourly averages per station
  python maia_to_csv_hourly.py FOLDER_WITH_NC_FILES --start 2025-01-15 --end 2025-01-20 --hourly

Needs: pip install netCDF4 pandas
"""
import argparse
import glob
import os
import re

import netCDF4
import numpy as np
import pandas as pd

VERSION_RE = re.compile(r"_(\d{8})T\d{6}Z_F(\d+)_V(\d+)p(\d+)p(\d+)(?:_V(\d+))?\.nc$")


def to_list(var):
    """Read a netCDF variable as a plain list, with missing values as NaN / ''."""
    vals = var[:]
    if np.ma.isMaskedArray(vals):
        vals = vals.astype(float).filled(np.nan) if vals.dtype.kind in "fiu" else vals.filled("")
    return [("" if v is None else v) for v in np.asarray(vals, dtype=object).tolist()]


def table_from_group(group, prefix=""):
    """All 1-D variables of a group (and its subgroups) as columns."""
    cols = {}
    for name, var in group.variables.items():
        cols[prefix + name] = to_list(var)
    for sub in group.groups.values():
        cols.update(table_from_group(sub, prefix))
    return cols


def daily_rows(path):
    """Yield one dict per (pollutant, monitor) daily value in a file."""
    ds = netCDF4.Dataset(path)
    try:
        def walk(g):
            if "pm_mass_concentration" in g.variables and "date_LST" in g.variables:
                cols = {k: to_list(v) for k, v in g.variables.items()}
                if "Monitor_Information" in g.groups:
                    cols.update(table_from_group(g["Monitor_Information"]))
                if "Quality_Metrics" in g.groups:
                    cols.update(table_from_group(g["Quality_Metrics"], "qc_"))
                n = len(cols["pm_mass_concentration"])
                parts = g.path.strip("/").split("/")   # e.g. PM_2.5/Sulfate, PM_10/Total
                for i in range(n):
                    row = {k: v[i] for k, v in cols.items() if len(v) == n}
                    if row.get("maia_surface_monitor_id", "") == "":
                        continue   # pollutant not measured that day
                    row["pm_size"] = parts[0]
                    row["pollutant"] = "/".join(parts[1:])
                    row["source_file"] = os.path.basename(path)
                    yield row
            for sub in g.groups.values():
                if sub.name not in ("Monitor_Information", "Quality_Metrics", "Source_Input_Data"):
                    yield from walk(sub)
        yield from walk(ds)
    finally:
        ds.close()


def raw_rows(path):
    """Yield the sub-daily readings stored in Source_Input_Data groups."""
    ds = netCDF4.Dataset(path)
    try:
        def walk(g):
            if g.name == "Source_Input_Data" or g.path.startswith("/PM_2.5/Dust/Source_Input_Data/"):
                cols = {k: to_list(v) for k, v in g.variables.items()}
                if cols:
                    n = len(next(iter(cols.values())))
                    parts = g.path.strip("/").split("/")
                    pollutant = [p for p in parts[1:] if p != "Source_Input_Data"]
                    for i in range(n):
                        row = {k: v[i] for k, v in cols.items() if len(v) == n}
                        if row.get("maia_surface_monitor_id", "") == "":
                            continue
                        # dust elements name their value column e.g. iron_pm_mass_concentration
                        for k in list(row):
                            if k.endswith("_pm_mass_concentration"):
                                row["pm_mass_concentration"] = row.pop(k)
                        row["pm_size"] = parts[0]
                        row["pollutant"] = "/".join(pollutant)
                        row["source_file"] = os.path.basename(path)
                        yield row
            for sub in g.groups.values():
                if sub.name != "Monitor_Information":
                    yield from walk(sub)
        yield from walk(ds)
    finally:
        ds.close()


def newest_versions(paths):
    """If a day was downloaded in more than one version, keep only the newest."""
    best = {}
    for p in paths:
        name = os.path.basename(p)
        m = VERSION_RE.search(name)
        if not m:
            best[name] = ((0,), p)
            continue
        key = name[: m.start()] + m.group(1)          # product + city + date
        ver = tuple(int(x) for x in m.groups()[1:5]) + (int(m.group(6) or 0),)
        if key not in best or ver > best[key][0]:
            best[key] = (ver, p)
    return sorted(p for _, p in best.values())


def to_hourly(raw, daily):
    """Average the raw readings into one value per station, pollutant and hour,
    with all the columns of the daily file attached.

    The raw readings are MAIA's input data, before its QC: negative or stuck values
    are still included. n_readings shows how many readings went into each hour.
    The qc_ columns are MAIA's daily QC for that station-day, repeated on each hour.
    """
    raw = raw.dropna(subset=["pm_mass_concentration"]).copy()
    # consecutive daily files overlap by a few hours, so the same reading can appear
    # in two files; keep it once (from the later file)
    raw = raw.drop_duplicates(["pm_size", "pollutant", "maia_surface_monitor_id",
                               "date_time_UTC"], keep="last")
    raw["hour_UTC"] = pd.to_datetime(raw["date_time_UTC"], utc=True).dt.floor("h")
    # date_time_LST is local standard time (its trailing "Z" is not UTC)
    lst = pd.to_datetime(raw["date_time_LST"].astype(str).str[:19])
    raw["hour_LST"] = lst.dt.floor("h")
    raw["date_LST"] = lst.dt.strftime("%Y%m%d")
    keys = ["date_LST", "hour_LST", "hour_UTC", "pm_size", "pollutant",
            "maia_surface_monitor_id"]
    hourly = (raw.groupby(keys)["pm_mass_concentration"]
                 .agg(pm_mass_concentration="mean", n_readings="size")
                 .reset_index())

    day = daily.rename(
        columns={"pm_mass_concentration": "daily_pm_mass_concentration"})
    day["date_LST"] = day["date_LST"].astype(str)
    hourly = hourly.merge(day, on=["date_LST", "pm_size", "pollutant",
                                   "maia_surface_monitor_id"], how="left")
    # fill station details for hours whose day has no daily value
    station_cols = [c for c in ["site_name", "latitude", "longitude", "elevation", "height",
                                "station_location_type", "source_name", "source_url",
                                "station_country", "station_state_or_province"] if c in day]
    stations = day.drop_duplicates("maia_surface_monitor_id").set_index(
        "maia_surface_monitor_id")[station_cols]
    for c in station_cols:
        hourly[c] = hourly[c].fillna(hourly["maia_surface_monitor_id"].map(stations[c]))

    hourly = hourly.sort_values(["hour_UTC", "pm_size", "pollutant", "maia_surface_monitor_id"])
    hourly["hour_UTC"] = hourly["hour_UTC"].dt.strftime("%Y-%m-%d %H:00")
    hourly["hour_LST"] = hourly["hour_LST"].dt.strftime("%Y-%m-%d %H:00")
    front = ["hour_UTC", "hour_LST", "date_LST", "pm_size", "pollutant",
             "pm_mass_concentration", "n_readings", "daily_pm_mass_concentration"]
    return hourly[front + [c for c in hourly if c not in front]]


FRONT = ["date_LST", "pm_size", "pollutant", "pm_mass_concentration",
         "maia_surface_monitor_id", "site_name", "latitude", "longitude",
         "station_location_type", "source_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing the downloaded MAIA .nc files")
    ap.add_argument("--out", default="maia_daily.csv")
    ap.add_argument("--raw", action="store_true",
                    help="also write <out>_raw.csv with every sub-daily reading")
    ap.add_argument("--hourly", action="store_true",
                    help="also write <out>_hourly.csv with hourly averages per station")
    ap.add_argument("--start", help="first day to include, YYYY-MM-DD")
    ap.add_argument("--end", help="last day to include, YYYY-MM-DD")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.folder, "**", "MAIA_*.nc"), recursive=True)
    if args.start or args.end:
        lo = (args.start or "0000-00-00").replace("-", "")
        hi = (args.end or "9999-99-99").replace("-", "")
        paths = [p for p in paths
                 if (m := VERSION_RE.search(os.path.basename(p))) and lo <= m.group(1) <= hi]
    if not paths:
        raise SystemExit(f"No MAIA_*.nc files found in {args.folder} for those dates")
    kept = newest_versions(paths)
    print(f"Found {len(paths)} files, using {len(kept)} (newest version of each day)")

    daily = pd.DataFrame([r for p in kept for r in daily_rows(p)])
    if daily.empty:
        raise SystemExit("No daily values found in these files")
    daily = daily[[c for c in FRONT if c in daily] + [c for c in daily if c not in FRONT]]
    daily = daily.sort_values(["date_LST", "pm_size", "pollutant", "maia_surface_monitor_id"])
    daily.to_csv(args.out, index=False)
    print(f"Wrote {len(daily)} rows to {args.out}")
    print(daily.groupby(["pm_size", "pollutant"]).size().rename("rows").to_string())

    if args.raw or args.hourly:
        raw = pd.DataFrame([r for p in kept for r in raw_rows(p)])
        base = os.path.splitext(args.out)[0]
        if args.raw:
            raw.to_csv(base + "_raw.csv", index=False)
            print(f"Wrote {len(raw)} raw readings to {base}_raw.csv")
        if args.hourly:
            hourly = to_hourly(raw, daily)
            hourly.to_csv(base + "_hourly.csv", index=False)
            print(f"Wrote {len(hourly)} hourly rows to {base}_hourly.csv")


if __name__ == "__main__":
    main()
