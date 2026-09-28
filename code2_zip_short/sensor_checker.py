"""
Scan all sensor_*_history.csv files in this directory and report each file's
actual date range (min/max time_stamp). Use this to spot a leftover file
from an old download that didn't get overwritten by a newer one (e.g. a
sensor that was in your old download but got filtered out of the new
LA-city-polygon-filtered download, so its old file is still sitting there
untouched).

Run this in the same directory as your sensor_*_history.csv files:
    python sensor_date_range_check.py
"""

import glob
import os
import pandas as pd
from collections import Counter

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "sensor_*_history.csv")))

    if not files:
        print("No sensor_*_history.csv files found in this directory.")
        return

    print(f"Scanning {len(files)} files...\n")

    results = []
    errors = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, usecols=["time_stamp"])
        except Exception as e:
            errors.append((fname, str(e)))
            continue

        if df.empty:
            results.append({"file": fname, "n_rows": 0, "min_date": None, "max_date": None})
            continue

        ts = pd.to_datetime(df["time_stamp"], utc=False, errors="coerce")
        results.append({
            "file": fname,
            "n_rows": len(df),
            "min_date": ts.min(),
            "max_date": ts.max(),
        })

    results_df = pd.DataFrame(results)

    if errors:
        print(f"=== {len(errors)} files could not be read ===")
        for fname, err in errors:
            print(f"  {fname}: {err}")
        print()

    # Round to date only (drop time-of-day) so minor hour differences don't
    # create false mismatches -- we care about which DAY range each file covers
    results_df["min_day"] = pd.to_datetime(results_df["min_date"]).dt.date
    results_df["max_day"] = pd.to_datetime(results_df["max_date"]).dt.date

    # Find the most common (min_day, max_day) pair -- that's your "expected" range
    range_counts = Counter(zip(results_df["min_day"], results_df["max_day"]))
    most_common_range, most_common_count = range_counts.most_common(1)[0]

    print(f"Most common date range across files: {most_common_range[0]} to {most_common_range[1]} "
          f"({most_common_count} of {len(results_df)} files)\n")

    mismatches = results_df[
        (results_df["min_day"] != most_common_range[0]) |
        (results_df["max_day"] != most_common_range[1])
    ]

    if mismatches.empty:
        print("No mismatches found -- every file's date range matches. All clear.")
    else:
        print(f"=== {len(mismatches)} file(s) with a DIFFERENT date range -- these are your suspects ===")
        print(mismatches[["file", "n_rows", "min_day", "max_day"]].to_string(index=False))

    # Save full report either way, useful for a paper trail
    out_path = os.path.join(directory, "_sensor_date_range_report.csv")
    results_df.drop(columns=["min_day", "max_day"]).to_csv(out_path, index=False)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()