"""
Sort all PurpleAir sensor_*_history.csv files chronologically by time_stamp.
Flags empty files (header only / zero data rows) and reports before doing anything.

Usage:
    python3 sort_purpleair_files.py /path/to/your/sensor/csv/directory

By default this OVERWRITES each file in place, sorted. If you'd rather write
to a separate output directory and leave originals untouched, set
OVERWRITE_IN_PLACE = False below and set OUTPUT_DIR.
"""

import sys
import glob
import os
import pandas as pd

OVERWRITE_IN_PLACE = True
OUTPUT_DIR = None  # e.g. "/path/to/output" -- only used if OVERWRITE_IN_PLACE is False


def main():
    # If no directory is given as an argument, default to the current
    # working directory (i.e. wherever this script is being run from).
    if len(sys.argv) < 2:
        directory = os.getcwd()
        print(f"No directory argument given -- defaulting to current directory: {directory}\n")
    else:
        directory = sys.argv[1]
    pattern = os.path.join(directory, "sensor_*_history.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching sensor_*_history.csv found in {directory}")
        sys.exit(1)

    print(f"Found {len(files)} sensor files.\n")

    empty_files = []
    sorted_ok = []
    error_files = []

    if not OVERWRITE_IN_PLACE:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath)
        except pd.errors.EmptyDataError:
            empty_files.append(fname)
            continue
        except Exception as e:
            error_files.append((fname, str(e)))
            continue

        if df.shape[0] == 0:
            empty_files.append(fname)
            continue

        if "time_stamp" not in df.columns:
            error_files.append((fname, f"no 'time_stamp' column (columns: {list(df.columns)})"))
            continue

        # Parse timestamps -- they already carry a UTC offset (e.g. -08:00),
        # so this correctly captures local time, no conversion needed.
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=False)

        # Check for duplicate timestamps within a file (can happen with overlapping pulls)
        n_dupes = df["time_stamp"].duplicated().sum()

        df = df.sort_values("time_stamp").reset_index(drop=True)

        out_path = fpath if OVERWRITE_IN_PLACE else os.path.join(OUTPUT_DIR, fname)
        df.to_csv(out_path, index=False)

        sorted_ok.append((fname, df.shape[0], n_dupes))

    # --- Report ---
    print(f"=== Successfully sorted: {len(sorted_ok)} files ===")
    total_dupes = sum(d for _, _, d in sorted_ok)
    if total_dupes:
        print(f"  ({total_dupes} duplicate timestamp rows found across all files -- "
              f"not removed, just flagged; inspect if this matters for your analysis)")
        for fname, n_rows, n_dupes in sorted_ok:
            if n_dupes > 0:
                print(f"    {fname}: {n_rows} rows, {n_dupes} duplicate timestamps")
    print()

    print(f"=== Empty files (0 data rows): {len(empty_files)} ===")
    for fname in empty_files:
        print(f"  {fname}")
    print()

    if error_files:
        print(f"=== Files with errors: {len(error_files)} ===")
        for fname, err in error_files:
            print(f"  {fname}: {err}")
        print()

    print("=== Summary ===")
    print(f"Total files found:     {len(files)}")
    print(f"Sorted successfully:   {len(sorted_ok)}")
    print(f"Empty (no data):       {len(empty_files)}")
    print(f"Errored:               {len(error_files)}")


if __name__ == "__main__":
    main()