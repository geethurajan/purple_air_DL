import glob
import os
import pandas as pd

MIN_HOURS = 120


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "sensor_*_history.csv")))

    if not files:
        print("No sensor_*_history.csv files found in this directory.")
        return

    print(f"Scanning {len(files)} files...\n")

    records = []
    errors = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, usecols=["time_stamp"])
        except Exception as e:
            errors.append((fname, str(e)))
            continue

        n_rows = len(df)
        records.append({"file": fname, "n_hours": n_rows})

    if errors:
        print(f"=== {len(errors)} file(s) could not be read ===")
        for fname, err in errors:
            print(f"  {fname}: {err}")
        print()

    all_df = pd.DataFrame(records)

    has_data = all_df[all_df["n_hours"] > 0].sort_values("file").reset_index(drop=True)
    has_enough = has_data[has_data["n_hours"] >= MIN_HOURS].sort_values("file").reset_index(drop=True)

    print(f"=== Sensors with any data: {len(has_data)} of {len(all_df)} files ===")
    print(has_data.to_string(index=False))
    print()

    print(f"=== Sensors with at least {MIN_HOURS} hours of data: {len(has_enough)} ===")
    print(has_enough.to_string(index=False))
    print()

    # Save both lists to CSV for reference
    has_data_path = os.path.join(directory, "sensors_with_data.csv")
    has_enough_path = os.path.join(directory, f"sensors_with_at_least_{MIN_HOURS}_hours.csv")

    has_data.to_csv(has_data_path, index=False)
    has_enough.to_csv(has_enough_path, index=False)

    print(f"Saved: {has_data_path}")
    print(f"Saved: {has_enough_path}")


if __name__ == "__main__":
    main()
