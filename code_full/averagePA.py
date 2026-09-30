import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(directory, "sensor_*_history.csv")))

    if not files:
        print("No sensor_*_history.csv files found in this directory.")
        return

    print(f"Loading {len(files)} sensor files...\n")

    all_data = []
    skipped = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, usecols=["time_stamp", "pm2.5_cf_1"])
        except Exception as e:
            skipped.append((fname, str(e)))
            continue

        if df.empty:
            skipped.append((fname, "empty file"))
            continue

        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.dropna(subset=["time_stamp", "pm2.5_cf_1"])

        if df.empty:
            skipped.append((fname, "no valid rows after parsing"))
            continue

        # If a single sensor has more than one reading at the same timestamp
        # (e.g. overlapping downloads), average those together FIRST so this
        # one sensor doesn't get double-counted relative to other sensors
        # when we average across sensors in the next step.
        df = df.groupby("time_stamp", as_index=False)["pm2.5_cf_1"].mean()
        df["sensor_file"] = fname

        all_data.append(df)

    if skipped:
        print(f"=== Skipped {len(skipped)} file(s) ===")
        for fname, reason in skipped:
            print(f"  {fname}: {reason}")
        print()

    if not all_data:
        print("No usable data found across any files.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Now average across sensors at each timestamp. pandas' .mean() ignores
    # NaN automatically, and since each sensor only contributes a row for
    # timestamps it actually has data for, a missing sensor at a given time
    # simply isn't included in that timestamp's average -- exactly the
    # "ignore if missing" behavior requested.
    result = (
        combined.groupby("time_stamp")["pm2.5_cf_1"]
        .agg(avg_pm25="mean", n_sensors="count")
        .reset_index()
        .sort_values("time_stamp")
        .reset_index(drop=True)
    )

    print(f"Computed averages across {len(result)} distinct timestamps, "
          f"from {len(all_data)} usable sensor files.\n")

    print(result.to_string(index=False))

    out_path = os.path.join(directory, "average_pm25_by_timestamp.csv")
    result.to_csv(out_path, index=False)
    print(f"\nSaved full result to: {out_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result["time_stamp"], result["avg_pm25"], linewidth=0.8, color="firebrick")

    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (\u00b5g/m\u00b3)")
    ax.set_title("Citywide average PM2.5 across all sensors, by timestamp")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()

    plot_path = os.path.join(directory, "average_pm25_by_timestamp.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
