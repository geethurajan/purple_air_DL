"""
Plot EPA PM2.5 readings from LA_Site_1103.csv:
  1. Hourly PM2.5 over time (to spot elevated periods)
  2. Daily-averaged PM2.5 over time

This is READ-ONLY on the input CSV -- it only reads LA_Site_1103.csv and
writes two new PNG files (plus prints summary info to the console).

Run this in the same directory as LA_Site_1103.csv:
    python plot_epa_pm25.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INPUT_FILE = "LA_Site_1103.csv"

# Only plot this date range
START_DATE = "2025-01-01"
END_DATE = "2025-03-01"


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(directory, INPUT_FILE)

    if not os.path.exists(csv_path):
        print(f"Could not find {INPUT_FILE} in {directory}. "
              f"Make sure this script is in the same folder as the CSV.")
        return

    df = pd.read_csv(csv_path)
    df = df[["Date Local", "Time Local", "Sample Measurement"]].copy()

    # Combine date + time into a single timestamp column
    df["timestamp"] = pd.to_datetime(
        df["Date Local"] + " " + df["Time Local"], errors="coerce"
    )
    df["Sample Measurement"] = pd.to_numeric(df["Sample Measurement"], errors="coerce")

    df = df.dropna(subset=["timestamp", "Sample Measurement"]).sort_values("timestamp")

    # Filter to the requested date range
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]

    if df.empty:
        print(f"No data found between {START_DATE} and {END_DATE} -- check the date range "
              f"or confirm the CSV actually covers this period.")
        return

    print(f"Loaded {len(df)} valid hourly readings.")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"PM2.5 range: {df['Sample Measurement'].min():.1f} to "
          f"{df['Sample Measurement'].max():.1f} ug/m3")
    print()

    # -----------------------------------------------------------------------
    # Plot 1: hourly PM2.5 over the full range
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["timestamp"], df["Sample Measurement"], linewidth=0.7, color="firebrick")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("PM2.5 (\u00b5g/m\u00b3)")
    ax1.set_title("EPA PM2.5 -- Hourly Readings (LA Site 1103)")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig1.autofmt_xdate(rotation=45)
    fig1.tight_layout()

    hourly_path = os.path.join(directory, "epa_pm25_hourly.png")
    fig1.savefig(hourly_path, dpi=150)
    print(f"Saved hourly plot to: {hourly_path}")

    # -----------------------------------------------------------------------
    # Plot 2: daily average PM2.5
    # -----------------------------------------------------------------------
    daily = (
        df.set_index("timestamp")["Sample Measurement"]
        .resample("D")
        .mean()
        .reset_index()
        .rename(columns={"Sample Measurement": "avg_pm25"})
    )

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(daily["timestamp"], daily["avg_pm25"], marker="o", markersize=3,
              linewidth=1.2, color="darkorange")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Average PM2.5 (\u00b5g/m\u00b3)")
    ax2.set_title("EPA PM2.5 -- Daily Average (LA Site 1103)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig2.autofmt_xdate(rotation=45)
    fig2.tight_layout()

    daily_path = os.path.join(directory, "epa_pm25_daily_avg.png")
    fig2.savefig(daily_path, dpi=150)
    print(f"Saved daily average plot to: {daily_path}")

    daily_csv_path = os.path.join(directory, "epa_pm25_daily_avg.csv")
    daily.to_csv(daily_csv_path, index=False)
    print(f"Saved daily average data to: {daily_csv_path}")

    plt.show()


if __name__ == "__main__":
    main()