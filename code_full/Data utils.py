"""
Data preparation utilities shared across Algorithms 1-3.

Handles:
  - synchronizing PurpleAir <-> EPA readings by timestamp
  - constructing lagged (t-1) meteorological features
  - contiguous time-block train/test splitting (leakage-resistant)
"""

import numpy as np
import pandas as pd

from config import CURRENT_FEATURES, LAGGED_FEATURES, TIMESTAMP_COL


def synchronize_calibration_data(purpleair_df: pd.DataFrame, epa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Algorithm 1, Step 1.

    Synchronize the closest PurpleAir sensor's readings with the EPA reference
    monitor's readings by timestamp, and drop missing/invalid rows.

    Parameters
    ----------
    purpleair_df : DataFrame with columns [timestamp, pa_pm25, pa_temp, pa_humidity, pa_pressure]
    epa_df       : DataFrame with columns [timestamp, epa_pm25]

    Returns
    -------
    DataFrame, one row per synchronized hour, sorted chronologically.
    """
    merged = pd.merge(purpleair_df, epa_df, on=TIMESTAMP_COL, how="inner")
    merged = merged.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    required_cols = ["pa_pm25", "pa_temp", "pa_humidity", "pa_pressure", "epa_pm25"]
    merged = merged.dropna(subset=required_cols)

    # basic sanity filtering: PM2.5 / met readings shouldn't be negative
    merged = merged[(merged["pa_pm25"] >= 0) & (merged["epa_pm25"] >= 0)]

    return merged.reset_index(drop=True)


def add_lagged_meteorology(df: pd.DataFrame, sensor_id_col: str = None) -> pd.DataFrame:
    """
    Algorithm 1, Step 2 / Algorithm 2, Step 1 / Algorithm 3, Step 3.

    Add tempt-1, humidityt-1, pressuret-1 columns using the immediately
    preceding hour's readings. The first hour of each series has no t-1
    and is dropped.

    Parameters
    ----------
    df : DataFrame sorted chronologically, containing pa_temp, pa_humidity, pa_pressure.
    sensor_id_col : optional column name identifying distinct sensors. If provided,
        lagging is done independently within each sensor's time series (used when
        applying calibration to *all* PurpleAir sensors in Algorithms 2 & 3, so one
        sensor's history doesn't leak into another's lag).

    Returns
    -------
    DataFrame with LAGGED_FEATURES columns added; first hour(s) dropped.
    """
    df = df.copy()

    def _lag_group(g):
        g = g.sort_values(TIMESTAMP_COL)
        g["pa_temp_lag1"] = g["pa_temp"].shift(1)
        g["pa_humidity_lag1"] = g["pa_humidity"].shift(1)
        g["pa_pressure_lag1"] = g["pa_pressure"].shift(1)
        return g

    if sensor_id_col is not None:
        df = df.groupby(sensor_id_col, group_keys=False).apply(_lag_group)
    else:
        df = _lag_group(df)

    # drop rows with no t-1 (first hour of each series)
    df = df.dropna(subset=LAGGED_FEATURES).reset_index(drop=True)
    return df


def contiguous_time_block_split(df: pd.DataFrame, min_test_days: int = 3, max_test_days: int = 10,
                                 rng: np.random.Generator = None):
    """
    Algorithm 1, Step 5.

    Draw a random *contiguous* span of consecutive days as the test block,
    with the remainder as training data. Deliberately NOT a row-wise random
    split, to avoid leakage from autocorrelated adjacent hours inflating R^2
    (especially for the MLP).

    Parameters
    ----------
    df : synchronized, chronologically sorted DataFrame with a timestamp column.
    min_test_days, max_test_days : bounds on the length of the held-out block.
    rng : numpy random Generator for reproducibility; if None, a fresh one is used.

    Returns
    -------
    (train_df, test_df)
    """
    if rng is None:
        rng = np.random.default_rng()

    ts = pd.to_datetime(df[TIMESTAMP_COL])
    start_date = ts.min().normalize()
    end_date = ts.max().normalize()
    total_days = (end_date - start_date).days + 1

    test_len = int(rng.integers(min_test_days, max_test_days + 1))
    test_len = min(test_len, max(1, total_days - 1))  # leave at least 1 day for training

    latest_start_offset = total_days - test_len
    test_start_offset = int(rng.integers(0, max(1, latest_start_offset)))

    test_start = start_date + pd.Timedelta(days=test_start_offset)
    test_end = test_start + pd.Timedelta(days=test_len)  # exclusive end

    is_test = (ts >= test_start) & (ts < test_end)

    test_df = df[is_test].reset_index(drop=True)
    train_df = df[~is_test].reset_index(drop=True)

    return train_df, test_df


def build_feature_matrix(df: pd.DataFrame):
    """Convenience helper: returns (X, y) numpy arrays for the 7 predictors + target."""
    from config import ALL_FEATURES, TARGET_COL
    X = df[ALL_FEATURES].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float) if TARGET_COL in df.columns else None
    return X, y
