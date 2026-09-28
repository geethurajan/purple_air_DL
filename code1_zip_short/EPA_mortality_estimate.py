import pandas as pd
import numpy as np

# =============================================================================
# USER-EDITABLE PARAMETERS
# =============================================================================

# Files
PM_FILE = "LA_Site_1103.csv"
POP_FILE = "la_city_tracts_final.csv"

# Wildfire period (inclusive)
FIRE_START = "2025-01-07"
FIRE_END   = "2025-01-31"

# Concentration-response coefficient
BETA = 0.00094
BETA_LOW = 0.00073
BETA_HIGH = 0.00116

# Baseline mortality (per 100,000 people)
BD = 780

# =============================================================================
# LOAD DATA
# =============================================================================

pm = pd.read_csv(PM_FILE)
pop = pd.read_csv(POP_FILE)

# =============================================================================
# PREPARE PM2.5 DATA
# =============================================================================

# Combine date and time into a datetime
pm["datetime"] = pd.to_datetime(
    pm["Date Local"] + " " + pm["Time Local"]
)

pm["date"] = pm["datetime"].dt.date

# Ensure measurements are numeric
pm["Sample Measurement"] = pd.to_numeric(
    pm["Sample Measurement"],
    errors="coerce"
)

# -----------------------------------------------------------------------------
# DAILY AVERAGES
#
# Missing hourly observations are automatically ignored.
# Each day's average is computed from all available observations.
# -----------------------------------------------------------------------------

daily_pm = (
    pm.groupby("date")
      .agg(
          daily_mean=("Sample Measurement", "mean"),
          n_obs=("Sample Measurement", "count")
      )
      .reset_index()
)

daily_pm["date"] = pd.to_datetime(daily_pm["date"])
daily_pm = daily_pm[
    (daily_pm["date"] >= "2025-01-01") &
    (daily_pm["date"] <= "2025-03-01")
]

# =============================================================================
# CLASSIFY FIRE VS NON-FIRE DAYS
# =============================================================================

fire_start = pd.to_datetime(FIRE_START)
fire_end = pd.to_datetime(FIRE_END)

daily_pm["fire"] = (
    (daily_pm["date"] >= fire_start) &
    (daily_pm["date"] <= fire_end)
)

pm_fire = daily_pm.loc[daily_pm.fire, "daily_mean"].mean()
pm_nofire = daily_pm.loc[~daily_pm.fire, "daily_mean"].mean()

delta_pm = pm_fire - pm_nofire

# =============================================================================
# RELATIVE RISK
# =============================================================================

def RR(beta, delta):
    return np.exp(beta * delta)

rr = RR(BETA, delta_pm)
rr_low = RR(BETA_LOW, delta_pm)
rr_high = RR(BETA_HIGH, delta_pm)

# =============================================================================
# ATTRIBUTABLE FRACTION
# =============================================================================

def AF(rr):
    return (rr - 1) / rr

af = AF(rr)
af_low = AF(rr_low)
af_high = AF(rr_high)

# =============================================================================
# POPULATION
# =============================================================================

population = pop["population"].sum()

# =============================================================================
# EXCESS MORTALITY
# =============================================================================

baseline_deaths = BD / 100000 * population

M = baseline_deaths * af
M_low = baseline_deaths * af_low
M_high = baseline_deaths * af_high

# =============================================================================
# RESULTS
# =============================================================================

print("\n==============================")
print("Average Daily PM2.5")
print("==============================")
print(f"Non-fire days : {pm_nofire:.2f} µg/m³")
print(f"Fire days     : {pm_fire:.2f} µg/m³")
print(f"Difference    : {delta_pm:.2f} µg/m³")

print("\n==============================")
print("Relative Risk")
print("==============================")
print(f"RR      = {rr:.5f}")
print(f"95% CI  = ({rr_low:.5f}, {rr_high:.5f})")

print("\n==============================")
print("Attributable Fraction")
print("==============================")
print(f"AF      = {af:.5f}")
print(f"95% CI  = ({af_low:.5f}, {af_high:.5f})")

print("\n==============================")
print("Population")
print("==============================")
print(f"{population:,.0f}")

print("\n==============================")
print("Excess Mortality")
print("==============================")
print(f"Baseline mortality rate = {BD} per 100,000")
print(f"Expected excess deaths = {M:.2f}")
print(f"95% CI = ({M_low:.2f}, {M_high:.2f})")

daily_summary = (
    pm.groupby("date")
      .agg(
          n_obs=("Sample Measurement", "count"),
          mean_pm25=("Sample Measurement", "mean"),
          median_pm25=("Sample Measurement", "median"),
          min_pm25=("Sample Measurement", "min"),
          max_pm25=("Sample Measurement", "max"),
          std_pm25=("Sample Measurement", "std"),
      )
      .reset_index()
)

daily_summary["date"] = pd.to_datetime(daily_summary["date"])

daily_summary["fire"] = (
    (daily_summary["date"] >= fire_start) &
    (daily_summary["date"] <= fire_end)
)

daily_summary = daily_summary.round({
    "mean_pm25": 2,
    "median_pm25": 2,
    "min_pm25": 2,
    "max_pm25": 2,
    "std_pm25": 2
})

daily_summary = daily_summary[
    (daily_summary["date"] >= "2025-01-01") &
    (daily_summary["date"] <= "2025-02-28")
]

print("\nDaily PM2.5 Summary")
print(daily_summary.to_string(index=False))