"""
Keep only the downtown LA EPA monitor 06-037-1103 (POC 3, Met One BAM-1020, the same instrument
as in the training file LA_Site_1103.csv) for January 2026 from EPA's hourly PM2.5 file for 2026,
and save it as LA_Site_1103_Jan2026.csv, in the same format as LA_Site_1103.csv.

    python extract_epa_1103_2026.py
"""

import os

import pandas as pd

# EPA AirData hourly PM2.5 FRM/FEM file for 2026: next to this script, or in the
# hourly_88101_2026 folder that Windows "Extract All" creates
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "hourly_88101_2026.csv"),
    os.path.join(HERE, "hourly_88101_2026", "hourly_88101_2026.csv"),
]
SRC = next((p for p in CANDIDATES if os.path.exists(p)), None)

if SRC is None:
    raise SystemExit("hourly_88101_2026.csv not found. Unzip hourly_88101_2026.zip into this folder.")

OUT = os.path.join(HERE, "LA_Site_1103_Jan2026.csv")
POC = "3"                       # the instrument used in the training file

print("Reading", SRC)

parts = []

def code(column):
    # Codes as numbers ("06" -> 6); non-numeric codes (e.g. "CC" for Canada) become NaN
    return pd.to_numeric(column, errors="coerce")


for ch in pd.read_csv(SRC, chunksize=1_000_000, dtype=str):
    parts.append(ch[(code(ch["State Code"]) == 6)        # California
                    & (code(ch["County Code"]) == 37)     # Los Angeles County
                    & (code(ch["Site Num"]) == 1103)])    # downtown LA (North Main Street)

site = pd.concat(parts)

if site.empty:
    raise SystemExit(f"Site 06-037-1103 is not in {SRC}.")

print("Instruments at site 1103 in this file:")
print(site.groupby(["POC", "Method Name"]).size().to_string())

site = site[code(site["POC"]) == int(POC)]

# January 2026 only
day = pd.to_datetime(site["Date Local"])
site = site[(day >= "2026-01-01") & (day < "2026-02-01")]

if site.empty:
    raise SystemExit(f"No POC {POC} rows for January 2026 in {SRC}. "
                     "The 2026 file may not include January yet; see the instructions.")

site.to_csv(OUT, index=False)

print(f"\n{len(site)} hours | {site['Date Local'].min()} -> {site['Date Local'].max()} -> {OUT}")
print(f"Expected about 744 hours for January (31 days x 24).")
