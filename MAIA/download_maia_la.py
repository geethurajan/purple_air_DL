"""
Download MAIA Surface Monitor (ground-based PM) files for the Los Angeles
target area from NASA ASDC.

Products (NASA catalog short names, version C01):
  MAIA_ANC_SURFACEMONITOR_PM_TOTAL       daily total PM2.5 and PM10, per monitor
  MAIA_ANC_SURFACEMONITOR_PM_2.5_SPECIES daily speciated PM2.5 (sulfate, nitrate,
                                         EC, OC, BC, dust), mostly every 3rd/6th day

Needs a free NASA Earthdata login. Easiest is a ~/.netrc file containing:
  machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
(then: chmod 600 ~/.netrc)

Usage:
  python download_maia_la.py                       # both products, all dates
  python download_maia_la.py --start 2023-01-01 --end 2025-12-31
  python download_maia_la.py --products total      # only total PM
  python download_maia_la.py --list-only           # just print what would download
"""
import argparse
import os
import re
from collections import defaultdict

import earthaccess

PRODUCTS = {
    "total": "MAIA_ANC_SURFACEMONITOR_PM_TOTAL",
    "species": "MAIA_ANC_SURFACEMONITOR_PM_2.5_SPECIES",
}
TARGET = "USA-LosAngeles"

# ..._20210104T000000Z_F01_V2026p06p10.nc  or  ..._F01_V01p01p01_V001.nc
NAME_RE = re.compile(r"_(\d{8})T\d{6}Z_F(\d+)_V(\d+)p(\d+)p(\d+)(?:_V(\d+))?\.nc$")


def granule_name(g):
    return g["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]


def latest_per_day(granules):
    """ASDC keeps older reprocessed versions of the same day; keep only the newest."""
    by_day = defaultdict(list)
    for g in granules:
        m = NAME_RE.search(granule_name(g))
        if not m:
            by_day[granule_name(g)].append(((0,), g))
            continue
        day = m.group(1)
        version = tuple(int(x) for x in m.groups()[1:5]) + (int(m.group(6) or 0),)
        by_day[day].append((version, g))
    return [max(v, key=lambda t: t[0])[1] for _, v in sorted(by_day.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2030-12-31")
    ap.add_argument("--products", nargs="+", choices=list(PRODUCTS), default=list(PRODUCTS))
    ap.add_argument("--out", default="maia_la_data")
    ap.add_argument("--all-versions", action="store_true",
                    help="keep every reprocessed version instead of the newest per day")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    # Uses ~/.netrc if present, otherwise EARTHDATA_USERNAME/PASSWORD env vars,
    # otherwise asks interactively.
    earthaccess.login()

    for key in args.products:
        short_name = PRODUCTS[key]
        results = earthaccess.search_data(
            short_name=short_name,
            version="C01",
            granule_name=f"*{TARGET}*",
            temporal=(args.start, args.end),
        )
        found = len(results)
        if not args.all_versions:
            results = latest_per_day(results)
        # .size is a property in newer earthaccess, a method in older ones
        size_mb = sum(r.size() if callable(r.size) else r.size for r in results)
        print(f"{short_name}: {found} files found, {len(results)} to download "
              f"(~{size_mb:.0f} MB)")

        if args.list_only:
            for r in results[:5]:
                print("   ", granule_name(r))
            continue

        out_dir = os.path.join(args.out, key)
        os.makedirs(out_dir, exist_ok=True)
        earthaccess.download(results, local_path=out_dir)
        print(f"  saved to {out_dir}/")


if __name__ == "__main__":
    main()
