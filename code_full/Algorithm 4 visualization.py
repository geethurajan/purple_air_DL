# -*- coding: utf-8 -*-
"""
Algorithm4_Visualization.py

Wildfire PM2.5 Calibration Project - Algorithm 4
==================================================

Purpose
-------
This script generates the publication-quality figures for the project,
drawing on the mortality estimates produced by Algorithm 2 (log-linear)
and Algorithm 3 (MLP):

    Required
    --------
    Figure 1: Log-linear estimated excess mortality per 100,000 by tract.
    Figure 2: MLP estimated excess mortality per 100,000 by tract.

    Optional (also produced here)
    ------------------------------
    Figure 3: Difference map, Log-linear minus EPA-only (spatially uniform).
    Figure 4: Difference map, MLP minus EPA-only (spatially uniform).
    Figure 5: Histogram of tract-level mortality (log-linear vs. MLP).
    Figure 6: Histogram of tract-level wildfire-period PM2.5
              (log-linear-calibrated interpolation).
    Figure 7: Interpolation diagnostic - sensor locations used in the
              spline vs. the resulting tract-centroid surface, for a
              representative wildfire day.
    Figure 8: Calibration diagnostic - log-linear coefficient magnitudes.

Tract polygons come from tl_2024_06_tract.shp (filtered to
STATEFP == "06", COUNTYFP == "037") joined on GEOID; maps are clipped to
City_Boundary.geojson, which is used ONLY for the city outline / clip
mask and never as a source of tract geometry, per the project
specification.

This script depends on outputs from Algorithm 2 and Algorithm 3; run those
first. All figures are written to ./outputs/figures/ at publication
resolution (300 DPI).
"""

# =============================================================================
# Imports
# =============================================================================
import os
import sys
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires geopandas (and its dependencies shapely, "
        "fiona/pyogrio, pyproj). Install with:\n"
        "    pip install geopandas --break-system-packages\n"
        "(or `conda install geopandas` if using Anaconda/Spyder)."
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pyogrio")

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CALIB_DIR = os.path.join(OUTPUT_DIR, "calibration")
MORTALITY_DIR = os.path.join(OUTPUT_DIR, "mortality")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _dir in (OUTPUT_DIR, FIGURE_DIR, LOG_DIR):
    os.makedirs(_dir, exist_ok=True)

TRACT_SHAPEFILE = os.path.join(DATA_DIR, "tl_2024_06_tract.shp")
CITY_BOUNDARY_FILE = os.path.join(DATA_DIR, "City_Boundary.geojson")

LOGLINEAR_MORTALITY_FILE = os.path.join(MORTALITY_DIR, "tract_mortality_loglinear.csv")
MLP_MORTALITY_FILE = os.path.join(MORTALITY_DIR, "tract_mortality_mlp.csv")
EPA_ONLY_MORTALITY_FILE = os.path.join(MORTALITY_DIR, "epa_only_mortality.csv")
DAILY_TRACT_PM_FILE = os.path.join(MORTALITY_DIR, "daily_tract_pm_loglinear.csv")
DAILY_SENSOR_PM_FILE = os.path.join(MORTALITY_DIR, "daily_calibrated_pm_by_sensor.csv")
LOGLINEAR_COEF_FILE = os.path.join(CALIB_DIR, "loglinear_coefficients.csv")

FIGURE_DPI = 300
MAP_CRS = "EPSG:4326"  # WGS84 lat/lon, matches typical TIGER/Line + GeoJSON input


# =============================================================================
# Logging
# =============================================================================
def setup_logging() -> logging.Logger:
    """Configure console + file logging for this script."""
    log_filename = os.path.join(
        LOG_DIR, f"algorithm4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("Algorithm4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# =============================================================================
# Loading: mortality tables
# =============================================================================
def load_mortality_csv(filepath: str, label: str) -> pd.DataFrame:
    """Load a tract-level mortality CSV produced by Algorithm 2 or 3.

    Parameters
    ----------
    filepath : str
        Path to the mortality CSV.
    label : str
        Human-readable label used only for logging/error messages.

    Returns
    -------
    pd.DataFrame
        The loaded table, with 'GEOID' coerced to a zero-padded 11-digit
        string to guarantee a clean join against the tract shapefile.

    Raises
    ------
    FileNotFoundError
        If the required upstream script has not been run.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{label} mortality file not found at {filepath}. "
            "Run the corresponding upstream script first."
        )
    df = pd.read_csv(filepath, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].str.zfill(11)
    logger.info("Loaded %s mortality table: %d tracts", label, len(df))
    return df


# =============================================================================
# Loading: geometry
# =============================================================================
def load_tract_polygons(shapefile_path: str) -> "gpd.GeoDataFrame":
    """Load census tract polygons, filtered to LA County, California.

    Parameters
    ----------
    shapefile_path : str
        Path to tl_2024_06_tract.shp.

    Returns
    -------
    gpd.GeoDataFrame
        Columns include 'GEOID' and 'geometry', reprojected to MAP_CRS.
    """
    if not os.path.isfile(shapefile_path):
        raise FileNotFoundError(f"Tract shapefile not found: {shapefile_path}")

    logger.info("Loading tract polygons from %s", shapefile_path)
    tracts = gpd.read_file(shapefile_path)

    required = {"STATEFP", "COUNTYFP", "GEOID"}
    missing = required - set(tracts.columns)
    if missing:
        raise ValueError(f"Tract shapefile is missing required columns: {missing}")

    tracts = tracts[(tracts["STATEFP"] == "06") & (tracts["COUNTYFP"] == "037")].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str).str.zfill(11)

    if tracts.crs is not None and str(tracts.crs) != MAP_CRS:
        tracts = tracts.to_crs(MAP_CRS)
    elif tracts.crs is None:
        logger.warning("Tract shapefile has no defined CRS; assuming %s", MAP_CRS)
        tracts = tracts.set_crs(MAP_CRS)

    logger.info("Filtered to %d LA County (STATEFP=06, COUNTYFP=037) tracts", len(tracts))
    return tracts[["GEOID", "geometry"]]


def load_city_boundary(geojson_path: str) -> "gpd.GeoDataFrame":
    """Load the City of Los Angeles boundary polygon, used only for
    clipping / drawing the city outline (never as tract geometry).

    Parameters
    ----------
    geojson_path : str
        Path to City_Boundary.geojson.

    Returns
    -------
    gpd.GeoDataFrame
        Single-polygon (or multipolygon) geodataframe, reprojected to
        MAP_CRS.
    """
    if not os.path.isfile(geojson_path):
        raise FileNotFoundError(f"City boundary file not found: {geojson_path}")

    logger.info("Loading city boundary from %s", geojson_path)
    boundary = gpd.read_file(geojson_path)

    if boundary.crs is not None and str(boundary.crs) != MAP_CRS:
        boundary = boundary.to_crs(MAP_CRS)
    elif boundary.crs is None:
        logger.warning("City boundary file has no defined CRS; assuming %s", MAP_CRS)
        boundary = boundary.set_crs(MAP_CRS)

    return boundary


def build_clipped_tract_geometry(tracts: "gpd.GeoDataFrame", city_boundary: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Clip tract polygons to the City of Los Angeles boundary.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Full LA County tract polygons with 'GEOID'.
    city_boundary : gpd.GeoDataFrame
        City boundary, used only as a clip mask.

    Returns
    -------
    gpd.GeoDataFrame
        Tract polygons clipped to the city outline.
    """
    logger.info("Clipping tract polygons to the City of Los Angeles boundary")
    clipped = gpd.clip(tracts, city_boundary)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
    logger.info("Retained %d tract polygons after clipping", len(clipped))
    return clipped


# =============================================================================
# Choropleth mapping
# =============================================================================
def merge_mortality_geometry(
    clipped_tracts: "gpd.GeoDataFrame",
    mortality_df: pd.DataFrame,
    value_col: str,
    population_col: str = "population",
) -> "gpd.GeoDataFrame":
    """Merge a mortality table onto clipped tract polygons and compute a
    mortality-per-100,000-population rate for mapping.

    Parameters
    ----------
    clipped_tracts : gpd.GeoDataFrame
        City-clipped tract polygons with 'GEOID'.
    mortality_df : pd.DataFrame
        Mortality table with 'GEOID', value_col, population_col.
    value_col : str
        Column holding total excess mortality (count of deaths) for the
        wildfire window.
    population_col : str
        Column holding tract population.

    Returns
    -------
    gpd.GeoDataFrame
        Clipped tracts merged with mortality data and a computed
        'mortality_per_100k' column.
    """
    merged = clipped_tracts.merge(mortality_df, on="GEOID", how="left")
    merged["mortality_per_100k"] = (
        merged[value_col] / merged[population_col].replace(0, np.nan) * 100_000
    )
    n_missing = merged["mortality_per_100k"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d tract(s) had no matching mortality data or zero population "
            "after the join; these will render blank on the map.", n_missing,
        )
    return merged


def plot_mortality_choropleth(
    merged_gdf: "gpd.GeoDataFrame",
    city_boundary: "gpd.GeoDataFrame",
    title: str,
    output_filename: str,
    cmap: str = "OrRd",
) -> str:
    """Plot a publication-quality choropleth of mortality per 100,000.

    Parameters
    ----------
    merged_gdf : gpd.GeoDataFrame
        Output of merge_mortality_geometry(), containing
        'mortality_per_100k'.
    city_boundary : gpd.GeoDataFrame
        City outline, drawn as a bold border for context.
    title : str
        Figure title.
    output_filename : str
        Filename (not full path) to save under FIGURE_DIR.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=FIGURE_DPI)

    merged_gdf.plot(
        column="mortality_per_100k",
        cmap=cmap,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        legend_kwds={
            "label": "Excess mortality per 100,000 population\n(Jan 7-31 wildfire window)",
            "shrink": 0.65,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    city_boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    legend_elements = [Line2D([0], [0], color="black", lw=1.2, label="City of Los Angeles boundary")]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, output_filename)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved choropleth figure to %s", out_path)
    return out_path


def plot_difference_choropleth(
    merged_gdf: "gpd.GeoDataFrame",
    city_boundary: "gpd.GeoDataFrame",
    model_col: str,
    epa_col: str,
    title: str,
    output_filename: str,
) -> str:
    """Plot a diverging difference map: model estimate minus EPA-only
    (spatially uniform) estimate, both expressed per 100,000 population.

    Parameters
    ----------
    merged_gdf : gpd.GeoDataFrame
        Tract polygons merged with both the model and EPA-only
        mortality-per-100k columns.
    city_boundary : gpd.GeoDataFrame
        City outline for context.
    model_col : str
        Column with the model (log-linear or MLP) mortality per 100k.
    epa_col : str
        Column with the EPA-only mortality per 100k.
    title : str
        Figure title.
    output_filename : str
        Filename (not full path) to save under FIGURE_DIR.

    Returns
    -------
    str
        Path to the saved figure.
    """
    merged_gdf = merged_gdf.copy()
    merged_gdf["difference"] = merged_gdf[model_col] - merged_gdf[epa_col]

    max_abs = np.nanmax(np.abs(merged_gdf["difference"].values))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(9, 9), dpi=FIGURE_DPI)
    merged_gdf.plot(
        column="difference",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        legend_kwds={
            "label": "Difference in mortality per 100,000\n(model minus spatially uniform EPA-only)",
            "shrink": 0.65,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
        ax=ax,
    )
    city_boundary.boundary.plot(ax=ax, color="black", linewidth=1.2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, output_filename)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved difference map to %s", out_path)
    return out_path


# =============================================================================
# Optional diagnostic figures
# =============================================================================
def plot_mortality_histograms(loglinear_df: pd.DataFrame, mlp_df: pd.DataFrame) -> str:
    """Overlaid histogram comparing tract-level mortality-per-100k
    distributions between the log-linear and MLP estimates.

    Parameters
    ----------
    loglinear_df : pd.DataFrame
        Log-linear mortality table with 'M_point' and 'population'.
    mlp_df : pd.DataFrame
        MLP mortality table with 'M_mean' and 'population'.

    Returns
    -------
    str
        Path to the saved figure.
    """
    ll_rate = loglinear_df["M_point"] / loglinear_df["population"].replace(0, np.nan) * 100_000
    mlp_rate = mlp_df["M_mean"] / mlp_df["population"].replace(0, np.nan) * 100_000

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    bins = np.linspace(
        0,
        np.nanmax([ll_rate.max(), mlp_rate.max()]) * 1.05,
        30,
    )
    ax.hist(ll_rate.dropna(), bins=bins, alpha=0.55, label="Log-linear", color="firebrick")
    ax.hist(mlp_rate.dropna(), bins=bins, alpha=0.55, label="MLP", color="steelblue")
    ax.set_xlabel("Excess mortality per 100,000 population")
    ax.set_ylabel("Number of census tracts")
    ax.set_title("Distribution of Tract-Level Excess Mortality Estimates")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_histogram_tract_mortality.png")
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved mortality histogram to %s", out_path)
    return out_path


def plot_pm_histogram(daily_tract_pm: pd.DataFrame) -> str:
    """Histogram of mean wildfire-period PM2.5 by tract.

    Parameters
    ----------
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'], covering the full study window.

    Returns
    -------
    str
        Path to the saved figure.
    """
    wildfire_start = pd.Timestamp("2025-01-07").date()
    wildfire_end = pd.Timestamp("2025-01-31").date()
    mask = (daily_tract_pm["date"] >= wildfire_start) & (daily_tract_pm["date"] <= wildfire_end)
    wildfire_mean = daily_tract_pm.loc[mask].groupby("GEOID")["pm25"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    ax.hist(wildfire_mean.dropna(), bins=30, color="darkorange", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean wildfire-period PM$_{2.5}$ ($\\mu g/m^3$), Jan 7-31")
    ax.set_ylabel("Number of census tracts")
    ax.set_title("Distribution of Tract-Level Wildfire-Period PM$_{2.5}$")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_histogram_tract_pm.png")
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved PM histogram to %s", out_path)
    return out_path


def plot_interpolation_diagnostic(
    daily_sensor_pm: pd.DataFrame,
    daily_tract_pm: pd.DataFrame,
    tracts_xy: pd.DataFrame,
) -> str:
    """Diagnostic scatter: sensor locations (colored by calibrated PM) and
    tract centroids (colored by interpolated PM) for a representative
    wildfire day, in the shared x_km/y_km projection.

    Parameters
    ----------
    daily_sensor_pm : pd.DataFrame
        Columns: ['sensor', 'date', 'x_km', 'y_km', 'calibrated_pm'].
    daily_tract_pm : pd.DataFrame
        Columns: ['date', 'GEOID', 'pm25'].
    tracts_xy : pd.DataFrame
        Columns: ['GEOID', 'x_km', 'y_km'].

    Returns
    -------
    str
        Path to the saved figure.
    """
    wildfire_start = pd.Timestamp("2025-01-07").date()
    wildfire_end = pd.Timestamp("2025-01-31").date()
    candidate_days = sorted(
        d for d in daily_tract_pm["date"].unique() if wildfire_start <= d <= wildfire_end
    )
    if not candidate_days:
        logger.warning("No wildfire-window days available for interpolation diagnostic; skipping.")
        return ""
    diagnostic_day = candidate_days[len(candidate_days) // 2]  # representative mid-window day

    sensors_today = daily_sensor_pm.loc[daily_sensor_pm["date"] == diagnostic_day]
    tract_today = daily_tract_pm.loc[daily_tract_pm["date"] == diagnostic_day].merge(
        tracts_xy, on="GEOID", how="left"
    )

    vmin = min(sensors_today["calibrated_pm"].min(), tract_today["pm25"].min())
    vmax = max(sensors_today["calibrated_pm"].max(), tract_today["pm25"].max())

    fig, ax = plt.subplots(figsize=(8, 7), dpi=FIGURE_DPI)
    tract_scatter = ax.scatter(
        tract_today["x_km"], tract_today["y_km"], c=tract_today["pm25"],
        cmap="viridis", vmin=vmin, vmax=vmax, s=18, marker="s", alpha=0.6,
        label="Tract centroids (interpolated)",
    )
    ax.scatter(
        sensors_today["x_km"], sensors_today["y_km"], c=sensors_today["calibrated_pm"],
        cmap="viridis", vmin=vmin, vmax=vmax, s=80, marker="o", edgecolor="black",
        linewidth=0.8, label="PurpleAir sensors (calibrated)",
    )
    fig.colorbar(tract_scatter, ax=ax, shrink=0.7, label="Calibrated PM$_{2.5}$ ($\\mu g/m^3$)")
    ax.set_xlabel("x (km, relative to EPA monitor)")
    ax.set_ylabel("y (km, relative to EPA monitor)")
    ax.set_title(f"Thin Plate Spline Interpolation Diagnostic - {diagnostic_day}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_interpolation_diagnostic.png")
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved interpolation diagnostic (day=%s) to %s", diagnostic_day, out_path)
    return out_path


def plot_calibration_coefficients(coef_table: pd.DataFrame) -> str:
    """Bar chart of log-linear calibration coefficient magnitudes.

    Parameters
    ----------
    coef_table : pd.DataFrame
        Columns: ['term', 'coefficient'], from loglinear_coefficients.csv.

    Returns
    -------
    str
        Path to the saved figure.
    """
    plot_df = coef_table[coef_table["term"] != "intercept"].copy()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    colors = ["firebrick" if v < 0 else "steelblue" for v in plot_df["coefficient"]]
    ax.barh(plot_df["term"], plot_df["coefficient"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value (log-EPA scale)")
    ax.set_title("Log-Linear Calibration Model Coefficients")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_calibration_coefficients.png")
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved calibration coefficient chart to %s", out_path)
    return out_path


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    """Run the full Algorithm 4 visualization pipeline."""
    logger.info("=" * 70)
    logger.info("ALGORITHM 4: Visualization - START")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: Load mortality tables
    # ------------------------------------------------------------------
    loglinear_mortality = load_mortality_csv(LOGLINEAR_MORTALITY_FILE, "log-linear")
    mlp_mortality = load_mortality_csv(MLP_MORTALITY_FILE, "MLP")
    epa_only_mortality = load_mortality_csv(EPA_ONLY_MORTALITY_FILE, "EPA-only")

    # ------------------------------------------------------------------
    # Step 1: Load and clip tract geometry
    # ------------------------------------------------------------------
    tract_polygons = load_tract_polygons(TRACT_SHAPEFILE)
    city_boundary = load_city_boundary(CITY_BOUNDARY_FILE)
    clipped_tracts = build_clipped_tract_geometry(tract_polygons, city_boundary)

    # ------------------------------------------------------------------
    # Step 2 (REQUIRED): Figure 1 - Log-linear mortality choropleth
    # ------------------------------------------------------------------
    ll_merged = merge_mortality_geometry(clipped_tracts, loglinear_mortality, value_col="M_point")
    plot_mortality_choropleth(
        ll_merged, city_boundary,
        title="Estimated Wildfire-Attributable Excess Mortality\n(Log-Linear Calibration Model)",
        output_filename="fig1_mortality_loglinear.png",
        cmap="OrRd",
    )

    # ------------------------------------------------------------------
    # Step 3 (REQUIRED): Figure 2 - MLP mortality choropleth
    # ------------------------------------------------------------------
    mlp_merged = merge_mortality_geometry(clipped_tracts, mlp_mortality, value_col="M_mean")
    plot_mortality_choropleth(
        mlp_merged, city_boundary,
        title="Estimated Wildfire-Attributable Excess Mortality\n(MLP Calibration Model)",
        output_filename="fig2_mortality_mlp.png",
        cmap="OrRd",
    )

    # ------------------------------------------------------------------
    # Step 4 (OPTIONAL): Difference maps vs. EPA-only
    # ------------------------------------------------------------------
    epa_merged = merge_mortality_geometry(clipped_tracts, epa_only_mortality, value_col="M_point")

    ll_vs_epa = ll_merged.copy()
    ll_vs_epa["epa_mortality_per_100k"] = epa_merged["mortality_per_100k"].values
    plot_difference_choropleth(
        ll_vs_epa, city_boundary,
        model_col="mortality_per_100k", epa_col="epa_mortality_per_100k",
        title="Log-Linear minus EPA-Only Mortality (per 100,000)",
        output_filename="fig3_diff_loglinear_vs_epa.png",
    )

    mlp_vs_epa = mlp_merged.copy()
    mlp_vs_epa["epa_mortality_per_100k"] = epa_merged["mortality_per_100k"].values
    plot_difference_choropleth(
        mlp_vs_epa, city_boundary,
        model_col="mortality_per_100k", epa_col="epa_mortality_per_100k",
        title="MLP minus EPA-Only Mortality (per 100,000)",
        output_filename="fig4_diff_mlp_vs_epa.png",
    )

    # ------------------------------------------------------------------
    # Step 5 (OPTIONAL): Histograms
    # ------------------------------------------------------------------
    plot_mortality_histograms(loglinear_mortality, mlp_mortality)

    if os.path.isfile(DAILY_TRACT_PM_FILE):
        daily_tract_pm = pd.read_csv(DAILY_TRACT_PM_FILE, dtype={"GEOID": str})
        daily_tract_pm["GEOID"] = daily_tract_pm["GEOID"].str.zfill(11)
        daily_tract_pm["date"] = pd.to_datetime(daily_tract_pm["date"]).dt.date
        plot_pm_histogram(daily_tract_pm)
    else:
        daily_tract_pm = None
        logger.warning("Daily tract PM file not found; skipping PM histogram.")

    # ------------------------------------------------------------------
    # Step 6 (OPTIONAL): Interpolation diagnostic
    # ------------------------------------------------------------------
    if daily_tract_pm is not None and os.path.isfile(DAILY_SENSOR_PM_FILE):
        daily_sensor_pm = pd.read_csv(DAILY_SENSOR_PM_FILE)
        daily_sensor_pm["date"] = pd.to_datetime(daily_sensor_pm["date"]).dt.date

        # Tract centroid x_km/y_km are needed for this diagnostic; pull
        # them from the census tract source file directly.
        tract_source_file = os.path.join(DATA_DIR, "la_city_tracts_final.csv")
        if os.path.isfile(tract_source_file):
            tracts_xy = pd.read_csv(tract_source_file, dtype={"GEOID": str})[["GEOID", "x_km", "y_km"]]
            tracts_xy["GEOID"] = tracts_xy["GEOID"].str.zfill(11)
            plot_interpolation_diagnostic(daily_sensor_pm, daily_tract_pm, tracts_xy)
        else:
            logger.warning("la_city_tracts_final.csv not found; skipping interpolation diagnostic.")
    else:
        logger.warning("Daily sensor PM file not found; skipping interpolation diagnostic.")

    # ------------------------------------------------------------------
    # Step 7 (OPTIONAL): Calibration coefficient chart
    # ------------------------------------------------------------------
    if os.path.isfile(LOGLINEAR_COEF_FILE):
        coef_table = pd.read_csv(LOGLINEAR_COEF_FILE)
        plot_calibration_coefficients(coef_table)
    else:
        logger.warning("Log-linear coefficients file not found; skipping calibration chart.")

    logger.info("=" * 70)
    logger.info("ALGORITHM 4 COMPLETE - all figures saved to %s", FIGURE_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()