"""Build fixed-grid hex-cell maps of stand-disturbance chronology.

Two independent national maps are written:

* the median detected-event count per inventory polygon in each occupied hexagon;
* the median polygon-specific adjacent inter-disturbance interval in each
  occupied hexagon.

The map frame, hillshade background, national extent, locator inset and detail
callouts follow the national panel of the study-area figure.  The hexagon grid
is deliberately frozen: both metrics use the same 25-km equal-area grid and a
recorded grid origin.  The script does not modify the legacy study-area or
chronology figures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, MultipleLocator
from shapely.geometry import Polygon


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "build_study_area_figure.py"
CHRONOLOGY_SCRIPT = Path(__file__).with_name("build_figure6_chronology_interval_maps.py")
CONFIG_PATH = REPO_ROOT / "configs" / "study_area_figure_v1.json"

DEFAULT_EVENTS = (
    REPO_ROOT
    / "outputs"
    / "windowed_cusum_n1_full_40tile_v1"
    / "national"
    / "events"
    / "VV_dB.ground.parquet"
)
DEFAULT_TRACKS = (
    REPO_ROOT
    / "outputs"
    / "forestry_sample_rebuild"
    / "20260825_balsa_validated_40tiles_v1_phase_d_full_cache_v1"
    / "polygon_tracks.parquet"
)
DEFAULT_BALSA = (
    REPO_ROOT
    / "shp"
    / "balsa_polygons"
    / "FC001_PLANTACION_FORESTAL_2025_A_BALSA"
    / "FC001_PLANTACION_FORESTAL_2025_A_BALSA_ID.shp"
)
DEFAULT_BOUNDARY = REPO_ROOT / "shp" / "Ecuador_0_without_islands" / "Ecuador_0_without_islands.shp"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "jag_codex_20260903" / "hex_disturbance_maps"

EXPECTED_POLYGONS = 3498
EXPECTED_EVENTS = 6658
EXPECTED_INTERVALS = 3509
EXPECTED_INTERVAL_POLYGONS = 2194

# Frozen 25-km pointy-top grid anchor.  It is the anchor used by the existing
# chronology map release, retained here so this new count map and interval map
# are directly comparable.  Coordinates are in EPSG:6933 metres.
HEX_RADIUS_M = 25_000.0
GRID_ORIGIN_X_M = -7_840_161.330176934
GRID_ORIGIN_Y_M = -656_936.0631179377

COUNT_COLORS = ["#F1D27A", "#C7DE9A", "#80C28D", "#3D996F", "#0C5B4E"]
INTERVAL_COLORS = ["#F1D27A", "#C5DB91", "#79BC8C", "#32928A", "#176B70"]
NO_SUPPORT = "#D9D9D9"
TEXT = "#263238"
BOUNDARY = "#4D4D4D"
GRID = "#9A9A9A"
HEX_EDGE = "#FFFFFF"
SUBMISSION_DPI = 600


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--polygon-tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--balsa-shp", type=Path, default=DEFAULT_BALSA)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hex-radius-m", type=float, default=HEX_RADIUS_M)
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow a deliberately different frozen input release.",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _event_and_polygon_metrics(args: argparse.Namespace, chronology: Any) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tracks = pd.read_parquet(args.polygon_tracks)
    polygons = chronology._choose_one_track_per_polygon(tracks)
    polygons["polygon_id"] = polygons["polygon_id"].astype(str)
    if polygons["polygon_id"].nunique() != EXPECTED_POLYGONS and not args.allow_count_mismatch:
        raise ValueError(
            f"Expected {EXPECTED_POLYGONS} validated polygons, found "
            f"{polygons['polygon_id'].nunique()}"
        )

    events, intervals, event_qa = chronology._event_metrics(args.events)
    event_counts = events.groupby("polygon_id", as_index=False).size().rename(columns={"size": "event_count"})
    interval_summary = intervals.groupby("polygon_id", as_index=False).agg(
        interval_median_months=("interval_months", "median"),
        interval_count=("interval_months", "size"),
    )
    polygons = polygons.merge(event_counts, on="polygon_id", how="left")
    polygons = polygons.merge(interval_summary, on="polygon_id", how="left")
    polygons["event_count"] = polygons["event_count"].fillna(0).astype(int)
    polygons["interval_count"] = polygons["interval_count"].fillna(0).astype(int)
    polygons["has_estimable_interval"] = polygons["interval_median_months"].notna()

    if len(events) != EXPECTED_EVENTS and not args.allow_count_mismatch:
        raise ValueError(f"Expected {EXPECTED_EVENTS} event rows, found {len(events)}")
    if len(intervals) != EXPECTED_INTERVALS and not args.allow_count_mismatch:
        raise ValueError(f"Expected {EXPECTED_INTERVALS} intervals, found {len(intervals)}")
    if int(polygons["has_estimable_interval"].sum()) != EXPECTED_INTERVAL_POLYGONS and not args.allow_count_mismatch:
        raise ValueError(
            f"Expected {EXPECTED_INTERVAL_POLYGONS} interval-eligible polygons, found "
            f"{int(polygons['has_estimable_interval'].sum())}"
        )

    polygons, geometry_qa = chronology._map_geometries(polygons, args.balsa_shp)
    polygon_points = polygons.to_crs("EPSG:6933").copy()
    polygon_points["geometry"] = polygon_points.geometry.centroid
    return polygon_points, intervals, {
        "events": events,
        "event_qa": event_qa,
        "geometry_qa": geometry_qa,
    }


def _fixed_hex_grid(points: gpd.GeoDataFrame, radius_m: float) -> gpd.GeoDataFrame:
    if radius_m <= 0:
        raise ValueError("--hex-radius-m must be positive")
    dx = math.sqrt(3.0) * radius_m
    dy = 1.5 * radius_m
    xmin, ymin, xmax, ymax = map(float, points.total_bounds)
    iy_min = math.floor((ymin - radius_m - GRID_ORIGIN_Y_M) / dy) - 1
    iy_max = math.ceil((ymax + radius_m - GRID_ORIGIN_Y_M) / dy) + 1
    ix_min = math.floor((xmin - radius_m - GRID_ORIGIN_X_M) / dx) - 1
    ix_max = math.ceil((xmax + radius_m - GRID_ORIGIN_X_M) / dx) + 1

    polygons: list[Polygon] = []
    ids: list[str] = []
    rows: list[int] = []
    cols: list[int] = []
    for iy in range(iy_min, iy_max + 1):
        cy = GRID_ORIGIN_Y_M + iy * dy
        x_offset = 0.5 * dx if iy % 2 else 0.0
        for ix in range(ix_min, ix_max + 1):
            cx = GRID_ORIGIN_X_M + ix * dx + x_offset
            vertices = [
                (
                    cx + radius_m * math.cos(math.radians(30.0 + k * 60.0)),
                    cy + radius_m * math.sin(math.radians(30.0 + k * 60.0)),
                )
                for k in range(6)
            ]
            polygons.append(Polygon(vertices))
            ids.append(f"y{iy:+06d}_x{ix:+06d}")
            rows.append(iy)
            cols.append(ix)
    return gpd.GeoDataFrame(
        {"cell_id": ids, "grid_row": rows, "grid_col": cols},
        geometry=polygons,
        crs="EPSG:6933",
    )


def _assign_hexes(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(
        points[["polygon_id", "event_count", "interval_median_months", "geometry"]],
        grid[["cell_id", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined["cell_id"].isna().any():
        missing = joined.loc[joined["cell_id"].isna(), ["polygon_id", "geometry"]]
        nearest = gpd.sjoin_nearest(missing, grid[["cell_id", "geometry"]], how="left")
        nearest = nearest.sort_values(["polygon_id", "cell_id"], kind="mergesort")
        fill = nearest.drop_duplicates("polygon_id", keep="first").set_index("polygon_id")["cell_id"]
        joined.loc[joined["cell_id"].isna(), "cell_id"] = joined.loc[
            joined["cell_id"].isna(), "polygon_id"
        ].map(fill)
    if joined["cell_id"].isna().any():
        raise RuntimeError("Some polygon centroids could not be assigned to the fixed hex grid")
    # A centroid on a hex boundary can create two spatial-join rows.  The
    # deterministic cell-id tie break ensures one polygon contributes once.
    return (
        joined.sort_values(["polygon_id", "cell_id"], kind="mergesort")
        .drop_duplicates("polygon_id", keep="first")
        .drop(columns="geometry")
        .reset_index(drop=True)
    )


def _aggregate_metrics(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, radius_m: float) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    assigned = _assign_hexes(points, grid)
    occupied_ids = assigned["cell_id"].drop_duplicates()
    occupied_grid = grid[grid["cell_id"].isin(occupied_ids)].copy()

    count_metrics = assigned.groupby("cell_id", as_index=False).agg(
        median_event_count=("event_count", "median"),
        inventory_polygon_n=("polygon_id", "nunique"),
        total_event_count=("event_count", "sum"),
    )
    count_hexes = occupied_grid.merge(count_metrics, on="cell_id", how="left")

    interval_support = assigned[assigned["interval_median_months"].notna()].copy()
    interval_metrics = interval_support.groupby("cell_id", as_index=False).agg(
        median_interval_months=("interval_median_months", "median"),
        interval_polygon_n=("polygon_id", "nunique"),
    )
    total_support = assigned.groupby("cell_id", as_index=False).agg(
        inventory_polygon_n=("polygon_id", "nunique")
    )
    interval_hexes = occupied_grid.merge(total_support, on="cell_id", how="left")
    interval_hexes = interval_hexes.merge(interval_metrics, on="cell_id", how="left")
    qa = {
        "hex_radius_m": float(radius_m),
        "grid_origin_epsg6933_m": [float(GRID_ORIGIN_X_M), float(GRID_ORIGIN_Y_M)],
        "grid_spacing_m": [float(math.sqrt(3.0) * radius_m), float(1.5 * radius_m)],
        "hex_cells_generated": int(len(grid)),
        "hex_cells_occupied": int(len(occupied_grid)),
        "count_hex_cells": int(len(count_hexes)),
        "interval_hex_cells_with_support": int(interval_metrics["cell_id"].nunique()),
        "count_polygon_n_min": int(count_hexes["inventory_polygon_n"].min()),
        "count_polygon_n_max": int(count_hexes["inventory_polygon_n"].max()),
        "interval_support_n_min": int(interval_hexes["interval_polygon_n"].fillna(0).min()),
        "interval_support_n_max": int(interval_hexes["interval_polygon_n"].fillna(0).max()),
        "polygon_assignments": int(len(assigned)),
        "polygon_assignment_unique": int(assigned["polygon_id"].nunique()),
    }
    if qa["polygon_assignments"] != qa["polygon_assignment_unique"]:
        raise RuntimeError("A polygon was assigned to more than one fixed hex cell")
    return count_hexes, interval_hexes, qa


def _format_lon(value: float, position: int | None = None) -> str:
    return source_module.format_longitude(value, position, 0)  # type: ignore[name-defined]


def _format_lat(value: float, position: int | None = None) -> str:
    return source_module.format_latitude(value, position, 0)  # type: ignore[name-defined]


def _draw_locator(ax: plt.Axes, extent: tuple[float, float, float, float], image: np.ndarray | None, countries: gpd.GeoDataFrame) -> None:
    colors = {"ocean": "#DCEEF7"}
    ax.set_facecolor(colors["ocean"])
    if image is not None:
        ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear", alpha=0.25, zorder=0, aspect="auto")
    local = source_module.clipped(countries, extent)  # type: ignore[name-defined]
    local.plot(ax=ax, facecolor="#F7F6F1", edgecolor="#777777", linewidth=0.28, alpha=0.78, zorder=1)
    ecuador = source_module.select_locator_ecuador(countries)  # type: ignore[name-defined]
    ecuador = source_module.clipped(ecuador, extent)  # type: ignore[name-defined]
    ecuador.plot(ax=ax, facecolor="#F2C744", edgecolor="#246B45", linewidth=0.38, alpha=0.72, zorder=3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    source_module.set_geographic_aspect(ax, extent)  # type: ignore[name-defined]
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#555555")


def _draw_map_context(
    ax: plt.Axes,
    ecuador: gpd.GeoDataFrame,
    amazonia: gpd.GeoDataFrame,
    ocean: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame,
    basemap: np.ndarray | None,
    locator_basemap: np.ndarray | None,
    config: Mapping[str, Any],
    extent: tuple[float, float, float, float],
) -> None:
    colors = dict(config["figure"]["colors"])
    source_module.draw_basemap(ax, basemap, extent, colors, float(config["basemap"]["alpha"]))  # type: ignore[name-defined]
    local_ocean = source_module.clipped(ocean, extent)  # type: ignore[name-defined]
    if not local_ocean.empty:
        local_ocean.plot(ax=ax, facecolor="#DCEEF7", edgecolor="none", alpha=0.72, zorder=1)
    local_amazonia = source_module.clipped(amazonia, extent)  # type: ignore[name-defined]
    if not local_amazonia.empty:
        local_amazonia.plot(ax=ax, facecolor="#DCEFD2", edgecolor="none", alpha=0.30, zorder=2)
    ecuador.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.85, zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.xaxis.set_major_locator(MultipleLocator(float(config["national_panel"]["longitude_tick_step"])))
    ax.yaxis.set_major_locator(MultipleLocator(float(config["national_panel"]["latitude_tick_step"])))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_lat))
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, direction="out", length=2.6, width=0.5, pad=1.5, labelsize=8)
    ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, direction="out", length=2.6, width=0.5, pad=1.5, labelsize=8)
    for tick in ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_rotation_mode("anchor")
        tick.set_va("center")
    ax.grid(color=GRID, linestyle="--", linewidth=0.35, alpha=0.28, zorder=3)
    source_module.set_geographic_aspect(ax, extent)  # type: ignore[name-defined]

    for label in config["national_panel"]["regional_labels"]:
        text = ax.text(
            float(label["longitude"]),
            float(label["latitude"]),
            str(label["text"]),
            ha="center",
            va="center",
            rotation=float(label.get("rotation", 0.0)),
            rotation_mode="anchor",
            fontsize=8,
            fontstyle="italic",
            color="#4F5660",
            alpha=0.92,
            zorder=20,
        )
        source_module.apply_text_halo(text, width=1.6)  # type: ignore[name-defined]

    # The standalone metric maps retain the study-area background but omit the
    # old (b)--(d) detail callout frames.  Those frames were useful in the
    # inventory composite, but would obscure hex-cell support labels here.
    locator_bounds = tuple(float(v) for v in config["national_panel"]["locator_axes"])
    locator_extent = tuple(float(v) for v in config["national_panel"]["locator_extent"])
    locator_ax = ax.inset_axes(locator_bounds, zorder=94)
    _draw_locator(locator_ax, locator_extent, locator_basemap, countries)


def _draw_metric_map(
    hexes: gpd.GeoDataFrame,
    ecuador: gpd.GeoDataFrame,
    amazonia: gpd.GeoDataFrame,
    ocean: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame,
    basemap: np.ndarray | None,
    locator_basemap: np.ndarray | None,
    config: Mapping[str, Any],
    extent: tuple[float, float, float, float],
    metric: str,
    cmap_colors: list[str],
    title: str,
    colorbar_label: str,
    support_column: str,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    display = hexes.to_crs("EPSG:4326")
    values = display.loc[display[metric].notna(), metric]
    cmap = LinearSegmentedColormap.from_list(metric, cmap_colors, N=256)
    if values.empty:
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        vmin = 0.0 if metric == "median_event_count" else float(values.min())
        vmax = max(vmin + 1.0, float(values.max()))
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(180.0 / 25.4, 112.0 / 25.4), facecolor="white")
    ax = fig.add_axes([0.055, 0.065, 0.89, 0.88])
    _draw_map_context(ax, ecuador, amazonia, ocean, countries, basemap, locator_basemap, config, extent)

    unsupported = display[display[metric].isna()]
    supported = display[display[metric].notna()]
    if not unsupported.empty:
        unsupported.plot(ax=ax, facecolor=NO_SUPPORT, edgecolor=HEX_EDGE, linewidth=0.42, alpha=0.86, zorder=6)
    if not supported.empty:
        supported.plot(ax=ax, column=metric, cmap=cmap, norm=norm, edgecolor=HEX_EDGE, linewidth=0.42, alpha=0.88, zorder=7)

    for _, row in display.iterrows():
        value = row[support_column]
        if pd.isna(value):
            label = "n=0"
        else:
            label = f"n={int(value)}"
        point = row.geometry.representative_point()
        ax.text(point.x, point.y, label, ha="center", va="center", fontsize=5.1, color=TEXT, zorder=12)

    ax.text(0.012, 0.978, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color=TEXT, zorder=100, bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88})
    ax.set_title(title, loc="left", pad=3, fontsize=9, fontweight="bold", color=TEXT)

    # The original panel used this lower-right quiet area for the area key;
    # it remains free of inventory cells and is therefore a stable metric key.
    cax = ax.inset_axes([0.57, 0.045, 0.18, 0.105], zorder=95)
    cax.set_facecolor((1.0, 1.0, 1.0, 0.91))
    for spine in cax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.4)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label(colorbar_label, fontsize=6.5, labelpad=1.2)
    cbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.4, pad=1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / ("hex_stand_disturbance_count" if metric == "median_event_count" else "hex_inter_disturbance_interval")
    paths = {
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "svg": str(stem.with_suffix(".svg").resolve()),
        "png": str(stem.with_suffix(".png").resolve()),
        "tiff": str(stem.with_suffix(".tiff").resolve()),
    }
    metadata = {"Title": title, "Creator": "build_hex_disturbance_maps.py"}
    fig.savefig(paths["pdf"], dpi=dpi, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(paths["svg"], dpi=dpi, bbox_inches="tight", pad_inches=0.04, metadata={"Title": title, "Creator": metadata["Creator"]})
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(paths["tiff"], dpi=dpi, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def _write_source_tables(output_dir: Path, count_hexes: gpd.GeoDataFrame, interval_hexes: gpd.GeoDataFrame) -> dict[str, str]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    count = count_hexes.to_crs("EPSG:4326").copy()
    count_centroids = count_hexes.to_crs("EPSG:6933").geometry.centroid.to_crs("EPSG:4326")
    count["longitude"] = count_centroids.x.to_numpy()
    count["latitude"] = count_centroids.y.to_numpy()
    interval = interval_hexes.to_crs("EPSG:4326").copy()
    interval_centroids = interval_hexes.to_crs("EPSG:6933").geometry.centroid.to_crs("EPSG:4326")
    interval["longitude"] = interval_centroids.x.to_numpy()
    interval["latitude"] = interval_centroids.y.to_numpy()
    count_path = source_dir / "hex_disturbance_count.csv"
    interval_path = source_dir / "hex_inter_disturbance_interval.csv"
    count.drop(columns="geometry").to_csv(count_path, index=False)
    interval.drop(columns="geometry").to_csv(interval_path, index=False)
    return {"count": str(count_path.resolve()), "interval": str(interval_path.resolve())}


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in [
        (args.events, "event parquet"),
        (args.polygon_tracks, "polygon-track parquet"),
        (args.balsa_shp, "BALSA shapefile"),
        (args.boundary, "Ecuador boundary"),
    ]:
        _require_file(path, label)

    global source_module
    source_module = _load_module(SOURCE_SCRIPT, "study_area_renderer_for_hex_maps")
    chronology = _load_module(CHRONOLOGY_SCRIPT, "chronology_metrics_for_hex_maps")
    config = source_module.apply_layout_variant(source_module.load_config(CONFIG_PATH), "v8")
    polygons, intervals, metric_qa = _event_and_polygon_metrics(args, chronology)

    boundary = gpd.read_file(args.boundary)
    if boundary.crs is None:
        raise ValueError("Ecuador boundary has no CRS")
    boundary = boundary.to_crs("EPSG:4326")
    ecuador = boundary
    amazonia_path = source_module.resolve_path(REPO_ROOT, config["data"]["amazonia_boundary"])
    amazonia = gpd.read_file(amazonia_path).to_crs("EPSG:4326")
    ocean, _ = source_module.natural_earth_layer("10m", "physical", "ocean", REPO_ROOT)
    countries, _ = source_module.natural_earth_layer("110m", "cultural", "admin_0_countries", REPO_ROOT)

    national_extent = source_module.national_extent(ecuador, float(config["national_panel"]["margin_fraction"]))
    national_extent = tuple(float(v) for v in national_extent)
    cache_dir = source_module.resolve_path(REPO_ROOT, config["outputs"]["basemap_cache"])
    basemap, _, basemap_hash = source_module.build_basemap_mosaic(
        national_extent,
        int(config["basemap"]["national_zoom"]),
        "cache",
        cache_dir,
        config["basemap"],
        REPO_ROOT,
    )
    locator_extent = tuple(float(v) for v in config["national_panel"]["locator_extent"])
    locator, _, locator_hash = source_module.build_basemap_mosaic(
        locator_extent,
        int(config["basemap"]["locator_zoom"]),
        "cache",
        cache_dir,
        config["basemap"],
        REPO_ROOT,
    )

    grid = _fixed_hex_grid(polygons, float(args.hex_radius_m))
    count_hexes, interval_hexes, hex_qa = _aggregate_metrics(polygons, grid, float(args.hex_radius_m))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _write_source_tables(args.output_dir, count_hexes, interval_hexes)
    count_paths = _draw_metric_map(
        count_hexes,
        ecuador,
        amazonia,
        ocean,
        countries,
        basemap,
        locator,
        config,
        national_extent,
        "median_event_count",
        COUNT_COLORS,
        "Hex-cell stand-disturbance count",
        "Median events per polygon",
        "inventory_polygon_n",
        args.output_dir,
        int(args.dpi),
    )
    interval_paths = _draw_metric_map(
        interval_hexes,
        ecuador,
        amazonia,
        ocean,
        countries,
        basemap,
        locator,
        config,
        national_extent,
        "median_interval_months",
        INTERVAL_COLORS,
        "Hex-cell inter-disturbance interval",
        "Median adjacent interval (months)",
        "interval_polygon_n",
        args.output_dir,
        int(args.dpi),
    )

    grid_spec = {
        "crs": "EPSG:6933",
        "geometry": "regular pointy-top hexagon",
        "radius_m": float(args.hex_radius_m),
        "origin_x_m": float(GRID_ORIGIN_X_M),
        "origin_y_m": float(GRID_ORIGIN_Y_M),
        "dx_m": float(math.sqrt(3.0) * args.hex_radius_m),
        "dy_m": float(1.5 * args.hex_radius_m),
        "assignment": "inventory polygon centroid within fixed hexagon",
        "count_aggregation": "median event_count across all inventory polygons in the cell",
        "interval_aggregation": "median polygon-specific median adjacent interval among interval-eligible polygons",
        "interval_support": "polygons with fewer than two detected events are not estimable and remain gray",
    }
    (args.output_dir / "grid_spec.json").write_text(json.dumps(grid_spec, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "events": str(args.events.resolve()),
            "polygon_tracks": str(args.polygon_tracks.resolve()),
            "balsa_shp": str(args.balsa_shp.resolve()),
            "boundary": str(args.boundary.resolve()),
        },
        "input_sha256": {key: _sha256(path) for key, path in {
            "events": args.events,
            "polygon_tracks": args.polygon_tracks,
            "balsa_shp": args.balsa_shp,
            "boundary": args.boundary,
        }.items()},
        "definitions": {
            "count": "median detected-event count per inventory polygon, aggregated by hex cell",
            "interval": "median polygon-specific adjacent inter-disturbance interval, aggregated by hex cell",
            "n_count": "number of inventory polygons contributing to the count median",
            "n_interval": "number of interval-eligible polygons contributing to the interval median",
        },
        "counts": {
            "inventory_polygons": int(len(polygons)),
            "event_rows": int(len(metric_qa["events"])),
            "interval_rows": int(len(intervals)),
            "polygons_with_events": int(polygons["event_count"].gt(0).sum()),
            "polygons_with_estimable_interval": int(polygons["has_estimable_interval"].sum()),
        },
        "event_qa": metric_qa["event_qa"],
        "geometry_qa": metric_qa["geometry_qa"],
        "hex_qa": hex_qa,
        "national_extent_lonlat": list(national_extent),
        "basemap": {"national_sha256": basemap_hash, "locator_sha256": locator_hash},
        "grid_spec": str((args.output_dir / "grid_spec.json").resolve()),
        "source_data": source_paths,
        "outputs": {"count": count_paths, "interval": interval_paths},
        "parameters": {"hex_radius_m": float(args.hex_radius_m), "dpi": int(args.dpi)},
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    manifest = build(_parse_args())
    print(json.dumps({"outputs": manifest["outputs"], "counts": manifest["counts"], "hex_qa": manifest["hex_qa"], "manifest": manifest["manifest"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()




