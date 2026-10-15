"""Build fixed-width national hex maps of BALSA disturbance chronology.

The fixed EPSG:6933 grid covers mainland Ecuador and is independent of the
plantation locations. Each validated inventory polygon contributes once, via
an interior representative point, to (1) the median detected-event count and
(2) the median polygon-specific adjacent inter-disturbance interval.

This workflow supersedes the diagnostic 25-km map without modifying it.
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
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = Path(__file__).with_name("build_hex_disturbance_maps.py")
CHRONOLOGY_SCRIPT = Path(__file__).with_name("build_figure6_chronology_interval_maps.py")
SOURCE_SCRIPT = REPO_ROOT / "scripts" / "build_study_area_figure.py"
CONFIG_PATH = REPO_ROOT / "configs" / "study_area_figure_v1.json"

DEFAULT_EVENTS = REPO_ROOT / "outputs" / "windowed_cusum_n1_full_40tile_v1" / "national" / "events" / "VV_dB.ground.parquet"
DEFAULT_TRACKS = REPO_ROOT / "outputs" / "forestry_sample_rebuild" / "20260825_balsa_validated_40tiles_v1_phase_d_full_cache_v1" / "polygon_tracks.parquet"
DEFAULT_BALSA = REPO_ROOT / "shp" / "balsa_polygons" / "FC001_PLANTACION_FORESTAL_2025_A_BALSA" / "FC001_PLANTACION_FORESTAL_2025_A_BALSA_ID.shp"
DEFAULT_BOUNDARY = REPO_ROOT / "shp" / "Ecuador_0_without_islands" / "Ecuador_0_without_islands.shp"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "jag_codex_20260903" / "hex_disturbance_maps_1km2_v1"

EXPECTED_POLYGONS = 3498
EXPECTED_EVENTS = 6658
EXPECTED_INTERVALS = 3509
EXPECTED_INTERVAL_POLYGONS = 2194
EXPECTED_OVERALL_INTERVAL_MEDIAN = 36.0
EXPECTED_POLYGON_INTERVAL_MEDIAN = 37.0

GRID_CRS = "EPSG:6933"
GRID_CELL_AREA_M2 = 1_000_000.0
GRID_RADIUS_M = math.sqrt(2.0 * GRID_CELL_AREA_M2 / (3.0 * math.sqrt(3.0)))
DEFAULT_FLAT_WIDTH_M = math.sqrt(3.0) * GRID_RADIUS_M
GRID_ORIGIN_X_M = 0.0
GRID_ORIGIN_Y_M = 0.0
SUBMISSION_DPI = 600

COUNT_COLORS = ["#F1D27A", "#D9DB82", "#AFCF87", "#73B982", "#32906F", "#0C5B4E"]
INTERVAL_COLORS = ["#F1D27A", "#D3DA85", "#9CCB89", "#5EAD88", "#277F79"]
EMPTY_CELL = "#F2F0E8"
NOT_ESTIMABLE = "#BDBDBD"
HEX_EDGE = "#FFFFFF"
TEXT = "#263238"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_vector(path: Path | str) -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


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
    parser.add_argument(
        "--hex-flat-width-m",
        type=float,
        default=DEFAULT_FLAT_WIDTH_M,
        help="Distance in metres between opposite parallel sides of each hexagon.",
    )
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def _load_polygon_metrics(args: argparse.Namespace, chronology: Any) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tracks = pd.read_parquet(args.polygon_tracks)
    polygons = chronology._choose_one_track_per_polygon(tracks)
    polygons["polygon_id"] = polygons["polygon_id"].astype(str)
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
    if args.balsa_shp.suffix.lower() in {".parquet", ".geoparquet"}:
        inventory = gpd.read_parquet(args.balsa_shp)
        inventory["polygon_id"] = inventory["polygon_id"].astype(str)
        geometry = inventory[["polygon_id", "geometry"]].drop_duplicates("polygon_id")
        polygons = gpd.GeoDataFrame(
            polygons.merge(geometry, on="polygon_id", how="left", validate="one_to_one"),
            geometry="geometry",
            crs=inventory.crs,
        )
        if polygons.geometry.isna().any():
            raise RuntimeError("Validated GeoParquet does not cover every polygon ID")
        geometry_qa = {
            "source_features_read": int(len(inventory)),
            "validated_polygons_mapped": int(len(polygons)),
            "geometry_duplicate_assignments": 0,
            "mapping": "direct polygon_id join from validated GeoParquet",
        }
    else:
        polygons, geometry_qa = chronology._map_geometries(polygons, args.balsa_shp)

    expected = {
        "polygons": (polygons["polygon_id"].nunique(), EXPECTED_POLYGONS),
        "events": (len(events), EXPECTED_EVENTS),
        "intervals": (len(intervals), EXPECTED_INTERVALS),
        "interval_polygons": (int(polygons["has_estimable_interval"].sum()), EXPECTED_INTERVAL_POLYGONS),
    }
    if not args.allow_count_mismatch:
        bad = {key: pair for key, pair in expected.items() if pair[0] != pair[1]}
        if bad:
            raise ValueError(f"Frozen-input count mismatch: {bad}")
    if not math.isclose(float(intervals["interval_months"].median()), EXPECTED_OVERALL_INTERVAL_MEDIAN):
        raise ValueError("Overall adjacent-interval median is no longer 36 months")
    polygon_interval_median = float(polygons.loc[polygons["has_estimable_interval"], "interval_median_months"].median())
    if not math.isclose(polygon_interval_median, EXPECTED_POLYGON_INTERVAL_MEDIAN):
        raise ValueError("Median of polygon-specific interval medians is no longer 37 months")

    ea = polygons.to_crs(GRID_CRS)
    points = ea.copy()
    points["geometry"] = ea.geometry.representative_point()
    if not np.all(ea.geometry.covers(points.geometry)):
        raise RuntimeError("One or more representative points fall outside their source polygons")
    qa = {"event_qa": event_qa, "geometry_qa": geometry_qa, "expected_counts": expected}
    return points, events, intervals, qa


def _build_mainland_grid(boundary: gpd.GeoDataFrame, flat_width_m: float) -> gpd.GeoDataFrame:
    if flat_width_m <= 0:
        raise ValueError("--hex-flat-width-m must be positive")
    radius_m = flat_width_m / math.sqrt(3.0)
    mainland = boundary.to_crs(GRID_CRS).geometry.union_all()
    xmin, ymin, xmax, ymax = mainland.bounds
    dx = flat_width_m
    dy = 1.5 * radius_m
    row_min = math.floor((ymin - radius_m - GRID_ORIGIN_Y_M) / dy) - 1
    row_max = math.ceil((ymax + radius_m - GRID_ORIGIN_Y_M) / dy) + 1
    col_min = math.floor((xmin - radius_m - GRID_ORIGIN_X_M) / dx) - 2
    col_max = math.ceil((xmax + radius_m - GRID_ORIGIN_X_M) / dx) + 2

    row_values = np.arange(row_min, row_max + 1, dtype=np.int32)
    col_values = np.arange(col_min, col_max + 1, dtype=np.int32)
    rows = np.repeat(row_values, len(col_values))
    cols = np.tile(col_values, len(row_values))
    centers_x = GRID_ORIGIN_X_M + cols * dx + (rows % 2) * (0.5 * dx)
    centers_y = GRID_ORIGIN_Y_M + rows * dy
    angles = np.deg2rad(30.0 + np.arange(6, dtype=float) * 60.0)
    coordinates = np.empty((len(rows), 6, 2), dtype=np.float64)
    coordinates[:, :, 0] = centers_x[:, None] + radius_m * np.cos(angles)[None, :]
    coordinates[:, :, 1] = centers_y[:, None] + radius_m * np.sin(angles)[None, :]
    candidates = shapely.polygons(coordinates)
    keep = shapely.intersects(candidates, mainland)
    rows = rows[keep]
    cols = cols[keep]
    geometries = candidates[keep]
    ids = [f"r{row:+07d}_c{col:+07d}" for row, col in zip(rows, cols)]
    grid = gpd.GeoDataFrame({"hex_id": ids, "grid_row": rows, "grid_col": cols}, geometry=geometries, crs=GRID_CRS)
    if grid["hex_id"].duplicated().any():
        raise RuntimeError("Duplicate hex IDs")
    return grid.sort_values(["grid_row", "grid_col"], kind="mergesort").reset_index(drop=True)


def _assign_polygons(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
    columns = ["polygon_id", "event_count", "interval_median_months", "interval_count", "geometry"]
    joined = gpd.sjoin(points[columns], grid[["hex_id", "geometry"]], how="left", predicate="within")
    if joined["hex_id"].isna().any():
        missing = joined.loc[joined["hex_id"].isna(), columns]
        nearest = gpd.sjoin_nearest(missing, grid[["hex_id", "geometry"]], how="left")
        nearest = nearest.sort_values(["polygon_id", "hex_id"], kind="mergesort").drop_duplicates("polygon_id")
        lookup = nearest.set_index("polygon_id")["hex_id"]
        mask = joined["hex_id"].isna()
        joined.loc[mask, "hex_id"] = joined.loc[mask, "polygon_id"].map(lookup)
    assigned = joined.sort_values(["polygon_id", "hex_id"], kind="mergesort").drop_duplicates("polygon_id")
    if assigned["hex_id"].isna().any() or len(assigned) != points["polygon_id"].nunique():
        raise RuntimeError("Each polygon was not assigned to exactly one mainland hex")
    return assigned.drop(columns=["geometry", "index_right"]).reset_index(drop=True)


def _aggregate(grid: gpd.GeoDataFrame, assigned: pd.DataFrame) -> gpd.GeoDataFrame:
    count = assigned.groupby("hex_id", as_index=False).agg(
        median_event_count=("event_count", "median"),
        inventory_polygon_n=("polygon_id", "nunique"),
        total_detected_events=("event_count", "sum"),
    )
    eligible = assigned[assigned["interval_median_months"].notna()]
    interval = eligible.groupby("hex_id", as_index=False).agg(
        median_interval_months=("interval_median_months", "median"),
        interval_polygon_n=("polygon_id", "nunique"),
    )
    metrics = grid.merge(count, on="hex_id", how="left").merge(interval, on="hex_id", how="left")
    metrics["inventory_polygon_n"] = metrics["inventory_polygon_n"].fillna(0).astype(int)
    metrics["interval_polygon_n"] = metrics["interval_polygon_n"].fillna(0).astype(int)
    metrics["total_detected_events"] = metrics["total_detected_events"].fillna(0).astype(int)
    return metrics


def _grid_hash(grid: gpd.GeoDataFrame) -> str:
    digest = hashlib.sha256()
    for row in grid[["hex_id", "geometry"]].itertuples(index=False):
        digest.update(row.hex_id.encode("ascii"))
        digest.update(row.geometry.wkb)
    return digest.hexdigest()


def _grid_tag(flat_width_m: float) -> str:
    if math.isclose(flat_width_m, DEFAULT_FLAT_WIDTH_M, rel_tol=0.0, abs_tol=1e-6):
        return "1km2"
    width_km = flat_width_m / 1000.0
    return f"{width_km:g}km_width".replace(".", "p")


def _prepare_display(metrics: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Keep complete, fixed hexagons in the plotting frame. Border cells extend
    # by at most one 620-m radius beyond the coastline, below the national-map
    # line width, while avoiding an expensive and irreproducible clipped-cell
    # geometry variant.
    return metrics.to_crs("EPSG:4326")


def _plot_layer(ax: plt.Axes, frame: gpd.GeoDataFrame, **kwargs: Any) -> None:
    if frame.empty:
        return
    frame.plot(ax=ax, rasterized=True, **kwargs)


def _draw_map(
    display: gpd.GeoDataFrame,
    metric: str,
    output_stem: Path,
    title: str,
    source: Any,
    legacy: Any,
    context: dict[str, Any],
    dpi: int,
) -> dict[str, str]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    # 180 x 112 mm, suitable for a double-column manuscript figure.
    fig = plt.figure(figsize=(7.0866, 4.4094), facecolor="white")
    ax = fig.add_axes([0.055, 0.065, 0.89, 0.88])
    legacy._draw_map_context(
        ax,
        context["ecuador"],
        context["amazonia"],
        context["ocean"],
        context["countries"],
        context["basemap"],
        context["locator"],
        context["config"],
        context["extent"],
    )

    empty = display[display["inventory_polygon_n"].eq(0)]
    occupied = display[display["inventory_polygon_n"].gt(0)]
    _plot_layer(ax, empty, facecolor=EMPTY_CELL, edgecolor="#D8D6CE", linewidth=0.035, alpha=0.34, zorder=6)

    support_items = [(EMPTY_CELL, "No inventory polygon")]
    if metric == "median_event_count":
        supported = occupied[occupied[metric].notna()]
        cmap = LinearSegmentedColormap.from_list("event_count", COUNT_COLORS, N=256)
        norm = Normalize(vmin=0.0, vmax=7.0)
        _plot_layer(ax, supported, column=metric, cmap=cmap, norm=norm, edgecolor=HEX_EDGE, linewidth=0.08, alpha=0.92, zorder=8)
        cax = ax.inset_axes([0.865, 0.315, 0.024, 0.255], zorder=95)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=np.arange(0, 8))
        cbar.ax.set_title("Median events\nper polygon", fontsize=6.4, loc="left", pad=3)
    else:
        not_estimable = occupied[occupied[metric].isna()]
        supported = occupied[occupied[metric].notna()]
        _plot_layer(ax, not_estimable, facecolor=NOT_ESTIMABLE, edgecolor=HEX_EDGE, linewidth=0.08, alpha=0.82, zorder=7)
        bounds = np.array([0.0, 24.0, 36.0, 48.0, 60.0, 102.0])
        cmap = LinearSegmentedColormap.from_list("interval", INTERVAL_COLORS, N=5)
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        _plot_layer(ax, supported, column=metric, cmap=cmap, norm=norm, edgecolor=HEX_EDGE, linewidth=0.08, alpha=0.92, zorder=8)
        cax = ax.inset_axes([0.865, 0.315, 0.024, 0.255], zorder=95)
        ticks = (bounds[:-1] + bounds[1:]) / 2.0
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=ticks)
        cbar.ax.set_yticklabels(["<24", "24–35", "36–47", "48–59", "≥60"])
        cbar.ax.set_title("Median interval\n(months)", fontsize=6.4, loc="left", pad=3)
        support_items.append((NOT_ESTIMABLE, "Not estimable (<2 events)"))

    cax.set_facecolor((1, 1, 1, 0.94))
    for spine in cax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=5.8, length=1.8, width=0.4, pad=1.5)
    box_y = 0.665
    box_height = 0.034 + 0.028 * len(support_items)
    ax.add_patch(Rectangle((0.740, box_y), 0.250, box_height, transform=ax.transAxes, facecolor=(1, 1, 1, 0.92), edgecolor="#777777", linewidth=0.4, zorder=96))
    for index, (color, label) in enumerate(support_items):
        y = box_y + box_height - 0.023 - index * 0.028
        ax.add_patch(Rectangle((0.750, y - 0.0065), 0.014, 0.013, transform=ax.transAxes, facecolor=color, edgecolor="#9A9890", linewidth=0.35, zorder=97))
        ax.text(0.773, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=5.7, color=TEXT, zorder=97)
    ax.set_title(title, loc="left", pad=3, fontsize=9, fontweight="bold", color=TEXT)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")
    paths = {
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "png": str(png_path.resolve()),
        "tiff": str(tiff_path.resolve()),
    }
    metadata = {"Title": title, "Creator": Path(__file__).name}
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, metadata={"Title": title, "Creator": metadata["Creator"]})
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def _context(args: argparse.Namespace, source: Any, legacy: Any) -> dict[str, Any]:
    config_path = Path(getattr(args, "study_area_config", CONFIG_PATH))
    config = source.apply_layout_variant(source.load_config(config_path), "v8")
    boundary = _read_vector(args.boundary)
    if boundary.crs is None:
        raise ValueError("Ecuador mainland boundary has no CRS")
    ecuador = boundary.to_crs("EPSG:4326")
    amazonia_path = getattr(args, "amazonia_boundary", None)
    if amazonia_path is None:
        amazonia_path = source.resolve_path(REPO_ROOT, config["data"]["amazonia_boundary"])
    amazonia = _read_vector(amazonia_path).to_crs("EPSG:4326")
    ocean, _ = source.natural_earth_layer("10m", "physical", "ocean", REPO_ROOT)
    countries, _ = source.natural_earth_layer("110m", "cultural", "admin_0_countries", REPO_ROOT)
    extent = tuple(float(v) for v in source.national_extent(ecuador, float(config["national_panel"]["margin_fraction"])))
    cache_dir = getattr(args, "basemap_cache", None)
    if cache_dir is None:
        cache_dir = source.resolve_path(REPO_ROOT, config["outputs"]["basemap_cache"])
    cache_dir = Path(cache_dir)
    basemap, _, basemap_hash = source.build_basemap_mosaic(extent, int(config["basemap"]["national_zoom"]), "cache", cache_dir, config["basemap"], REPO_ROOT)
    locator_extent = tuple(float(v) for v in config["national_panel"]["locator_extent"])
    locator, _, locator_hash = source.build_basemap_mosaic(locator_extent, int(config["basemap"]["locator_zoom"]), "cache", cache_dir, config["basemap"], REPO_ROOT)
    legacy.source_module = source
    return {
        "boundary": boundary,
        "ecuador": ecuador,
        "amazonia": amazonia,
        "ocean": ocean,
        "countries": countries,
        "extent": extent,
        "basemap": basemap,
        "locator": locator,
        "config": config,
        "basemap_hash": basemap_hash,
        "locator_hash": locator_hash,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.events, args.polygon_tracks, args.balsa_shp, args.boundary):
        if not path.exists():
            raise FileNotFoundError(path)
    source = _load_module(SOURCE_SCRIPT, "study_area_source_1km2")
    legacy = _load_module(LEGACY_SCRIPT, "legacy_hex_context_1km2")
    chronology = _load_module(CHRONOLOGY_SCRIPT, "chronology_source_1km2")
    context = _context(args, source, legacy)
    points, events, intervals, input_qa = _load_polygon_metrics(args, chronology)

    flat_width_m = float(args.hex_flat_width_m)
    radius_m = flat_width_m / math.sqrt(3.0)
    cell_area_m2 = math.sqrt(3.0) * flat_width_m * flat_width_m / 2.0
    grid_tag = _grid_tag(flat_width_m)
    grid = _build_mainland_grid(context["boundary"], flat_width_m)
    assigned = _assign_polygons(points, grid)
    metrics = _aggregate(grid, assigned)
    display = _prepare_display(metrics, context["boundary"])
    grid_hash = _grid_hash(grid)

    occupied = metrics[metrics["inventory_polygon_n"].gt(0)]
    interval_supported = metrics[metrics["interval_polygon_n"].gt(0)]
    qa = {
        "mainland_hex_cells": int(len(grid)),
        "hex_flat_width_m": flat_width_m,
        "hex_cell_area_km2": cell_area_m2 / 1_000_000.0,
        "occupied_count_cells": int(len(occupied)),
        "interval_supported_cells": int(len(interval_supported)),
        "polygon_assignments": int(len(assigned)),
        "unique_assigned_polygons": int(assigned["polygon_id"].nunique()),
        "event_count_polygon_min_median_max": [int(points["event_count"].min()), float(points["event_count"].median()), int(points["event_count"].max())],
        "event_count_hex_min_median_max": [float(occupied["median_event_count"].min()), float(occupied["median_event_count"].median()), float(occupied["median_event_count"].max())],
        "interval_polygon_min_median_max_months": [float(points["interval_median_months"].min()), float(points["interval_median_months"].median()), float(points["interval_median_months"].max())],
        "interval_hex_min_median_max_months": [float(interval_supported["median_interval_months"].min()), float(interval_supported["median_interval_months"].median()), float(interval_supported["median_interval_months"].max())],
        "overall_adjacent_interval_median_months": float(intervals["interval_months"].median()),
        "polygon_interval_median_of_medians_months": float(points["interval_median_months"].median()),
        "count_cell_support": {str(int(k)): int(v) for k, v in occupied["inventory_polygon_n"].value_counts().sort_index().items()},
        "interval_cell_support": {str(int(k)): int(v) for k, v in interval_supported["interval_polygon_n"].value_counts().sort_index().items()},
    }
    if qa["polygon_assignments"] != EXPECTED_POLYGONS or qa["unique_assigned_polygons"] != EXPECTED_POLYGONS:
        raise RuntimeError("Polygon assignment conservation failed")
    if qa["event_count_hex_min_median_max"][0] < 0 or qa["event_count_hex_min_median_max"][2] > 7:
        raise RuntimeError("Hex event-count median is outside the polygon-level 0–7 range")
    if int(metrics["inventory_polygon_n"].sum()) != EXPECTED_POLYGONS:
        raise RuntimeError("Count support does not sum to 3,498 polygons")
    if int(metrics["interval_polygon_n"].sum()) != EXPECTED_INTERVAL_POLYGONS:
        raise RuntimeError("Interval support does not sum to 2,194 polygons")

    output = args.output_dir
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    grid_path = source_dir / f"ecuador_mainland_hex_{grid_tag}_epsg6933.parquet"
    metrics_path = source_dir / f"hex_metrics_{grid_tag}_epsg6933.parquet"
    assignments_path = source_dir / "polygon_hex_assignments.csv"
    count_path = source_dir / "hex_event_count_metrics.csv"
    interval_path = source_dir / "hex_interval_metrics.csv"
    grid.to_parquet(grid_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    assigned.sort_values("polygon_id").to_csv(assignments_path, index=False)
    occupied.drop(columns="geometry").to_csv(count_path, index=False)
    interval_supported.drop(columns="geometry").to_csv(interval_path, index=False)

    count_outputs = _draw_map(display, "median_event_count", output / f"hex_median_detected_event_count_{grid_tag}", "Median stand-disturbance count", source, legacy, context, args.dpi)
    interval_outputs = _draw_map(display, "median_interval_months", output / f"hex_median_inter_disturbance_interval_{grid_tag}", "Median inter-disturbance interval", source, legacy, context, args.dpi)

    grid_spec = {
        "version": "ecuador-mainland-equal-area-hex/1.0",
        "crs": GRID_CRS,
        "cell_area_m2": cell_area_m2,
        "cell_area_km2": cell_area_m2 / 1_000_000.0,
        "radius_and_side_m": radius_m,
        "flat_to_flat_width_m": flat_width_m,
        "row_center_spacing_m": 1.5 * radius_m,
        "origin_m": [GRID_ORIGIN_X_M, GRID_ORIGIN_Y_M],
        "orientation": "pointy-top",
        "mainland_selection": "retain full fixed hexagons intersecting the mainland boundary; preserve complete cells in source data and display",
        "polygon_assignment": "one interior representative point per validated polygon; one polygon contributes to one hex",
        "grid_sha256": grid_hash,
    }
    (output / "grid_spec.json").write_text(json.dumps(grid_spec, indent=2) + "\n", encoding="utf-8")
    (output / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {"events": str(args.events.resolve()), "polygon_tracks": str(args.polygon_tracks.resolve()), "balsa_shp": str(args.balsa_shp.resolve()), "boundary": str(args.boundary.resolve())},
        "input_sha256": {name: _sha256(path) for name, path in {"events": args.events, "polygon_tracks": args.polygon_tracks, "balsa_shp": args.balsa_shp, "boundary": args.boundary}.items()},
        "definitions": {
            "count": "median polygon detected-event count among all validated inventory polygons assigned to each cell",
            "interval": "median of polygon-specific median adjacent-event intervals among polygons with at least two events assigned to each cell",
            "support": "polygon contributor counts are retained in source data and are not printed as map values",
        },
        "counts": {"polygons": len(points), "events": len(events), "intervals": len(intervals), "interval_polygons": int(points["interval_median_months"].notna().sum())},
        "input_qa": input_qa,
        "grid_spec": grid_spec,
        "qa": qa,
        "source_data": {"grid": str(grid_path.resolve()), "metrics": str(metrics_path.resolve()), "assignments": str(assignments_path.resolve()), "count": str(count_path.resolve()), "interval": str(interval_path.resolve())},
        "outputs": {"count": count_outputs, "interval": interval_outputs},
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    result = build(_parse_args())
    print(json.dumps({"outputs": result["outputs"], "qa": result["qa"], "manifest": result["manifest"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
