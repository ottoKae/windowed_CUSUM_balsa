"""Build an independent 2 x 2 chronology/interval figure for the JAG manuscript.

This script deliberately does not import or modify the legacy Figure 5/Figure 6
workflows.  It reads the frozen national VV event table and the validated BALSA
polygon-track table, joins each validated polygon to one source geometry, and
writes a new figure directory containing editable vector output, a raster
preview, source tables, and a machine-readable manifest.

The four panels are:
    (a) detected-event count (chronology depth) for all validated polygons;
    (b) polygon-specific median adjacent interval, where >=2 events are
        required;
    (c) an equal-area hexagon aggregation of (b), with polygon counts printed
        in occupied cells; and
    (d) the distribution of all adjacent event intervals.

The defaults target the current 40-tile release.  All paths can be overridden
from the command line so that the script remains usable with a later frozen
release without editing source code.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer


REPO_ROOT = Path(__file__).resolve().parents[2]
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

EXPECTED_VALIDATED_POLYGONS = 3498
EXPECTED_INTERVAL_POLYGONS = 2194
EXPECTED_INTERVALS = 3509

# A green/gold/blue-green palette is readable for common red-green deficiencies
# and is consistent with the existing manuscript figures.  No red is used.
GREEN_GOLD = ["#F1D27A", "#B8D88D", "#62B780", "#21865F", "#0A554A"]
INTERVAL_COLORS = ["#F1D27A", "#C5DB91", "#79BC8C", "#32928A", "#176B70"]
NO_EVENT = "#D9D9D9"
OUTLINE = "#737373"
TEXT = "#263238"
SUBMISSION_DPI = 600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--polygon-tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--balsa-shp", type=Path, default=DEFAULT_BALSA)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "jag_codex_20260901" / "figure_06_chronology_interval_maps_2x2",
    )
    parser.add_argument("--hex-radius-m", type=float, default=25000.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Continue with a warning if frozen-release counts differ from the defaults",
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _parse_affine(value: object) -> tuple[float, float, float, float, float, float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    values = tuple(float(v) for v in value)
    if len(values) != 6:
        raise ValueError(f"Expected six affine coefficients, got {value!r}")
    return values  # type: ignore[return-value]


def _representative_point(row: pd.Series) -> Point:
    a, b, c, d, e, f = _parse_affine(row["transform"])
    col = float(row["representative_native_col"])
    r = float(row["representative_native_row"])
    return Point(a * (col + 0.5) + b * (r + 0.5) + c, d * (col + 0.5) + e * (r + 0.5) + f)


def _source_crs(row: pd.Series) -> CRS:
    value = row.get("track_crs")
    if pd.notna(value) and str(value).strip():
        return CRS.from_user_input(value)
    tile = str(row.get("tile_id", ""))
    # UTM zone 17S for 17M tiles and 17N tiles are in the northern hemisphere.
    # This fallback is only used when track_crs is absent; normal records carry it.
    zone = int(tile[:2]) if tile[:2].isdigit() else 17
    northern = len(tile) > 2 and tile[2].upper() >= "N"
    return CRS.from_dict({"proj": "utm", "zone": zone, "south": not northern})


def _choose_one_track_per_polygon(tracks: pd.DataFrame) -> pd.DataFrame:
    required = {
        "polygon_id",
        "source_objectid",
        "tile_id",
        "track_crs",
        "transform",
        "representative_native_row",
        "representative_native_col",
        "representative_distance_to_boundary_m",
        "polygon_area_m2",
    }
    missing = sorted(required.difference(tracks.columns))
    if missing:
        raise ValueError(f"polygon_tracks is missing required columns: {missing}")
    out = tracks.copy()
    out["_distance"] = pd.to_numeric(out["representative_distance_to_boundary_m"], errors="coerce").fillna(-np.inf)
    out["_track_key"] = out.get("polygon_track_id", out.index.astype(str)).astype(str)
    out = out.sort_values(["polygon_id", "_distance", "_track_key"], ascending=[True, False, True], kind="mergesort")
    return out.drop_duplicates("polygon_id", keep="first").drop(columns=["_distance", "_track_key"])


def _assignment_for_group(
    polygon_group: pd.DataFrame,
    shape_group: gpd.GeoDataFrame,
    shape_crs: CRS,
) -> tuple[dict[str, int], dict[str, str]]:
    """Assign source geometries one-to-one within each OBJECTID group.

    A representative point contained by a candidate geometry is preferred.  A
    relative-area cost resolves the remaining ambiguities.  The rectangular
    Hungarian assignment prevents one shapefile part from being reused by two
    validated polygons that share an OBJECTID.
    """
    if len(polygon_group) > len(shape_group):
        raise ValueError(
            f"Source OBJECTID {polygon_group['source_objectid'].iloc[0]} has "
            f"{len(polygon_group)} validated polygons but only {len(shape_group)} geometries"
        )
    points: list[Point] = []
    contains: np.ndarray = np.zeros((len(polygon_group), len(shape_group)), dtype=bool)
    for i, (_, row) in enumerate(polygon_group.iterrows()):
        point = _representative_point(row)
        src_crs = _source_crs(row)
        if src_crs != shape_crs:
            transformer = Transformer.from_crs(src_crs, shape_crs, always_xy=True)
            point = shapely_transform(transformer.transform, point)
        points.append(point)
        contains[i, :] = np.asarray([geom.covers(point) for geom in shape_group.geometry], dtype=bool)

    polygon_areas = pd.to_numeric(polygon_group["polygon_area_m2"], errors="coerce").to_numpy(float)
    shape_areas = shape_group.geometry.area.to_numpy(float)
    relative = np.abs(shape_areas[None, :] - polygon_areas[:, None]) / np.maximum(np.abs(polygon_areas[:, None]), 1.0)
    cost = np.log1p(100.0 * relative)
    cost[contains] -= 10.0
    rows, cols = linear_sum_assignment(cost)
    if len(rows) != len(polygon_group):
        raise RuntimeError("Geometry assignment did not assign every validated polygon")
    ids = polygon_group["polygon_id"].astype(str).tolist()
    shape_indices = list(shape_group.index)
    selected: dict[str, int] = {}
    methods: dict[str, str] = {}
    for i, j in zip(rows, cols):
        selected[ids[int(i)]] = int(shape_indices[int(j)])
        methods[ids[int(i)]] = "point_contains" if bool(contains[int(i), int(j)]) else "area_fallback"
    return selected, methods


def _map_geometries(polygons: pd.DataFrame, balsa_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    source = gpd.read_file(balsa_path)
    if source.crs is None:
        raise ValueError("BALSA shapefile has no CRS")
    if "OBJECTID" not in source.columns:
        raise ValueError("BALSA shapefile does not contain OBJECTID")
    source = source[source.geometry.notna() & ~source.geometry.is_empty].copy()
    source["_objectid"] = pd.to_numeric(source["OBJECTID"], errors="coerce").astype("Int64")
    polygons = polygons.copy()
    polygons["_objectid"] = pd.to_numeric(polygons["source_objectid"], errors="coerce").astype("Int64")
    shape_groups = {key: grp for key, grp in source.groupby("_objectid", dropna=False)}
    selected: dict[str, int] = {}
    methods: dict[str, str] = {}
    group_sizes: Counter[str] = Counter()
    for objectid, pgroup in polygons.groupby("_objectid", dropna=False):
        sgroup = shape_groups.get(objectid)
        if sgroup is None or pd.isna(objectid):
            raise ValueError(f"No source geometry for OBJECTID {objectid}")
        result, result_methods = _assignment_for_group(pgroup, sgroup, CRS.from_user_input(source.crs))
        selected.update(result)
        methods.update(result_methods)
        group_sizes[str(objectid)] = len(pgroup)
    if len(selected) != polygons["polygon_id"].nunique():
        raise RuntimeError("Not all validated polygons received a geometry")

    geometry = [source.loc[selected[str(pid)], "geometry"] for pid in polygons["polygon_id"].astype(str)]
    out = gpd.GeoDataFrame(polygons.copy(), geometry=geometry, crs=source.crs)
    out["geometry_match"] = [methods[str(pid)] for pid in polygons["polygon_id"].astype(str)]
    duplicate_assignments = int(pd.Series(list(selected.values())).duplicated().sum())
    qa = {
        "source_features_read": int(len(source)),
        "source_objectid_groups": int(source["_objectid"].nunique()),
        "validated_polygons_mapped": int(len(out)),
        "geometry_duplicate_assignments": duplicate_assignments,
        "geometry_match_counts": dict(Counter(methods.values())),
        "max_validated_per_objectid": int(max(group_sizes.values())),
    }
    if duplicate_assignments:
        raise RuntimeError(f"Geometry assignment was not one-to-one ({duplicate_assignments} duplicates)")
    return out, qa


def _month_number(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError("Event table contains unparseable change_time values")
    return dates.dt.year * 12 + dates.dt.month


def _event_metrics(events_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    events = pd.read_parquet(events_path)
    required = {"polygon_id", "change_time"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"Event table is missing required columns: {missing}")
    events = events.copy()
    events["polygon_id"] = events["polygon_id"].astype(str)
    events["change_time"] = pd.to_datetime(events["change_time"], errors="coerce")
    if events["change_time"].isna().any():
        raise ValueError("Event table contains unparseable change_time values")
    events = events.sort_values(["polygon_id", "change_time"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for polygon_id, group in events.groupby("polygon_id", sort=True):
        months = _month_number(group["change_time"]).to_numpy(int)
        dates = group["change_time"].dt.strftime("%Y-%m").tolist()
        for idx in range(1, len(months)):
            rows.append(
                {
                    "polygon_id": polygon_id,
                    "interval_index": idx,
                    "start_month": dates[idx - 1],
                    "end_month": dates[idx],
                    "interval_months": int(months[idx] - months[idx - 1]),
                }
            )
    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise ValueError("No adjacent intervals could be computed")
    if (intervals["interval_months"] <= 0).any():
        raise ValueError("Event table is not strictly ordered within one or more polygons")
    summary = intervals.groupby("polygon_id", as_index=False).agg(
        interval_median_months=("interval_months", "median"),
        interval_count=("interval_months", "size"),
    )
    summary["event_count"] = summary["interval_count"] + 1
    qa = {
        "event_rows": int(len(events)),
        "event_polygons": int(events["polygon_id"].nunique()),
        "interval_count": int(len(intervals)),
        "interval_polygons": int(len(summary)),
        "overall_interval_median_months": float(intervals["interval_months"].median()),
        "polygon_median_of_medians_months": float(summary["interval_median_months"].median()),
    }
    return events, intervals, qa


def _hex_grid(bounds: Iterable[float], radius: float) -> gpd.GeoDataFrame:
    if radius <= 0:
        raise ValueError("--hex-radius-m must be positive")
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    dx = math.sqrt(3.0) * radius
    dy = 1.5 * radius
    polygons: list[Polygon] = []
    ids: list[str] = []
    y_values = np.arange(ymin - 2 * radius, ymax + 2 * radius, dy)
    for iy, y in enumerate(y_values):
        x_offset = 0.5 * dx if iy % 2 else 0.0
        for ix, x in enumerate(np.arange(xmin - 2 * radius, xmax + 2 * radius, dx)):
            cx = x + x_offset
            vertices = [
                (cx + radius * math.cos(math.radians(30 + k * 60)), y + radius * math.sin(math.radians(30 + k * 60)))
                for k in range(6)
            ]
            polygons.append(Polygon(vertices))
            ids.append(f"h{iy:04d}_{ix:04d}")
    return gpd.GeoDataFrame({"cell_id": ids}, geometry=polygons, crs="EPSG:6933")


def _aggregate_hex(polygons: gpd.GeoDataFrame, radius: float) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    eligible = polygons[polygons["interval_median_months"].notna()].copy()
    ea = eligible.to_crs("EPSG:6933")
    points = gpd.GeoDataFrame(
        ea[["polygon_id", "interval_median_months"]].copy(),
        geometry=ea.geometry.centroid,
        crs=ea.crs,
    )
    grid = _hex_grid(ea.total_bounds, radius)
    joined = gpd.sjoin(points, grid[["cell_id", "geometry"]], how="left", predicate="within")
    if joined["cell_id"].isna().any():
        raise RuntimeError("Some eligible polygon centroids were not assigned to a hexagon")
    metrics = (
        joined.groupby("cell_id", as_index=False)
        .agg(interval_median_months=("interval_median_months", "median"), polygon_count=("polygon_id", "nunique"))
    )
    occupied = grid.merge(metrics, on="cell_id", how="inner").to_crs("EPSG:4326")
    qa = {
        "hex_radius_m": float(radius),
        "hex_cells_total": int(len(grid)),
        "hex_cells_occupied": int(len(occupied)),
        "hex_polygon_count_min": int(occupied["polygon_count"].min()),
        "hex_polygon_count_max": int(occupied["polygon_count"].max()),
    }
    return occupied, qa


def _set_map_extent(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7, length=2, width=0.5, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#9E9E9E")


def _base_map(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
    *,
    show_x_labels: bool = True,
) -> None:
    boundary.boundary.plot(ax=ax, color=OUTLINE, linewidth=0.45, zorder=5)
    _set_map_extent(ax, extent)
    ax.set_xlabel("Longitude" if show_x_labels else "", fontsize=8, color=TEXT, labelpad=2)
    if not show_x_labels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Latitude", fontsize=8, color=TEXT, labelpad=2)
    ax.grid(color="#C7C7C7", linewidth=0.3, alpha=0.55, linestyle=":")
    ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.015, 0.985, label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color=TEXT)


def _write_csvs(
    output_dir: Path,
    polygons: gpd.GeoDataFrame,
    intervals: pd.DataFrame,
    hexes: gpd.GeoDataFrame,
) -> dict[str, str]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    polygon_export = polygons.to_crs("EPSG:4326").copy()
    polygon_centroids = polygons.to_crs("EPSG:6933").geometry.centroid.to_crs("EPSG:4326")
    polygon_export["longitude"] = polygon_centroids.x.to_numpy()
    polygon_export["latitude"] = polygon_centroids.y.to_numpy()
    polygon_export.drop(columns="geometry").to_csv(source_dir / "polygon_metrics.csv", index=False)
    intervals.to_csv(source_dir / "event_intervals.csv", index=False)
    hex_export = hexes.copy()
    hex_centroids = hexes.to_crs("EPSG:6933").geometry.centroid.to_crs("EPSG:4326")
    hex_export["longitude"] = hex_centroids.x.to_numpy()
    hex_export["latitude"] = hex_centroids.y.to_numpy()
    hex_export.drop(columns="geometry").to_csv(source_dir / "hex_metrics.csv", index=False)
    return {
        "polygon_metrics": str((source_dir / "polygon_metrics.csv").resolve()),
        "event_intervals": str((source_dir / "event_intervals.csv").resolve()),
        "hex_metrics": str((source_dir / "hex_metrics.csv").resolve()),
    }


def _draw_figure(
    polygons: gpd.GeoDataFrame,
    intervals: pd.DataFrame,
    hexes: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    output_dir: Path,
    dpi: int,
) -> dict[str, str]:
    display = polygons.to_crs("EPSG:4326")
    boundary = boundary.to_crs("EPSG:4326")
    bounds = display.total_bounds
    bnd_bounds = boundary.total_bounds
    xmin = min(float(bounds[0]), float(bnd_bounds[0]))
    ymin = min(float(bounds[1]), float(bnd_bounds[1]))
    xmax = max(float(bounds[2]), float(bnd_bounds[2]))
    ymax = max(float(bounds[3]), float(bnd_bounds[3]))
    pad_x = max(0.02, 0.025 * (xmax - xmin))
    pad_y = max(0.02, 0.025 * (ymax - ymin))
    extent = (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 7.05), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.965, wspace=0.08, hspace=0.12)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    cmap_count = LinearSegmentedColormap.from_list("chronology_depth", GREEN_GOLD, N=256)
    cmap_interval = LinearSegmentedColormap.from_list("interval_months", INTERVAL_COLORS, N=256)

    _base_map(ax_a, boundary, extent, show_x_labels=False)
    no_event = display[display["event_count"].eq(0)]
    positive = display[display["event_count"].gt(0)]
    if not no_event.empty:
        no_event.plot(ax=ax_a, color=NO_EVENT, edgecolor="none", zorder=1)
    vmax_count = max(1, int(positive["event_count"].max()))
    norm_count = Normalize(vmin=1, vmax=vmax_count)
    if not positive.empty:
        positive.plot(ax=ax_a, column="event_count", cmap=cmap_count, norm=norm_count, edgecolor="none", zorder=2)
    _panel_label(ax_a, "a")
    ax_a.set_title("Chronology depth", loc="left", pad=3, color=TEXT, fontweight="bold")
    sm_a = ScalarMappable(norm=norm_count, cmap=cmap_count)
    sm_a.set_array([])
    cb_a = fig.colorbar(sm_a, ax=ax_a, fraction=0.035, pad=0.015, shrink=0.78)
    cb_a.set_label("Detected-event count", fontsize=7, labelpad=3)
    cb_a.ax.tick_params(labelsize=6, length=2)
    ax_a.legend(handles=[Patch(facecolor=NO_EVENT, edgecolor="none", label="No detected event")], loc="lower left", fontsize=6.5, frameon=True, framealpha=0.92, borderpad=0.35, handlelength=1.0)

    _base_map(ax_b, boundary, extent, show_x_labels=False)
    not_estimable = display[display["interval_median_months"].isna()]
    estimable = display[display["interval_median_months"].notna()]
    if not not_estimable.empty:
        not_estimable.plot(ax=ax_b, color=NO_EVENT, edgecolor="none", zorder=1)
    if not estimable.empty:
        norm_interval = Normalize(vmin=float(estimable["interval_median_months"].min()), vmax=float(estimable["interval_median_months"].max()))
        estimable.plot(ax=ax_b, column="interval_median_months", cmap=cmap_interval, norm=norm_interval, edgecolor="none", zorder=2)
    else:
        norm_interval = Normalize(vmin=0, vmax=1)
    _panel_label(ax_b, "b")
    ax_b.set_title("Polygon-level median interval", loc="left", pad=3, color=TEXT, fontweight="bold")
    sm_b = ScalarMappable(norm=norm_interval, cmap=cmap_interval)
    sm_b.set_array([])
    cb_b = fig.colorbar(sm_b, ax=ax_b, fraction=0.035, pad=0.015, shrink=0.78)
    cb_b.set_label("Adjacent interval (months)", fontsize=7, labelpad=3)
    cb_b.ax.tick_params(labelsize=6, length=2)
    ax_b.legend(handles=[Patch(facecolor=NO_EVENT, edgecolor="none", label="Not estimable (<2 events)")], loc="lower left", fontsize=6.5, frameon=True, framealpha=0.92, borderpad=0.35, handlelength=1.0)

    _base_map(ax_c, boundary, extent)
    hexes.plot(ax=ax_c, column="interval_median_months", cmap=cmap_interval, norm=norm_interval, edgecolor="#FFFFFF", linewidth=0.35, zorder=2)
    for _, row in hexes.iterrows():
        point = row.geometry.representative_point()
        ax_c.text(point.x, point.y, f"n={int(row['polygon_count'])}", ha="center", va="center", fontsize=5.3, color=TEXT, zorder=4)
    _panel_label(ax_c, "c")
    ax_c.set_title("Spatially aggregated interval", loc="left", pad=3, color=TEXT, fontweight="bold")
    sm_c = ScalarMappable(norm=norm_interval, cmap=cmap_interval)
    sm_c.set_array([])
    cb_c = fig.colorbar(sm_c, ax=ax_c, fraction=0.035, pad=0.015, shrink=0.78)
    cb_c.set_label("Regional median interval (months)", fontsize=7, labelpad=3)
    cb_c.ax.tick_params(labelsize=6, length=2)

    ax_d.hist(intervals["interval_months"].to_numpy(), bins=np.arange(0.5, intervals["interval_months"].max() + 1.5, 1), color="#4E9F83", edgecolor="#FFFFFF", linewidth=0.25, alpha=0.92, label="All adjacent intervals")
    overall = float(intervals["interval_months"].median())
    estimable_intervals = polygons.loc[polygons["interval_median_months"].notna(), "interval_median_months"]
    polygon_median = float(estimable_intervals.median())
    ax_d.axvline(overall, color="#0A554A", linestyle="--", linewidth=1.1, label=f"Overall median = {overall:.0f} months")
    ax_d.axvline(polygon_median, color="#C49A2E", linestyle="-.", linewidth=1.1, label=f"Median polygon median = {polygon_median:.0f} months")
    ax_d.set_xlabel("Adjacent inter-disturbance interval (months)", fontsize=8)
    ax_d.set_ylabel("Intervals (n)", fontsize=8)
    ax_d.set_title("Distribution and statistical support", loc="left", pad=3, color=TEXT, fontweight="bold")
    _panel_label(ax_d, "d")
    ax_d.grid(axis="y", color="#D0D0D0", linewidth=0.35, linestyle=":")
    ax_d.set_axisbelow(True)
    ax_d.tick_params(labelsize=7, length=2, width=0.5, colors=TEXT)
    ax_d.legend(loc="upper right", fontsize=6.6, frameon=True, framealpha=0.94, borderpad=0.35, handlelength=1.6)
    for spine in ax_d.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#9E9E9E")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_06_chronology_interval_maps_2x2"
    outputs = {
        "pdf": str(stem.with_suffix(".pdf").resolve()),
        "svg": str(stem.with_suffix(".svg").resolve()),
        "png": str(stem.with_suffix(".png").resolve()),
    }
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outputs["png"], dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    outputs["tiff"] = str(stem.with_suffix(".tiff").resolve())
    fig.savefig(outputs["tiff"], dpi=dpi, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return outputs


def build(args: argparse.Namespace) -> dict[str, object]:
    for path, label in [
        (args.events, "events parquet"),
        (args.polygon_tracks, "polygon tracks parquet"),
        (args.balsa_shp, "BALSA shapefile"),
        (args.boundary, "Ecuador boundary"),
    ]:
        _require_file(path, label)
    tracks = pd.read_parquet(args.polygon_tracks)
    polygons = _choose_one_track_per_polygon(tracks)
    if polygons["polygon_id"].nunique() != EXPECTED_VALIDATED_POLYGONS and not args.allow_count_mismatch:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATED_POLYGONS} validated polygons, found {polygons['polygon_id'].nunique()}. "
            "Use --allow-count-mismatch only for a deliberately different frozen release."
        )
    events, intervals, event_qa = _event_metrics(args.events)
    interval_summary = intervals.groupby("polygon_id", as_index=False).agg(
        interval_median_months=("interval_months", "median"),
        interval_count=("interval_months", "size"),
    )
    interval_summary["event_count_from_intervals"] = interval_summary["interval_count"] + 1
    event_counts = events.groupby("polygon_id", as_index=False).size().rename(columns={"size": "event_count"})
    polygons = polygons.copy()
    polygons["polygon_id"] = polygons["polygon_id"].astype(str)
    polygons = polygons.merge(event_counts, on="polygon_id", how="left")
    polygons = polygons.merge(interval_summary[["polygon_id", "interval_median_months", "interval_count"]], on="polygon_id", how="left")
    polygons["event_count"] = polygons["event_count"].fillna(0).astype(int)
    polygons["interval_count"] = polygons["interval_count"].fillna(0).astype(int)
    polygons["has_estimable_interval"] = polygons["interval_median_months"].notna()
    if int(polygons["event_count"].sum()) != len(events):
        raise RuntimeError("Event-to-polygon join changed the event count")
    if int(polygons["event_count"].gt(0).sum()) != event_qa["event_polygons"]:
        raise RuntimeError("Event polygon count is inconsistent after the join")
    if len(intervals) != EXPECTED_INTERVALS and not args.allow_count_mismatch:
        raise ValueError(f"Expected {EXPECTED_INTERVALS} intervals, found {len(intervals)}")
    if int(polygons["has_estimable_interval"].sum()) != EXPECTED_INTERVAL_POLYGONS and not args.allow_count_mismatch:
        raise ValueError(
            f"Expected {EXPECTED_INTERVAL_POLYGONS} polygons with >=2 events, found {int(polygons['has_estimable_interval'].sum())}"
        )
    polygons, geometry_qa = _map_geometries(polygons, args.balsa_shp)
    boundary = gpd.read_file(args.boundary)
    if boundary.crs is None:
        raise ValueError("Ecuador boundary has no CRS")
    hexes, hex_qa = _aggregate_hex(polygons, args.hex_radius_m)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _write_csvs(args.output_dir, polygons, intervals, hexes)
    figure_paths = _draw_figure(polygons, intervals, hexes, boundary, args.output_dir, args.dpi)
    rounded = {
        "overall_interval_median_months_rounded": int(round(float(intervals["interval_months"].median()))),
        "polygon_median_of_medians_months_rounded": int(round(float(polygons.loc[polygons["interval_median_months"].notna(), "interval_median_months"].median()))),
    }
    if rounded["overall_interval_median_months_rounded"] != 36 or rounded["polygon_median_of_medians_months_rounded"] != 37:
        print("WARNING: computed medians differ from the requested 36/37-month annotations", file=sys.stderr)
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
        "figure_definition": {
            "panel_a": "Validated BALSA polygons colored by detected-event count; zero-event polygons are light gray",
            "panel_b": "Polygon-specific median adjacent interval; polygons with fewer than two events are light gray",
            "panel_c": "25-km equal-area hexagons colored by median polygon interval and annotated with polygon count",
            "panel_d": "All adjacent event intervals with overall and polygon-median reference lines",
        },
        "counts": {
            "validated_polygons": int(len(polygons)),
            "polygons_with_events": int(polygons["event_count"].gt(0).sum()),
            "polygons_without_events": int(polygons["event_count"].eq(0).sum()),
            "polygons_with_estimable_interval": int(polygons["has_estimable_interval"].sum()),
            "polygons_not_estimable": int(polygons["has_estimable_interval"].eq(False).sum()),
            **event_qa,
        },
        "geometry_qa": geometry_qa,
        "hex_qa": hex_qa,
        "statistics": {**rounded, **event_qa},
        "source_data": source_paths,
        "outputs": figure_paths,
        "parameters": {"hex_radius_m": float(args.hex_radius_m), "dpi": int(args.dpi)},
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = build(args)
    print(json.dumps({"outputs": manifest["outputs"], "counts": manifest["counts"], "manifest": manifest["manifest"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



