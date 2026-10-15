"""Build a compact national recurrence figure on a fixed 10-km hex grid.

The four panels show (a) median detected-event count among validated BALSA
inventory polygons in each occupied hexagon, (b) annual event
counts, (c) the polygon-level event-count distribution, and (d) the adjacent
inter-disturbance interval distribution. The map uses a fixed, mainland-wide
EPSG:6933 grid and assigns each polygon exactly once by an interior
representative point.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
HEX_HELPER = Path(__file__).with_name("build_hex_disturbance_maps_1km2.py")
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "jag_codex_20260904" / "figure6_recurrence_10km_v21_ylabels_aligned"
STUDY_AREA_REFERENCE_PDF = REPO_ROOT / "manuscript_JAG_v2" / "figures" / "fig02_study_area.pdf"

EXPECTED_POLYGONS = 3498
EXPECTED_EVENTS = 6658
EXPECTED_EVENT_POLYGONS = 3149
EXPECTED_YEARS = tuple(range(2017, 2026))
HEX_FLAT_WIDTH_M = 10_000.0
SUBMISSION_DPI = 600
FIG_WIDTH_MM = 180.0
FIG_HEIGHT_MM = 107.0
RIGHT_COLUMN_SHIFT_MM = 9.0
RIGHT_COLUMN_WIDTH_REDUCTION_MM = 4.5

YELLOW = "#F1D27A"
PALE_YELLOW = "#EADB9B"
PALE_GREEN = "#AFCF87"
GREEN = "#73B982"
DARK_GREEN = "#0C5B4E"
GRID_EDGE = "#FFFFFF"
EMPTY_CELL = "#F2F0E8"
TEXT = "#263238"
NEUTRAL = "#777777"


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


def _pdf_media_box_inches(path: Path) -> tuple[float, float]:
    """Read the first PDF MediaBox without adding a PDF dependency."""
    if not path.exists():
        raise FileNotFoundError(path)
    match = re.search(
        rb"/MediaBox\s*\[\s*0(?:\.0*)?\s+0(?:\.0*)?\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        path.read_bytes(),
    )
    if match is None:
        raise RuntimeError(f"Cannot read PDF MediaBox: {path}")
    return float(match.group(1)) / 72.0, float(match.group(2)) / 72.0


def _parse_args() -> argparse.Namespace:
    helper = _load_module(HEX_HELPER, "hex_helper_defaults")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=helper.DEFAULT_EVENTS)
    parser.add_argument("--polygon-tracks", type=Path, default=helper.DEFAULT_TRACKS)
    parser.add_argument("--balsa-shp", type=Path, default=helper.DEFAULT_BALSA)
    parser.add_argument("--boundary", type=Path, default=helper.DEFAULT_BOUNDARY)
    parser.add_argument("--vh-events", type=Path)
    parser.add_argument("--vh-tolerance-months", type=int, default=2)
    parser.add_argument("--study-area-config", type=Path, default=helper.CONFIG_PATH)
    parser.add_argument("--amazonia-boundary", type=Path)
    parser.add_argument("--basemap-cache", type=Path)
    parser.add_argument(
        "--study-area-reference-pdf",
        type=Path,
        default=STUDY_AREA_REFERENCE_PDF,
        help="PDF whose MediaBox fixes the exact Figure 8 export dimensions.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=SUBMISSION_DPI)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def _aggregate_10km(grid: gpd.GeoDataFrame, assigned: pd.DataFrame) -> gpd.GeoDataFrame:
    summary = assigned.groupby("hex_id", as_index=False).agg(
        mean_event_count=("event_count", "mean"),
        median_event_count=("event_count", "median"),
        inventory_polygon_n=("polygon_id", "nunique"),
        total_detected_events=("event_count", "sum"),
    )
    result = grid.merge(summary, on="hex_id", how="left")
    result["inventory_polygon_n"] = result["inventory_polygon_n"].fillna(0).astype(int)
    result["total_detected_events"] = result["total_detected_events"].fillna(0).astype(int)
    return result


def _annual_counts(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["change_time"] = pd.to_datetime(frame["change_time"], errors="raise")
    counts = frame.groupby(frame["change_time"].dt.year).size().reindex(EXPECTED_YEARS, fill_value=0)
    result = counts.rename("annual_event_count").rename_axis("year").reset_index()
    result["cumulative_event_count"] = result["annual_event_count"].cumsum()
    return result


def _polygon_distribution(assigned: pd.DataFrame) -> pd.DataFrame:
    maximum = int(assigned["event_count"].max())
    counts = assigned["event_count"].value_counts().reindex(range(maximum + 1), fill_value=0).sort_index()
    result = counts.rename("polygon_count").rename_axis("event_count").reset_index()
    result["polygon_percent"] = 100.0 * result["polygon_count"] / result["polygon_count"].sum()
    return result


def _interval_distribution(intervals: pd.DataFrame) -> pd.DataFrame:
    """Summarize all adjacent intervals in fixed six-month bins."""
    bin_edges = np.arange(0.0, 109.0, 6.0)
    values = intervals["interval_months"].to_numpy(dtype=float)
    counts, edges = np.histogram(values, bins=bin_edges)
    result = pd.DataFrame(
        {
            "bin_start_month": edges[:-1],
            "bin_end_month": edges[1:],
            "bin_midpoint_month": (edges[:-1] + edges[1:]) / 2.0,
            "interval_count": counts,
        }
    )
    result["interval_percent"] = 100.0 * result["interval_count"] / result["interval_count"].sum()
    return result


def _polarization_evidence(
    vv_path: Path,
    vh_path: Path,
    tolerance_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | float]]:
    """Build VH support overlays without altering the primary VV chronology."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from lr_cusum.disturbance_intervals import (  # noqa: PLC0415
        build_adjacent_intervals,
        match_polygon_polarizations,
    )

    vv = pd.read_parquet(vv_path)
    vh = pd.read_parquet(vh_path)
    annotated_vv, vh_only, matched = match_polygon_polarizations(
        pd.concat([vv, vh], ignore_index=True),
        tolerance_months=tolerance_months,
    )
    intervals = build_adjacent_intervals(annotated_vv)
    years = pd.Index(EXPECTED_YEARS, name="year")
    annual_supported = (
        annotated_vv.loc[annotated_vv["vh_supported"]]
        .assign(year=lambda frame: frame["change_time"].dt.year)
        .groupby("year")
        .size()
        .reindex(years, fill_value=0)
        .rename("vh_supported_event_count")
        .reset_index()
    )
    supported_intervals = intervals.loc[
        intervals["both_endpoints_vh_supported"]
    ].copy()
    supported_distribution = _interval_distribution(supported_intervals)
    qa = {
        "primary_vv_events": int(len(annotated_vv)),
        "polygon_first_vh_events": int(len(vh)),
        "vv_events_with_vh_support": int(annotated_vv["vh_supported"].sum()),
        "vv_support_fraction": float(annotated_vv["vh_supported"].mean()),
        "unmatched_vh_events": int(len(vh_only)),
        "matched_pairs": int(len(matched)),
        "primary_vv_intervals": int(len(intervals)),
        "intervals_with_both_endpoints_supported": int(len(supported_intervals)),
        "supported_interval_fraction": float(
            intervals["both_endpoints_vh_supported"].mean()
        ),
        "primary_interval_median_months": float(intervals["interval_months"].median()),
        "supported_interval_median_months": float(
            supported_intervals["interval_months"].median()
        ),
    }
    return annotated_vv, annual_supported, supported_distribution, qa


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.11,
    y: float = 1.025,
    y_offset_pt: float = 0.0,
) -> None:
    transform = ax.transAxes + ScaledTranslation(
        0.0,
        y_offset_pt / 72.0,
        ax.figure.dpi_scale_trans,
    )
    ax.text(x, y, label, transform=transform, ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT)


def _black_frame(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.55)


def _plot_figure(
    display: gpd.GeoDataFrame,
    annual: pd.DataFrame,
    distribution: pd.DataFrame,
    interval_distribution: pd.DataFrame,
    helper: Any,
    legacy: Any,
    context: dict[str, Any],
    output_stem: Path,
    dpi: int,
    reference_pdf: Path,
    annual_supported: pd.DataFrame | None = None,
    supported_interval_distribution: pd.DataFrame | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    _style()
    # Reuse the study-area figure's physical canvas, two-column ratio, margins,
    # and map subplot.  This is stronger than visually tuning two independent
    # axes: panel (a) is now defined by the same GridSpec as Fig. 2(a), so its
    # geographic frame has the same physical width, height, and aspect ratio.
    settings = context["config"]["figure"]
    layout = settings["layout"]
    margins = settings["margins"]
    width_mm = float(settings["width_mm"])
    height_mm = float(settings["height_mm"])
    if not math.isclose(width_mm, FIG_WIDTH_MM) or not math.isclose(
        height_mm, FIG_HEIGHT_MM
    ):
        raise RuntimeError("Study-area canvas changed from the frozen 180 x 107 mm contract")
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor="white")
    main_grid = fig.add_gridspec(
        1,
        2,
        width_ratios=list(layout["main_width_ratios"]),
        left=float(margins["left"]),
        right=float(margins["right"]),
        bottom=float(margins["bottom"]),
        top=float(margins["top"]),
        wspace=float(layout["main_wspace"]),
    )
    ax_map = fig.add_subplot(main_grid[0, 0])
    right_box = main_grid[0, 1].get_position(fig)
    right_x = right_box.x0 + RIGHT_COLUMN_SHIFT_MM / width_mm
    right_width = right_box.width - RIGHT_COLUMN_WIDTH_REDUCTION_MM / width_mm
    if right_width <= 0.0:
        raise RuntimeError("Right-column width reduction is too large")
    panel_top = right_box.y1
    panel_bottom = right_box.y0
    right_span = panel_top - panel_bottom
    panel_height = right_span * (0.255 / 0.895)
    secondary_panel_height = right_span * (0.230 / 0.895)
    panel_gap = right_span * (0.090 / 0.895)
    interval_bottom = panel_bottom
    dist_bottom = interval_bottom + secondary_panel_height + panel_gap
    year_bottom = dist_bottom + secondary_panel_height + panel_gap
    if not math.isclose(year_bottom + panel_height, panel_top, abs_tol=1e-9):
        raise RuntimeError("Right-column layout no longer shares panel (a)'s vertical extent")
    ax_year = fig.add_axes([right_x, year_bottom, right_width, panel_height])
    ax_dist = fig.add_axes([right_x, dist_bottom, right_width, secondary_panel_height])
    ax_interval = fig.add_axes([right_x, interval_bottom, right_width, secondary_panel_height])

    legacy._draw_map_context(
        ax_map,
        context["ecuador"],
        context["amazonia"],
        context["ocean"],
        context["countries"],
        context["basemap"],
        context["locator"],
        context["config"],
        context["extent"],
    )
    # The national locator duplicates the main map at this compact figure size.
    # Remove only the locator created by the shared map-context helper.
    for child_ax in list(ax_map.child_axes):
        child_ax.remove()
    for tick_label in ax_map.get_yticklabels():
        tick_label.set_ha("center")
        tick_label.set_va("center")
        tick_label.set_rotation_mode("anchor")
    ax_map.xaxis.set_ticks_position("bottom")
    ax_map.tick_params(
        axis="x",
        which="both",
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    # Province boundaries are contextual only: above the basemap, below hexes.
    context["provinces"].plot(
        ax=ax_map,
        color=legacy.BOUNDARY,
        linewidth=0.48,
        alpha=0.66,
        zorder=5.5,
    )
    empty = display[display["inventory_polygon_n"].eq(0)]
    occupied = display[display["inventory_polygon_n"].gt(0)]
    helper._plot_layer(
        ax_map,
        empty,
        facecolor=(0.78, 0.78, 0.78, 0.10),
        edgecolor="#A8A8A8",
        linewidth=0.075,
        zorder=6,
    )
    cmap = matplotlib.colormaps["YlGnBu"]
    norm = Normalize(vmin=0.0, vmax=7.0)
    helper._plot_layer(
        ax_map,
        occupied,
        column="median_event_count",
        cmap=cmap,
        norm=norm,
        edgecolor="#7F7F7F",
        linewidth=0.10,
        alpha=0.95,
        zorder=8,
    )
    # Restore a crisp national outline above the analytical grid. Province
    # lines remain beneath the hexagons and are deliberately thinner.
    context["boundary"].boundary.plot(
        ax=ax_map,
        color=legacy.BOUNDARY,
        linewidth=0.85,
        zorder=9.5,
    )
    ax_map.text(0.012, 0.985, "(a)", transform=ax_map.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT, zorder=110)

    legend_backdrop = Rectangle(
            (0.475, 0.018),
            0.500,
            0.170,
            transform=ax_map.transAxes,
            facecolor=(1.0, 1.0, 1.0, 0.92),
            edgecolor="#777777",
            linewidth=0.4,
            zorder=94,
        )
    ax_map.add_patch(legend_backdrop)
    cax = ax_map.inset_axes([0.560, 0.127, 0.330, 0.023], zorder=95)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=np.arange(0, 8))
    cbar.ax.set_xlabel(
        "Median detected-event count\nwithin each 10-km-wide hexagon",
        fontsize=7.8,
        linespacing=1.05,
        labelpad=2.4,
        color=TEXT,
    )
    cbar.ax.tick_params(labelsize=7.6, length=2.2, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.45)

    years = annual["year"].to_numpy()
    annual_values = annual["annual_event_count"].to_numpy()
    ax_year.bar(years, annual_values, width=0.72, color=PALE_GREEN, edgecolor=DARK_GREEN, linewidth=0.45, zorder=2)
    if annual_supported is not None:
        supported = annual_supported.set_index("year")["vh_supported_event_count"].reindex(years, fill_value=0).to_numpy()
        ax_year.bar(
            years,
            supported,
            width=0.42,
            color=YELLOW,
            edgecolor="#8A6D13",
            linewidth=0.38,
            zorder=3,
        )
        ax_year.legend(
            handles=[
                Patch(facecolor=PALE_GREEN, edgecolor=DARK_GREEN, label="Primary VV events"),
                Patch(facecolor=YELLOW, edgecolor="#8A6D13", label="VH-supported subset"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.59, 0.995),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#A8A8A8",
            framealpha=0.92,
            borderpad=0.25,
            labelspacing=0.15,
            handlelength=0.85,
            handletextpad=0.25,
            columnspacing=0.55,
            fontsize=5.4,
        )
    ax_year.set_ylabel("Annual events", rotation=90, rotation_mode="anchor", labelpad=4)
    ax_year.set_xticks([2017, 2019, 2021, 2023, 2025])
    ax_year.set_xlim(2016.45, 2025.55)
    ax_year.set_ylim(0, max(annual_values) * 1.17)
    ax_year.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75, zorder=0)
    _black_frame(ax_year)
    _panel_label(ax_year, "(b)", x=0.012, y=0.985, y_offset_pt=-3.0)

    count_x = distribution["event_count"].to_numpy()
    percentages = distribution["polygon_percent"].to_numpy()
    bar_colors = cmap(norm(count_x.astype(float)))
    ax_dist.bar(count_x, percentages, width=0.74, color=bar_colors, edgecolor="white", linewidth=0.45, zorder=2)
    ax_dist.set_xlabel("Detected events per polygon", labelpad=0)
    ax_dist.set_ylabel("Polygons (%)")
    ax_dist.set_xticks(count_x)
    ax_dist.set_xlim(-0.55, float(count_x.max()) + 0.55)
    ax_dist.set_ylim(0, max(percentages) * 1.24)
    ax_dist.yaxis.set_major_locator(MultipleLocator(10))
    ax_dist.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75, zorder=0)
    _black_frame(ax_dist)
    for x_value, percent in zip(count_x, percentages):
        if percent >= 3.0:
            ax_dist.text(x_value, percent + ax_dist.get_ylim()[1] * 0.022, f"{percent:.1f}", ha="center", va="bottom", fontsize=8.0, color=TEXT)
    _panel_label(ax_dist, "(c)", x=0.012, y=0.985, y_offset_pt=-3.0)

    interval_x = interval_distribution["bin_midpoint_month"].to_numpy()
    interval_percent = interval_distribution["interval_percent"].to_numpy()
    interval_width = interval_distribution["bin_end_month"].to_numpy() - interval_distribution["bin_start_month"].to_numpy()
    ax_interval.bar(
        interval_x,
        interval_percent,
        width=interval_width * 0.88,
        color=cmap(0.52),
        edgecolor=cmap(0.82),
        linewidth=0.45,
        zorder=2,
    )
    if supported_interval_distribution is not None:
        supported_x = supported_interval_distribution["bin_midpoint_month"].to_numpy()
        supported_percent = supported_interval_distribution["interval_percent"].to_numpy()
        supported_width = (
            supported_interval_distribution["bin_end_month"].to_numpy()
            - supported_interval_distribution["bin_start_month"].to_numpy()
        )
        ax_interval.step(
            np.r_[supported_x - supported_width / 2.0, supported_x[-1] + supported_width[-1] / 2.0],
            np.r_[supported_percent, supported_percent[-1]],
            where="post",
            color="#8A6D13",
            linewidth=1.05,
            zorder=4,
            label="VH-supported endpoint pairs",
        )
    ax_interval.axvline(36.0, color=cmap(0.95), linestyle=(0, (3, 2)), linewidth=1.0, zorder=3)
    if supported_interval_distribution is not None:
        ax_interval.axvline(40.0, color="#8A6D13", linestyle=(0, (2, 2)), linewidth=0.9, zorder=4)
    ax_interval.set_xlabel("Inter-disturbance interval (months)", labelpad=2)
    ax_interval.set_ylabel("Intervals (%)")
    ax_interval.set_xlim(0, 108)
    ax_interval.set_xticks([0, 24, 48, 72, 96])
    interval_top = float(max(interval_percent))
    if supported_interval_distribution is not None:
        interval_top = max(
            interval_top,
            float(supported_interval_distribution["interval_percent"].max()),
        )
    ax_interval.set_ylim(0, interval_top * 1.20)
    ax_interval.yaxis.set_major_locator(MultipleLocator(5))
    ax_interval.grid(axis="y", color="#D9D9D9", linewidth=0.4, alpha=0.75, zorder=0)
    _black_frame(ax_interval)
    ax_interval.text(
        38.5,
        ax_interval.get_ylim()[1] * 0.96,
        (
            "VV median = 36 months"
            if supported_interval_distribution is not None
            else "Median = 36 months"
        ),
        color=cmap(0.95),
        fontsize=8.0,
        ha="left",
        va="top",
    )
    if supported_interval_distribution is not None:
        ax_interval.text(
            42.5,
            ax_interval.get_ylim()[1] * 0.78,
            "VH-supported median = 40 months",
            color="#8A6D13",
            fontsize=6.2,
            ha="left",
            va="top",
        )
        ax_interval.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="#A8A8A8",
            framealpha=0.92,
            borderpad=0.25,
            labelspacing=0.20,
            handlelength=1.4,
            handletextpad=0.35,
            fontsize=5.4,
        )
    _panel_label(ax_interval, "(d)", x=0.012, y=0.985, y_offset_pt=-3.0)

    # Align the three vertical y-axis titles by their rendered horizontal
    # centres. Automatic label placement depends on tick-label width, so the
    # four-digit ticks in panel (b) otherwise push its title farther left than
    # the titles in panels (c) and (d).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_label_bbox = ax_dist.yaxis.label.get_window_extent(renderer)
    target_label_center_x = (target_label_bbox.x0 + target_label_bbox.x1) / 2.0
    for axis in (ax_year, ax_dist, ax_interval):
        axis.yaxis.label.set_rotation(90)
        axis.yaxis.label.set_rotation_mode("anchor")
        display_y = axis.transAxes.transform((0.0, 0.5))[1]
        aligned_x = axis.transAxes.inverted().transform(
            (target_label_center_x, display_y)
        )[0]
        axis.yaxis.set_label_coords(aligned_x, 0.5)

    # Align complete rendered subfigures rather than only their axes frames.
    # Panel (d) includes an x-axis title, whereas panel (a) has longitude ticks;
    # using tight bounding boxes removes the apparent blank strip below the map.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    map_tight_bbox = ax_map.get_tightbbox(renderer)
    interval_tight_bbox = ax_interval.get_tightbbox(renderer)
    interval_shift = (map_tight_bbox.y0 - interval_tight_bbox.y0) / fig.bbox.height
    interval_position = ax_interval.get_position()
    ax_interval.set_position(
        [
            interval_position.x0,
            interval_position.y0 + interval_shift,
            interval_position.width,
            interval_position.height,
        ]
    )

    # Centre panel (c) between the rendered extents of panels (b) and (d), so
    # the two visible inter-panel gaps are equal after bottom alignment.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    year_tight_bbox = ax_year.get_tightbbox(renderer)
    dist_tight_bbox = ax_dist.get_tightbbox(renderer)
    interval_tight_bbox = ax_interval.get_tightbbox(renderer)
    free_vertical_space = year_tight_bbox.y0 - interval_tight_bbox.y1 - dist_tight_bbox.height
    if free_vertical_space <= 0.0:
        raise RuntimeError("Insufficient vertical space for the three right-column panels")
    target_dist_tight_y0 = interval_tight_bbox.y1 + free_vertical_space / 2.0
    dist_shift = (target_dist_tight_y0 - dist_tight_bbox.y0) / fig.bbox.height
    dist_position = ax_dist.get_position()
    ax_dist.set_position(
        [
            dist_position.x0,
            dist_position.y0 + dist_shift,
            dist_position.width,
            dist_position.height,
        ]
    )

    # Validate the rendered colourbar, including its title and tick labels.
    # This detects actual overflow instead of relying on inset coordinates.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    map_bbox = ax_map.get_window_extent(renderer)
    map_tight_bbox = ax_map.get_tightbbox(renderer)
    legend_bbox = cbar.ax.get_tightbbox(renderer)
    backdrop_bbox = legend_backdrop.get_window_extent(renderer)
    tolerance_px = 0.0
    legend_within_map = bool(
        legend_bbox.x0 >= map_bbox.x0 - tolerance_px
        and legend_bbox.y0 >= map_bbox.y0 - tolerance_px
        and legend_bbox.x1 <= map_bbox.x1 + tolerance_px
        and legend_bbox.y1 <= map_bbox.y1 + tolerance_px
    )
    legend_within_backdrop = bool(
        legend_bbox.x0 >= backdrop_bbox.x0 - tolerance_px
        and legend_bbox.y0 >= backdrop_bbox.y0 - tolerance_px
        and legend_bbox.x1 <= backdrop_bbox.x1 + tolerance_px
        and legend_bbox.y1 <= backdrop_bbox.y1 + tolerance_px
    )
    if not legend_within_map:
        raise RuntimeError(
            "Panel (a) colourbar exceeds the map axes: "
            f"legend={legend_bbox.bounds}, map={map_bbox.bounds}"
        )
    if not legend_within_backdrop:
        raise RuntimeError(
            "Panel (a) colourbar exceeds its legend backdrop: "
            f"legend={legend_bbox.bounds}, backdrop={backdrop_bbox.bounds}"
        )
    year_bbox = ax_year.get_tightbbox(renderer)
    dist_bbox = ax_dist.get_tightbbox(renderer)
    interval_bbox = ax_interval.get_tightbbox(renderer)
    right_tight_gaps_px = [
        float(year_bbox.y0 - dist_bbox.y1),
        float(dist_bbox.y0 - interval_bbox.y1),
    ]
    right_horizontal_gaps_px = [
        float(year_bbox.x0 - map_tight_bbox.x1),
        float(dist_bbox.x0 - map_tight_bbox.x1),
        float(interval_bbox.x0 - map_tight_bbox.x1),
    ]
    yaxis_label_centres_px = []
    for axis in (ax_year, ax_dist, ax_interval):
        label_bbox = axis.yaxis.label.get_window_extent(renderer)
        yaxis_label_centres_px.append(float((label_bbox.x0 + label_bbox.x1) / 2.0))
    yaxis_label_alignment_spread_px = max(yaxis_label_centres_px) - min(
        yaxis_label_centres_px
    )
    if min(right_tight_gaps_px) < 0.0:
        raise RuntimeError(f"Right-column panel content overlaps: {right_tight_gaps_px}")
    if min(right_horizontal_gaps_px) < 2.0:
        raise RuntimeError(
            "Right-column content does not clear panel (a): "
            f"{right_horizontal_gaps_px}"
        )
    if yaxis_label_alignment_spread_px > 0.1:
        raise RuntimeError(
            "Right-column y-axis titles are not horizontally aligned: "
            f"{yaxis_label_centres_px}"
        )
    panel_a_and_d_bottom_aligned = math.isclose(map_tight_bbox.y0, interval_bbox.y0, abs_tol=0.1)
    if not panel_a_and_d_bottom_aligned:
        raise RuntimeError("Panels (a) and (d) no longer share the same rendered bottom")
    layout_qa = {
        "layout_source": "configs/study_area_figure_v1.json, layout variant v8",
        "source_canvas_width_mm": width_mm,
        "source_canvas_height_mm": height_mm,
        "main_width_ratios": [float(value) for value in layout["main_width_ratios"]],
        "main_wspace": float(layout["main_wspace"]),
        "right_column_shift_mm": RIGHT_COLUMN_SHIFT_MM,
        "right_column_width_reduction_mm": RIGHT_COLUMN_WIDTH_REDUCTION_MM,
        "main_margins": {
            key: float(margins[key])
            for key in ("left", "right", "bottom", "top")
        },
        "panel_a_axes_bbox_mm": [
            float(ax_map.get_position().x0 * width_mm),
            float(ax_map.get_position().y0 * height_mm),
            float(ax_map.get_position().width * width_mm),
            float(ax_map.get_position().height * height_mm),
        ],
        "panel_a_and_d_rendered_bottom_aligned": panel_a_and_d_bottom_aligned,
        "panel_a_rendered_bottom_pixels": float(map_tight_bbox.y0),
        "panel_d_rendered_bottom_pixels": float(interval_bbox.y0),
        "panel_d_axes_bottom_offset_above_panel_a_fraction": float(
            ax_interval.get_position().y0 - ax_map.get_position().y0
        ),
        "panel_a_and_b_top_aligned": math.isclose(
            ax_map.get_position().y1,
            ax_year.get_position().y1,
            abs_tol=1e-9,
        ),
        "panel_c_and_d_height_fraction": float(secondary_panel_height),
        "panel_a_colourbar_within_map": legend_within_map,
        "panel_a_colourbar_within_backdrop": legend_within_backdrop,
        "panel_a_colourbar_bbox_pixels": [float(value) for value in legend_bbox.bounds],
        "panel_a_colourbar_backdrop_bbox_pixels": [float(value) for value in backdrop_bbox.bounds],
        "panel_a_map_bbox_pixels": [float(value) for value in map_bbox.bounds],
        "right_panel_axes_gap_fractions": [
            float(ax_year.get_position().y0 - ax_dist.get_position().y1),
            float(ax_dist.get_position().y0 - ax_interval.get_position().y1),
        ],
        "right_panel_tight_content_gaps_pixels": right_tight_gaps_px,
        "panel_a_to_right_tight_content_gaps_pixels": right_horizontal_gaps_px,
        "right_yaxis_label_centres_pixels": yaxis_label_centres_px,
        "right_yaxis_label_alignment_spread_pixels": yaxis_label_alignment_spread_px,
    }

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")
    metadata = {"Title": "National BALSA disturbance recurrence", "Creator": Path(__file__).name}
    # Use the exact reference MediaBox rather than accepting a content-dependent
    # tight crop.  The left/bottom origin still follows Matplotlib's normal
    # tight-bbox padding, while width and height come directly from Fig. 2.
    # Consequently, equal LaTeX include widths produce equal physical scaling.
    fig.canvas.draw()
    content_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    pad_inches = float(plt.rcParams["savefig.pad_inches"])
    reference_width_inches, reference_height_inches = _pdf_media_box_inches(reference_pdf)
    export_bbox = Bbox.from_bounds(
        content_bbox.x0 - pad_inches,
        content_bbox.y0 - pad_inches,
        reference_width_inches,
        reference_height_inches,
    )
    layout_qa["reference_pdf"] = str(reference_pdf.resolve())
    layout_qa["reference_pdf_sha256"] = _sha256(reference_pdf)
    layout_qa["target_media_box_points"] = [
        reference_width_inches * 72.0,
        reference_height_inches * 72.0,
    ]
    layout_qa["export_bbox_inches"] = [float(value) for value in export_bbox.bounds]
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=export_bbox, metadata=metadata)
    fig.savefig(svg_path, dpi=dpi, bbox_inches=export_bbox, metadata=metadata)
    fig.savefig(png_path, dpi=dpi, bbox_inches=export_bbox)
    fig.savefig(tiff_path, dpi=dpi, bbox_inches=export_bbox, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {
        "pdf": str(pdf_path.resolve()),
        "svg": str(svg_path.resolve()),
        "png": str(png_path.resolve()),
        "tiff": str(tiff_path.resolve()),
    }, layout_qa


def build(args: argparse.Namespace) -> dict[str, Any]:
    helper = _load_module(HEX_HELPER, "hex_helper_10km")
    source = helper._load_module(helper.SOURCE_SCRIPT, "study_area_source_10km")
    legacy = helper._load_module(helper.LEGACY_SCRIPT, "legacy_hex_context_10km")
    chronology = helper._load_module(helper.CHRONOLOGY_SCRIPT, "chronology_source_10km")
    for path in (args.events, args.polygon_tracks, args.balsa_shp, args.boundary):
        if not path.exists():
            raise FileNotFoundError(path)

    context = helper._context(args, source, legacy)
    province_lines, province_records = source.natural_earth_layer(
        "10m",
        "cultural",
        "admin_1_states_provinces_lines",
        REPO_ROOT,
    )
    provinces = province_lines.loc[
        province_lines["ADM0_A3"].astype(str).eq("ECU")
    ].copy()
    provinces = gpd.clip(provinces, context["boundary"])
    if provinces.empty:
        raise RuntimeError("Natural Earth returned no mainland Ecuador province boundaries")
    context["provinces"] = provinces
    context["province_records"] = province_records
    points, events, intervals, input_qa = helper._load_polygon_metrics(args, chronology)
    grid = helper._build_mainland_grid(context["boundary"], HEX_FLAT_WIDTH_M)
    assigned = helper._assign_polygons(points, grid)
    metrics = _aggregate_10km(grid, assigned)
    display = metrics.to_crs("EPSG:4326")
    annual = _annual_counts(events)
    distribution = _polygon_distribution(assigned)
    interval_distribution = _interval_distribution(intervals)
    annual_supported = None
    supported_interval_distribution = None
    polarization_qa = None
    annotated_vv = None
    if args.vh_events is not None:
        if not args.vh_events.exists():
            raise FileNotFoundError(args.vh_events)
        annotated_vv, annual_supported, supported_interval_distribution, polarization_qa = (
            _polarization_evidence(
                args.events,
                args.vh_events,
                args.vh_tolerance_months,
            )
        )
        if polarization_qa["primary_vv_events"] != len(events):
            raise RuntimeError(
                "The VV-primary evidence ledger does not match the national chronology"
            )

    if len(assigned) != EXPECTED_POLYGONS or assigned["polygon_id"].nunique() != EXPECTED_POLYGONS:
        raise RuntimeError("Polygon assignment conservation failed")
    if len(events) != EXPECTED_EVENTS or int(annual["annual_event_count"].sum()) != EXPECTED_EVENTS:
        raise RuntimeError("Event conservation failed")
    if int((assigned["event_count"] > 0).sum()) != EXPECTED_EVENT_POLYGONS:
        raise RuntimeError("Event-bearing polygon count changed")
    if int(distribution["polygon_count"].sum()) != EXPECTED_POLYGONS:
        raise RuntimeError("Polygon event-count distribution does not sum to 3,498")
    if int(interval_distribution["interval_count"].sum()) != helper.EXPECTED_INTERVALS:
        raise RuntimeError("Adjacent-interval distribution does not sum to 3,509")
    if not math.isclose(float(assigned["event_count"].median()), 2.0):
        raise RuntimeError("Polygon event-count median is no longer 2")

    occupied = metrics[metrics["inventory_polygon_n"] > 0].copy()
    abs_mean_median_difference = (occupied["mean_event_count"] - occupied["median_event_count"]).abs()
    n1_fraction = float((occupied["inventory_polygon_n"] == 1).mean())
    cell_area_km2 = math.sqrt(3.0) * HEX_FLAT_WIDTH_M**2 / 2.0 / 1_000_000.0
    qa = {
        "mainland_hex_cells": int(len(grid)),
        "occupied_hex_cells": int(len(occupied)),
        "hex_flat_width_km": HEX_FLAT_WIDTH_M / 1000.0,
        "hex_area_km2": cell_area_km2,
        "polygon_assignments": int(len(assigned)),
        "unique_polygons": int(assigned["polygon_id"].nunique()),
        "events": int(len(events)),
        "event_bearing_polygons": int((assigned["event_count"] > 0).sum()),
        "polygon_event_count_min_median_max": [int(assigned["event_count"].min()), float(assigned["event_count"].median()), int(assigned["event_count"].max())],
        "occupied_hex_mean_event_count_min_median_max": [float(occupied["mean_event_count"].min()), float(occupied["mean_event_count"].median()), float(occupied["mean_event_count"].max())],
        "occupied_hex_median_event_count_min_median_max": [float(occupied["median_event_count"].min()), float(occupied["median_event_count"].median()), float(occupied["median_event_count"].max())],
        "mean_vs_median_abs_difference_median_max": [float(abs_mean_median_difference.median()), float(abs_mean_median_difference.max())],
        "mean_vs_median_spearman": float(occupied[["mean_event_count", "median_event_count"]].corr(method="spearman").iloc[0, 1]),
        "occupied_cells_with_one_polygon": int((occupied["inventory_polygon_n"] == 1).sum()),
        "occupied_cells_with_one_polygon_fraction": n1_fraction,
        "annual_event_counts": {str(row.year): int(row.annual_event_count) for row in annual.itertuples(index=False)},
        "annual_event_count_sum": int(annual["annual_event_count"].sum()),
        "polygon_event_count_distribution": {str(row.event_count): int(row.polygon_count) for row in distribution.itertuples(index=False)},
        "adjacent_interval_count": int(interval_distribution["interval_count"].sum()),
        "ecuador_province_boundary_features": int(len(provinces)),
        "adjacent_interval_min_median_max_months": [
            float(intervals["interval_months"].min()),
            float(intervals["interval_months"].median()),
            float(intervals["interval_months"].max()),
        ],
    }

    output = args.output_dir
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    grid_path = source_dir / "ecuador_mainland_hex_10km_width_epsg6933.parquet"
    metrics_path = source_dir / "hex_event_count_summary_10km_epsg6933.parquet"
    assignments_path = source_dir / "polygon_hex_assignments_10km.csv"
    annual_path = source_dir / "annual_detected_event_counts_2017_2025.csv"
    distribution_path = source_dir / "polygon_detected_event_count_distribution.csv"
    interval_distribution_path = source_dir / "adjacent_interval_distribution_6month_bins.csv"
    grid.to_parquet(grid_path, index=False)
    metrics.to_parquet(metrics_path, index=False)
    assigned.sort_values("polygon_id").to_csv(assignments_path, index=False)
    annual.to_csv(annual_path, index=False)
    distribution.to_csv(distribution_path, index=False)
    interval_distribution.to_csv(interval_distribution_path, index=False)
    evidence_source_data: dict[str, str] = {}
    if annual_supported is not None and supported_interval_distribution is not None:
        annual_supported_path = source_dir / "annual_vv_events_with_vh_support_pm2.csv"
        supported_interval_path = source_dir / "vh_supported_interval_distribution_6month_bins.csv"
        annotated_vv_path = source_dir / "vv_primary_events_with_vh_support_pm2.parquet"
        annual_supported.to_csv(annual_supported_path, index=False)
        supported_interval_distribution.to_csv(supported_interval_path, index=False)
        assert annotated_vv is not None
        annotated_vv.to_parquet(annotated_vv_path, index=False)
        evidence_source_data = {
            "annual_vh_support": str(annual_supported_path.resolve()),
            "supported_interval_distribution": str(supported_interval_path.resolve()),
            "annotated_primary_vv_events": str(annotated_vv_path.resolve()),
        }

    outputs, layout_qa = _plot_figure(
        display,
        annual,
        distribution,
        interval_distribution,
        helper,
        legacy,
        context,
        output / "figure_06_national_recurrence_10km",
        args.dpi,
        args.study_area_reference_pdf,
        annual_supported,
        supported_interval_distribution,
    )
    qa["layout"] = layout_qa
    grid_spec = {
        "crs": helper.GRID_CRS,
        "orientation": "pointy-top",
        "flat_to_flat_width_m": HEX_FLAT_WIDTH_M,
        "side_and_radius_m": HEX_FLAT_WIDTH_M / math.sqrt(3.0),
        "row_center_spacing_m": 1.5 * HEX_FLAT_WIDTH_M / math.sqrt(3.0),
        "cell_area_km2": cell_area_km2,
        "origin_m": [helper.GRID_ORIGIN_X_M, helper.GRID_ORIGIN_Y_M],
        "mainland_selection": "full fixed hexagons intersecting the mainland boundary",
        "polygon_assignment": "one polygon-interior representative point; one polygon contributes to exactly one hexagon",
        "map_statistic": "median of polygon detected-event counts, including zero-event polygons",
        "grid_sha256": helper._grid_hash(grid),
    }
    (output / "grid_spec.json").write_text(json.dumps(grid_spec, indent=2) + "\n", encoding="utf-8")
    (output / "qa_summary.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
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
            "vh_events": str(args.vh_events.resolve()) if args.vh_events else None,
            "vh_tolerance_months": int(args.vh_tolerance_months),
            "study_area_reference_pdf": str(args.study_area_reference_pdf.resolve()),
            "province_boundaries": province_records,
        },
        "input_sha256": {
            "events": _sha256(args.events),
            "polygon_tracks": _sha256(args.polygon_tracks),
            "balsa_shp": _sha256(args.balsa_shp),
            "boundary": _sha256(args.boundary),
        },
        "definitions": {
            "panel_a": "median polygon-level detected-event count in each occupied 10-km-wide hexagon; zero-event polygons are included",
            "panel_a_province_boundaries": "Natural Earth 1:10m Ecuador admin-1 boundary lines, rendered above the basemap and below the analytical hexagons",
            "panel_b": "number of frozen VV ground-route events by detected change year",
            "panel_c": "distribution of detected-event counts across all 3,498 validated inventory polygons, including zero-event polygons",
            "panel_d": "distribution of all 3,509 polygon-level adjacent detected-event intervals in fixed six-month bins; the line marks the overall 36-month median",
            "polarization_rule": "VV defines the primary chronology; Polygon-first VH is matched one-to-one within the configured calendar-month tolerance and remains complementary evidence, never an input average",
        },
        "polarization_qa": polarization_qa,
        "input_qa": input_qa,
        "grid_spec": grid_spec,
        "qa": qa,
        "source_data": {
            "grid": str(grid_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "assignments": str(assignments_path.resolve()),
            "annual_counts": str(annual_path.resolve()),
            "polygon_distribution": str(distribution_path.resolve()),
            "interval_distribution": str(interval_distribution_path.resolve()),
            **evidence_source_data,
        },
        "outputs": outputs,
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path.resolve()), "outputs": outputs, "qa": qa}


def main() -> None:
    print(json.dumps(build(_parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
