#!/usr/bin/env python3
"""Rebuild the national event-count and interval summary from local data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lr_cusum.rotation import lifecycle_summary, rotation_periods  # noqa: E402


def hexagons(bounds, width=10_000.0) -> gpd.GeoDataFrame:
    """Return a globally anchored, pointy-top grid with fixed flat width."""
    radius = width / np.sqrt(3.0)
    vertical = 1.5 * radius
    minx, miny, maxx, maxy = bounds
    rows = range(int(np.floor(miny / vertical)) - 1, int(np.ceil(maxy / vertical)) + 2)
    cells = []
    for row in rows:
        y = row * vertical
        offset = width / 2 if row % 2 else 0.0
        columns = range(
            int(np.floor((minx - offset) / width)) - 1,
            int(np.ceil((maxx - offset) / width)) + 2,
        )
        for column in columns:
            x = column * width + offset
            vertices = [
                (x + radius * np.cos(np.deg2rad(30 + 60 * k)),
                 y + radius * np.sin(np.deg2rad(30 + 60 * k)))
                for k in range(6)
            ]
            cells.append((f"r{row}_c{column}", Polygon(vertices)))
    return gpd.GeoDataFrame(cells, columns=["hex_id", "geometry"], crs="EPSG:6933")


def load_inputs(data: Path):
    inventory = gpd.read_parquet(data / "inventory/balsa_validated.parquet")
    events = pd.read_parquet(data / "frozen_results/national/ground_polygon_vv_primary.parquet")
    events = events.loc[events.route.eq("polygon_first")].copy()
    admin = data / "admin/cultural"
    countries = gpd.read_file(admin / "ne_110m_admin_0_countries.shp")
    country = countries.loc[countries.ADM0_A3.eq("ECU")].to_crs("EPSG:6933")
    provinces = gpd.read_file(admin / "ne_10m_admin_1_states_provinces_lines.shp")
    provinces = provinces.loc[provinces.ADM0_A3.eq("ECU")].to_crs("EPSG:6933")
    return inventory, events, country, provinces


def build(data: Path, output: Path) -> None:
    inventory, events, country, provinces = load_inputs(data)
    inventory = inventory.to_crs("EPSG:6933")
    counts = lifecycle_summary(events, inventory.polygon_id, id_column="polygon_id")
    inventory = inventory.merge(counts[["polygon_id", "detected_events"]], on="polygon_id")

    grid = hexagons(country.total_bounds)
    grid = grid.loc[grid.intersects(country.geometry.union_all())].copy()
    points = inventory.copy()
    points.geometry = points.geometry.representative_point()
    assigned = gpd.sjoin(points, grid, predicate="within", how="left")
    medians = assigned.groupby("hex_id").detected_events.median()
    grid["median_events"] = grid.hex_id.map(medians)

    intervals = rotation_periods(events, id_column="polygon_id")
    annual = (
        events.assign(year=pd.to_datetime(events.change_time).dt.year)
        .groupby("year").size().reindex(range(2017, 2026), fill_value=0)
    )
    distribution = counts.groupby("detected_events").size()

    width = 7.1
    figure = plt.figure(figsize=(width, width * 0.60), constrained_layout=True)
    layout = figure.add_gridspec(3, 2, width_ratios=(1.55, 1.0))
    axes = [
        figure.add_subplot(layout[:, 0]),
        figure.add_subplot(layout[0, 1]),
        figure.add_subplot(layout[1, 1]),
        figure.add_subplot(layout[2, 1]),
    ]

    ax = axes[0]
    grid.plot(ax=ax, color="#E5E5E2", edgecolor="#B9B9B4", linewidth=0.10)
    occupied = grid.loc[grid.median_events.notna()]
    bounds = np.arange(-0.5, 8.5, 1)
    occupied.plot(
        ax=ax,
        column="median_events",
        cmap="YlGnBu",
        norm=BoundaryNorm(bounds, 256),
        edgecolor="#8A8A85",
        linewidth=0.12,
        legend=True,
        legend_kwds={"label": "Median event count\nper 10-km hexagon", "shrink": 0.50, "pad": 0.01},
    )
    provinces.plot(ax=ax, color="#444444", linewidth=0.25)
    country.boundary.plot(ax=ax, color="#222222", linewidth=0.65)
    ax.set_axis_off()

    axes[1].bar(annual.index, annual.values, color="#73B982", edgecolor="#0C5B4E", linewidth=0.5)
    axes[1].set_ylabel("Detected events")
    axes[1].set_xticks(range(2017, 2026, 2))

    axes[2].bar(distribution.index, distribution.values, color="#F1D27A", edgecolor="#7C6A32", linewidth=0.5)
    axes[2].set_ylabel("Polygons")
    axes[2].set_xlabel("Detected-event count")

    bins = np.arange(0, max(78, intervals.rotation_months.max() + 6), 6)
    axes[3].hist(intervals.rotation_months.dropna(), bins=bins, color="#AFCF87", edgecolor="#315C48", linewidth=0.5)
    median = float(intervals.rotation_months.median())
    axes[3].axvline(median, color="#263238", linestyle="--", linewidth=1)
    axes[3].text(0.98, 0.88, f"Median = {median:.0f} months", ha="right", transform=axes[3].transAxes)
    axes[3].set_ylabel("Intervals")
    axes[3].set_xlabel("Inter-disturbance interval (months)")

    for label, ax in zip("abcd", axes, strict=True):
        ax.text(0.01, 0.99, f"({label})", transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.4)
        ax.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"figure={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/local/paper_v1",
        help="Prepared authorized dataset root.",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.data, args.output)


if __name__ == "__main__":
    main()
