#!/usr/bin/env python3
"""Build manuscript Figure 2 from the authorized inventory and references."""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd

from _common import DATA_ROOT, GREEN, INK, YELLOW, configure_style, panel_label, save_figure


DETAILS = (
    (-80.02, -79.88, 0.55, 0.68),
    (-79.47, -79.22, -0.58, -0.42),
    (-77.08, -76.78, -0.31, -0.15),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("runs/figures/fig02_study_area.pdf"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    validated = gpd.read_parquet(args.data_root / "inventory/balsa_validated.parquet")
    pending = gpd.read_parquet(args.data_root / "inventory/balsa_excluded_pending.parquet")
    inventory = gpd.GeoDataFrame(
        pd.concat([validated, pending], ignore_index=True),
        geometry="geometry",
        crs=validated.crs,
    ).to_crs(4326)
    if len(inventory) != 3890:
        raise ValueError(f"Expected 3,890 official BALSA polygons, found {len(inventory):,}")
    reference_table = pd.read_parquet(args.data_root / "references/reference_ground_events.parquet")
    references = gpd.GeoDataFrame(
        reference_table,
        geometry=gpd.points_from_xy(
            reference_table.longitude,
            reference_table.latitude,
        ), crs=4326,
    )
    dated_ids = set(references.sample_id.astype(str))
    dated = inventory.loc[inventory.sample_id.astype(str).isin(dated_ids)]
    countries = gpd.read_file(
        args.data_root / "admin/cultural/ne_110m_admin_0_countries.shp"
    ).to_crs(4326)
    ecuador = countries.loc[countries.ADMIN.eq("Ecuador")]

    figure = plt.figure(figsize=(7.2, 5.0))
    grid = figure.add_gridspec(3, 2, width_ratios=(1.18, 1.0), hspace=0.17, wspace=0.08)
    national = figure.add_subplot(grid[:, 0])
    detail_axes = [figure.add_subplot(grid[i, 1]) for i in range(3)]

    ecuador.plot(ax=national, color="#F7F7F2", edgecolor=INK, linewidth=0.8)
    inventory_points = gpd.GeoSeries(
        gpd.points_from_xy(inventory.longitude, inventory.latitude), crs=4326,
    )
    dated_points = inventory_points.loc[dated.index]
    inventory_points.plot(
        ax=national, color=GREEN, edgecolor=GREEN, markersize=inventory.polygon_area_ha.clip(0.2, 16) * 1.4,
        alpha=0.72, linewidth=0.15,
    )
    dated_points.plot(
        ax=national, color=YELLOW, edgecolor="#8A711B", markersize=dated.polygon_area_ha.clip(0.2, 16) * 1.6,
        alpha=0.72, linewidth=0.25,
    )
    national.set_xlim(-81.2, -75.0); national.set_ylim(-5.1, 1.6)
    national.set_xlabel("Longitude"); national.set_ylabel("Latitude")
    national.grid(color="#D8D8D8", linewidth=0.35, alpha=0.7)
    panel_label(national, "a")

    for letter, axis, extent in zip("bcd", detail_axes, DETAILS):
        xmin, xmax, ymin, ymax = extent
        local = inventory.cx[xmin:xmax, ymin:ymax]
        local_dated = dated.cx[xmin:xmax, ymin:ymax]
        ecuador.plot(ax=axis, color="#FAFAF7", edgecolor="#777777", linewidth=0.35)
        if not local.empty:
            local.plot(ax=axis, color=GREEN, edgecolor="#315E39", linewidth=0.25, alpha=0.65)
        if not local_dated.empty:
            local_dated.plot(ax=axis, color=YELLOW, edgecolor="#806C20", linewidth=0.35, alpha=0.72)
        axis.set_xlim(xmin, xmax); axis.set_ylim(ymin, ymax)
        axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        axis.grid(color="#DDDDDD", linewidth=0.3, alpha=0.7)
        panel_label(axis, letter)
        national.add_patch(Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor="#555555", linewidth=0.7,
        ))

    detail_axes[-1].legend(
        handles=[
            Patch(facecolor=GREEN, edgecolor=GREEN, alpha=0.72, label="Official balsa inventory"),
            Patch(facecolor=YELLOW, edgecolor="#806C20", alpha=0.72, label="Dated event subset"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
    )
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
