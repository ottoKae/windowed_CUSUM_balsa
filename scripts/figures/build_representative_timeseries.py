#!/usr/bin/env python3
"""Build manuscript Figure 6; licensed optical rasters are supplied locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import xarray as xr

from _common import DATA_ROOT, GREEN, INK, ROOT, YELLOW, configure_style, load_case_series, panel_label, require, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("runs/figures/fig06_representative_timeseries.pdf"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--historical-id", type=int, default=1499)
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/figure_sources.local.json")
    parser.add_argument("--optical-before", type=Path)
    parser.add_argument("--optical-after", type=Path)
    return parser.parse_args()


def bounds_around(geometry, minimum_width: float = 2000.0) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = geometry.bounds
    width = max(xmax - xmin, ymax - ymin, minimum_width)
    center_x, center_y = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * width
    return center_x - half, center_y - half, center_x + half, center_y + half


def optical(path: Path, bounds, target_crs) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as source:
        geometry_bounds = gpd.GeoSeries([box(*bounds)], crs=target_crs).to_crs(source.crs).total_bounds
        window = from_bounds(*geometry_bounds, transform=source.transform)
        raw = source.read([1, 2, 3], window=window, boundless=True, fill_value=0).transpose(1, 2, 0).astype(float)
        extent = tuple(geometry_bounds[[0, 2, 1, 3]])
    if raw.max() > 1:
        raw /= 255.0 if raw.max() <= 255 else np.nanpercentile(raw, 99.5)
    return np.clip(raw, 0, 1), extent


def sar_pair(zarr_path: Path, bounds) -> tuple[list[np.ndarray], tuple[float, float, float, float]]:
    dataset = xr.open_zarr(zarr_path, consolidated=False)
    xmin, ymin, xmax, ymax = bounds
    crop = dataset.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    months = ["2024-02", "2024-03"]
    composites = []
    channels = []
    for month in months:
        selected = crop.sel(time=month)
        vv = np.asarray(selected.VV_dB.median("time", skipna=True).compute(), float)
        vh = np.asarray(selected.VH_dB.median("time", skipna=True).compute(), float)
        channels.append((vv, vh, vv - vh))
    ranges = []
    for index in range(3):
        combined = np.concatenate([item[index][np.isfinite(item[index])] for item in channels])
        ranges.append(np.quantile(combined, [0.02, 0.98]))
    for item in channels:
        rgb = np.stack([
            np.clip((item[index] - ranges[index][0]) / max(1e-6, ranges[index][1] - ranges[index][0]), 0, 1)
            for index in range(3)
        ], axis=-1)
        composites.append(rgb)
    x = np.asarray(crop.x); y = np.asarray(crop.y)
    return composites, (float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def map_panel(axis, image, extent, geometry, letter, date, crs) -> None:
    axis.imshow(image, extent=extent, origin="upper")
    polygon = gpd.GeoSeries([geometry], crs=crs).to_crs(crs)
    polygon.boundary.plot(ax=axis, color="white", linewidth=1.4)
    polygon.boundary.plot(ax=axis, color=INK, linewidth=0.45)
    axis.text(0.98, 0.96, date, transform=axis.transAxes, ha="right", va="top", fontsize=6.5, color="white",
              path_effects=[])
    length = 1000.0
    x0 = extent[1] - 0.08 * (extent[1] - extent[0]) - length
    y0 = extent[2] + 0.07 * (extent[3] - extent[2])
    axis.plot([x0, x0 + length], [y0, y0], color="white", linewidth=2.2)
    axis.plot([x0, x0 + length], [y0, y0], color=INK, linewidth=0.7)
    axis.text(x0 + length / 2, y0, "1 km", ha="center", va="bottom", fontsize=5.8, color="white")
    axis.set_xticks([]); axis.set_yticks([]); panel_label(axis, letter)


def main() -> None:
    args = parse_args(); configure_style()
    local_sources = {}
    if args.sources.is_file():
        local_sources = json.loads(args.sources.read_text(encoding="utf-8"))
    before_value = args.optical_before or local_sources.get("figure6_optical_before")
    after_value = args.optical_after or local_sources.get("figure6_optical_after")
    if not before_value or not after_value:
        raise FileNotFoundError(
            "Figure 6 needs locally licensed optical rasters. Copy "
            "configs/figure_sources.example.json to configs/figure_sources.local.json "
            "and set both paths."
        )
    optical_before = Path(before_value)
    optical_after = Path(after_value)
    require([optical_before, optical_after])
    case = load_case_series(args.data_root, args.historical_id)
    inventory = gpd.read_parquet(args.data_root / "inventory/balsa_validated.parquet")
    polygon = inventory.loc[inventory.sample_id.astype(str).eq(case.sample_id)].iloc[0]
    bounds = bounds_around(polygon.geometry)
    source = json.loads((args.data_root / "s1grits/SOURCE.local.json").read_text(encoding="utf-8"))
    catalog = pd.read_parquet(args.data_root / "s1grits/catalog.parquet")
    entry = catalog.loc[catalog.geometry_group_id.eq(case.track_key) & catalog.product_type.eq("smonthly")].iloc[0]
    zarr_path = Path(source["path"]) / entry.tile_id / entry.zarr_path
    require([zarr_path])
    before_optical, before_extent = optical(optical_before, bounds, inventory.crs)
    after_optical, after_extent = optical(optical_after, bounds, inventory.crs)
    sar_images, sar_extent = sar_pair(zarr_path, bounds)

    figure = plt.figure(figsize=(7.2, 3.55))
    grid = figure.add_gridspec(2, 4, height_ratios=(1.45, 1.0), hspace=0.24, wspace=0.03)
    axis = figure.add_subplot(grid[0, :])
    q25 = np.nanquantile(case.pixels, 0.25, axis=0); q75 = np.nanquantile(case.pixels, 0.75, axis=0)
    for index, (label, color) in enumerate((("VV", GREEN), ("VH", YELLOW))):
        axis.fill_between(case.calendar, q25[index], q75[index], color=color, alpha=0.18, linewidth=0)
        axis.plot(case.calendar, case.median[index], color=color, marker="o", markersize=1.5, linewidth=1.0, label=f"Polygon-first {label} median")
    for event in case.references:
        axis.axvline(event, color=INK, linestyle="--", linewidth=0.65)
        axis.text(event, -4.8, event.strftime("%Y-%m"), ha="center", va="bottom", fontsize=6.1)
    axis.set_xlim(case.calendar[3], case.calendar[-1]); axis.set_ylim(-24, -4); axis.set_ylabel("Backscatter (dB)")
    axis.xaxis.set_major_locator(mdates.YearLocator()); axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(color="#D8D8D8", linewidth=0.35); axis.legend(loc="lower left", frameon=True, ncol=1); panel_label(axis, "a")

    axes = [figure.add_subplot(grid[1, index]) for index in range(4)]
    map_panel(axes[0], before_optical, before_extent, polygon.geometry, "b", "17 Feb. 2024", inventory.crs)
    map_panel(axes[1], sar_images[0], sar_extent, polygon.geometry, "c", "Feb. 2024", inventory.crs)
    map_panel(axes[2], after_optical, after_extent, polygon.geometry, "d", "28 Mar. 2024", inventory.crs)
    map_panel(axes[3], sar_images[1], sar_extent, polygon.geometry, "e", "Mar. 2024", inventory.crs)
    figure.text(0.26, 0.405, "Before stand disturbance", ha="center", fontsize=7.2, fontweight="bold")
    figure.text(0.75, 0.405, "After stand disturbance", ha="center", fontsize=7.2, fontweight="bold")
    save_figure(figure, args.output)


if __name__ == "__main__":
    main()
