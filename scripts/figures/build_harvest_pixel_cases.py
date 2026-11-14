"""Three FC001 sites: monthly pixel maps and the same pixels' VH histories.

Frozen observations and detections are read only. Run from any working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, to_hex
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

from _common import ROOT, DATA_ROOT


WORKSPACE = ROOT.parents[1]
PRODUCT = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
REFERENCES = ROOT / "runs/validation/jag_reference_score_closeout_20261015_v1/evidence/verified_events.csv"
DEFAULT_OUTPUT = ROOT / "runs/figures/harvest_pixel_cases_20261104_v1"
CSS = WORKSPACE / "BALSA/webapp/Global_Plantation_Explorer/prototype_v11/frontend/src/hedgerow.css"
FC001 = WORKSPACE / "BALSA/dataset/webapp/plantation_explorer/v2/canonical/plantation_features.parquet"
STANDS = (716, 728, 788)
VH_COLOR = "#367aa8"
DETECTION_COLOR, REFERENCE_COLOR = "#a64730", "#75529a"
SCALE_BARS_M = {716: 200, 728: 200, 788: 200}
SCALE_BAR_MM = 8
NAME = "fig_harvest_pixel_cases"
WIDTH_MM, HEIGHT_MM = 180, 134


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    planet_audit = json.loads((DEFAULT_OUTPUT / "input_audit_planet.json").read_text(encoding="utf-8"))
    palette = read_palette()
    units = pd.read_parquet(PRODUCT / "master_units.parquet").merge(
        pd.read_parquet(PRODUCT / "national_membership.parquet"), on="unit_id", validate="one_to_one",
    )
    events = pd.read_parquet(PRODUCT / "national/pixel_events.parquet")
    references = pd.read_csv(REFERENCES, keep_default_na=False)
    assert events.candidate_id.is_unique
    events["month"] = pd.to_datetime(events.change_time).dt.strftime("%Y-%m")
    assert events.month.between("2017-01", "2025-12").all()
    geo = []
    for crs, group in units.groupby("track_crs", sort=False):
        geo.append(gpd.GeoDataFrame(group, geometry=gpd.GeoSeries.from_wkb(
            group.native_geometry_wkb, index=group.index, crs=crs)).to_crs(32717))
    pixels = gpd.GeoDataFrame(pd.concat(geo), crs=32717)
    inventory = gpd.read_parquet(FC001)
    inventory = inventory.loc[inventory.source_dataset.eq("fc001_2025")].to_crs(32717)
    master = np.load(PRODUCT / "master_values.npy", mmap_mode="r")
    records = {r["source_row_original"]: r for r in planet_audit["records"]}
    additional_planet_audit = json.loads((DEFAULT_OUTPUT / "input_audit_planet_728_788.json").read_text(encoding="utf-8"))
    records.update({r["source_row_original"]: r for r in additional_planet_audit["records"]})
    context_audit = json.loads((DEFAULT_OUTPUT / "input_audit_planet_context.json").read_text(encoding="utf-8"))
    records.update({r["source_row_original"]: r for r in context_audit["records"]})
    scale_audit = json.loads((DEFAULT_OUTPUT / "input_audit_planet_uniform_scale.json").read_text(encoding="utf-8"))
    records.update({r["source_row_original"]: r for r in scale_audit["records"]})
    cases, curve_tables, all_curve_tables, map_tables, event_tables, numbering_tables, reference_tables = [], [], [], [], [], [], []
    for number in STANDS:
        record = records[number]
        stand_references = references.loc[references.sample_id.eq(record["sample_id"])].copy()
        assert stand_references.timing_type.eq("month").all()
        reference_tables.append(stand_references.assign(source_row_original=number, spatial_scale="stand"))
        stand_units = units.loc[units.sample_id.eq(record["sample_id"])].copy()
        assert stand_units.track_key.nunique() == 1
        track = stand_units.track_key.iloc[0]
        cache_path = DATA_ROOT / f"polygon_pixels/cache/{track}.npz"
        with np.load(cache_path) as cache:
            calendar = pd.DatetimeIndex(cache["calendar"])
            idx = stand_units.cache_index.to_numpy(int)
            valid = cache["valid"][idx]
            values = np.where(valid[:, None, :], cache["values"][idx], np.nan)
            counts = cache["n_obs"][idx] if "n_obs" in cache.files else None
        assert calendar.equals(pd.date_range("2016-01-01", "2026-06-01", freq="MS"))
        np.testing.assert_array_equal(values[:, 0], master[stand_units.series_index.to_numpy(int)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            median = np.nanmedian(values, axis=0)
            distance = np.nanmedian(np.abs(values - median), axis=(1, 2))
        eligible = (valid.sum(axis=1) >= np.median(valid.sum(axis=1))) & stand_units.unit_id.isin(events.unit_id)
        selected_index = int(np.argmin(np.where(eligible, distance, np.inf)))
        chosen = stand_units.iloc[selected_index]
        numbering = stand_units.sort_values(["native_row", "native_col"])[[
            "source_row_original", "sample_id", "unit_id", "pixel_track_id", "native_row", "native_col", "x", "y",
        ]].copy()
        numbering["stand_pixel_number"] = np.arange(1, len(numbering) + 1)
        numbering["selected"] = numbering.unit_id.eq(chosen.unit_id)
        pixel_number = int(numbering.loc[numbering.selected, "stand_pixel_number"].iloc[0])
        numbering_tables.append(numbering)
        selected_events = events.loc[events.unit_id.eq(chosen.unit_id)].sort_values("change_time")
        extent = record.get("recommended_viewport_left_bottom_right_top", record["display_bounds_left_bottom_right_top"])
        np.testing.assert_allclose((extent[2] - extent[0]) * SCALE_BAR_MM / 32, SCALE_BARS_M[number])
        frame = box(*extent)
        visible = pixels.loc[pixels.geometry.intersects(frame)].copy()
        local_events = events.loc[events.unit_id.isin(visible.unit_id)].copy()
        months = local_events.groupby("unit_id").month.agg(list)
        visible["event_months"] = visible.unit_id.map(months).apply(lambda x: sorted(x) if isinstance(x, list) else [])
        visible["last_month"] = visible.event_months.map(lambda x: x[-1] if x else None)
        visible["response_count"] = visible.event_months.map(len)
        boundaries = inventory.loc[inventory.geometry.intersects(frame)]
        assert len(boundaries) == record.get("recommended_viewport_intersecting_fc001_count", record["intersecting_fc001_count"])
        focal = boundaries.loc[boundaries.source_feature_id.eq(f"fc001_2025:{number:08d}")]
        assert len(focal) == 1
        focal_fully_visible = frame.covers(focal.geometry.iloc[0])
        selected_geometry = pixels.loc[pixels.unit_id.eq(chosen.unit_id)].geometry.iloc[0]
        assert frame.covers(selected_geometry)
        curve = pd.DataFrame({
            "source_row_original": number, "sample_id": chosen.sample_id,
            "unit_id": chosen.unit_id, "pixel_track_id": chosen.pixel_track_id,
            "month": calendar.strftime("%Y-%m"), "VV_dB": values[selected_index, 0],
            "VH_dB": values[selected_index, 1], "valid": valid[selected_index],
            "detected": calendar.strftime("%Y-%m").isin(selected_events.month),
        })
        if counts is not None:
            curve["n_obs"] = counts[selected_index]
        curve_tables.append(curve)
        all_curve_tables.append(pd.DataFrame({
            "source_row_original": number,
            "unit_id": np.repeat(stand_units.unit_id.to_numpy(), len(calendar)),
            "pixel_track_id": np.repeat(stand_units.pixel_track_id.to_numpy(), len(calendar)),
            "month": np.tile(calendar.strftime("%Y-%m"), len(stand_units)),
            "VV_dB": values[:, 0, :].ravel(), "VH_dB": values[:, 1, :].ravel(),
            "valid": valid.ravel(),
        }))
        selected_events = selected_events.assign(source_row_original=number, sample_id=chosen.sample_id)
        event_tables.append(selected_events)
        mapped = visible.drop(columns=["geometry", "native_geometry_wkb", "transform"]).copy()
        mapped["panel_source_row"] = number
        mapped["geometry_wkt_epsg32717"] = visible.geometry.to_wkt()
        mapped["event_months"] = mapped.event_months.map(lambda x: ";".join(x))
        map_tables.append(mapped)
        cases.append(dict(record=record, calendar=calendar, values=values[selected_index], all_values=values,
                          chosen=chosen, pixel_number=pixel_number, selected_events=selected_events, visible=visible,
                          boundaries=boundaries, focal=focal, geometry=selected_geometry,
                          viewport=extent, focal_fully_visible=focal_fully_visible,
                          references=stand_references,
                          distance=float(distance[selected_index]), cache_path=cache_path))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    figure = make_figure(cases, palette)
    figure.savefig(out / f"{NAME}.pdf", dpi=600, facecolor="white")
    figure.savefig(out / f"{NAME}.svg", dpi=600, facecolor="white")
    figure.savefig(out / f"{NAME}.png", dpi=600, facecolor="white")
    figure.savefig(out / f"{NAME}.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(out / f"{NAME}_preview.png", dpi=300, facecolor="white")
    for row in range(3):
        for col in range(2):
            ax = figure.axes[2 * row + col]
            bbox = ax.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.dpi_scale_trans.inverted())
            figure.savefig(out / f"panel_{chr(97 + 2 * row + col)}.png", dpi=300, bbox_inches=bbox.padded(0.015), pad_inches=0)
    plt.close(figure)
    pd.concat(curve_tables).to_csv(out / "source_pixel_timeseries.csv", index=False)
    pd.concat(all_curve_tables).to_csv(out / "source_all_stand_pixel_timeseries.csv", index=False)
    pd.concat(event_tables).to_csv(out / "source_selected_pixel_events.csv", index=False)
    pd.concat(map_tables).to_csv(out / "source_map_pixels.csv", index=False)
    pd.concat(numbering_tables).to_csv(out / "source_stand_pixel_numbering.csv", index=False)
    pd.concat(reference_tables).to_csv(out / "source_reference_labels.csv", index=False)
    pd.DataFrame(palette["month_hex"].items(), columns=["month", "hex"]).to_csv(out / "source_month_palette.csv", index=False)
    write_manifest(cases, palette, out)
    print(json.dumps({"figure": str(out / f"{NAME}.pdf"), "cases": [
        {"stand": c["record"]["source_row_original"], "pixel": c["chosen"].unit_id,
         "detections": c["selected_events"].month.tolist()} for c in cases]}, indent=2))


def read_palette() -> dict:
    css = CSS.read_text(encoding="utf-8")
    tokens = dict(re.findall(r"--([\w-]+)\s*:\s*([^;\n}]+)", css))
    anchors = ["#54C7C9", "#7FD4C8", "#A9DEBF", "#D5E7C5", "#F2E8B3",
               "#F7C98D", "#F79E69", "#F56C4A", "#EF3B2C"]
    cmap = LinearSegmentedColormap.from_list("detection_month", anchors, N=4096)
    months = pd.date_range("2017-01-01", "2025-12-01", freq="MS").strftime("%Y-%m")
    return {"source": "user-requested smooth full-period gradient through the nine supplied anchor colors",
            "anchors": anchors,
            "month_hex": {month: to_hex(cmap(index / 107)).upper() for index, month in enumerate(months)},
            "month_rule": "continuous RGB interpolation over 2017-01 to 2025-12; no annual lightness reset; opaque fills",
            "boundary": tokens["data-boundary"], "selected": tokens["data-selected"]}


def event_color(month: str, palette: dict) -> str:
    return palette["month_hex"][month]


def polygon_parts(geometry):
    return [geometry] if geometry.geom_type == "Polygon" else geometry.geoms


def make_figure(cases: list, palette: dict, history_band: str = "VH") -> plt.Figure:
    band_index = {"VV": 0, "VH": 1}[history_band]
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    def axes_mm(left, top, width, height):
        return fig.add_axes([left / WIDTH_MM, 1 - (top + height) / HEIGHT_MM,
                             width / WIDTH_MM, height / HEIGHT_MM])
    ylow, yhigh = (-20, 5) if history_band == "VV" else (-30, -5)
    geo_transform = Transformer.from_crs(32717, 4326, always_xy=True)
    for row, case in enumerate(cases):
        top = 5 + row * 37
        record = case["record"]
        number = record["source_row_original"]
        ax = axes_mm(10, top, 32, 32)
        history = axes_mm(46, top, 119, 32)
        image_xmin, image_ymin, image_xmax, image_ymax = record["display_bounds_left_bottom_right_top"]
        xmin, ymin, xmax, ymax = record.get("recommended_viewport_left_bottom_right_top", record["display_bounds_left_bottom_right_top"])
        image = plt.imread(record["source_png"])
        ax.imshow(image, extent=(image_xmin, image_xmax, image_ymin, image_ymax), interpolation="nearest", origin="upper")
        # Whole native support cells match the GUI; do not clip them to inventory boundaries.
        for pixel in case["visible"].itertuples():
            for poly in polygon_parts(pixel.geometry):
                fill = event_color(pixel.last_month, palette) if pixel.response_count else "#c4c4c4"
                ax.add_patch(Polygon(np.asarray(poly.exterior.coords), facecolor=fill,
                                     edgecolor="none", linewidth=0, zorder=2))
                ax.add_patch(Polygon(np.asarray(poly.exterior.coords), fill=False,
                                     edgecolor="black", linewidth=0.3,
                                     linestyle=(0, (1.5, 1.1)), zorder=3))
        for boundary in case["boundaries"].itertuples():
            is_focal = boundary.source_feature_id == f"fc001_2025:{number:08d}"
            for poly in polygon_parts(boundary.geometry):
                rings = [poly.exterior, *poly.interiors]
                for ring in rings:
                    coords = np.asarray(ring.coords)
                    line, = ax.plot(coords[:, 0], coords[:, 1], color=palette["boundary"],
                                    lw=0.75 if is_focal else 0.45, zorder=4)
                    line.set_path_effects([pe.Stroke(linewidth=1.1 if is_focal else 0.8, foreground="#23342b"), pe.Normal()])
        for poly in polygon_parts(case["geometry"]):
            patch = Polygon(np.asarray(poly.exterior.coords), fill=False, edgecolor="white", linewidth=0.8, zorder=6)
            patch.set_path_effects([pe.Stroke(linewidth=1.4, foreground="#151515"), pe.Normal()])
            ax.add_patch(patch)
        center = case["geometry"].centroid
        pixel_label_offset = (8, -14) if number == 788 else (8, 8)
        ax.annotate(f"P{case['pixel_number']}", xy=(center.x, center.y), xytext=pixel_label_offset, textcoords="offset points",
                    fontsize=6.3, color="white", weight="bold", zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.1, foreground="#161616")],
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8,
                                    path_effects=[pe.withStroke(linewidth=1.7, foreground="#161616")]))
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
        x_ticks = [xmin + (xmax - xmin) * f for f in (0.23, 0.77)]
        y_ticks = [ymin + (ymax - ymin) * f for f in (0.23, 0.77)]
        ax.set_xticks(x_ticks, [f"{abs(geo_transform.transform(x, (ymin + ymax) / 2)[0]):.3f}°W" for x in x_ticks])
        ax.set_yticks(y_ticks, [f"{abs(geo_transform.transform((xmin + xmax) / 2, y)[1]):.3f}°S" for y in y_ticks])
        ax.tick_params(axis="both", labelsize=5.6, length=2, pad=1.5)
        ax.tick_params(axis="y", labelrotation=90)
        for tick in ax.get_yticklabels():
            tick.set_va("center")
        scale_m = SCALE_BARS_M[number]
        sx, sy = xmax - (xmax - xmin) * 0.07 - scale_m, ymin + (ymax - ymin) * 0.075
        line, = ax.plot([sx, sx + scale_m], [sy, sy], color="white", lw=1.6, solid_capstyle="butt", zorder=8)
        line.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#151515"), pe.Normal()])
        ax.text(sx + scale_m / 2, sy + (ymax - ymin) * 0.03, f"{scale_m} m", fontsize=5.8, color="white", ha="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="#161616")])
        detail = " detail" if not case["focal_fully_visible"] else ""
        ax.text(0.03, 0.96, f"({chr(97 + row * 2)}) Stand {number}{detail}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, weight="bold", color="white", zorder=10,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#151515")])
        history.text(0.01, 0.96, f"({chr(98 + row * 2)}) Stand {number}  Pixel P{case['pixel_number']}", transform=history.transAxes,
                     ha="left", va="top", fontsize=8, weight="bold", color="#151515", zorder=10,
                     path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
        history.set_facecolor("white")
        history.plot(case["calendar"], case["all_values"][:, band_index, :].T,
                     color="#bdc3bf", alpha=0.38, linewidth=0.3, zorder=1)
        history.plot(case["calendar"], case["values"][band_index], color=VH_COLOR, lw=0.85,
                     marker="o", markersize=1.45, markeredgewidth=0, label=history_band, zorder=3)
        for event_index, event in enumerate(case["selected_events"].itertuples()):
            date = pd.Timestamp(event.change_time)
            history.axvline(date, color=DETECTION_COLOR, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
            label_y = 0.82 + 0.10 * (event_index % 2)
            history.annotate(event.month, xy=(mdates.date2num(date), label_y), xycoords=history.get_xaxis_transform(),
                             xytext=(-3, 0), textcoords="offset points", ha="right", va="top",
                             fontsize=6.2, color=DETECTION_COLOR,
                             path_effects=[pe.withStroke(linewidth=1.2, foreground="white")], zorder=6)
        for reference in case["references"].itertuples():
            date = pd.Timestamp(reference.reference_month)
            history.axvline(date, color=REFERENCE_COLOR, linewidth=0.8, linestyle=(0, (1, 2)), zorder=5)
            history.annotate(reference.reference_month, xy=(mdates.date2num(date), 0.67),
                             xycoords=history.get_xaxis_transform(), xytext=(-3, 0), textcoords="offset points",
                             ha="right", va="top", fontsize=6.2, color=REFERENCE_COLOR,
                             path_effects=[pe.withStroke(linewidth=1.2, foreground="white")], zorder=7)
        if case["references"].empty:
            history.text(0.012, 0.035, "No dated reference", transform=history.transAxes,
                         fontsize=5.8, color="#666666", va="bottom")
        observed_months = case["calendar"][np.isfinite(case["all_values"][:, band_index, :]).any(axis=0)]
        history.set_xlim(observed_months[0], observed_months[-1])
        history.set_ylim(ylow, yhigh)
        history.set_yticks(np.arange(ylow, yhigh + 1, 5))
        history.set_ylabel(f"{history_band} (dB)", fontsize=7.2, labelpad=4)
        history.yaxis.tick_right()
        history.yaxis.set_label_position("right")
        history.xaxis.set_major_locator(mdates.YearLocator())
        history.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        history.tick_params(axis="both", labelsize=6.4, length=2.5, pad=2)
        history.spines[["top", "left"]].set_visible(False)
        history.grid(False)
    month_dates = mdates.date2num(pd.to_datetime(list(palette["month_hex"])))
    # January 2026 marks the end of December 2025, not another reporting month.
    month_dates = np.append(month_dates, mdates.date2num(pd.Timestamp("2026-01-01")))
    month_index = np.arange(len(month_dates))
    history_start, history_end = history.get_xlim()
    bar_left, bar_right = 46 + 119 * (month_dates[[0, -1]] - history_start) / (history_end - history_start)
    fig.text((bar_left - 2) / WIDTH_MM, 1 - 118 / HEIGHT_MM, "Latest detection month",
             fontsize=7.2, weight="bold", ha="right")
    legend = axes_mm(bar_left, 115.8, bar_right - bar_left, 2.6)
    bar_colors = list(palette["month_hex"].values())
    cmap = LinearSegmentedColormap.from_list("detection_month",
        list(zip(month_index / month_index[-1], bar_colors + [bar_colors[-1]])), N=4096)
    # Preserve the monthly LUT while placing the bar on the actual calendar axis.
    norm = FuncNorm((lambda dates: np.interp(dates, month_dates, month_index),
                     lambda indices: np.interp(indices, month_index, month_dates)),
                    vmin=month_dates[0], vmax=month_dates[-1])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=legend,
                           orientation="horizontal", drawedges=False)
    colorbar.ax.set_xscale("linear")
    colorbar.set_ticks(month_dates[::12])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(which="major", length=2.5, top=True, bottom=False,
                           labeltop=False, labelbottom=False)
    colorbar.ax.minorticks_off()
    colorbar.solids.set_edgecolor("face")
    handles = [Line2D([], [], color=palette["boundary"], lw=1.2, label="FC001 boundary"),
               Line2D([], [], color="#bdc3bf", lw=0.7, label=f"Stand {history_band}"),
               Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor="#202020",
                      color="none", markersize=4, label="Selected pixel"),
               Line2D([], [], color=VH_COLOR, lw=1, label=f"Selected {history_band}"),
               Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=0.7,
                         linestyle=(0, (1.5, 1.1)), label="Native grid"),
               Rectangle((0, 0), 1, 1, facecolor="#c4c4c4", edgecolor="none", label="No detection"),
               Line2D([], [], color=REFERENCE_COLOR, lw=0.8, linestyle=(0, (1, 2)), label="Dated Reference"),
               Line2D([], [], color=DETECTION_COLOR, lw=1.1, linestyle=(0, (4, 2)), label="Pixelwise detection")]
    fig.legend(handles=handles, loc="upper left", ncol=4, mode="expand", frameon=False,
               bbox_to_anchor=(10 / WIDTH_MM, 1 - 124 / HEIGHT_MM, 165 / WIDTH_MM, 0),
               fontsize=6.4, handlelength=1.7, handletextpad=0.6, borderaxespad=0, labelspacing=0.4)
    return fig


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(cases: list, palette: dict, out: Path) -> None:
    sources = [CSS, FC001, REFERENCES, DEFAULT_OUTPUT / "input_audit_reference_labels.json",
               PRODUCT / "protocol.json", PRODUCT / "master_units.parquet",
               PRODUCT / "master_values.npy", PRODUCT / "national_membership.parquet",
               PRODUCT / "national/pixel_events.parquet", Path(__file__).resolve(),
               DEFAULT_OUTPUT / "input_audit_planet.json", DEFAULT_OUTPUT / "input_audit_planet_728_788.json",
               DEFAULT_OUTPUT / "input_audit_planet_context.json", DEFAULT_OUTPUT / "input_audit_planet_uniform_scale.json",
               DEFAULT_OUTPUT / "input_audit_first_case_reference.json"]
    sources += list(dict.fromkeys(c["cache_path"] for c in cases))
    for c in cases:
        sources.extend([Path(c["record"]["source_png"]), Path(c["record"]["source_json"])])
    manifest = {
        "figure": NAME, "dimensions_mm": [WIDTH_MM, HEIGHT_MM], "raster_export_dpi": 600,
        "palette": palette, "input_period": ["2016-01", "2026-06"],
        "display_period": "first through last finite month in each target stand; 2016-10 through 2026-06 for these cases",
        "layout": {"column_gap_mm": 4, "row_step_mm": 37, "row_gap_mm": 5,
                   "map_size_mm": [32, 32], "history_size_mm": [119, 32],
                   "panel_labels": "inside upper left: stand ID on maps; stand ID and stand-local pixel number on curves",
                   "scale_bars": "lower right; identical 8 mm physical length; map extents set by labelled distance",
                   "scale_bar_mm": SCALE_BAR_MM, "compass_arrows": False, "column_headings": False,
                   "history_y_axis": "right; VH only; -30 to -5 dB", "history_background": "white; no shaded context or grid",
                   "legend": "smooth colorbar positioned on panel (f)'s actual date coordinates; annual ticks without repeated year labels; shared symbols in two rows",
                   "colorbar_period": ["2017-01-01", "2026-01-01"],
                   "colorbar_end_semantics": "2026 tick is the upper boundary of December 2025; no 2026 detections added",
                   "colorbar_tick_side": "top", "colorbar_title_gap_mm": 2,
                   "colorbar_top_mm": 115.8, "colorbar_height_mm": 2.6,
                   "colorbar_title_baseline_top_mm": 118, "symbol_legend_top_mm": 124},
        "report_period": ["2017-01", "2025-12"],
        "planet_months": {str(c["record"]["source_row_original"]): c["record"]["month"] for c in cases},
        "series_statistic": "VH of all target-stand native pixels in light grey; one representative native pixel highlighted; both original bands retained in exported source tables",
        "detector_band": "VV_dB", "VH_role": "descriptive companion, not a detector input band",
        "reference_semantics": "independent stand-level optical labels; no pixel-level removal footprint; absence of a dated label is not evidence of no event",
        "event_styles": {"pixel_detection": {"label": "Pixelwise detection", "color": DETECTION_COLOR,
                                              "line": "dashed", "width_pt": 1.1, "zorder": 4},
                         "dated_reference": {"label": "Dated Reference", "color": REFERENCE_COLOR,
                                             "line": "dotted", "width_pt": 0.8, "zorder": 5},
                         "date_labels": "inside each history panel, 3 pt left of the corresponding line; detection labels alternate y=0.82/0.92 to separate nearby dates; references at y=0.67 in axes coordinates"},
        "selection": "among detected pixels with at least median valid-month count, smallest median absolute distance over both bands to stand monthly medians; no p-value ranking",
        "pixel_numbering": "1-based within all native support cells assigned to the focal stand; native_row ascending then native_col ascending (top to bottom, left to right); includes no-detection cells, excludes neighbors",
        "map_rule": "last accepted date across 2017-2025; whole native cells; uniform black dashed cell edges; no-response cells filled #c4c4c4",
        "map_projection": "EPSG:32717; geographic tick labels; identical 200 m scales for 716/728/788; north-up",
        "image_processing": "real same-month PlanetScope RGB windows, nearest-neighbor display grid 3 m, no extra contrast stretch; exact 800 m viewports; Stand 788 is a selected-pixel-centered detail; source raster extent remains distinct from viewport",
        "QA": {"VV_exact_master_equality_all_case_pixels": True, "calendar_126_positions": True,
               "unique_event_ids": True, "all_selected_events_drawn": True,
               "all_selected_pixels_inside_maps": True,
               "whole_stands_visible": {str(c["record"]["source_row_original"]): bool(c["focal_fully_visible"]) for c in cases},
               "smooth_palette_from_user_anchors": True, "boundary_colors_from_GUI_CSS": True},
        "cases": [{"source_row_original": c["record"]["source_row_original"],
                   "sample_id": c["chosen"].sample_id, "unit_id": c["chosen"].unit_id,
                   "pixel_track_id": c["chosen"].pixel_track_id,
                   "stand_pixel_number": c["pixel_number"],
                   "planet_month": c["record"]["month"],
                   "map_bounds_left_bottom_right_top": c["viewport"],
                   "map_width_m": c["viewport"][2] - c["viewport"][0],
                   "scale_bar_m": SCALE_BARS_M[c["record"]["source_row_original"]],
                   "scale_bar_mm": SCALE_BAR_MM,
                   "focal_fully_visible": bool(c["focal_fully_visible"]),
                   "focal_longest_edge_fraction": float(max(c["focal"].total_bounds[2:] - c["focal"].total_bounds[:2]) / (c["viewport"][2] - c["viewport"][0])),
                   "cache_index": int(c["chosen"].cache_index),
                   "median_distance_db": c["distance"],
                   "valid_months": int(np.isfinite(c["values"][0]).sum()),
                   "target_stand_pixel_count": len(c["all_values"]),
                   "events": c["selected_events"].month.tolist(),
                   "reference_months": c["references"].reference_month.tolist(),
                   "reference_ids": c["references"].verified_event_id.tolist(),
                   "visible_FC001_stands": len(c["boundaries"]), "visible_native_pixels": len(c["visible"]),
                   "visible_pixels_with_detections": int(c["visible"].response_count.gt(0).sum()),
                   "visible_pixels_with_multiple_detections": int(c["visible"].response_count.gt(1).sum())}
                  for c in cases],
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
