"""Link native pixel detections, complete stand histories and latest STP footprints.

The frozen detector and aggregation products are read without rerunning them.
Spatial case selection is fixed independently of event dates or STP outcomes.
"""
from pathlib import Path
import json
import subprocess

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box

import build_harvest_pixel_cases as base


ROOT = base.ROOT
PRODUCT = ROOT / "runs/validation/harvest_inventory_20261104_v1"
OUT = ROOT / "runs/figures/harvest_stand_histories_20261112_v5"
CONTEXT = ROOT / "runs/figures/harvest_pixel_tda_stp_cases_20261111_v3_context"
CASES = {716: [719, 717, 718, 716, 715],
         728: [730, 728, 727, 724, 25621],
         788: [788, 2285, 2284, 2279, 789]}
WIDTH_MM, HEIGHT_MM = 180, 168
TOP_MM, ROW_STEP_MM = 9, 49
HISTORY_LEFT_MM, HISTORY_WIDTH_MM = 65, 64
LEFT_MAP_MM, RIGHT_MAP_MM = 9, 134
TDA_COLOR = "#197663"
STP_COLORS = ["#76BCA7", "#C3DDD3", "#438E79"]
NO_DETECTION = "#c4c4c4"
DRAW_EMPTY_CELL_BORDERS = False
OUTSIDE_LATEST = "#e1e4e1"
HISTORY_DATE_MARKERS = True
SHOW_BASEMAP_MONTH = True
SCALE_BAR_Y_FRACTION = .07


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [base.PRODUCT / "master_units.parquet", base.PRODUCT / "national_membership.parquet",
               base.PRODUCT / "national/pixel_events.parquet", PRODUCT / "octa/reports.parquet",
               PRODUCT / "cstr/rounds.parquet", PRODUCT / "cstr/node_ownership.parquet", base.FC001]
    source_hashes = {str(p): base.sha256(p) for p in sources}
    palette = base.read_palette()
    records = {r["source_row_original"]: r for r in json.loads(
        (CONTEXT / "imagery/input_imagery.json").read_text(encoding="utf-8"))["records"]}
    expanded = json.loads((OUT / "imagery/input_imagery.json").read_text(encoding="utf-8"))
    records.update({r["source_row_original"]: r for r in expanded["records"]})
    stand_ids = [s for ids in CASES.values() for s in ids]
    inventory = gpd.read_parquet(base.FC001).to_crs(32717)
    inventory = inventory.loc[inventory.source_dataset.eq("fc001_2025")].copy()
    inventory["stand_id"] = inventory.source_feature_id.str.rsplit(":", n=1).str[-1].astype(int)
    boundaries = inventory.loc[inventory.stand_id.isin(stand_ids)].copy()
    assert len(boundaries) == len(stand_ids)
    members = pd.read_parquet(base.PRODUCT / "national_membership.parquet")
    members = members.loc[members.source_row_original.isin(stand_ids)].copy()
    units = pd.read_parquet(base.PRODUCT / "master_units.parquet").merge(members, on="unit_id", validate="one_to_one")
    geo = []
    for crs, group in units.groupby("track_crs", sort=False):
        geo.append(gpd.GeoDataFrame(group, geometry=gpd.GeoSeries.from_wkb(
            group.native_geometry_wkb, index=group.index, crs=crs)).to_crs(32717))
    pixels = gpd.GeoDataFrame(pd.concat(geo), crs=32717)
    events = pd.read_parquet(base.PRODUCT / "national/pixel_events.parquet")
    events = events.loc[events.unit_id.isin(members.unit_id) & events.Band.eq("VV_dB")].copy()
    events["month"] = pd.to_datetime(events.change_time).dt.strftime("%Y-%m")
    assert events.month.between("2017-01", "2025-12").all()
    pixels["last_month"] = pixels.unit_id.map(events.groupby("unit_id").month.max())
    parents = pd.read_parquet(PRODUCT / "octa/reports.parquet")
    parents = parents.loc[parents.sample_id.isin(members.sample_id)].copy()
    parents = parents.sort_values(["sample_id", "representative_month", "prediction_id"])
    latest = parents.groupby("sample_id", sort=False).tail(1)
    rounds = pd.read_parquet(PRODUCT / "cstr/rounds.parquet")
    rounds = rounds.loc[rounds.parent_prediction_id.isin(latest.prediction_id)].copy()
    ownership = pd.read_parquet(PRODUCT / "cstr/node_ownership.parquet")
    ownership = ownership.loc[ownership.prediction_id.isin(latest.prediction_id)].copy()
    owners = ownership[["sample_id", "unit_id", "prediction_id", "round_id"]].drop_duplicates()
    assert not owners.duplicated("unit_id").any()
    owners = owners.merge(rounds[["round_id", "round_index"]], on="round_id", validate="many_to_one")
    pixels = pixels.merge(owners[["unit_id", "prediction_id", "round_id", "round_index"]], on="unit_id", how="left", validate="one_to_one")
    for report in parents.itertuples():
        assert members.loc[members.sample_id.eq(report.sample_id), "unit_id"].nunique() == report.total_pixels
    for report in latest.itertuples():
        assert owners.loc[owners.prediction_id.eq(report.prediction_id), "unit_id"].nunique() == report.response_pixels
    assert int(rounds.round_index.max()) < len(STP_COLORS)
    cases, stand_rows = [], []
    for row, (focal, ids) in enumerate(CASES.items()):
        record = records[focal]
        extent = record["recommended_viewport_left_bottom_right_top"]
        frame = box(*extent)
        shapes = boundaries.set_index("stand_id").loc[ids].reset_index()
        assert all(frame.covers(g) for g in shapes.geometry), f"Cropped stand in {focal}"
        local_members = members.loc[members.source_row_original.isin(ids)]
        local_parents = parents.loc[parents.sample_id.isin(local_members.sample_id)].copy()
        local_latest = latest.loc[latest.sample_id.isin(local_members.sample_id)].copy()
        local_pixels = pixels.loc[pixels.source_row_original.isin(ids)].copy()
        for index, stand in enumerate(ids):
            sample_id = members.loc[members.source_row_original.eq(stand), "sample_id"].iloc[0]
            n = len(parents.loc[parents.sample_id.eq(sample_id)])
            last = latest.loc[latest.sample_id.eq(sample_id)]
            stand_rows.append({"case": focal, "letter": chr(65 + index), "stand_id": stand,
                "sample_id": sample_id, "fixed_pixels": int(members.sample_id.eq(sample_id).sum()),
                "parent_count": n, "latest_prediction_id": last.prediction_id.iloc[0] if n else "",
                "latest_representative_month": last.representative_month.iloc[0] if n else "",
                "latest_stp_groups": int(rounds.parent_prediction_id.isin(last.prediction_id).sum()),
                "polygon_fully_inside_view": True})
        cases.append({"focal": focal, "ids": ids, "record": record, "shapes": shapes,
                      "pixels": local_pixels, "parents": local_parents, "latest": local_latest})
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": .6,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)
    for row, case in enumerate(cases):
        top = TOP_MM + row * ROW_STEP_MM
        left = fig.add_axes([LEFT_MAP_MM / WIDTH_MM, 1 - (top + 40) / HEIGHT_MM, 40 / WIDTH_MM, 40 / HEIGHT_MM])
        history = fig.add_axes([HISTORY_LEFT_MM / WIDTH_MM, 1 - (top + 40) / HEIGHT_MM,
                                HISTORY_WIDTH_MM / WIDTH_MM, 40 / HEIGHT_MM])
        right = fig.add_axes([RIGHT_MAP_MM / WIDTH_MM, 1 - (top + 40) / HEIGHT_MM, 40 / WIDTH_MM, 40 / HEIGHT_MM])
        draw_map(left, case, palette, chr(97 + row * 3), stp=False)
        history_handles = draw_history(history, case, chr(98 + row * 3)) or []
        draw_map(right, case, palette, chr(99 + row * 3), stp=True)
        np.testing.assert_array_equal(left.get_xlim(), right.get_xlim())
        np.testing.assert_array_equal(left.get_ylim(), right.get_ylim())
    draw_decorations(fig, palette, parents, rounds, history_handles)
    stem = OUT / "fig_harvest_stand_histories"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), dpi=600, metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-scale-to", "2300", "-singlefile", "-png", str(stem.with_suffix(".pdf")), str(OUT / "preview")], check=True)
    labels = pd.DataFrame(stand_rows)
    labels.to_csv(OUT / "source_stands.csv", index=False)
    parents.merge(labels[["case", "letter", "stand_id", "sample_id"]], on="sample_id", validate="many_to_one").to_csv(OUT / "source_tda_reports.csv", index=False)
    rounds.merge(labels[["case", "letter", "stand_id", "sample_id"]], on="sample_id", validate="many_to_one").to_csv(OUT / "source_latest_stp_groups.csv", index=False)
    pixels[["source_row_original", "sample_id", "unit_id", "last_month", "prediction_id", "round_id", "round_index", "geometry"]].to_parquet(OUT / "source_pixels.parquet", index=False)
    manifest = {"dimensions_mm": [WIDTH_MM, HEIGHT_MM], "case_stands": CASES,
        "parent_count": len(parents), "latest_parent_count": len(latest), "latest_stp_group_count": len(rounds),
        "fixed_pixels": len(pixels), "history_period": "2017-01 to 2025-12",
        "latest_rule": "latest representative_month in each stand, then prediction_id; not a common anchor month",
        "history_spans": "first response month start to month after last response month; full stand reports",
        "selection": "five fixed spatial neighbours per original focal area; fixed A-E labels shared across panels",
        "stp_colours": "final round_index within each parent; not a shared group identity across stands",
        "map_y_labels": "left for pixel maps, right for STP maps", "algorithm_rerun": False,
        "image_records": records, "source_sha256": source_hashes,
        "script_sha256": base.sha256(Path(__file__)), "all_polygons_fully_visible": True}
    assert source_hashes == {str(p): base.sha256(p) for p in sources}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(labels.to_string(index=False))
    print(stem.with_suffix(".pdf"))


def draw_decorations(fig, palette, parents, rounds, history_handles):
    for x, title in [(29, "Pixelwise detections"), (97, "Stand harvest histories"), (154, "Latest-event STP")]:
        fig.text(x / WIDTH_MM, 1 - 2 / HEIGHT_MM, title, ha="center", va="top", fontsize=7.5, weight="bold")
    cax = fig.add_axes([12 / WIDTH_MM, 10 / HEIGHT_MM, 73 / WIDTH_MM, 2.6 / HEIGHT_MM])
    cmap = LinearSegmentedColormap.from_list("detection_month", palette["anchors"], N=4096)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 107), cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_ticks([0, 24, 48, 72, 96, 107], labels=["2017", "2019", "2021", "2023", "2025", "Dec."])
    bar.ax.tick_params(labelsize=5.8, length=2, pad=1)
    bar.outline.set_visible(False)
    cax.text(.5, 1.75, "Latest pixel-detection month", transform=cax.transAxes, ha="center", fontsize=6.3)
    groups = int(rounds.round_index.max()) + 1
    handles = [Patch(facecolor=NO_DETECTION, label="No pixel detection"),
               Patch(facecolor=OUTSIDE_LATEST, label="Outside latest event")]
    if HISTORY_DATE_MARKERS:
        handles.append(Line2D([], [], color=TDA_COLOR, lw=1.1, marker="D", markersize=3, label="TDA range / representative month"))
    if parents[["left_censored", "right_censored"]].any().any():
        handles.append(Line2D([], [], color=TDA_COLOR, lw=0, marker=">", markerfacecolor="white", markersize=4, label="Censored end"))
    handles.extend(Patch(facecolor=STP_COLORS[i], label=f"STP {i+1}") for i in range(groups))
    handles.extend(history_handles)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(94 / WIDTH_MM, 16 / HEIGHT_MM),
               ncol=2, fontsize=5.7, frameon=False, handlelength=1.4, handletextpad=.4,
               columnspacing=.9, labelspacing=.5, borderaxespad=0)


def draw_map(ax, case, palette, panel, stp, show_labels=True):
    record = case["record"]
    xmin, ymin, xmax, ymax = record["recommended_viewport_left_bottom_right_top"]
    ix0, iy0, ix1, iy1 = record["display_bounds_left_bottom_right_top"]
    ax.imshow(plt.imread(record["source_png"]), extent=(ix0, ix1, iy0, iy1), origin="upper", interpolation="nearest")
    for pixel in case["pixels"].itertuples():
        if stp:
            fill = STP_COLORS[int(pixel.round_index)] if pd.notna(pixel.round_index) else OUTSIDE_LATEST
        else:
            fill = base.event_color(pixel.last_month, palette) if pd.notna(pixel.last_month) else NO_DETECTION
            if fill == "none" and not DRAW_EMPTY_CELL_BORDERS:
                continue
        for polygon in base.polygon_parts(pixel.geometry):
            ax.add_patch(Polygon(np.asarray(polygon.exterior.coords), facecolor=fill,
                                edgecolor="black" if fill == "none" else "#202523", lw=.22, zorder=2))
    halo = [pe.withStroke(linewidth=1.8, foreground="#1c2b24")]
    for index, stand in enumerate(case["shapes"].itertuples()):
        for polygon in base.polygon_parts(stand.geometry):
            for ring in [polygon.exterior, *polygon.interiors]:
                coords = np.asarray(ring.coords)
                line, = ax.plot(coords[:, 0], coords[:, 1], color=palette["boundary"], lw=.75, zorder=4)
                line.set_path_effects([pe.Stroke(linewidth=1.3, foreground="#23342b"), pe.Normal()])
        if show_labels:
            point = stand.geometry.representative_point()
            offset = {2285: (-150, 260), 2284: (100, -260)}.get(stand.stand_id, (0, 0))
            if offset != (0, 0):
                line, = ax.plot([point.x, point.x+offset[0]], [point.y, point.y+offset[1]],
                                color="white", lw=.6, zorder=5)
                line.set_path_effects([pe.Stroke(linewidth=1.2, foreground="#1c2b24"), pe.Normal()])
            ax.text(point.x+offset[0], point.y+offset[1], chr(65 + index), ha="center", va="center", color="white",
                    fontsize=8.7, weight="bold", path_effects=halo, zorder=6)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
    transform = Transformer.from_crs(32717, 4326, always_xy=True)
    ticks_x = [xmin + (xmax - xmin) * f for f in (.23, .77)]
    ticks_y = [ymin + (ymax - ymin) * f for f in (.23, .77)]
    ax.set_xticks(ticks_x, [f"{abs(transform.transform(x, (ymin+ymax)/2)[0]):.3f}°W" for x in ticks_x])
    ax.set_yticks(ticks_y, [f"{abs(transform.transform((xmin+xmax)/2, y)[1]):.3f}°S" for y in ticks_y])
    ax.tick_params(axis="both", labelsize=5.7, length=2, pad=1.5)
    ax.tick_params(axis="y", labelrotation=90)
    if stp:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    for tick in ax.get_yticklabels():
        tick.set_va("center")
    scale = record.get("recommended_scale_bar_m", 200)
    sx, sy = xmax - (xmax-xmin)*.06-scale, ymin+(ymax-ymin)*SCALE_BAR_Y_FRACTION
    line, = ax.plot([sx, sx+scale], [sy, sy], color="white", lw=1.6, zorder=6, solid_capstyle="butt")
    line.set_path_effects([pe.Stroke(linewidth=2.7, foreground="#171d19"), pe.Normal()])
    ax.text(sx+scale/2, sy+(ymax-ymin)*.035, f"{scale:g} m", color="white", fontsize=5.7,
            ha="center", path_effects=halo, zorder=6)
    ax.text(.03, .97, f"({panel})", transform=ax.transAxes, color="white", fontsize=8, weight="bold",
            va="top", path_effects=halo, zorder=7)
    if SHOW_BASEMAP_MONTH:
        ax.text(.03, .035, f"PS {record['month']}", transform=ax.transAxes, color="white", fontsize=5.7,
                va="bottom", path_effects=halo, zorder=7)


def draw_history(ax, case, panel):
    ax.set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2026-01-01"))
    ax.set_ylim(4.65, -.85)
    ax.set_yticks(range(5), [f"{chr(65+i)}  {s}" for i, s in enumerate(case["ids"])])
    ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in [2017, 2019, 2021, 2023, 2025]]))
    ax.tick_params(axis="x", labelsize=5.9, length=2, pad=2)
    ax.tick_params(axis="y", labelsize=6.3, length=0, pad=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#7d8983")
    ax.grid(axis="x", color="#e0e6e2", linewidth=.5, zorder=0)
    for index, stand in enumerate(case["ids"]):
        ax.axhspan(index-.38, index+.38, facecolor="#f2f5f3", edgecolor="none", zorder=-1)
        sid = case["pixels"].loc[case["pixels"].source_row_original.eq(stand), "sample_id"].iloc[0]
        reports = case["parents"].loc[case["parents"].sample_id.eq(sid)]
        latest_ids = set(case["latest"].prediction_id)
        if reports.empty:
            ax.text(pd.Timestamp("2021-07-01"), index, "No retained event", ha="center", va="center", fontsize=5.8, color="#747b77")
        for report in reports.itertuples():
            first = pd.Timestamp(report.first_response_month)
            end = pd.Timestamp(report.last_response_month) + pd.offsets.MonthBegin(1)
            is_latest = report.prediction_id in latest_ids
            alpha = 1 if is_latest else .6
            ax.plot([first, end], [index, index], color=TDA_COLOR, lw=1.2, alpha=alpha, solid_capstyle="butt", zorder=2)
            for edge, censored, marker in [(first, report.left_censored, "<"), (end, report.right_censored, ">")]:
                if censored:
                    ax.scatter([edge], [index], marker=marker, s=20, facecolor="white",
                               edgecolor=TDA_COLOR, linewidth=.8, zorder=4, clip_on=False)
                else:
                    ax.vlines(edge, index-.065, index+.065, color=TDA_COLOR, linewidth=.6, alpha=alpha, zorder=2)
            ax.scatter([pd.Timestamp(report.representative_month)], [index], marker="D", s=15,
                       facecolor=TDA_COLOR, edgecolor="#142c23" if is_latest else "white", linewidth=.45,
                       alpha=alpha, zorder=3)
    ax.text(.01, .98, f"({panel})", transform=ax.transAxes, fontsize=8, weight="bold", va="top")


if __name__ == "__main__":
    main()
