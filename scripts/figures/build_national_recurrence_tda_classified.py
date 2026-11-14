"""Classified map colours for the frozen enhanced national figure."""
from pathlib import Path
import hashlib
import json
import subprocess

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as effects
import numpy as np
import pandas as pd
from pypdf import PdfReader
from shapely.geometry import box

from build_national_recurrence_tda_draft import ROOT, PRODUCT, OLD_GRID, ADMIN, BLUE, PURPLE, INK, GREY


OUT = ROOT / "runs/figures/national_recurrence_tda_20261110_v3_classified"
PREVIOUS = ROOT / "runs/figures/national_recurrence_tda_20261110_v2_enhanced"
WIDTH, HEIGHT = 180, 216
COUNT_COLORS = [GREY, "#A6CEE3", "#4C95B8", "#164E70"]
INTERVAL_COLORS = ["#E6DDF2", "#C6ADDF", "#A07AC5", "#7851A6", "#542D80", "#32134F"]
COUNT_LABELS = ["0", "(0, 1]", "(1, 2]", "(2, 3]"]
INTERVAL_LABELS = ["<12", "12–<24", "24–<36", "36–<48", "48–<60", "≥60"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data_out = OUT / "source_data"
    data_out.mkdir(exist_ok=True)
    protected = [*sorted((ROOT/"manuscript_JAG").rglob("*.tex")),
                 ROOT/"manuscript_JAG/figures/fig07_national_harvest_patterns.pdf",
                 ROOT/"runs/figures/national_recurrence_tda_20261109_v1/national_recurrence_five_panel.pdf",
                 PREVIOUS/"national_recurrence_enhanced.pdf"]
    original = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    stands = gpd.read_file(PRODUCT/"harvest_inventory_2017_2025.gpkg", layer="stand_history").to_crs(32717)
    assignments = pd.read_csv(OLD_GRID/"polygon_hex_assignments_10km.csv", usecols=["polygon_id", "hex_id"])
    stands = stands.merge(assignments, left_on="sample_id", right_on="polygon_id", validate="one_to_one")
    grid = gpd.read_parquet(OLD_GRID/"ecuador_mainland_hex_10km_width_epsg6933.parquet")
    reports = pd.read_csv(PRODUCT/"event_reports.csv")
    reports = reports.loc[reports.method.eq("OCTA")].copy()
    pairs = pd.read_csv(PRODUCT/"adjacent_report_intervals.csv")
    pairs = pairs.loc[pairs.method.eq("OCTA") & pairs.report_level.eq("report")]
    assert len(stands) == 3498 and stands.sample_id.is_unique
    assert stands.model_evaluable.sum() == 3364 and len(reports) == 6303 and len(pairs) == 3372
    eligible = stands.loc[stands.model_evaluable]
    metrics = stands.groupby("hex_id").agg(n_inventory=("sample_id", "size"),
        n_evaluable=("model_evaluable", "sum"), n_interval=("octa_interval_median_months", "count"),
        median_count=("octa_count", "median"), median_interval=("octa_interval_median_months", "median"))
    cells = grid.merge(metrics.reset_index(), on="hex_id", how="inner").to_crs(4326)
    # Count classes are right-closed; interval classes are left-closed.
    cells["count_class"] = pd.cut(cells.median_count, [0, 1, 2, 3], labels=[1, 2, 3], right=True).astype(float)
    cells.loc[cells.median_count.eq(0), "count_class"] = 0
    cells["interval_class"] = pd.cut(cells.median_interval, [0, 12, 24, 36, 48, 60, np.inf],
                                      labels=range(6), right=False).astype(float)
    assert cells.count_class.notna().equals(cells.median_count.notna())
    assert cells.interval_class.notna().equals(cells.median_interval.notna())
    annual = reports.groupby(pd.to_datetime(reports.representative_month).dt.year).size().reindex(range(2017, 2026), fill_value=0)
    counts = eligible.octa_count.astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    intervals = pairs.interval_months.to_numpy()
    edges = np.arange(0, 109, 6)
    hist, _ = np.histogram(intervals, bins=edges)
    q1, median, q3 = np.quantile(intervals, [.25, .5, .75])
    assert (annual.sum(), counts.sum(), hist.sum(), q1, median, q3) == (6303, 3364, 3372, 23, 34, 43)

    # Select by inventory density and distance only, before reading local histories.
    ranked = metrics.reset_index().sort_values(["n_inventory", "hex_id"], ascending=[False, True])
    local_hex = ranked.iloc[0].hex_id
    centre = grid.loc[grid.hex_id.eq(local_hex)].to_crs(32717).geometry.iloc[0].centroid
    local = stands.loc[stands.hex_id.eq(local_hex)].copy()
    local["distance_to_hex_centre_m"] = local.representative_point().distance(centre)
    local = local.sort_values(["distance_to_hex_centre_m", "sample_id"]).head(8).copy()
    local["northing"] = local.representative_point().y
    local["easting"] = local.representative_point().x
    local = local.sort_values(["northing", "easting", "sample_id"], ascending=[False, True, True]).reset_index(drop=True)
    local["map_label"] = list("ABCDEFGH")
    local_reports = reports.merge(local[["sample_id", "map_label"]], on="sample_id", validate="many_to_one")
    assert len(local) == 8 and len(local_reports) == 20
    assert local.source_row_original.astype(int).tolist() == [22769, 22768, 22766, 22764, 22763, 22762, 22761, 22760]
    assert len(local.loc[~local.model_evaluable]) == 1
    selection = {"rule": "highest inventory-count fixed hex; ties by hex_id; nearest eight interior representative points to hex centre in EPSG:32717; ties by sample_id; includes NA",
                 "hex_id": local_hex, "inventory_stands_in_hex": int(ranked.iloc[0].n_inventory),
                 "label_order": "north to south, then west to east",
                 "n_selected": len(local), "n_local_parents": len(local_reports),
                 "local_bounds_utm": local.total_bounds.tolist(),
                 "interpretation": "spatial illustration, not a probability sample or independent accuracy test",
                 "shared_source_objectid": local.OBJECTID.astype(int).unique().tolist()}

    boundary = gpd.read_parquet(ADMIN/"analysis_context/ecuador_mainland.parquet").to_crs(4326)
    countries = gpd.read_file(ADMIN/"cultural/ne_110m_admin_0_countries.shp", bbox=(-82, -6, -74, 3)).to_crs(4326)
    provinces = gpd.read_file(ADMIN/"cultural/ne_10m_admin_1_states_provinces_lines.shp")
    provinces = gpd.clip(provinces.loc[provinces.ADM0_A3.eq("ECU")].to_crs(4326), boundary)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "text.color": INK,
                         "axes.labelcolor": INK, "axes.linewidth": .5,
                         "xtick.major.width": .5, "ytick.major.width": .5,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "national-tda-classified-v3", "hatch.linewidth": .35})
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), facecolor="white")
    count_cmap = ListedColormap(COUNT_COLORS, name="count_classes")
    interval_cmap = ListedColormap(INTERVAL_COLORS, name="interval_classes")
    count_norm = BoundaryNorm(np.arange(-.5, 4), count_cmap.N)
    interval_norm = BoundaryNorm(np.arange(-.5, 6), interval_cmap.N)
    locator = local.to_crs(4326).geometry.union_all().centroid
    for index, (field, title, cmap, norm, label) in enumerate([
        ("count_class", "(a) Detected-event count", count_cmap, count_norm, "Median reports per stand"),
        ("interval_class", "(b) Recurrence interval", interval_cmap, interval_norm, "Median within-stand interval (months)")]):
        left = 9 + index*88
        ax = fig.add_axes([left/WIDTH, 129/HEIGHT, 75/WIDTH, 79/HEIGHT])
        draw_map(ax, cells, field, boundary, countries, provinces, cmap, norm, index)
        fig.text(left/WIDTH, 212/HEIGHT, title, fontsize=9, weight="bold", va="top")
        if index == 0:
            for text in list(ax.texts):
                if text.get_text() == "PERU":
                    text.remove()
            ax.scatter([locator.x], [locator.y], s=34, marker="s", facecolor="none", edgecolor="#B26C2E", linewidth=.85, zorder=9)
            ax.annotate("Detail (e–f)", xy=(locator.x, locator.y), xytext=(-77.45, -.70),
                        fontsize=6.7, color="#905325", ha="left", va="center",
                        arrowprops={"arrowstyle": "-", "linewidth": .6, "color": "#905325"}, zorder=10)
            inset = ax.inset_axes([.57, .065, .39, .235], zorder=12)
            inset.set_facecolor("#FFFFFF")
            compact = np.r_[counts.iloc[:5].values, counts.iloc[5:].sum()]
            inset.bar(range(6), compact/counts.sum()*100, width=.72,
                      color=[GREY]+[BLUE]*5, edgecolor="white", linewidth=.25)
            inset.set(ylim=(0, 36), xlim=(-.7, 5.7), ylabel="Stands (%)", xlabel="Parent reports")
            inset.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
            inset.set_yticks([0, 15, 30])
            inset.set_title("n = 3,364 stands", fontsize=6.2, loc="left", pad=3)
            inset.tick_params(labelsize=5.8, length=1.5, pad=1.5)
            inset.xaxis.label.set_size(5.8)
            inset.yaxis.label.set_size(5.8)
            inset.xaxis.labelpad = 1
            inset.yaxis.labelpad = 1
            inset.spines[["top", "right"]].set_visible(False)
            for spine in inset.spines.values():
                spine.set_linewidth(.4)
        cax = fig.add_axes([(left+8)/WIDTH, 121/HEIGHT, 59/WIDTH, 2/HEIGHT])
        labels = COUNT_LABELS if index == 0 else INTERVAL_LABELS
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          ticks=np.arange(len(labels)), spacing="uniform", drawedges=True)
        cb.set_ticklabels(labels)
        cb.dividers.set_color("white")
        cb.dividers.set_linewidth(.65)
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(length=0, pad=2, labelsize=6.0)
        cb.set_label(label, fontsize=7, labelpad=1.2)
    fig.legend(handles=[Patch(facecolor="#FFFFFF", edgecolor="#8E979E", linewidth=.5, label="(b) No estimable interval"),
                        Patch(facecolor="#FFFFFF", edgecolor="#80808A", hatch="///", linewidth=.5, label="(a, b) No evaluable stands")],
               loc="lower center", bbox_to_anchor=(.5, 109.8/HEIGHT), ncol=2, frameon=False,
               fontsize=6.3, handlelength=1.6, columnspacing=2)
    annual_ax = fig.add_axes([13/WIDTH, 78/HEIGHT, 69/WIDTH, 25/HEIGHT])
    interval_ax = fig.add_axes([105/WIDTH, 78/HEIGHT, 69/WIDTH, 25/HEIGHT])
    for ax, title in [(annual_ax, "(c) Annual reports"), (interval_ax, "(d) Adjacent-report intervals")]:
        ax.set_title(title, fontsize=8.5, weight="bold", loc="left", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#75808A")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E3E7E9", lw=.4)
        ax.tick_params(length=2.5, pad=2, colors=INK)
    annual_ax.bar(annual.index, annual.values, width=.74, color=BLUE, edgecolor="white", lw=.35)
    annual_ax.set(xlim=(2016.35, 2025.65), ylim=(0, 1290), ylabel="Parent reports", xlabel="Representative year")
    annual_ax.set_xticks(list(range(2017, 2026)), [str(y) for y in range(2017, 2026)])
    annual_ax.set_yticks([0, 500, 1000])
    annual_ax.text(.99, .97, "6,303 parents", transform=annual_ax.transAxes, ha="right", va="top", fontsize=6.8)
    interval_ax.axvspan(q1, q3, facecolor="#EAE4F0", zorder=0)
    interval_ax.bar(edges[:-1]+3, hist/len(intervals)*100, width=5.5, color=PURPLE, edgecolor="white", lw=.3)
    interval_ax.axvline(median, color="#493266", lw=1, ls=(0, (3, 2)))
    interval_ax.set(xlim=(0, 108), ylim=(0, 24), ylabel="Intervals (%)", xlabel="Interval (months)")
    interval_ax.set_xticks([0, 12, 24, 36, 48, 60, 72, 84, 96])
    interval_ax.text(.99, .97, "Median 34 months · IQR 23–43\nn = 3,372; 2,005 stands",
                     transform=interval_ax.transAxes, ha="right", va="top", fontsize=6.7, linespacing=1.4)
    fig.lines.append(Line2D([9/WIDTH, 175/WIDTH], [65/HEIGHT, 65/HEIGHT], transform=fig.transFigure,
                            color="#E0E5E7", linewidth=.6))
    fig.text(9/WIDTH, 62/HEIGHT, "(e) Local inventory polygons", fontsize=8.5, weight="bold", va="top")
    fig.text(80/WIDTH, 62/HEIGHT, "(f) Linked stand histories", fontsize=8.5, weight="bold", va="top")
    fig.text(9/WIDTH, 57.7/HEIGHT, "Cotopaxi · spatially selected neighbourhood", fontsize=6.4, va="top", color="#677780")
    fig.text(80/WIDTH, 57.7/HEIGHT, "Same polygons A–H · 20 parent reports", fontsize=6.4, va="top", color="#677780")
    detail = fig.add_axes([9/WIDTH, 17/HEIGHT, 59/WIDTH, 36/HEIGHT])
    draw_neighbourhood(detail, stands, local)
    histories = fig.add_axes([88/WIDTH, 17/HEIGHT, 87/WIDTH, 36/HEIGHT])
    draw_histories(histories, local, local_reports)
    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", lw=0, markersize=3.2, label="Representative month"),
        Line2D([], [], color="#79B2C1", lw=2.2, label="First–last response range"),
        Patch(facecolor="#F2F3F4", edgecolor="#9AA1A7", hatch="///", label="Unassessable")],
        loc="lower center", bbox_to_anchor=(.5, 1.5/HEIGHT), ncol=3, frameon=False,
        fontsize=6.5, handlelength=1.7, columnspacing=1.7)
    fig.canvas.draw()
    stem = OUT/"national_recurrence_classified"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600,
                metadata={"Title": "National recurrence and linked local stand histories", "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), dpi=600, metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-scale-to", "2600", "-singlefile", "-png",
                    str(stem.with_suffix(".pdf")), str(OUT/"preview")], check=True)
    metrics.to_csv(data_out/"hex_statistics.csv")
    cells[["hex_id", "median_count", "median_interval", "n_evaluable", "n_interval", "count_class", "interval_class"]].to_csv(data_out/"hex_display_classes.csv", index=False)
    annual.rename("n_parent_reports").rename_axis("year").to_csv(data_out/"annual_reports.csv")
    counts.rename("n_stands").rename_axis("report_count").to_csv(data_out/"full_count_distribution.csv")
    pairs.to_csv(data_out/"adjacent_intervals.csv", index=False)
    local.drop(columns="geometry").to_csv(data_out/"selected_stands.csv", index=False)
    local.to_parquet(data_out/"selected_geometries.parquet", index=False)
    local_reports.to_csv(data_out/"selected_parent_reports.csv", index=False)
    (OUT/"case_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert current == original
    pdf = PdfReader(stem.with_suffix(".pdf"))
    fonts = [str(v.get_object()["/BaseFont"]) for v in pdf.pages[0]["/Resources"]["/Font"].get_object().values()]
    assert all("Arial" in f for f in fonts)
    qa = {"dimensions_mm": [WIDTH, HEIGHT], "fonts": fonts, "stand_count": len(stands),
          "evaluable_count": len(eligible), "parent_report_count": len(reports), "interval_count": len(pairs),
          "local_parent_reports": len(local_reports), "local_unassessable": int((~local.model_evaluable).sum()),
          "manuscript_v1_and_v2_unchanged": True, "protected_sha256": original,
          "map_classes": {"count_labels": COUNT_LABELS, "count_colors": COUNT_COLORS,
                          "interval_labels": INTERVAL_LABELS, "interval_colors": INTERVAL_COLORS,
                          "count_class_hex_counts": cells.count_class.value_counts().sort_index().to_dict(),
                          "interval_class_hex_counts": cells.interval_class.value_counts().sort_index().to_dict()},
          "selection": selection, "inferential_status": "descriptive national census of frozen outputs; local illustration only",
          "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
              PRODUCT/"harvest_inventory_2017_2025.gpkg", PRODUCT/"event_reports.csv", PRODUCT/"adjacent_report_intervals.csv", OLD_GRID/"polygon_hex_assignments_10km.csv"]}}
    (OUT/"qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (OUT/"CAPTION_EN.md").write_text(
        "Detected disturbance recurrence and linked stand histories in the Ecuadorian balsa inventory, 2017–2025. "
        "(a) Median TDA parent-report count among evaluable stands in fixed 10-km-wide hexagons; "
        "the inset summarizes all 3,364 evaluable stands, grouping counts of five or more. "
        "(b) Median of within-stand median adjacent-report intervals, using stands with at least two parents. "
        "Each stand contributes to one hexagon through its archived interior representative point. "
        "In (a), grey denotes zero; the positive-count classes are (0,1], (1,2] and (2,3]. "
        "In (b), the interval classes are <12, 12–<24, 24–<36, 36–<48, 48–<60 and ≥60 months. "
        "White cells with grey outlines in (b) have no estimable interval; hatched cells in either map have no evaluable stands. "
        "Empty-inventory cells are omitted, and mixed cells use eligible stands only. "
        "(c) Annual counts of 6,303 parents, each assigned once by its representative month. "
        "(d) Six-month-bin distribution of 3,372 adjacent-parent intervals from 2,005 stands; "
        "the dashed line marks the median, and shading marks the IQR rather than a confidence interval. "
        "(e) Eight polygons nearest the centre of the fixed hexagon with the most inventory polygons; "
        "selection used spatial information only. Labels A–H follow north-to-south order, then west-to-east order. "
        "Unselected nearby polygons provide grey spatial context. These polygons share a source inventory OBJECTID "
        "and are not asserted to represent independent management entities. "
        "(f) All 20 retained parents from the same eight polygons. Points mark representative months and horizontal lines "
        "span first to last response months; ranges are not verified harvest-operation durations. "
        "Numbers above the first two recurrent rows give consecutive representative-month intervals. "
        "G has no native SAR pixel support and remains unassessable, not a zero-report stand. "
        "Unmarked months do not establish stable canopy. The local panels illustrate model histories and are not an independent accuracy test. "
        "The input archive ends in June 2026; STP children do not add events to the parent histories.\n", encoding="utf-8")
    print(json.dumps({"pdf": str(stem.with_suffix('.pdf')), "parents": len(reports), "local_parents": len(local_reports),
                      "local_labels": local[["map_label", "source_row_original", "octa_count"]].to_dict('records'),
                      "protected_unchanged": True}, indent=2))


def draw_map(ax, cells, field, boundary, countries, provinces, cmap, norm, index):
    ax.set_facecolor("#F1F6F8")
    countries.plot(ax=ax, color="#F3F2EF", edgecolor="#D6D7D4", linewidth=.35, zorder=1)
    boundary.plot(ax=ax, color="#FCFCF9", edgecolor="none", zorder=2)
    provinces.plot(ax=ax, color="#D5D7D3", linewidth=.25, zorder=3)
    # Missingness is categorical: never fill unassessable values with zero.
    missing = cells.loc[cells.n_evaluable.eq(0)]
    if not missing.empty:
        missing.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#80808A", linewidth=.25, hatch="///", zorder=6)
    empty = cells.loc[cells.n_evaluable.gt(0) & (cells[field].isna() | (cells[field].eq(0) if index == 0 else False))]
    if not empty.empty:
        empty.plot(ax=ax, facecolor=GREY if index == 0 else "#FFFFFF", edgecolor="#FFFFFF" if index == 0 else "#8E979E", linewidth=.07 if index == 0 else .22, zorder=4)
    values = cells.loc[cells[field].notna() & (cells[field].gt(0) if index == 0 else True)]
    values.plot(ax=ax, column=field, cmap=cmap, norm=norm, edgecolor="#FFFFFF", linewidth=.07, zorder=5)
    boundary.boundary.plot(ax=ax, color="#5D686D", linewidth=.6, zorder=7)
    ax.set(xlim=(-81.35, -74.95), ylim=(-5.25, 1.70), aspect="equal")
    ax.set_xticks([-81, -79, -77, -75], ["81°W", "79°W", "77°W", "75°W"])
    ax.set_yticks([-4, -2, 0], ["4°S", "2°S", "0°"])
    ax.tick_params(length=2, width=.4, pad=2, labelsize=6.5, colors="#5E6570")
    for spine in ax.spines.values():
        spine.set_color("#8D999E")
        spine.set_linewidth(.4)
    ax.text(-80.8, .30, "PACIFIC\nOCEAN", fontsize=6.5, color="#84929A", style="italic", ha="center", linespacing=1.3)
    ax.text(-76.10, 1.15, "COLOMBIA", fontsize=6.5, color="#969A97", ha="center")
    ax.text(-76.20, -4.62, "PERU", fontsize=6.5, color="#969A97", ha="center")
    lon_width = 100/(111.32*np.cos(np.deg2rad(-5.04)))
    ax.plot([-80.8, -80.8+lon_width], [-5.04, -5.04], color=INK, lw=1.1, solid_capstyle="butt", zorder=8)
    ax.text(-80.8+lon_width/2, -4.91, "100 km", fontsize=6.2, ha="center", va="bottom")


def draw_neighbourhood(ax, stands, local):
    xmin, ymin, xmax, ymax = local.total_bounds
    cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
    width = (xmax-xmin)*1.21
    height = width*36/59
    extent = (cx-width/2, cy-height/2, cx+width/2, cy+height/2)
    context = stands.loc[stands.geometry.intersects(box(*extent)) & ~stands.sample_id.isin(local.sample_id)]
    context.plot(ax=ax, facecolor="#F0F2F3", edgecolor="#D9DFE2", linewidth=.45)
    local.loc[local.model_evaluable].plot(ax=ax, facecolor="#BEDBE3", edgecolor=BLUE, linewidth=.65)
    local.loc[~local.model_evaluable].plot(ax=ax, facecolor="#FFFFFF", edgecolor="#69767C", linewidth=.65, hatch="///")
    offsets = {"A": (0, 0), "B": (0, 0), "C": (0, 0), "D": (0, 0),
               "E": (0, 0), "F": (0, 0), "G": (-16, -12), "H": (0, 0)}
    for row in local.itertuples():
        point = row.geometry.representative_point()
        offset = offsets[row.map_label]
        ax.annotate(row.map_label, xy=(point.x, point.y), xytext=offset, textcoords="offset points",
                    fontsize=8, weight="bold", color=INK, ha="center", va="center",
                    path_effects=[effects.withStroke(linewidth=1.7, foreground="white")],
                    arrowprops={"arrowstyle": "-", "linewidth": .6, "color": INK} if offset != (0,0) else None)
    ax.set(xlim=(extent[0], extent[2]), ylim=(extent[1], extent[3]), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#9AA6AB")
        spine.set_linewidth(.45)
    sx, sy = extent[0]+width*.07, extent[1]+height*.08
    ax.plot([sx, sx+200], [sy, sy], color=INK, lw=1.2, solid_capstyle="butt")
    ax.text(sx+100, sy+height*.045, "200 m", ha="center", va="bottom", fontsize=6.5)
    ax.annotate("N", xy=(.07,.90), xytext=(.07,.74), xycoords="axes fraction", fontsize=6.5,
                ha="center", arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": .7})


def draw_histories(ax, local, reports):
    origin = pd.Period("2017-01", freq="M").ordinal
    annotated_rows = 0
    for i, row in enumerate(local.itertuples()):
        ax.axhspan(i-.46, i+.46, facecolor="#F6F8F9" if i % 2 == 0 else "#FFFFFF", zorder=0)
        if not row.model_evaluable:
            ax.add_patch(Rectangle((0, i-.42), 108, .84, facecolor="#F2F3F4", edgecolor="#CCD1D5", hatch="///", lw=0, zorder=1))
            ax.text(54, i, "No native-pixel support", ha="center", va="center", fontsize=6.3, color="#68757C", zorder=5)
            continue
        history = reports.loc[reports.sample_id.eq(row.sample_id)].sort_values(["representative_month", "prediction_id"])
        first = pd.PeriodIndex(history.first_response_month, freq="M").asi8-origin
        last = pd.PeriodIndex(history.last_response_month, freq="M").asi8-origin+1
        dates = pd.PeriodIndex(history.representative_month, freq="M").asi8-origin+.5
        ax.hlines(np.full(len(history), i), first, last, color="#79B2C1", linewidth=2.4, zorder=3)
        ax.scatter(dates, np.full(len(history), i), s=12, color=BLUE, edgecolor="#FFFFFF", linewidth=.35, zorder=4)
        if len(dates) >= 2 and annotated_rows < 2:
            for left, right in zip(dates[:-1], dates[1:]):
                ax.text((left+right)/2, i-.23, f"{int(right-left)} mo", fontsize=5.8, color="#697780", ha="center", va="bottom")
            annotated_rows += 1
        if history.empty:
            ax.text(54, i, "No retained parent reports", ha="center", va="center", fontsize=6.3, color="#68757C")
    ax.set(xlim=(0,108), ylim=(7.6,-.65), xlabel="Representative year")
    ax.set_yticks(range(8), [f"{r.map_label}  {int(r.source_row_original)}" for r in local.itertuples()])
    ax.tick_params(axis="y", length=0, pad=4, labelsize=6.3)
    ax.set_xticks([.5,24.5,48.5,72.5,96.5], ["2017", "2019", "2021", "2023", "2025"])
    ax.set_xticks(np.arange(0,109,12), minor=True)
    ax.tick_params(axis="x", length=2, pad=2, labelsize=6.5)
    ax.grid(axis="x", which="minor", color="#E7EBEE", linewidth=.4)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#89969D")
    ax.xaxis.label.set_size(7)


if __name__ == "__main__":
    main()
