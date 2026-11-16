"""Map Canton TDA totals and the complete mainland interval hex grid."""
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle, Circle, PathPatch
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from pypdf import PdfReader, PdfWriter, Transformation

from build_national_recurrence_tda_classified import (
    ROOT, PRODUCT, OLD_GRID, ADMIN, INK, PURPLE,
)


OUT = ROOT / "runs/figures/harvest_context_20261116_v9_final"
FROZEN = ROOT / "runs/figures/national_recurrence_tda_20261112_v4_overview/source_data"
CONTEXT = ROOT / "runs/validation/harvest_context_20261112_v1"
DATA = ROOT / "data/local/paper_v1/context/administrative_climate_20261112"
WIDTH, HEIGHT = 180, 134
CLIMATE_COLORS = {"Af": "#A9CEB9", "Am": "#B1D2E5", "Aw": "#EFD59B"}
CANTON_COLORS = ["#D9A15C", "#C97877", "#B5AF64", "#7BA88B", "#A58CA5"]
OTHER_COLOR = "#8FAFCA"
CLIMATE_NAMES = {"Af": "Tropical rainforest", "Am": "Tropical monsoon", "Aw": "Tropical savannah"}
CLIMATE_TEXT_COLORS = {"Af": "#6CA68C", "Am": "#679CB9", "Aw": "#C7A467"}
BIN_MONTHS = 3
TOTAL_COLORS = ["#D3D6D8", "#FEE8A8", "#FEC44F", "#FE9929", "#E87519", "#B54E10", "#75300B"]
TOTAL_LABELS = ["0", "1–10", "11–50", "51–200", "201–500", "501–1,000", ">1,000"]


def main(out=OUT, number_cantons=False):
    out.mkdir(parents=True, exist_ok=True)
    sources = out / "source_data"
    sources.mkdir(exist_ok=True)
    protected = [*sorted((ROOT / "manuscript_JAG").rglob("*.tex")),
                 PRODUCT / "event_reports.csv"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    stands = pd.read_parquet(CONTEXT / "assignments/stand_context.parquet",
                            columns=["sample_id", "canton", "nom_can", "kg_class"])
    assert stands.sample_id.is_unique and len(stands) == 3498
    reports = pd.read_csv(PRODUCT / "event_reports.csv")
    reports = reports.loc[reports.method.eq("OCTA")].merge(stands, on="sample_id", validate="many_to_one")
    reports["year"] = pd.to_datetime(reports.representative_month).dt.year
    assert len(reports) == 6303 and reports.prediction_id.is_unique
    ranking = reports.groupby(["canton", "nom_can"]).size().rename("reports").reset_index()
    ranking = ranking.sort_values(["reports", "canton"], ascending=[False, True])
    top = ranking.head(5)
    assert top.canton.tolist() == ["2301", "1211", "0503", "0804", "1210"]
    names = {row.canton: row.nom_can.title() for row in top.itertuples()}
    reports["display_group"] = reports.canton.map(names).fillna("Other cantons")
    order = ["Other cantons"] + list(names.values())
    bar_colors = [OTHER_COLOR] + CANTON_COLORS
    canton_numbers = {name: i for i, name in enumerate(reversed(order[1:]), 1)}
    annual = reports.groupby(["year", "display_group"]).size().unstack(fill_value=0)
    annual = annual.reindex(index=range(2017, 2026), columns=order, fill_value=0)
    annual["National"] = annual.sum(axis=1)
    frozen_annual = pd.read_csv(FROZEN / "annual_reports.csv").set_index("year").n_parent_reports
    np.testing.assert_array_equal(annual.National, frozen_annual)
    assert annual.National.sum() == 6303 and annual[order[1:]].to_numpy().sum() == 4263
    reports["quarter"] = pd.PeriodIndex(reports.representative_month, freq="Q")
    quarters = pd.period_range("2017Q1", "2025Q4", freq="Q")
    quarterly = reports.groupby(["quarter", "display_group"]).size().unstack(fill_value=0)
    quarterly = quarterly.reindex(index=quarters, columns=order, fill_value=0)
    quarterly["National"] = quarterly.sum(axis=1)
    np.testing.assert_array_equal(quarterly.groupby(quarterly.index.year).sum(), annual)

    pairs = pd.read_csv(FROZEN / "adjacent_intervals.csv").merge(stands, on="sample_id", validate="many_to_one")
    assert len(pairs) == 3372 and pairs.sample_id.nunique() == 2005
    assert not pairs.duplicated(["sample_id", "left_prediction_id", "right_prediction_id"]).any()
    pairs = pairs.rename(columns={"interval_months": "representative_interval_months",
                                  "left_month": "left_representative_month", "right_month": "right_representative_month"})
    starts = reports.set_index("prediction_id").first_response_month
    pairs["left_first_month"] = pairs.left_prediction_id.map(starts)
    pairs["right_first_month"] = pairs.right_prediction_id.map(starts)
    pairs["interval_months"] = (pd.PeriodIndex(pairs.right_first_month, freq="M").asi8
                                - pd.PeriodIndex(pairs.left_first_month, freq="M").asi8)
    assert pairs.interval_months.between(1, 107).all()
    pairs["interval_bin"] = pairs.interval_months // BIN_MONTHS
    edges = np.arange(0, 109, BIN_MONTHS)
    distributions, quantiles = [], []
    for group in ["All", "Af", "Am", "Aw"]:
        sub = pairs if group == "All" else pairs.loc[pairs.kg_class.eq(group)]
        counts, _ = np.histogram(sub.interval_months, bins=edges)
        assert counts.sum() == len(sub)
        q1, median, q3 = np.quantile(sub.interval_months, [.25, .5, .75])
        quantiles.append({"group": group, "n_intervals": len(sub), "n_stands": sub.sample_id.nunique(),
                          "q1": q1, "median": median, "q3": q3,
                          "modal_bin_left_month": int(np.argmax(counts) * BIN_MONTHS),
                          "modal_bin_right_month": int((np.argmax(counts) + 1) * BIN_MONTHS)})
        distributions.extend({"group": group, "left_month": int(a), "right_month": int(b),
                              "n_intervals": int(n), "percent": float(n / len(sub) * 100)}
                             for a, b, n in zip(edges[:-1], edges[1:], counts))
    distribution = pd.DataFrame(distributions)
    summaries = pd.DataFrame(quantiles).set_index("group")
    assert summaries.n_intervals.tolist() == [3372, 150, 2861, 345]

    display = pd.read_csv(FROZEN / "hex_display_classes.csv")[["hex_id", "n_evaluable", "n_interval"]]
    inventory = gpd.read_file(PRODUCT / "harvest_inventory_2017_2025.gpkg", layer="stand_history",
                             columns=["sample_id", "model_evaluable", "octa_count"])
    assignments = pd.read_csv(OLD_GRID / "polygon_hex_assignments_10km.csv", usecols=["polygon_id", "hex_id"])
    inventory = inventory.merge(assignments, left_on="sample_id", right_on="polygon_id", validate="one_to_one")
    eligible = inventory.loc[inventory.model_evaluable].copy()
    assert len(eligible) == 3364
    observed_counts = reports.groupby("sample_id").size().reindex(eligible.sample_id, fill_value=0)
    np.testing.assert_array_equal(observed_counts, eligible.octa_count)
    pairs = pairs.merge(assignments, left_on="sample_id", right_on="polygon_id", validate="many_to_one")
    interval_summary = pairs.groupby("hex_id").agg(
        interval_median=("interval_months", "median"),
        interval_n_values=("interval_months", "size"),
        interval_n_contributing_stands=("sample_id", "nunique")).reset_index()
    assert len(interval_summary) == 171 and interval_summary.interval_n_values.sum() == 3372
    display = display.merge(interval_summary, on="hex_id", how="left", validate="one_to_one")
    medians = display.interval_median
    display["interval_class"] = np.select(
        [medians.lt(20), medians.lt(30), medians.lt(40), medians.le(50), medians.gt(50)],
        [0, 1, 2, 3, 4], default=np.nan)
    grid = gpd.read_parquet(OLD_GRID / "ecuador_mainland_hex_10km_width_epsg6933.parquet")
    boundary = gpd.read_parquet(ADMIN / "analysis_context/ecuador_mainland.parquet").to_crs(4326)
    cells = grid.merge(display, on="hex_id", how="left", validate="one_to_one").to_crs(4326)
    cells = gpd.clip(cells, boundary)
    assert (len(cells), int(cells.n_evaluable.gt(0).sum()), int(cells.n_interval.gt(0).sum())) == (3066, 286, 171)
    countries = gpd.read_file(ADMIN / "cultural/ne_110m_admin_0_countries.shp", bbox=(-82, -6, -74, 3)).to_crs(4326)
    provinces = gpd.read_file(ADMIN / "cultural/ne_10m_admin_1_states_provinces_lines.shp")
    provinces = gpd.clip(provinces.loc[provinces.ADM0_A3.eq("ECU")].to_crs(4326), boundary)
    cantons = gpd.read_parquet(DATA / "admin/inec_cantons_2020.parquet")
    cantons = cantons.loc[~cantons.canton.str.startswith("20")].to_crs(6933)
    cantons.geometry = cantons.geometry.simplify(150, preserve_topology=True)
    cantons = cantons.to_crs(4326)
    canton_support = stands.groupby("canton").size().rename("n_inventory")
    cantons = cantons.merge(canton_support, on="canton", how="left", validate="one_to_one")
    cantons = cantons.merge(ranking[["canton", "reports"]].rename(columns={"reports": "total_reports"}),
                            on="canton", how="left", validate="one_to_one")
    covered = cantons.n_inventory.notna()
    cantons.loc[covered, "total_reports"] = cantons.loc[covered, "total_reports"].fillna(0)
    cantons["report_class"] = pd.cut(cantons.total_reports, [-.5, .5, 10.5, 50.5, 200.5, 500.5, 1000.5, np.inf],
                                    labels=range(len(TOTAL_COLORS))).astype(float)
    assert (len(cantons), int(covered.sum()), int(cantons.total_reports.eq(0).sum())) == (219, 81, 9)
    assert cantons.total_reports.sum() == 6303 and cantons.total_reports.notna().equals(covered)
    with rasterio.open(DATA / "climate/kg_1991_2020_ecuador.tif") as src:
        climate = src.read(1)
        mask = geometry_mask(boundary.geometry, out_shape=climate.shape, transform=src.transform, invert=True)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    # Climate fills stay pale beneath the quantitative interval hexagons.
    climate_colors = ["#FFFFFF"] + [CLIMATE_COLORS[k] for k in ["Af", "Am", "Aw"]] + ["#E1E3E1"] * 27

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 6.5,
                         "axes.linewidth": .5, "axes.labelcolor": INK, "text.color": INK,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .35})
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
    top_tick_probe = fig.text(0, 0, "81°W", fontsize=6.5)
    fig.canvas.draw()
    top_tick_space = 4 * 25.4 / 72 + top_tick_probe.get_window_extent().height / fig.dpi * 25.4
    top_tick_probe.remove()
    map_height = 72 / 6.4 * 6.95 / np.cos(np.deg2rad(np.mean(boundary.total_bounds[[1, 3]])))
    map_y = HEIGHT - 2 * 25.4 / 72 - top_tick_space - (81 + map_height) / 2
    map_shift = map_y - 61
    panel_gap = ((61 + (81 - map_height) / 2 - 51) / 3 + 1) / 3
    quarter_y = map_y + (81 - map_height) / 2 - panel_gap - 40
    # Skip the pale yellow centre so filled hexagons remain distinct from climate backgrounds.
    interval_colors = plt.get_cmap("Spectral")([.77, .18, .02, .90, .99])
    interval_labels = ["<20", "20–<30", "30–<40", "40–50", ">50"]
    for i, (field, colors) in enumerate([("report_class", TOTAL_COLORS), ("interval_class", interval_colors)]):
        left, width = [(5, 72), (80, 96)][i]
        ax = fig.add_axes([left / WIDTH, map_y / HEIGHT, width / WIDTH, 81 / HEIGHT])
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-.5, len(colors)), len(colors))
        draw_background(ax, boundary, countries)
        ax.text(.025, .98, f"({'ab'[i]})", transform=ax.transAxes,
                fontsize=8.5, weight="bold", va="top", ha="left", zorder=40)
        if i == 0:
            provinces.plot(ax=ax, color="#D5D7D3", linewidth=.25, zorder=3)
            cantons.plot(ax=ax, color="white", edgecolor="#9DAAB2", lw=.22, zorder=10)
            cantons.loc[covered].plot(ax=ax, column=field, cmap=cmap, norm=norm,
                                     edgecolor="#8A9AA5", lw=.25, zorder=11)
            if number_cantons:
                label_positions = {"1210": (-80.2572, -.7757), "1211": (-78.69, -.76),
                                   "0503": (-79.0591, -1.4361), "0804": (-79.498, .3178),
                                   "2301": (-79.17, -.29)}
                for row in cantons.loc[cantons.canton.isin(names)].itertuples():
                    point = row.geometry.representative_point()
                    x, y = label_positions[row.canton]
                    ax.plot([point.x, x], [point.y, y], color=INK, lw=.55, zorder=69)
                    badge = DrawingArea(10, 10, 0, 0)
                    badge.add_artist(Circle((5, 5), 4.7, fc="white", ec=INK, lw=.45))
                    digit = TextPath((0, 0), str(canton_numbers[names[row.canton]]),
                                     size=7, prop=FontProperties(family="Arial", weight="bold"))
                    bounds = digit.get_extents()
                    # Centre the visible glyph, excluding font ascent/descent padding.
                    digit = Affine2D().translate(5 - (bounds.x0 + bounds.x1) / 2,
                                                 5 - (bounds.y0 + bounds.y1) / 2).transform_path(digit)
                    badge.add_artist(PathPatch(digit, fc=INK, ec="none"))
                    ax.add_artist(AnnotationBbox(badge, (x, y), frameon=False, pad=0,
                                                box_alignment=(.5, .5), zorder=70))
            map_a = ax
            legend_a_frame = [53, 62.6 + map_shift, 23.7, 27.2]
            fig.add_artist(Rectangle((53 / WIDTH, (62.6 + map_shift) / HEIGHT), 23.7 / WIDTH, 27.2 / HEIGHT,
                                    transform=fig.transFigure, facecolor="white", edgecolor="#9DAAB2", lw=.4, zorder=1))
            fig.text(54 / WIDTH, (89 + map_shift) / HEIGHT, "Stand harvest count\n(2017–2025)", fontsize=6.5, va="top")
            cax = fig.add_axes([54 / WIDTH, (68 + map_shift) / HEIGHT, 2.7 / WIDTH, 16 / HEIGHT], zorder=2)
            cb = matplotlib.colorbar.ColorbarBase(cax, cmap=ListedColormap(TOTAL_COLORS[1:]),
                    norm=BoundaryNorm(np.arange(7), 6), boundaries=np.arange(7),
                    ticks=np.arange(6) + .5, spacing="uniform", drawedges=True)
            cb.ax.set_yticklabels(TOTAL_LABELS[1:])
            cb.ax.minorticks_off()
            cb.ax.tick_params(length=0, pad=3, labelsize=6.5)
            cb.dividers.set(color="white", linewidth=.6)
            cb.outline.set_linewidth(.35)
            fig.legend(handles=[Patch(facecolor=TOTAL_COLORS[0], label="0 reports"),
                                Patch(facecolor="white", edgecolor="#9DAAB2", lw=.4, label="No inventory stands")],
                       loc="upper left", bbox_to_anchor=(54 / WIDTH, (67.2 + map_shift) / HEIGHT), frameon=False,
                       fontsize=6, handlelength=1.1, handletextpad=.4, labelspacing=.05,
                       borderpad=0, borderaxespad=0)
        else:
            ax.imshow(np.ma.masked_where(~mask | (climate == 0), climate), extent=extent,
                      cmap=ListedColormap(climate_colors), norm=BoundaryNorm(np.arange(-.5, 31.5), 31),
                      interpolation="nearest", alpha=.65, zorder=2.5)
            values = cells.loc[cells.interval_class.notna()]
            values.plot(ax=ax, column=field, cmap=cmap, norm=norm, edgecolor="none", zorder=20)
            cells.boundary.plot(ax=ax, color="white", lw=.18, zorder=21)
            map_b = ax
            # Extra eastern context makes room for the inset without changing map scale.
            ax.set_xlim(-81.35, -81.35 + 6.4 * 96 / 72)
            ax.yaxis.tick_right()
            ax.set_xticks([-81, -79, -77, -75, -73], ["81°W", "79°W", "77°W", "75°W", "73°W"])
            for label in list(ax.texts):
                if label.get_text() in {"COLOMBIA", "PERU", "PACIFIC\nOCEAN"}:
                    label.remove()
            climate_annotations = []
            climate_background = Rectangle(
                (122 / WIDTH, (130.9 + map_shift) / HEIGHT), 27.3 / WIDTH, 8.3 / HEIGHT,
                transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=3)
            fig.add_artist(climate_background)
            climate_annotations.append(climate_background)
            for row, (code, name) in enumerate(CLIMATE_NAMES.items()):
                legend_y = 137.8 + map_shift - row * 2.8
                symbol = Line2D([123.4 / WIDTH], [legend_y / HEIGHT],
                                transform=fig.transFigure, marker="h", markersize=5.5,
                                linestyle="none", markerfacecolor=CLIMATE_COLORS[code],
                                markeredgecolor="#85928E", markeredgewidth=.35, alpha=.65, zorder=4)
                fig.add_artist(symbol)
                climate_annotations.append(symbol)
                climate_annotations.append(fig.text(
                    125.4 / WIDTH, legend_y / HEIGHT,
                    f"{name} ({code})", fontsize=6, weight="bold", color=CLIMATE_TEXT_COLORS[code],
                    ha="left", va="center", zorder=4))
            legend_b_frame = [150, 105.5 + map_shift, 25, 33.8]
            fig.add_artist(Rectangle((150 / WIDTH, (105.5 + map_shift) / HEIGHT), 25 / WIDTH, 33.8 / HEIGHT,
                                    transform=fig.transFigure, facecolor="white", edgecolor="#9DAAB2", lw=.4, zorder=1))
            interval_title = fig.text(151 / WIDTH, (137.8 + map_shift) / HEIGHT,
                                      "Harvest-event interval\nMedian (months)",
                                      fontsize=6.2, va="top", ha="left")
            cax = fig.add_axes([151 / WIDTH, (110 + map_shift) / HEIGHT, 3 / WIDTH, 22 / HEIGHT], zorder=2)
            cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                    norm=BoundaryNorm(np.arange(6), 5), boundaries=np.arange(6),
                    ticks=np.arange(5) + .5, spacing="uniform", drawedges=True)
            cb.ax.set_yticklabels(interval_labels)
            cb.ax.minorticks_off()
            cb.dividers.set(color="white", linewidth=.6)
            cb.outline.set_linewidth(.35)
            cb.ax.tick_params(length=0, pad=3, labelsize=6.5)
            fig.canvas.draw()
            title_bottom = interval_title.get_window_extent().transformed(fig.transFigure.inverted()).y0
            position = cax.get_position()
            old_title_gap = title_bottom - position.y1
            cax.set_position([position.x0, position.y0 + old_title_gap / 2,
                              position.width, position.height])
            interval_title_gap_mm = old_title_gap * HEIGHT / 2
            fig.legend(handles=[Patch(facecolor="none", edgecolor="#8E979E", lw=.5, label="No interval")],
                       loc="upper left", bbox_to_anchor=(151 / WIDTH, (108.5 + map_shift) / HEIGHT),
                       fontsize=6, frameon=False, handlelength=1, handletextpad=.4,
                       borderpad=0, borderaxespad=0)
        boundary.boundary.plot(ax=ax, color="black", linewidth=.9, zorder=60)
        for tick in ax.get_yticklabels():
            tick.set_rotation(90)
            tick.set_horizontalalignment("center")
            tick.set_verticalalignment("center")
        ax.tick_params(axis="y", pad=4)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)

    ax = fig.add_axes([5 / WIDTH, quarter_y / HEIGHT, 171 / WIDTH, 40 / HEIGHT])
    ax.text(.99, .98, "(d)", transform=ax.transAxes, fontsize=8.5, weight="bold", va="top", ha="right", zorder=10)
    for year in range(2017, 2026, 2):
        ax.axvspan(year, year + 1, facecolor="#F0F1F2", edgecolor="none", zorder=0)
    for year in range(2018, 2026):
        ax.axvline(year, color="black", lw=.3, zorder=1)
    quarter_x = 2017 + (np.arange(len(quarterly)) + .5) / 4
    bottom = np.zeros(len(quarterly))
    for name, color in zip(order, bar_colors):
        ax.bar(quarter_x, quarterly[name], bottom=bottom, width=.195, color=color, edgecolor="white", lw=.18, label=name)
        bottom += quarterly[name]
    np.testing.assert_array_equal(bottom, quarterly.National)
    ax.set(xlim=(2017, 2026), ylim=(0, np.ceil(quarterly.National.max() / 50) * 50))
    # One enclosed calendar strip, with annual dividers shared by plot and labels.
    strip_height = 8 / 40
    for year in range(2017, 2026):
        ax.add_patch(Rectangle((year, -strip_height), 1, strip_height,
                              transform=ax.get_xaxis_transform(), clip_on=False,
                              facecolor="#F0F1F2" if year % 2 else "white",
                              edgecolor="black", lw=.3, zorder=3))
    for x, q in zip(quarter_x, quarters):
        ax.text(x, -2 / 40, f"Q{q.quarter}", transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=6.5)
    for year in range(2017, 2026):
        ax.text(year + .5, -5.8 / 40, str(year), transform=ax.get_xaxis_transform(),
                ha="center", va="center", fontsize=7)
    ax.set_xticks([])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.tick_params(labelsize=7, length=2, pad=2)
    for tick in ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_horizontalalignment("center")
        tick.set_verticalalignment("center")
    ax.tick_params(axis="y", pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.yaxis.grid(True, color="#E5E8EA", lw=.35); ax.set_axisbelow(True)
    legend_order = list(reversed(list(zip(order, bar_colors))))
    # Matplotlib fills columns; transpose handles to keep stack order when read by row.
    legend_labels = {name: f"{canton_numbers[name]} {name}" if number_cantons and name in canton_numbers else name
                     for name in order}
    legend_handles = [Patch(facecolor=legend_order[k][1], label=legend_labels[legend_order[k][0]])
                      for k in [0, 3, 1, 4, 2, 5]]
    legend = ax.legend(handles=legend_handles,
                       loc="upper left", bbox_to_anchor=(.012, .96), ncol=3,
                       frameon=True, facecolor="white", edgecolor="#667580", framealpha=1,
                       fancybox=False,
                       fontsize=6.5, handlelength=1.4, handletextpad=.6, labelspacing=.32,
                       borderaxespad=.25)
    legend.get_frame().set_linewidth(.45)
    inset_frame = [137, 63.4 + map_shift, 38, 30]
    fig.add_artist(Rectangle((137 / WIDTH, (63.4 + map_shift) / HEIGHT), 38 / WIDTH, 30 / HEIGHT,
                            transform=fig.transFigure, facecolor="white", edgecolor="#667580", lw=.45, zorder=2))
    fig.text(175 / WIDTH, (94.4 + map_shift) / HEIGHT, "(c) Intervals (months)",
             fontsize=7, weight="bold", va="bottom", ha="right", zorder=4)
    group_colors = CLIMATE_TEXT_COLORS
    for row, group in enumerate(["Af", "Am", "Aw"]):
        bottom = 84.4 + map_shift - row * 8.5
        ax = fig.add_axes([142 / WIDTH, bottom / HEIGHT, 31 / WIDTH, 7.5 / HEIGHT], zorder=3)
        values = distribution.loc[distribution.group.eq(group), "percent"].to_numpy()
        color = group_colors[group]
        ax.stairs(values, edges, baseline=0, fill=True, color=color, alpha=.30, lw=0)
        ax.stairs(values, edges, baseline=None, color=color, lw=.65)
        median = summaries.loc[group, "median"]
        modal_left = int(summaries.loc[group, "modal_bin_left_month"])
        ax.axvspan(modal_left, modal_left + 3, color=color, alpha=.12, lw=0, zorder=0)
        ax.axvline(median, color=color, lw=.65, ls=(0, (2, 1.3)))
        ax.text(.98, .13, group, transform=ax.transAxes, fontsize=7, weight="bold", color=color, ha="right", va="bottom")
        ax.text(.99, .98, f"Med {median:.0f}; mode {modal_left}–{modal_left + 2}",
                transform=ax.transAxes, fontsize=6, ha="right", va="top")
        ax.set(ylim=(0, 15), xlim=(0, 108))
        ax.set_yticks([0, 10], ["0", "10%"])
        ax.set_xticks([0, 24, 48, 72, 96])
        ax.tick_params(axis="y", length=1, pad=1, labelsize=6)
        ax.tick_params(axis="x", length=1.5 if row == 2 else 0, pad=1.5, labelsize=6, labelbottom=row == 2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.35)
        ax.grid(axis="y", color="#E3E7E9", lw=.25, zorder=0)

    fig.canvas.draw()
    for label in climate_annotations:
        text_box = label.get_window_extent()
        corners = map_b.transData.inverted().transform(text_box.get_points())
        assert not boundary.geometry.intersects(box(*corners.ravel())).any(), "Climate text must not cover mainland"
        assert text_box.transformed(fig.transFigure.inverted()).x1 * WIDTH < 150
    inset_data_corners = map_b.transData.inverted().transform(
        fig.transFigure.transform([[137 / WIDTH, (63.4 + map_shift) / HEIGHT], [175 / WIDTH, (93.4 + map_shift) / HEIGHT]]))
    inset_box = box(*inset_data_corners.ravel())
    assert not boundary.geometry.intersects(inset_box).any(), "Inset must not cover mainland Ecuador"
    map_rects = [list(a.get_position().bounds * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT]))
                 for a in [map_a, map_b]]
    bx, by, bw, bh = map_rects[1]
    dx, dy, dw, dh = inset_frame
    assert abs(legend_b_frame[0] + legend_b_frame[2] - (dx + dw)) < 1e-9
    assert bx <= dx and by <= dy and dx + dw <= bx + bw and dy + dh <= by + bh
    top_tick_pixels = max(label.get_window_extent().y1 for a in [map_a, map_b] for label in a.get_xticklabels())
    top_margin_pt = (fig.bbox.y1 - top_tick_pixels) / fig.dpi * 72
    assert abs(top_margin_pt - 2) < .02
    for map_axis, frame in [(map_a, legend_a_frame), (map_b, legend_b_frame)]:
        x, y, w, h = frame
        corners = map_axis.transData.inverted().transform(fig.transFigure.transform(
            [[x / WIDTH, y / HEIGHT], [(x + w) / WIDTH, (y + h) / HEIGHT]]))
        assert not boundary.geometry.intersects(box(*corners.ravel())).any(), "Legend frame must not cover mainland"
    minimum_font_pt = min(a.get_fontsize() for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text())
    assert minimum_font_pt >= 6

    stem = out / "national_harvest_context"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), dpi=600, metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(out / "preview.png", dpi=300)
    writer = PdfWriter()
    page = writer.add_blank_page(width=210 * 72 / 25.4, height=297 * 72 / 25.4)
    page.merge_transformed_page(PdfReader(stem.with_suffix(".pdf")).pages[0],
                                Transformation().translate(15 * 72 / 25.4, (297 - HEIGHT) / 2 * 72 / 25.4))
    writer.write(out / "A4_actual_size_proof.pdf")
    plt.close(fig)
    annual.to_csv(sources / "annual_canton_reports.csv")
    winners = pd.DataFrame({"canton": annual[order[1:]].idxmax(axis=1),
                            "reports": annual[order[1:]].max(axis=1)})
    winners.to_csv(sources / "annual_top5_leaders.csv")
    quarterly.rename_axis("quarter").to_csv(sources / "quarterly_canton_reports.csv")
    ranking.to_csv(sources / "canton_ranking.csv", index=False)
    pairs.to_csv(sources / "intervals_with_climate.csv", index=False)
    distribution.to_csv(sources / "interval_distributions.csv", index=False)
    summaries.to_csv(sources / "interval_group_summary.csv")
    cells.drop(columns="geometry").to_csv(sources / "hex_display_classes.csv", index=False)
    cantons.drop(columns="geometry").to_csv(sources / "canton_total_reports.csv", index=False)
    class_summary = cantons.loc[covered].groupby("report_class").agg(
        n_cantons=("canton", "size"), reports=("total_reports", "sum")).reindex(range(7))
    class_summary["label"] = TOTAL_LABELS
    class_summary["color"] = TOTAL_COLORS
    np.testing.assert_array_equal(class_summary.n_cantons, [9, 39, 16, 10, 3, 2, 2])
    class_summary.to_csv(sources / "canton_color_classes.csv")
    eligible.drop(columns="geometry").to_csv(sources / "evaluable_stand_counts.csv", index=False)
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)]
    qa = {"canton_numbers": canton_numbers if number_cantons else {}, "status": "passed", "dimensions_mm": [WIDTH, HEIGHT], "hex_inventory": 287,
          "map_a_geometry": "Canton polygons only, no hexagons",
          "canton_admin_objects": len(cantons), "cantons_with_inventory": int(covered.sum()),
          "cantons_positive_reports": int(cantons.total_reports.gt(0).sum()),
          "cantons_zero_reports": int(cantons.total_reports.eq(0).sum()),
          "cantons_without_inventory": int((~covered).sum()),
          "mainland_hexes_drawn": len(cells), "hexes_without_interval_fill": int(cells.n_interval.fillna(0).eq(0).sum()),
          "hex_display_clip": "Original fixed hexagons clipped for display; pooled adjacent-interval median per hex",
          "map_b_hex_edgecolor": "white", "map_b_provincial_boundaries": False,
          "geographic_zorder": {"countries": 1, "climate": 2.5, "provinces": 3, "country_border": 60,
                                "hex_fill": 20, "hex_edges": 21, "labels": 40, "legends": 50},
          "hex_evaluable": 286, "hex_interval": 171, "reports": len(reports), "intervals": len(pairs),
          "top5_canton_report_sum": int(top.reports.sum()), "other_canton_reports": int(annual['Other cantons'].sum()),
          "interval_groups": summaries.reset_index().to_dict('records'), "hex_support_filter_added": False,
          "country_border": {"color": "black", "width_pt": .9, "zorder": 60},
          "map_rects_mm": map_rects, "map_b_latitude_axis": "right",
          "map_latitude_label_rotation_degrees": 90,
          "map_a_legend_title": "Stand harvest count (2017–2025)",
          "map_b_interval_legend_position": "upper right",
          "map_b_mark_position": "inside upper left",
          "map_scale_start_longitude": -81.05,
          "peru_labels_longitude_latitude": [[-77.9, -4.0]],
          "panel_b_country_ocean_labels": False,
          "panel_b_climate_names_position": "Upper-right of mainland, left of interval legend",
          "panel_b_climate_legend_symbol": "Hexagon matching climate background colours",
          "panel_b_climate_legend_row_spacing_mm": 2.8,
          "map_longitude_axis_position": "top",
          "map_legend_frames_mm": [legend_a_frame, legend_b_frame],
          "interval_title_colorbar_gap_mm": interval_title_gap_mm,
          "legend_b_right_aligned_with_panel_c": True,
          "map_legend_frames_overlap_mainland": False,
          "canton_color_classes": class_summary.reset_index().to_dict("records"),
          "panel_d_rect_mm": [5, quarter_y, 171, 40], "panel_d_calendar_strip_mm": 8,
          "panel_d_legend_row_major": [n for n, _ in legend_order],
          "panel_d_legend_rows_columns": [2, 3], "panel_d_y_axes": 1,
          "panel_d_mark_position": "inside upper right", "panel_d_y_title": None,
          "panel_d_y_tick_rotation_degrees": 90, "panel_d_legend_border_pt": .45,
          "panel_c_frame_mm": inset_frame, "panel_c_groups": list(group_colors),
          "panel_c_overlaps_mainland": False, "panel_c_y_scale_percent": [0, 15],
          "panel_c_fully_inside_map_b": True,
          "top_margin_pt": top_margin_pt, "top_margin_reference": "highest longitude tick text",
          "map_to_quarter_gap_mm": panel_gap,
          "bottom_calendar_margin_mm": quarter_y - 8,
          "minimum_font_size_pt_at_180mm": minimum_font_pt,
          "panel_c_row_height_mm": 7.5,
          "panel_c_title_position": "outside frame upper right",
          "panel_c_climate_label_position": "inside each row lower right",
          "map_b_longitude_ticks": [-81, -79, -77, -75, -73],
          "a4_proof": "A4_actual_size_proof.pdf; native 180 mm figure width, no scaling",
          "panel_c_mode_definition": "Most frequent three-month bin, shown as inclusive integer months",
          "climate_legend_names": CLIMATE_NAMES,
          "interval_definition": "right.first_response_month minus left.first_response_month, in calendar months",
          "interval_values_changed": int(pairs.interval_months.ne(pairs.representative_interval_months).sum()),
          "hexes_with_one_interval": int(interval_summary.interval_n_values.eq(1).sum()),
          "hexes_with_fewer_than_five_intervals": int(interval_summary.interval_n_values.lt(5).sum()),
          "map_b_legend_title": "Harvest-event interval / Median (months)",
          "interval_display_classes": interval_labels,
          "interval_display_colormap": "Spectral sampled at 0.77, 0.18, 0.02, 0.90, 0.99; 30–<40 months red",
          "interval_display_value": "Median of all adjacent TDA first-detection-month differences within each hex; no rounding",
          "map_interval_weighting": "Each adjacent interval once; no stand-level pre-aggregation",
          "interval_bin_months": BIN_MONTHS,
          "annual_date_definition": "Representative year, unchanged",
          "quarterly_date_definition": "Quarter containing representative_month",
          "quarters": len(quarterly), "quarterly_annual_reconciliation": "Exact for all six display groups",
          "annual_line": False,
          "annual_total_values": annual.National.to_dict(),
          "interval_exact_1_month_count": int(pairs.interval_months.eq(1).sum()),
          "interval_exact_2_month_count": int(pairs.interval_months.eq(2).sum()),
          "interval_3_to_5_month_count": int(pairs.interval_months.between(3, 5).sum()),
          "stack_order_bottom_to_top": order, "canton_colors": dict(zip(order, bar_colors)),
          "protected_files_unchanged": True, "protected_sha256": before,
          "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "recomputed_detection_or_inference": False}
    (out / "qa_summary.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "CAPTION_EN.md").write_text(
        "Detected balsa harvests across mainland Ecuador, 2017–2025. "
        "(a) TDA event totals by canton. "
        "(b) Median intervals between consecutive TDA first-detection months in 10-km hexagons, over Köppen–Geiger climate classes. "
        "(c) Interval distributions for Af (rainforest), Am (monsoon), and Aw (savannah); dashed lines mark medians and shading marks modal bins. "
        "(d) Quarterly TDA event counts by canton, assigned by representative month. "
        "Each adjacent interval contributes once to the median in (b). "
        + ("Numbers 1–5 link cantons in (a) and (d).\n" if number_cantons else "\n"),
        encoding="utf-8")
    (out / "README_ZH.md").write_text(
        "# 全国 TDA 图：Canton 编号与离散间隔色带预览\n\n"
        "本版 PDF/SVG/PNG 由 `scripts/figures/build_national_harvest_context_cantons.py` 生成；"
        "沿用冻结检出与TDA产品，保留Canton编号，(b)改为各hex内全部相邻TDA起始月差的中位数。(c)的分布曲线、众数与中位数保持不变。正式图与TeX尚未替换。\n\n"
        "- (a) 去掉 hex，按 Canton 统计 2017—2025 年 TDA 父报告总数。81 个有库存 Canton 中，"
        "72 个正数、9 个有效零值；138 个无库存行政对象不填成零。合计 6,303 条报告。"
        "图例恢复为用户指定的 Stand harvest count (2017–2025)，其统计量为Canton内TDA父报告总数。"
        "(a)完整图例保持白底细边框；(b)采用竖向五档色带，标题与色带左对齐，No interval单列。两个背景框均通过不遮挡大陆的几何检查。"
        "正值中位数仅8条、最高1358条，因此拆分原1—50档，采用1—10、11—50、51—200、201—500、501—1000、>1000六档，"
        "对应39、16、10、3、2、2个Canton。浅黄—橙—深棕色带按等宽离散档显示，零值灰色、无库存白色另列。"
        "不等宽数值区间的色块不是连续线性比例尺；分级台账见 source_data/canton_color_classes.csv。\n"
        "- (b) 每个hex汇集各林分的相邻TDA起始月差，直接取中位数，每个间隔贡献一次。不先计算各林分中位数，也不取整；偶数样本允许半个月结果。\n"
        "按中位数分为<20、20–<30、30–<40、40–50、>50个月五档，50归入40–50，50.5归入>50。等宽色块表示离散类别。\n"
        "三个月分组刻画间隔长短，不等于旱雨季或日历季节；气候底图仍为长期类别。\n"
        "完整 3,066 个 hex 在绘图时裁至大陆；2,895 个无间隔单元保持透明，气候底图可见。"
        "hex 边线为白色，(b) 无省级边界；Af/Am/Aw 显示气候全名。(a)(b) 大陆国界黑色加粗，zorder=60，位于全部图层上方。\n"
        "(a)左边距5 mm、宽72 mm，两地图间距3 mm；(b)宽度96 mm，右缘176 mm，扩展东侧背景而不拉伸国土，两地图地理比例保持一致，纬度标记位于右侧。"
        "两图纬度刻度文字逆时针旋转90°、经度刻度移至顶部；(b)的Harvest-event interval (months)色标保留在图内右上方，气候全名及背景色对应移入caption。\n"
        "两幅比例尺保持左移0.25°经度（约2.8 mm），100 km长度不变；两处PERU均移至(-77.9°, -4.0°)，位于大陆东南侧、右下图例或嵌图左侧，颜色和字体与COLOMBIA一致。\n"
        "- (d) 通栏宽度171 mm，左右边缘与(a)(b)总宽对齐。36 个季度按代表月归属，Other cantons 位于柱底，五个 Canton 的颜色沿用上一版。"
        "季度逐年汇总与旧年度图在六个地区组上完全一致，总计 6,303。"
        "去掉年度折线与右轴，保留一个从零开始的左轴，去掉其标题，季度报告数含义写入caption。"
        "Other cantons浅蓝，奇数年浅灰底，年界黑线；(d)标记在图内右上角，纵轴数值逆时针旋转90°。图例保留图内左上角三列两行，下移1.2 mm并加0.45 pt细边框，按逐行阅读顺序对应堆叠柱从上到下。"
        f"绘图区高度40 mm，画布{WIDTH}×{HEIGHT} mm；顶部留白从最高经度刻度文字计算，为{top_margin_pt:.3f} pt；地图与季度图的轴框间距{panel_gap:.2f} mm。底部日历栏至画布留白为{quarter_y - 8:.2f} mm。(b)使用Spectral五档色带，并压缩图例底部空白。"
        "季度和年份在底部8 mm高的封闭日历栏中：Q1—Q4上行、年份下行，年度界线延续至底边，无额外季度小格。\n"
        "- (c) 间隔 = 后一次 TDA 的 first_response_month − 前一次 TDA 的 first_response_month。"
        "仍为 3,372 对、2,005 林分；2,128 个间隔值改变，全国中位数 34 月、IQR 22—43 月。"
        "该日期是最早接受检测月，不直接等于实际采伐作业开始。"
        "1、2 月间隔均为 0；3—5 月合计 208 对，占 6.17%。"
        "只画Af、Am、Aw三组，共同0—15%纵轴，分别标出中位数和三月众数组；全国数据仍保留在源CSV。"
        "(c)白底框38×30 mm，完全置于(b)右下角。整体标题和(c)置于白框外右上方，Af/Am/Aw在各行右下角，统计注释放行内右上方。"
        "曲线行高由4.6增至7.5 mm，相邻行间隙1 mm；几何相交与轴框包含检查确认不遮大陆且不超出(b)。"
        "独立逐报告来源核验与短间隔明细见 ../harvest_context_20261112_v8_review/interval_audit/。\n\n"
        "季度柱按代表月统计；次数和间隔是不同统计量，年度总量保持一致。"
        "完整格网的中位数、贡献间隔数及贡献林分数见 source_data/hex_display_classes.csv；"
        "Canton 总数和库存支持见 source_data/canton_total_reports.csv；"
        "逐间隔新旧值见 source_data/intervals_with_climate.csv。\n\n"
        "(d)保留季度与年份双层信息，统一纳入年度分组栏。"
        "柱高是每季度报告数，不是逐季累计总量；年度汇总仅保留在源CSV，不再绘制。\n\n"
        f"A4_actual_size_proof.pdf为210×297 mm页面，居中放置原生180 mm宽矢量图，没有缩放。实际最小字号{minimum_font_pt:.1f} pt；"
        "主要刻度6.5—7 pt，面板标记8.5 pt。建议论文按约180 mm双栏通栏宽度插图，进一步缩小会同比缩小字号；可按100%打印A4校样复核。"
        "(b)补齐81°W、79°W、77°W、75°W、73°W经度刻度。\n",
        encoding="utf-8")
    print(json.dumps({"pdf": str(stem.with_suffix('.pdf')), "reports": len(reports), "intervals": len(pairs)}, indent=2))


def draw_background(ax, boundary, countries):
    ax.set_facecolor("#F1F6F8")
    countries.plot(ax=ax, color="#F3F2EF", edgecolor="#D6D7D4", linewidth=.35, zorder=1)
    boundary.plot(ax=ax, color="#FCFCF9", edgecolor="none", zorder=2)
    ax.set(xlim=(-81.35, -74.95), ylim=(-5.25, 1.70), aspect="equal")
    ax.set_xticks([-81, -79, -77, -75], ["81°W", "79°W", "77°W", "75°W"])
    ax.set_yticks([-4, -2, 0], ["4°S", "2°S", "0°"])
    ax.tick_params(length=2, width=.4, pad=2, labelsize=6.5, colors="#5E6570")
    for spine in ax.spines.values():
        spine.set_color("#8D999E")
        spine.set_linewidth(.4)
    ax.text(-80.8, .30, "PACIFIC\nOCEAN", fontsize=6.5, color="#84929A", style="italic", ha="center", linespacing=1.3, zorder=40)
    ax.text(-76.10, 1.15, "COLOMBIA", fontsize=6.5, color="#969A97", ha="center", zorder=40)
    ax.text(-77.9, -4.0, "PERU", fontsize=6.5, color="#969A97", ha="center", zorder=40)
    lon_width = 100 / (111.32 * np.cos(np.deg2rad(-5.04)))
    ax.plot([-81.05, -81.05 + lon_width], [-5.04, -5.04], color=INK, lw=1.1, solid_capstyle="butt", zorder=40)
    ax.text(-81.05 + lon_width / 2, -4.91, "100 km", fontsize=6.2, ha="center", va="bottom", zorder=40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--number-cantons", action="store_true")
    args = parser.parse_args()
    main(out=args.output_dir, number_cantons=args.number_cantons)
