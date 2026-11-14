"""Five-panel draft from frozen TDA parents; never publish into the manuscript."""
from pathlib import Path
import hashlib
import json
import subprocess

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/figures/national_recurrence_tda_20261109_v1"
PRODUCT = ROOT / "runs/validation/harvest_inventory_20261104_v1/products"
OLD_GRID = ROOT / "runs/figure_sources/figure08_national_recurrence/source_data"
ADMIN = ROOT / "data/local/paper_v1/admin"
WIDTH, HEIGHT = 180, 154
BLUE, PURPLE = "#287B9B", "#79639B"
INK, GREY = "#243743", "#D2D5D8"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = OUT / "source_data"
    sources.mkdir(exist_ok=True)
    protected = [*sorted((ROOT / "manuscript_JAG").rglob("*.tex")),
                 ROOT / "manuscript_JAG/figures/fig07_national_harvest_patterns.pdf"]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    stands = pd.read_csv(PRODUCT / "stand_history.csv")
    # Reuse spatial assignments only; all historical detector attributes are discarded.
    assignment = pd.read_csv(OLD_GRID / "polygon_hex_assignments_10km.csv",
                             usecols=["polygon_id", "hex_id"])
    stands = stands.merge(assignment, left_on="sample_id", right_on="polygon_id",
                          how="left", validate="one_to_one")
    reports = pd.read_csv(PRODUCT / "event_reports.csv")
    reports = reports.loc[reports.method.eq("OCTA")].copy()
    pairs = pd.read_csv(PRODUCT / "adjacent_report_intervals.csv")
    pairs = pairs.loc[pairs.method.eq("OCTA") & pairs.report_level.eq("report")].copy()
    assert len(stands) == 3498 and stands.hex_id.notna().all()
    assert stands.model_evaluable.sum() == 3364 and len(reports) == 6303
    assert len(pairs) == 3372 and pairs.sample_id.nunique() == 2005
    assert stands.loc[~stands.model_evaluable, "octa_count"].isna().all()
    assert reports.prediction_id.is_unique
    assert reports.groupby("sample_id").size().sum() == int(stands.octa_count.sum())
    eligible = stands.loc[stands.model_evaluable]
    summary = stands.groupby("hex_id").agg(
        n_inventory=("sample_id", "size"), n_evaluable=("model_evaluable", "sum"),
        n_interval=("octa_interval_median_months", "count"),
        median_count=("octa_count", "median"),
        median_interval=("octa_interval_median_months", "median"))
    grid = gpd.read_parquet(OLD_GRID / "ecuador_mainland_hex_10km_width_epsg6933.parquet")
    cells = grid.merge(summary.reset_index(), on="hex_id", how="inner").to_crs(4326)
    annual = reports.groupby(pd.to_datetime(reports.representative_month).dt.year).size().reindex(range(2017, 2026), fill_value=0)
    counts = eligible.octa_count.astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    intervals = pairs.interval_months.to_numpy()
    bins = np.arange(0, 109, 6)
    histogram, _ = np.histogram(intervals, bins=bins)
    q1, median, q3 = np.quantile(intervals, [.25, .5, .75])
    assert annual.sum() == 6303 and counts.sum() == 3364 and histogram.sum() == 3372
    assert (q1, median, q3) == (23, 34, 43)

    boundary = gpd.read_parquet(ADMIN / "analysis_context/ecuador_mainland.parquet").to_crs(4326)
    countries = gpd.read_file(ADMIN / "cultural/ne_110m_admin_0_countries.shp", bbox=(-82, -6, -74, 3)).to_crs(4326)
    provinces = gpd.read_file(ADMIN / "cultural/ne_10m_admin_1_states_provinces_lines.shp")
    provinces = gpd.clip(provinces.loc[provinces.ADM0_A3.eq("ECU")].to_crs(4326), boundary)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.5,
                         "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.linewidth": .5, "xtick.major.width": .5, "ytick.major.width": .5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "national-tda-five-panel-v1", "hatch.linewidth": .35})
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4), facecolor="white")
    count_cmap = LinearSegmentedColormap.from_list("count_blue", [(0, GREY), (1/6, "#D6E9EE"), (1/3, "#6CAFC0"), (2/3, "#176381"), (1, "#113C59")])
    interval_cmap = LinearSegmentedColormap.from_list("interval_purple", ["#EFEAF5", "#B5A5D0", "#7D639E", "#493266"])
    count_norm = Normalize(0, np.ceil(cells.median_count.max()))
    interval_norm = Normalize(0, np.ceil(cells.median_interval.max()/12)*12)
    for index, (field, title, cmap, norm, label) in enumerate([
        ("median_count", "(a) Detected-event count", count_cmap, count_norm, "Median reports per stand"),
        ("median_interval", "(b) Recurrence interval", interval_cmap, interval_norm, "Median within-stand interval (months)")]):
        left = 9 + index*88
        ax = fig.add_axes([left/WIDTH, 67/HEIGHT, 75/WIDTH, 80/HEIGHT])
        draw_map(ax, cells, field, boundary, countries, provinces, cmap, norm, index)
        fig.text(left/WIDTH, 151/HEIGHT, title, fontsize=9, weight="bold", va="top")
        cax = fig.add_axes([(left+8)/WIDTH, 59/HEIGHT, 59/WIDTH, 2/HEIGHT])
        ticks = np.arange(0, norm.vmax+1, 1 if index == 0 else 24)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
        bar.outline.set_linewidth(.4)
        bar.ax.tick_params(length=2, pad=1, labelsize=6.8)
        bar.set_label(label, fontsize=7, labelpad=1.4)
    fig.text(.5, 50.6/HEIGHT,
             "Grey: median 0 in (a), no interval in (b)   |   Hatched: no evaluable stands",
             ha="center", va="center", fontsize=6.5, color="#5E6570")

    axes = [fig.add_axes([x/WIDTH, 17/HEIGHT, 44/WIDTH, 27/HEIGHT]) for x in [12, 73, 134]]
    titles = ["(c) Annual reports", "(d) Reports per stand", "(e) Adjacent-report intervals"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontsize=8, weight="bold", pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#75808A")
        ax.tick_params(length=2.5, pad=2, colors=INK)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#E3E7E9", lw=.4)
    ax = axes[0]
    ax.bar(annual.index, annual.values, width=.75, color=BLUE, edgecolor="white", linewidth=.35)
    ax.set(xlim=(2016.35, 2025.65), ylim=(0, 1320), ylabel="Parent reports", xlabel="Representative year")
    ax.set_xticks([2017, 2019, 2021, 2023, 2025], ["2017", "2019", "2021", "2023", "2025"], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([0, 500, 1000])
    ax.text(.97, .98, "6,303 reports", transform=ax.transAxes, ha="right", va="top", fontsize=6.6)
    ax = axes[1]
    ax.bar(counts.index, counts.values/counts.sum()*100, width=.75,
           color=[GREY if value == 0 else BLUE for value in counts.index], edgecolor="white", linewidth=.3)
    ax.set(xlim=(-.65, 9.65), ylim=(0, 39), ylabel="Evaluable stands (%)", xlabel="Number of parent reports")
    ax.set_xticks(range(10))
    ax.set_yticks([0, 10, 20, 30])
    ax.text(.98, .97, "n = 3,364", transform=ax.transAxes, ha="right", va="top", fontsize=6.6)
    ax = axes[2]
    ax.axvspan(q1, q3, facecolor="#EAE4F0", zorder=0)
    ax.bar(bins[:-1]+3, histogram/len(intervals)*100, width=5.5, color=PURPLE, edgecolor="white", linewidth=.3)
    ax.axvline(median, color="#493266", linewidth=1, linestyle=(0, (3, 2)))
    ax.set(xlim=(0, 108), ylim=(0, max(histogram/len(intervals)*100)*1.28), ylabel="Intervals (%)", xlabel="Interval (months)")
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.text(.98, .98, "Median 34 months\nIQR 23–43", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, linespacing=1.3)
    fig.text(.5, 1.8/HEIGHT,
             "2017–2025  |  3,498 inventory stands; 134 unassessable  |  3,372 intervals from 2,005 stands",
             ha="center", fontsize=6.5, color="#5E6570")
    fig.canvas.draw()
    stem = OUT / "national_recurrence_five_panel"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600,
                metadata={"Title": "Detected recurrence in the Ecuadorian balsa inventory", "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), dpi=600, metadata={"Date": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-scale-to", "2400",
                    "-singlefile", "-png", str(stem.with_suffix(".pdf")), str(OUT/"preview")], check=True)
    summary.to_csv(sources / "hex_statistics.csv")
    stands[["sample_id", "hex_id", "model_evaluable", "octa_count", "octa_interval_median_months"]].to_csv(sources/"stand_hex_values.csv", index=False)
    annual.rename("n_parent_reports").rename_axis("year").to_csv(sources/"annual_reports.csv")
    counts.rename("n_stands").rename_axis("report_count").to_csv(sources/"count_distribution.csv")
    pairs.to_csv(sources/"adjacent_intervals.csv", index=False)
    pd.DataFrame({"left_month": bins[:-1], "right_month": bins[1:], "n_intervals": histogram}).to_csv(sources/"interval_histogram.csv", index=False)
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before == after
    pdf = PdfReader(stem.with_suffix(".pdf"))
    fonts = [str(f.get_object()["/BaseFont"]) for f in pdf.pages[0]["/Resources"]["/Font"].get_object().values()]
    assert all("Arial" in f for f in fonts)
    qa = {"status": "draft_not_installed", "n_stands": len(stands), "n_evaluable": len(eligible),
          "zero_report_stands": int((eligible.octa_count == 0).sum()), "n_reports": len(reports),
          "n_intervals": len(intervals), "interval_q1_median_q3": [q1, median, q3],
          "n_hex_inventory": len(cells), "n_hex_evaluable": int((cells.n_evaluable > 0).sum()),
          "n_hex_interval": int((cells.n_interval > 0).sum()), "fonts": fonts,
          "dimensions_mm": [WIDTH, HEIGHT], "tex_and_manuscript_figure_unchanged": before == after,
          "protected_sha256": before, "interval_map_weight": "one within-stand median per stand",
          "bootstrap": "not displayed; descriptive counts and IQR only",
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [PRODUCT/"stand_history.csv", PRODUCT/"event_reports.csv", PRODUCT/"adjacent_report_intervals.csv", OLD_GRID/"polygon_hex_assignments_10km.csv"]}}
    (OUT/"qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (OUT/"CAPTION_EN.md").write_text(
        "Detected disturbance recurrence in the Ecuadorian balsa inventory, 2017–2025. "
        "(a) Median TDA parent-report count among evaluable stands within each fixed 10-km-wide hexagon, including zero-report stands. "
        "(b) Median of within-stand median adjacent-report intervals; only stands with at least two reports contribute. "
        "Each stand is assigned to one hexagon by its archived interior representative point. "
        "Grey cells indicate a median count of zero in (a), or no estimable interval among evaluable stands in (b); "
        "hatched cells contain inventory stands but none is evaluable. Cells without inventory stands are omitted. "
        "Mixed cells summarize eligible stands only; per-cell sample sizes accompany the source data. "
        "(c) Annual counts of 6,303 parents assigned once by their representative month. "
        "(d) Report counts among 3,364 evaluable stands; 134 unassessable stands are excluded from this denominator. "
        "(e) Distribution of 3,372 adjacent-report intervals from 2,005 stands in six-month bins. "
        "The dashed line marks the median (34 months), and shading indicates the IQR (23–43 months). "
        "The observation archive ends in June 2026; reporting is restricted to 2017–2025. "
        "STP children do not add events. These are model-detected recurrence intervals, not independently verified biological rotation lengths.\n",
        encoding="utf-8")
    print(json.dumps({k:v for k,v in qa.items() if not k.endswith("sha256")}, indent=2))
    print(stem.with_suffix(".pdf"))


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
        empty.plot(ax=ax, facecolor=GREY, edgecolor="white", linewidth=.12, zorder=4)
    values = cells.loc[cells[field].notna() & (cells[field].gt(0) if index == 0 else True)]
    values.plot(ax=ax, column=field, cmap=cmap, norm=norm, edgecolor="#FFFFFF", linewidth=.12, zorder=5)
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


if __name__ == "__main__":
    main()
