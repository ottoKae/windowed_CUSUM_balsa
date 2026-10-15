"""Render a deterministic 12-polygon reference atlas, without detector outputs.

Selection is deliberately diagnostic, not a representative accuracy sample.
Reuse only the established plot palette; read every selected polygon's tracks
and interior extraction pixels from the frozen cache. Never merge tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

from _common import DATA_ROOT, ROOT, GREEN, YELLOW, INK, configure_style


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def longest_gap(valid):
    best = current = 0
    for value in valid:
        current = 0 if value else current + 1
        best = max(best, current)
    return best


def select_polygons(root, references, tracks):
    """Use reference metadata and signal summaries only, never detection success."""
    support = pd.read_parquet(root / "polygon_pixels/monthly_support_by_polygon_track.parquet")
    support = support[support.sample_id.isin(references.sample_id)
                      & support.month.between("2017-01", "2025-12")]
    track_quality = []
    for (sid, key), group in support.groupby(["sample_id", "track_key"]):
        group = group.sort_values("month")
        assert len(group) == 108 and group.month.nunique() == 108
        track_quality.append((sid, key, 1 - group.any_valid.mean(), longest_gap(group.any_valid)))
    quality = pd.DataFrame(track_quality, columns=["sample_id", "track_key", "missing_fraction", "longest_gap"])
    candidate = references.groupby("sample_id").agg(
        reference_events=("event_id", "size"), source_row=("source_row_original", "first"),
        province=("DPA_DESPRO", "first"), tile=("primary_tile_id", "first"))
    candidate["track_count"] = tracks.groupby("sample_id").track_key.nunique()
    # Tile-track objects can repeat one orbit family across overlapping tiles.
    families = tracks.assign(orbit_family=tracks.track_key.str.extract(r"TK(\d+)", expand=False))
    candidate["orbit_family_count"] = families.groupby("sample_id").orbit_family.nunique()
    candidate["worst_track_missing_fraction"] = quality.groupby("sample_id").missing_fraction.max()
    candidate["longest_gap_months"] = quality.groupby("sample_id").longest_gap.max()
    summary = pd.read_csv(root / "figure_inputs/fig04_population_signatures/historical_event_summary.csv")
    summary = summary[summary.band.eq("VV") & summary.event_id.isin(references.event_id)].copy()
    summary["absolute_delta"] = summary.post_delta_db.abs()
    candidate["median_abs_reference_vv_delta_db"] = summary.groupby("sample_id").absolute_delta.median()
    candidate["qualified_vv_events"] = summary.groupby("sample_id").event_id.nunique()
    candidate["qualified_vv_events"] = candidate.qualified_vv_events.fillna(0).astype(int)
    candidate = candidate.reset_index()
    chosen, selected_ids = [], set()

    def take(pool, n, reason, order, ascending, diversify=False):
        # Deterministic rank with source-row tie-break; no stochastic selection.
        pool = pool.sort_values(order + ["source_row"], ascending=ascending + [True])
        local_tiles = set()
        for _ in range(n):
            remaining = pool[~pool.sample_id.isin(selected_ids)]
            if diversify and (~remaining.tile.isin(local_tiles)).any():
                remaining = remaining[~remaining.tile.isin(local_tiles)]
            if remaining.empty:
                raise ValueError(f"Insufficient candidates for {reason}")
            row = remaining.iloc[0].to_dict()
            row["selection_reason"] = reason
            chosen.append(row); selected_ids.add(row["sample_id"]); local_tiles.add(row["tile"])

    take(candidate, 2, "More missing months", ["worst_track_missing_fraction", "longest_gap_months"], [False, False])
    take(candidate[candidate.qualified_vv_events.ge(2)], 2, "Weak VV response",
         ["median_abs_reference_vv_delta_db"], [True])
    take(candidate[candidate.track_count.gt(1)], 3, "Multiple tile-tracks",
         ["worst_track_missing_fraction"], [True], True)
    take(candidate[candidate.reference_events.ge(3)], 2, "Recurrent references",
         ["reference_events", "worst_track_missing_fraction"], [False, True], True)
    take(candidate[candidate.track_count.eq(1)], 3, "Single track",
         ["worst_track_missing_fraction"], [True], True)
    selected = pd.DataFrame(chosen).sort_values(["tile", "source_row"]).reset_index(drop=True)
    selected["atlas_id"] = [f"P{i + 1:02d}" for i in range(len(selected))]
    selected["track_relationship"] = np.where(selected.orbit_family_count.gt(1), "Distinct orbit families",
        np.where(selected.track_count.gt(1), "Same family, overlapping tiles", "Single tile-track"))
    assert len(selected) == selected.sample_id.nunique() == 12
    assert selected.orbit_family_count.gt(1).any(), "Pilot must include a genuine cross-family example"
    return candidate, selected


def extract(root, selected, tracks):
    associations = pd.read_parquet(root / "polygon_pixels/pixel_associations.parquet")
    subset = tracks[tracks.sample_id.isin(selected.sample_id)].copy()
    rows, sources, checks = [], {}, []
    for key, track_rows in subset.groupby("track_key", sort=True):
        path = root / f"polygon_pixels/cache/{key}.npz"
        index_path = path.with_name(f"{key}.pixels.parquet")
        sources[str(path.relative_to(root))] = digest(path)
        sources[str(index_path.relative_to(root))] = digest(index_path)
        mapping = pd.read_parquet(index_path).set_index("pixel_track_id").cache_index
        with np.load(path, allow_pickle=False) as cache:
            values = cache["values"]
            valid = cache["valid"].astype(bool) & (cache["n_obs"] > 0)
            calendar = pd.DatetimeIndex(cache["calendar"])
            present = cache["source_month_present"].astype(bool)
            assert calendar.equals(pd.date_range("2016-01-01", "2026-06-01", freq="MS"))
            for item in track_rows.itertuples():
                association = associations[(associations.sample_id == item.sample_id)
                                           & (associations.track_key == key)]
                assert association.pixel_track_id.is_unique and len(association) == item.interior_pixels
                indices = mapping.loc[association.pixel_track_id].to_numpy(int)
                pixels = values[indices].copy()
                assert pixels.shape == (len(indices), 2, 126)
                invalid = ~valid[indices] | ~present[None, :]
                finite_invalid = int((np.isfinite(pixels) & invalid[:, None, :]).sum())
                pixels[np.broadcast_to(invalid[:, None, :], pixels.shape)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    quantiles = np.nanquantile(pixels, [0.25, 0.5, 0.75], axis=0)
                checks.append({"sample_id": item.sample_id, "track_key": key,
                               "extraction_pixels": len(indices), "masked_finite_invalid_values": finite_invalid,
                               "support_mode": item.support_mode})
                for band_index, band in enumerate(["VV", "VH"]):
                    q25, median, q75 = quantiles[:, band_index]
                    counts = np.isfinite(pixels[:, band_index]).sum(axis=0)
                    assert np.array_equal(np.isnan(median), counts == 0)
                    assert np.all(q25[counts > 0] <= median[counts > 0])
                    assert np.all(median[counts > 0] <= q75[counts > 0])
                    for i, month in enumerate(calendar):
                        rows.append({"sample_id": item.sample_id, "track_key": key, "tile": item.tile_id,
                                     "band": band, "month": month, "q25_db": q25[i], "median_db": median[i],
                                     "q75_db": q75[i], "valid_pixels": int(counts[i]),
                                     "extraction_pixels": len(indices), "source_month_present": bool(present[i]),
                                     "support_mode": item.support_mode})
        print(f"Extracted {key}: {len(track_rows)} polygon-track units", flush=True)
    return pd.DataFrame(rows), sources, checks


def export_page(fig, pdf, output, number):
    fig.canvas.draw()
    # Audit canvas bounds. Annotation boxes are also inspected in the previews.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width + 1 or box.y1 > bounds.height + 1):
            raise ValueError(f"Text outside page: {text.get_text()}")
    pdf.savefig(fig)
    fig.savefig(output / f"page_{number:02d}.png", dpi=300)
    fig.savefig(output / f"page_{number:02d}.svg")
    plt.close(fig)


def render(output, selected, references, series):
    configure_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    units = series[["sample_id", "track_key"]].drop_duplicates().merge(
        selected[["sample_id", "atlas_id", "source_row", "selection_reason"]], on="sample_id", validate="many_to_one")
    units = units.sort_values(["atlas_id", "track_key"]).reset_index(drop=True)
    # Keep all views of one polygon on the same page; at most three units/page.
    page, occupied = 2, 0
    for _, group in units.groupby("atlas_id", sort=False):
        if len(group) > 3:
            raise ValueError("Pilot layout supports at most three tile-track units per polygon")
        if occupied + len(group) > 3:
            page, occupied = page + 1, 0
        units.loc[group.index, "pdf_page"] = page
        units.loc[group.index, "slot"] = np.arange(occupied + 1, occupied + len(group) + 1)
        occupied += len(group)
    units[["pdf_page", "slot"]] = units[["pdf_page", "slot"]].astype(int)
    selected = selected.copy()
    selected["pdf_pages"] = selected.sample_id.map(units.groupby("sample_id").pdf_page.agg(lambda x: ",".join(map(str, sorted(set(x))))))
    selected.to_csv(output / "selected_polygons.csv", index=False)
    units.to_csv(output / "page_index.csv", index=False)
    limits = {}
    for band, local in series.groupby("band"):
        lo = float(np.nanmin(local.q25_db)); hi = float(np.nanmax(local.q75_db))
        limits[band] = (math.floor((lo - .5) / 2) * 2, math.ceil((hi + .5) / 2) * 2)
    legend = [Line2D([], [], color=GREEN, marker="o", markersize=3, lw=1, label="VV median"),
              Line2D([], [], color=YELLOW, marker="o", markersize=3, lw=1, label="VH median"),
              Patch(facecolor="#999999", alpha=.20, label="Pixel P25-P75"),
              Line2D([], [], color=INK, ls="--", lw=.8, label="Dated reference")]
    page_count = int(units.pdf_page.max())
    with PdfPages(output / "reference_atlas_12_polygons.pdf", metadata={
        "Title": "BALSA dated-reference atlas: 12-polygon layout pilot",
        "Subject": "Track-specific monthly VV/VH observations; no detector outputs"}) as pdf:
        fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
        fig.text(.08, .945, "Dated-reference time-series atlas", fontsize=16, weight="bold")
        fig.text(.08, .914, f"Layout pilot | 12 polygons | {len(units)} polygon-track units | "
                 f"{len(references)} unique reference events", fontsize=10)
        fig.text(.08, .878, "Purpose", fontsize=11, weight="bold")
        fig.text(.08, .852, "Inspect how dated stand disturbances relate to monthly SAR observations.\n"
                 "This purposive diagnostic sample is not an accuracy estimate. No detections\n"
                 "or algorithm windows are displayed; all selected polygons' tracks are retained.", va="top", linespacing=1.5)
        fig.text(.08, .775, "Reading the pages", fontsize=11, weight="bold")
        fig.text(.08, .750, "Each unit is one tile-track: VV above VH, on a common 2016-2026 axis.\n"
                 "Points and lines show spatial medians; shading shows pixel P25-P75, not a CI.\n"
                 "Missing months remain gaps. Pale-grey bands show absent source months.\n"
                 "Black dashed lines mark every dated reference for the polygon. Reference\n"
                 "dates are not adjusted to the SAR minimum; some original dates used SAR\n"
                 "as supplementary interpretation evidence. No optical validation is implied.", va="top", linespacing=1.5)
        fig.text(.08, .625, "Polygon index", fontsize=11, weight="bold")
        ax = fig.add_axes([.08, .30, .85, .30]); ax.axis("off")
        body = [[r.atlas_id, f"{int(r.source_row):06d}", r.tile, str(r.reference_events),
                 str(r.track_count), r.selection_reason, r.pdf_pages] for r in selected.itertuples()]
        tab = ax.table(cellText=body, colLabels=["ID", "Source row", "Tile", "Events", "Units", "Selection", "Pages"],
                       colWidths=[.07, .14, .11, .09, .09, .28, .12], bbox=[0, 0, 1, 1], cellLoc="left")
        tab.auto_set_font_size(False); tab.set_fontsize(8)
        for (i, j), cell in tab.get_celld().items():
            cell.set_linewidth(.35); cell.set_edgecolor("#BBBBBB")
            if i == 0: cell.set_facecolor("#EDF1E6"); cell.set_text_props(weight="bold")
        fig.text(.08, .263, "Selection and identity", fontsize=11, weight="bold")
        fig.text(.08, .238, "The 462-polygon reference cohort contains 1,093 events and 474 trajectories.\n"
                 "Selection covers 3 single-unit, 3 multi-unit, 2 recurrent, 2 weak-response\n"
                 "and 2 higher-missingness cases. Categories can overlap. Weak response is\n"
                 "ranked by median absolute reference-aligned VV change; missingness uses\n"
                 "the worst track over 2017-2025. Both are diagnostic ranks, not new labels.\n"
                 "Source row is the immutable official-inventory row identifier. Full polygon\n"
                 "IDs, selection measures and event identities are supplied in the CSV index.\n"
                 "Multiple units need not be different orbits: P03/P09/P10 span overlapping\n"
                 "tiles in the TK18 family; P12 contrasts the TK18 and TK120 families.", va="top", linespacing=1.5)
        fig.text(.08, .045, "Frozen observations only | No smoothing, interpolation or cross-track averaging", fontsize=8)
        fig.text(.93, .045, f"1 / {page_count}", ha="right", fontsize=8)
        export_page(fig, pdf, output, 1)
        for page, group in units.groupby("pdf_page"):
            fig = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
            fig.text(.08, .963, "BALSA reference trajectories", fontsize=13, weight="bold")
            fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.075, .947), ncol=4,
                       frameon=False, handlelength=2, columnspacing=1.7)
            for unit in group.itertuples():
                top = .89 - (unit.slot - 1) * .276
                local = series[(series.sample_id == unit.sample_id) & (series.track_key == unit.track_key)]
                dates = pd.DatetimeIndex(references.loc[references.sample_id.eq(unit.sample_id), "event_month"].sort_values())
                n = int(local.extraction_pixels.iloc[0])
                fig.text(.08, top, f"{unit.atlas_id} | Polygon row {int(unit.source_row):06d} | {unit.track_key}", weight="bold", fontsize=9)
                valid = local[(local.band == "VV") & local.month.between("2017-01-01", "2025-12-01")].valid_pixels.gt(0)
                fig.text(.08, top - .018, f"{n} extraction pixels | {len(dates)} references | "
                         f"VV observed {int(valid.sum())}/108 months (2017-2025) | {unit.selection_reason}", fontsize=7.5)
                fig.text(.08, top - .035, "References: " + "; ".join(dates.strftime("%Y-%m")), fontsize=7.5)
                for bi, (band, color) in enumerate([("VV", GREEN), ("VH", YELLOW)]):
                    ax = fig.add_axes([.11, top - .131 - bi * .104, .825, .084])
                    d = local[local.band.eq(band)].sort_values("month")
                    for month in d.loc[~d.source_month_present, "month"]:
                        ax.axvspan(month, month + pd.offsets.MonthBegin(1), color="#F0F0F0", lw=0, zorder=0)
                    ax.fill_between(d.month, d.q25_db, d.q75_db, color=color, alpha=.20, lw=0, zorder=2)
                    ax.plot(d.month, d.median_db, color=color, lw=1, marker="o", markersize=2.1,
                            markeredgecolor="white", markeredgewidth=.25, zorder=3)
                    for date in dates: ax.axvline(date, color=INK, ls="--", lw=.65, zorder=4)
                    ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2026-07-01")); ax.set_ylim(limits[band])
                    ax.set_ylabel(f"{band} (dB)", labelpad=4)
                    tick_step = 4 if limits[band][1] - limits[band][0] > 16 else 2
                    ax.set_yticks(np.arange(limits[band][0], limits[band][1] + 1, tick_step))
                    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                    ax.tick_params(axis="x", labelbottom=bi == 1, length=2.5, pad=2)
                    ax.tick_params(axis="y", length=2.5, pad=2)
                    ax.grid(color="#DADADA", lw=.35); ax.set_axisbelow(True)
                    ax.spines[["top", "right"]].set_visible(False)
            fig.text(.08, .037, "IQR = within-polygon pixel spread, not a confidence interval. Gaps are not interpolated.", fontsize=7.5)
            fig.text(.935, .037, f"{page} / {page_count}", ha="right", fontsize=8)
            export_page(fig, pdf, output, int(page))
    # Preview grid is derived from the actual exported pages; no data resampling.
    thumbs = []
    for path in sorted(output.glob("page_*.png")):
        with Image.open(path) as im:
            thumb = im.convert("RGB"); thumb.thumbnail((420, 594)); thumbs.append(thumb.copy())
    sheet = Image.new("RGB", (3 * 440, math.ceil(len(thumbs) / 3) * 620), "#DCDCDC")
    for i, thumb in enumerate(thumbs): sheet.paste(thumb, ((i % 3) * 440 + 10, (i // 3) * 620 + 10))
    sheet.save(output / "contact_sheet.png")
    return {"pages": page_count, "polygon_track_units": len(units), "axis_limits_db": limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/previews/reference_atlas_12polygons_20260911")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = ["references/reference_ground_events.parquet", "polygon_pixels/polygon_tracks.parquet",
                    "polygon_pixels/pixel_associations.parquet", "polygon_pixels/monthly_support_by_polygon_track.parquet",
                    "figure_inputs/fig04_population_signatures/historical_event_summary.csv"]
    hashes = {name: digest(args.data_root / name) for name in source_files}
    references = pd.read_parquet(args.data_root / source_files[0])
    tracks = pd.read_parquet(args.data_root / source_files[1])
    assert len(references) == 1093 and references.sample_id.nunique() == 462
    assert not references.duplicated(["sample_id", "event_month"]).any()
    candidate, selected = select_polygons(args.data_root, references, tracks)
    candidate.to_csv(args.output / "selection_candidates.csv", index=False)
    refs = references[references.sample_id.isin(selected.sample_id)].copy()
    refs.drop(columns=["geometry"], errors="ignore").to_csv(args.output / "selected_reference_events.csv", index=False)
    series, cache_hashes, checks = extract(args.data_root, selected, tracks)
    hashes.update(cache_hashes)
    series.to_parquet(args.output / "monthly_plot_data.parquet", index=False)
    result = render(args.output, selected, refs, series)
    assert len(series) == result["polygon_track_units"] * 2 * 126
    result.update({"passed": True, "polygons": 12, "unique_reference_events": len(refs),
                   "input_cohort_polygons": 462, "input_cohort_events": 1093,
                   "selection_seed": None, "selection": "deterministic diagnostic ranking; not probability sampling",
                   "detector_results_read": False, "all_selected_tracks_retained": True,
                   "inputs_sha256": hashes, "pixel_checks": checks,
                   "script_sha256": digest(__file__)})
    (args.output / "QA.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    outputs = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "COMPLETE.json").write_text(json.dumps({"passed": True, "outputs_sha256": outputs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["polygons", "unique_reference_events", "polygon_track_units", "pages", "axis_limits_db"]}, indent=2))


if __name__ == "__main__":
    main()
