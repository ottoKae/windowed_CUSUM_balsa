#!/usr/bin/env python3
"""Create figures and an English report for the N1 polarization sensitivity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "outputs/windowed_cusum_n1_polarization_interval_sensitivity_40tile_v1"
)
GREEN = "#667A3E"
GOLD = "#D6A62A"
DARK = "#252525"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(axis, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left")


def panel_label_inside(axis, label: str) -> None:
    """Place a parenthesized panel label inside the axes bounding box."""
    axis.text(
        0.015, 0.985, f"({label})", transform=axis.transAxes, fontsize=8,
        fontweight="bold", va="top", ha="left",
    )


def save_bundle(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.025,
                   pil_kwargs={"compression": "tiff_lzw"})


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = np.sort(numeric[np.isfinite(numeric)])
    return x, np.arange(1, len(x) + 1) / len(x)


def main_figure(root: Path) -> None:
    (root / "figures").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    tolerance = int(manifest["vh_tolerance_months"])
    primary = pd.read_parquet(root / "intervals/vv_primary_adjacent_intervals.parquet")
    supported = pd.read_parquet(root / "intervals/vv_adjacent_both_endpoints_vh_supported.parquet")
    summary = pd.read_csv(root / "metrics/interval_summary.csv").iloc[:2]
    bootstrap = pd.read_csv(root / "metrics/spatial_block_bootstrap_intervals.csv")
    contrasts = pd.read_csv(root / "metrics/polarization_sensitivity_contrasts.csv")
    support_all = pd.read_csv(root / "metrics/vh_support_strata_bootstrap.csv")
    interval_order = ["<12", "12--23", "24--47", ">=48"]
    support = support_all.loc[support_all.interval_bin.isin(interval_order)].copy()
    support["interval_bin"] = pd.Categorical(
        support.interval_bin, interval_order, ordered=True
    )
    support = support.sort_values("interval_bin")
    overall = support_all.loc[support_all.interval_bin.eq("overall")].iloc[0]

    source = pd.concat([
        primary[["interval_id", "interval_months"]].assign(cohort="Primary VV intervals"),
        supported[["interval_id", "interval_months"]].assign(cohort="VH-supported subset"),
    ], ignore_index=True)
    source.to_csv(root / "figures/interval_distribution_source_data.csv", index=False)

    figure, axes = plt.subplots(
        1, 3, figsize=(7.20, 2.45),
        gridspec_kw={"width_ratios": [1.10, 1.20, 1.10]},
    )

    # (a) Selection mechanism: endpoint support conditional on the VV interval.
    x_positions = np.arange(len(support))
    rates = support.support_rate.to_numpy(dtype=float) * 100
    lower = (support.support_rate - support.ci95_low).to_numpy(dtype=float) * 100
    upper = (support.ci95_high - support.support_rate).to_numpy(dtype=float) * 100
    bars = axes[0].bar(
        x_positions, rates, color=GOLD, edgecolor=DARK, linewidth=0.4,
        width=0.66, zorder=2,
    )
    axes[0].errorbar(
        x_positions, rates, yerr=np.vstack([lower, upper]), fmt="none",
        ecolor=DARK, elinewidth=0.8, capsize=2.0, capthick=0.7, zorder=3,
    )
    axes[0].axhline(
        float(overall.support_rate) * 100, color=GREY, lw=0.75,
        ls=(0, (3, 2)), zorder=1,
    )
    axes[0].text(
        0.02, float(overall.support_rate) * 100 + 1.0,
        f"Overall {float(overall.support_rate) * 100:.1f}%",
        transform=axes[0].get_yaxis_transform(), ha="left", va="bottom",
        fontsize=5.8, color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
    )
    annotation_top = rates + upper
    for bar, item, top in zip(bars, support.itertuples(), annotation_top):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, top + 1.4,
            f"{item.support_rate * 100:.1f}%\n({int(item.supported)}/{int(item.total)})",
            ha="center", va="bottom", fontsize=5.5,
        )
    y_max = min(100.0, max(50.0, np.ceil((annotation_top.max() + 9) / 10) * 10))
    axes[0].set(
        xlabel="Primary VV interval (months)",
        ylabel="VH-supported at both endpoints (%)",
        xticks=x_positions,
        xticklabels=["<12", "12--23", "24--47", "≥48"],
        ylim=(0, y_max),
    )
    axes[0].grid(axis="y", color="#ECECEC", lw=0.45, zorder=0)
    panel_label_inside(axes[0], "a")

    # (b) Distributional consequence of selecting the supported subset.
    cohorts = [
        (primary, "Primary VV intervals", GREEN),
        (supported, "VH-supported subset", GOLD),
    ]
    for frame, label, color in cohorts:
        x, y = ecdf(frame.interval_months)
        axes[1].step(x, y, where="post", color=color, lw=1.35,
                     label=label)
    axes[1].set(xlabel="Adjacent-event interval (months)", ylabel="Empirical cumulative probability",
                xlim=(0, 108), ylim=(0, 1.01))
    axes[1].set_xticks(np.arange(0, 109, 24))
    axes[1].grid(axis="both", color="#ECECEC", lw=0.45, zorder=0)
    legend = axes[1].legend(
        loc="lower right", ncol=1, frameon=True, handlelength=1.7,
        handletextpad=0.55, borderpad=0.30, labelspacing=0.22,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#B8B8B8")
    legend.get_frame().set_linewidth(0.5)
    panel_label_inside(axes[1], "b")

    # (c) Effect size: central scale and paired median shift.
    names = ["Primary VV", "VH-supported"]
    colors = [GREEN, GOLD]
    y_positions = [1, 0]
    cohort_keys = ["primary_vv", "both_endpoints_vh_supported"]
    for index, (row, key, color, y) in enumerate(zip(summary.itertuples(), cohort_keys, colors, y_positions)):
        ci = bootstrap.loc[
            bootstrap.cohort.eq(key) & bootstrap.metric.eq("median_months")
        ].iloc[0]
        axes[2].hlines(y, row.q25_months, row.q75_months, color=color, lw=5, alpha=0.35)
        axes[2].errorbar(
            row.median_months, y,
            xerr=np.array([[row.median_months - ci.ci95_low], [ci.ci95_high - row.median_months]]),
            fmt="o", ms=4.2, color=color, mec=DARK, mew=0.35,
            elinewidth=1.1, capsize=2.2, capthick=0.7,
        )
        axes[2].text(
            row.q75_months + 1.5, y,
            f"{row.median_months:.0f} mo\nIQR {row.q25_months:.0f}--{row.q75_months:.0f}",
            va="center", fontsize=5.8,
        )
    delta = contrasts.loc[contrasts.metric.eq("median_months")].iloc[0]
    point_delta = float(summary.iloc[1].median_months - summary.iloc[0].median_months)
    axes[2].set(xlabel="Interval (months)", yticks=y_positions, yticklabels=names,
                xlim=(0, 66), ylim=(-0.72, 1.65))
    # The two labels are categorical descriptors rather than a continuous
    # response axis. Rotate them vertically so panel (c) stays compact while
    # the interval scale remains unobstructed.
    for tick in axes[2].get_yticklabels():
        tick.set_rotation(90)
        tick.set_rotation_mode("anchor")
        tick.set_va("center")
        tick.set_ha("center")
    axes[2].set_xticks(np.arange(0, 61, 12))
    axes[2].grid(axis="x", color="#ECECEC", lw=0.45, zorder=0)
    axes[2].text(
        0.02, 0.03,
        f"Δ median = {point_delta:+.0f} months\n"
        f"95% CI: {delta.ci95_low:+.0f} to {delta.ci95_high:+.0f}",
        transform=axes[2].transAxes, fontsize=5.8, color="#555555",
    )
    panel_label_inside(axes[2], "c")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.90, wspace=0.22)
    # Compensate for the visual weight of the long outer y-axis labels: move
    # the middle panel a small, fixed amount to the right so the two inter-panel
    # gaps read as balanced at final print size.
    b_x, b_y, b_width, b_height = axes[1].get_position().bounds
    axes[1].set_position([b_x + 0.010, b_y, b_width, b_height])
    save_bundle(
        figure,
        root / "figures" / f"polarization_interval_sensitivity_vh_pm{tolerance}_v6_centered_b",
    )
    plt.close(figure)


def diagnostic_figure(root: Path) -> None:
    (root / "figures").mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(root / "diagnostics/vh_only_events.parquet").copy()
    events["year"] = pd.to_datetime(events.change_time).dt.year
    years = np.arange(2017, 2026)
    annual = events.year.value_counts().reindex(years, fill_value=0).rename("events").reset_index()
    annual.columns = ["year", "events"]
    top_tiles = events.primary_tile_id.value_counts().head(10).index.tolist()
    local = events.loc[events.primary_tile_id.isin(top_tiles)]
    heatmap = pd.crosstab(local.primary_tile_id, local.year).reindex(index=top_tiles, columns=years, fill_value=0)
    annual.to_csv(root / "figures/vh_only_annual_source_data.csv", index=False)
    heatmap.to_csv(root / "figures/vh_only_tile_year_source_data.csv")

    figure, axes = plt.subplots(1, 2, figsize=(7.20, 2.75), gridspec_kw={"width_ratios": [1.0, 1.7]})
    axes[0].bar(annual.year, annual.events, color=GOLD, edgecolor=DARK, linewidth=0.35)
    axes[0].plot(annual.year, annual.events, color=DARK, marker="o", ms=2.8, lw=0.75)
    axes[0].set(xlabel="Year", ylabel="VH-only events", xticks=years)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    axes[0].grid(axis="y", color="#ECECEC", lw=0.45, zorder=0)
    axes[0].spines[["top", "right"]].set_visible(False)
    panel_label(axes[0], "a")

    image = axes[1].imshow(heatmap.to_numpy(), aspect="auto", cmap="YlOrBr")
    axes[1].set(xlabel="Year", ylabel="MGRS tile", xticks=np.arange(len(years)),
                xticklabels=years, yticks=np.arange(len(top_tiles)), yticklabels=top_tiles)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = int(heatmap.iloc[row, column])
            if value:
                axes[1].text(column, row, str(value), ha="center", va="center",
                             fontsize=5.2, color="white" if value > heatmap.to_numpy().max() * 0.55 else DARK)
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.028, pad=0.02)
    colorbar.set_label("Events", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=5.8)
    panel_label(axes[1], "b")
    figure.subplots_adjust(left=0.075, right=0.96, bottom=0.23, top=0.91, wspace=0.30)
    save_bundle(figure, root / "figures/vh_only_year_tile_diagnostics")
    plt.close(figure)


def fmt_ci(row) -> str:
    return f"{row.ci95_low:.1f}--{row.ci95_high:.1f}"


def report(root: Path) -> None:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    tolerance = int(manifest["vh_tolerance_months"])
    summaries = pd.read_csv(root / "metrics/interval_summary.csv")
    bootstrap = pd.read_csv(root / "metrics/spatial_block_bootstrap_intervals.csv")
    contrasts = pd.read_csv(root / "metrics/polarization_sensitivity_contrasts.csv")
    references = pd.read_csv(root / "metrics/vh_only_reference_summary.csv")
    strata = pd.read_csv(root / "metrics/vh_support_strata.csv")
    primary = summaries.iloc[0]
    supported = summaries.iloc[1]
    primary_ci = bootstrap.loc[
        bootstrap.cohort.eq("primary_vv") & bootstrap.metric.eq("median_months")
    ].iloc[0]
    supported_ci = bootstrap.loc[
        bootstrap.cohort.eq("both_endpoints_vh_supported") & bootstrap.metric.eq("median_months")
    ].iloc[0]
    delta = contrasts.loc[contrasts.metric.eq("median_months")].iloc[0]
    point_delta = float(supported.median_months - primary.median_months)
    interval_support = strata.loc[strata.stratum_type.eq("interval_bin")]
    support_map = dict(zip(interval_support.stratum_value, interval_support.support_rate))
    ref2 = references.loc[references.tolerance_months.eq(2)].iloc[0]
    parity_note = (
        "The frozen +/-1-month VH annotation was independently reproduced exactly."
        if tolerance == 1
        else "The archived +/-1-month result was preserved; frozen-count parity was not expected for this prespecified alternative tolerance."
    )

    text = f"""# N1 polarization sensitivity of recurrent-disturbance intervals

Date: {manifest['created_date']}  
Status: complete  
Detector outputs: frozen and read-only  
Primary chronology: Polygon-first VV N1 Windowed CUSUM events, 2017--2025
VV--VH support tolerance: +/-{tolerance} calendar months

## Purpose

This experiment tested whether requiring VH confirmation materially changes the
distribution of intervals between adjacent qualified VV disturbance candidates.
It does not interpret the candidates as causally verified harvests or biological
rotation cycles.

## Frozen design and denominator correction

The file named `ground_polygon_vv_primary.parquet` contains two distinct routes:
6,658 Polygon-first VV events and 6,241 frozen Pixel-first VV events. The present
analysis correctly excluded the Pixel-first route. Mixing both routes would have
created a scientifically invalid 12,899-event chronology.

VH was used only to confirm VV endpoints within +/-{tolerance} calendar months. The complete
VV chronology was formed first. An interval entered the high-confidence subset
only when both of its original adjacent VV endpoints were VH-supported; unsupported
intermediate VV events were never removed before interval construction.

## Results

The primary chronology contained {manifest['counts']['polygon_first_vv_events']:,}
events in {manifest['counts']['primary_event_polygons']:,} polygons. A total of
{manifest['counts']['primary_interval_polygons']:,} polygons had at least two events,
producing {manifest['counts']['primary_intervals']:,} completed adjacent-event
intervals. Their median was {primary.median_months:.0f} months (IQR
{primary.q25_months:.0f}--{primary.q75_months:.0f}); the 95% complete-spatial-block
bootstrap interval for the median was {fmt_ci(primary_ci)} months. The polygon-weighted
median was {primary.polygon_weighted_median_months:.0f} months.

Among the primary VV events, {manifest['counts']['vv_with_vh_support']:,} were
one-to-one VH-supported. Requiring support at both original adjacent endpoints left
{manifest['counts']['supported_intervals']:,} intervals from
{manifest['counts']['supported_interval_polygons']:,} polygons. Their median was
{supported.median_months:.0f} months (IQR {supported.q25_months:.0f}--{supported.q75_months:.0f}),
with a 95% block-bootstrap interval of {fmt_ci(supported_ci)} months. The observed
median difference was {point_delta:+.0f} months (95% paired spatial-block bootstrap interval
{delta.ci95_low:.0f} to {delta.ci95_high:.0f}).

The shift is explained primarily by selective confirmation of short intervals.
Both endpoints were VH-supported for only {support_map['<12'] * 100:.1f}% of intervals
shorter than 12 months and {support_map['12--23'] * 100:.1f}% of 12--23-month
intervals, compared with {support_map['24--47'] * 100:.1f}% of 24--47-month intervals
and {support_map['>=48'] * 100:.1f}% of intervals at least 48 months long. Thus the
longer high-confidence median must be interpreted as VH-support selection, not as
evidence that BALSA biological rotations are longer.

The independent one-to-one matching produced {manifest['counts']['polarization_matches']:,}
VV--VH pairs and identified {manifest['counts']['vh_only_events']:,}
VH-only events in {int(summaries.iloc[2].polygons):,} polygons. At a +/-2-month
historical tolerance, VH-only detections matched {int(ref2.vh_only_reference_matches)}
of 1,093 references; {int(ref2.vh_only_reference_matches_not_matched_by_vv)} of these
references were not matched by the primary VV chronology. These cases motivate
future review, but they were not inserted post hoc into the primary event sequence.

## Conclusion

Dual-polarization confirmation at +/-{tolerance} months does not behave as a neutral
precision filter. It preferentially removes short VV-to-VV intervals and changes the
observed median interval by {point_delta:+.0f} months. The manuscript should therefore retain all qualified
Polygon-first VV events as the primary recurrent-disturbance chronology, report the
both-endpoints VH-supported subset as a sensitivity analysis, and keep VH-only events
for diagnosis or manual review. Neither result alone establishes disturbance cause.

## Reproducibility and QA

- Complete spatial blocks were resampled in 5,000 paired bootstrap replicates
  (`seed=20260830`).
- {parity_note}
- Supported intervals are an exact subset of the primary adjacency table.
- Intervals bridging an unsupported intermediate VV event: 0.
- A 120-event VH-only audit sample was fixed without using optical outcomes or
  historical reference matches.
- Source paths, sizes, and SHA-256 hashes are recorded in `source_hashes.json`.

## Outputs

- Main interval figure: `figures/polarization_interval_sensitivity_vh_pm{tolerance}.*`
- VH-only diagnostic figure: `figures/vh_only_year_tile_diagnostics.*`
- Primary and supported interval tables: `intervals/`
- VH-only diagnostics and fixed audit sample: `diagnostics/`
- Bootstrap replicates and paired contrasts: `bootstrap/`
- Manuscript-facing numerical tables: `metrics/`

## Interpretation boundary

Use `recurrent-disturbance interval`, `inter-disturbance interval`, or `candidate
recurrence interval`. Do not call these unverified national candidates harvest
intervals or biological rotation cycles. VH agreement raises polarization support;
it does not independently identify the disturbance cause.
"""
    atomic_text(text, root / "REPORT.md")


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    style()
    main_figure(root)
    diagnostic_figure(root)
    report(root)
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
