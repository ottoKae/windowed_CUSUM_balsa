"""Render Fig. 3 v3 with the Bisan Gardenia palette and compact typography.

This is a visual-only successor to ``build_population_s1_signature_figure_v2``.
It reuses the frozen v1 signature products and the same event-aligned statistics;
only the palette, spacing, typography, and annotation placement change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.workflows import build_population_s1_signature_figure_v2 as base


PALETTE = {
    "VV": "#77964a",       # Bisan Gardenia green
    "VH": "#fbc040",       # Bisan Gardenia yellow
    "national": "#b8b8b8",
    "zero": "#333333",
    "text": "#333333",
}
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4
DEFAULT_SEED = 20260822


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def draw_violin(ax: plt.Axes, values: np.ndarray, position: float, colour: str,
                alpha: float, width: float = 0.62) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    violin = ax.violinplot(
        [finite], positions=[position], orientation="horizontal", widths=width,
        showmeans=False, showmedians=False, showextrema=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(colour)
    body.set_edgecolor("#4d4d4d")
    body.set_linewidth(0.35)
    body.set_alpha(alpha)
    q25, med, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
    ax.plot([q25, q75], [position, position], color="black", linewidth=1.1, zorder=4)
    ax.scatter([med], [position], s=11, color="white", edgecolor="black",
               linewidth=0.45, zorder=5)


def render_figure(
    national_values: np.ndarray,
    events: pd.DataFrame,
    table_summary: pd.DataFrame,
    curve: pd.DataFrame,
    seed: int,
) -> tuple[plt.Figure, dict[str, Any]]:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.4,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    fig.subplots_adjust(left=0.060, right=0.995, bottom=0.245, top=0.825, wspace=0.18)
    national = {
        band: national_values[:, i, :][np.isfinite(national_values[:, i, :])]
        for i, band in enumerate(base.BANDS)
    }

    # Panel A: absolute context.
    ax = axes[0]
    positions = [5, 4, 3, 1, 0, -1]
    labels = ["VV national", "VV pre", "VV event/+2",
              "VH national", "VH pre", "VH event/+2"]
    distributions = [
        national["VV"],
        events.loc[events.band.eq("VV"), "pre_db"].to_numpy(),
        events.loc[events.band.eq("VV"), "post_db"].to_numpy(),
        national["VH"],
        events.loc[events.band.eq("VH"), "pre_db"].to_numpy(),
        events.loc[events.band.eq("VH"), "post_db"].to_numpy(),
    ]
    colours = [PALETTE["national"], PALETTE["VV"], PALETTE["VV"],
               PALETTE["national"], PALETTE["VH"], PALETTE["VH"]]
    alphas = [0.36, 0.25, 0.78, 0.36, 0.25, 0.78]
    for values, pos, colour, alpha in zip(distributions, positions, colours, alphas):
        draw_violin(ax, values, pos, colour, alpha)
    ax.set_yticks(positions, labels)
    ax.set_xlim(-20, -3)
    ax.set_xticks([-20, -16, -12, -8, -4])
    ax.set_xlabel("Backscatter (dB)", labelpad=3)
    ax.text(0.0, 1.045, "(a) Absolute-state context", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=9.5)
    ax.axhline(2, color="#d0d0d0", linewidth=0.4)
    ax.grid(axis="x", color="#dedede", linewidth=0.45, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B: paired event response.
    ax = axes[1]
    for position, band in enumerate(base.BANDS, start=1):
        values = events.loc[events.band.eq(band), "post_delta_db"].to_numpy(dtype=float)
        violin = ax.violinplot([values], positions=[position], widths=0.66,
                               showmeans=False, showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(PALETTE[band])
        body.set_edgecolor(PALETTE[band])
        body.set_alpha(0.24)
        jitter = 0.16 * np.sin(np.arange(len(values), dtype=float) * 2.399963)
        ax.scatter(position + jitter, values,
                   s=2.2, color=PALETTE[band], alpha=0.20, linewidth=0,
                   rasterized=True)
        summary = table_summary.loc[table_summary.band.eq(band)].iloc[0]
        median = float(summary.paired_delta_median_db)
        low = float(summary.paired_delta_block_bootstrap_lower_db)
        high = float(summary.paired_delta_block_bootstrap_upper_db)
        ax.errorbar([position], [median], yerr=[[median - low], [high - median]],
                    fmt="D", color="black", markerfacecolor="white", markersize=3.6,
                    linewidth=1.0, capsize=2.2, zorder=5)
        proportion = float(summary.proportion_negative) * 100.0
        ax.text(position, 2.15, f"{proportion:.1f}% < 0", ha="center", va="top",
                fontsize=7.4, color=PALETTE["text"], fontweight="bold")
    ax.axhline(0, color=PALETTE["zero"], linestyle="--", linewidth=0.8)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(-7.2, 2.55)
    ax.set_xticks([1, 2], base.BANDS)
    ax.set_yticks([-6, -3, 0, 2])
    ax.set_ylabel("Paired Δ (dB)", labelpad=2)
    ax.text(0.0, 1.045, "(b) Paired event response", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=9.5)
    ax.grid(axis="y", color="#dedede", linewidth=0.45, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C: event-aligned response.
    ax = axes[2]
    for band in base.BANDS:
        local = curve.loc[curve.band.eq(band)].sort_values("relative_month")
        x = local.relative_month.to_numpy(dtype=float)
        colour = PALETTE[band]
        ax.fill_between(x, local["q25"], local["q75"], color=colour, alpha=0.18, linewidth=0)
        ax.fill_between(x, local["bootstrap_lower"], local["bootstrap_upper"],
                        color=colour, alpha=0.25, linewidth=0)
        ax.plot(x, local["median"], color=colour, linewidth=1.7, label=band)
    ax.axvline(0, color=PALETTE["zero"], linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#666666", linestyle=":", linewidth=0.7)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-3.5, 0.65)
    ax.set_xticks([-12, -6, 0, 6, 12])
    ax.set_yticks([-3, -2, -1, 0])
    ax.set_xlabel("Months relative to reference event", labelpad=3)
    ax.set_ylabel("Δ from pre-event (dB)", labelpad=2)
    ax.text(0.0, 1.045, "(c) Event-aligned trajectory", transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, ncol=2, handlelength=1.4,
              columnspacing=0.8, borderaxespad=0.1)
    ax.grid(color="#dedede", linewidth=0.45, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    diagnostics = {
        "figure_width_mm": FIGURE_WIDTH_MM,
        "figure_height_mm": FIGURE_HEIGHT_MM,
        "palette": "Bisan Gardenia",
        "palette_hex": {key: value for key, value in PALETTE.items()},
        "panels": {"a": "absolute context", "b": "paired event response",
                   "c": "event-aligned trajectory"},
        "panel_label_alignment": "axes left edge",
        "inter_panel_wspace": 0.18,
        "seed": int(seed),
    }
    return fig, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "national_metadata": input_dir / "national_balsa_series_metadata.parquet",
        "national_values": input_dir / "national_balsa_series.npz",
        "event_summary": input_dir / "historical_event_summary.csv",
        "event_aligned": input_dir / "historical_event_aligned.parquet",
        "table_summary": input_dir / "table_s1_signature_summary.csv",
        "source_manifest": input_dir / "signature_analysis_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    national_metadata = pd.read_parquet(paths["national_metadata"])
    with np.load(paths["national_values"]) as arrays:
        national_values = arrays["values"].copy()
    events_all = pd.read_csv(paths["event_summary"], parse_dates=["event_month"])
    aligned_all = pd.read_parquet(paths["event_aligned"])
    table_summary = pd.read_csv(paths["table_summary"])
    event_ids = base.strict_event_ids(events_all)
    events = events_all.loc[events_all.event_id.astype(str).isin(set(event_ids))].copy()
    aligned = base.filter_aligned(aligned_all, event_ids)
    metadata = base.event_metadata(events, event_ids)
    counts = {
        "national_polygons": int(national_metadata.sample_id.nunique()),
        "strict_events": int(len(event_ids)),
        "strict_reference_polygons": int(metadata.sample_id.nunique()),
        "strict_spatial_blocks": int(metadata.spatial_block_id.nunique()),
        "strict_tiles": int(metadata.tile_id.nunique()),
        "aligned_rows": int(len(aligned)),
    }
    for key, expected in base.EXPECTED.items():
        if counts[key] != expected:
            raise ValueError(f"Frozen count mismatch for {key}: {counts[key]} != {expected}")
    if national_values.shape[0] != counts["national_polygons"]:
        raise ValueError("National metadata and value arrays disagree on polygon count")
    monthly = base.monthly_summary(aligned)
    bootstrap = base.block_curve_bootstrap(aligned, metadata, args.bootstrap_replicates, args.seed)
    curve = monthly.merge(bootstrap, on=["band", "relative_month"], validate="one_to_one")
    fig, layout = render_figure(national_values, events, table_summary, curve, args.seed)
    stem = output_dir / "figure_03_population_s1_signatures_v3"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600,
                metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v3"})
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 3,
        "workflow": "population_s1_signatures_three_panel_visual_v3",
        "statistics_unchanged_from": "signature_analysis_40tile_v2",
        "counts": counts,
        "protocol": {
            "pre_event": "relative months -12 through -3",
            "event_post": "relative months 0 through +2",
            "aligned_curve": "relative months -12 through +12",
            "uncertainty": "5,000 complete spatial-block bootstrap resamples",
            "seed": int(args.seed),
        },
        "layout": layout,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    (output_dir / "figure_03_population_s1_signatures_v3_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
