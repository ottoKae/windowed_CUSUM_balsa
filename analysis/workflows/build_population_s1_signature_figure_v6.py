"""Render Fig. 3 v6 with combined panel headings and composite uncertainty legend.

This visual-only revision preserves the frozen v5 statistics.  Each panel
heading is rendered as one left-aligned string containing its panel label and
subtitle.  Panel C uses composite legend handles so that each polarization is
shown as the median line together with its translucent uncertainty band.
The script writes only a new output directory; it does not modify the
manuscript or any LaTeX source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.workflows import build_population_s1_signature_figure_v2 as base
from analysis.workflows import build_population_s1_signature_figure_v4 as v4


DEFAULT_SEED = 20260822
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_v6_layout(fig: plt.Figure, curve: pd.DataFrame) -> dict[str, object]:
    """Apply v6 headings, terminology, and a line-plus-band legend."""

    axes = list(fig.axes)
    if len(axes) != 3:
        raise ValueError(f"Expected three panel axes, found {len(axes)}")

    # Remove the v4 axes titles and figure-level panel tags before adding one
    # combined heading per panel.  The tight bounding box includes the
    # y-axis label/ticks, so the heading aligns to the complete subfigure edge.
    for ax in axes:
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
    for text in list(fig.texts):
        text.remove()

    axes[0].set_yticklabels([
        "VV all", "VV pre-event", "VV after-event",
        "VH all", "VH pre-event", "VH after-event",
    ])
    axes[1].set_ylabel("Pair difference (dB)", labelpad=2)
    axes[2].set_ylabel("Difference from pre-event (dB)", labelpad=2)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headings = [
        "(a) Absolute backscatter distributions",
        "(b) Paired event responses",
        "(c) Event-aligned trajectories",
    ]
    for ax, heading in zip(axes, headings):
        tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        fig.text(
            tight.x0,
            tight.y1 + 0.006,
            heading,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    # Rebuild Panel C's two interval fills explicitly in this v6 layer.  The
    # v4 renderer creates the same bands; replacing them here keeps the
    # visual encoding auditable from this standalone script as well.
    for collection in list(axes[2].collections):
        collection.remove()
    for band in base.BANDS:
        local = curve.loc[curve.band.eq(band)].sort_values("relative_month")
        x = local.relative_month.to_numpy(dtype=float)
        colour = v4.PALETTE[band]
        axes[2].fill_between(
            x, local["q25"], local["q75"],
            color=colour, alpha=0.18, linewidth=0,
        )
        axes[2].fill_between(
            x, local["bootstrap_lower"], local["bootstrap_upper"],
            color=colour, alpha=0.25, linewidth=0,
        )

    # Panel C now contains the two interval fills (within-event IQR and
    # block-bootstrap interval) and the median line.  These ``fill_between``
    # bands and the median ``errorbar`` are represented together in the
    # legend.  A tuple handle makes
    # the legend communicate the same visual encoding instead of showing a
    # line alone.
    legend = axes[2].get_legend()
    if legend is not None:
        legend.remove()
    handles = []
    for band in base.BANDS:
        handles.append((
            Patch(
                facecolor=v4.PALETTE[band],
                edgecolor="none",
                alpha=0.25,
            ),
            Line2D(
                [], [], color=v4.PALETTE[band], linewidth=1.7,
            ),
        ))
    axes[2].legend(
        handles,
        ["VV", "VH"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=2.0,
        labelspacing=0.35,
        borderaxespad=0.1,
    )

    return {
        "panel_headings": headings,
        "panel_heading_alignment": "left-aligned to each panel tight bounding box",
        "panel_a_labels": [
            "VV all", "VV pre-event", "VV after-event",
            "VH all", "VH pre-event", "VH after-event",
        ],
        "panel_b_ylabel": "Pair difference (dB)",
        "panel_c_ylabel": "Difference from pre-event (dB)",
        "panel_c_legend": "polarization-specific median line over translucent uncertainty band",
        "panel_c_legend_location": "lower-left, one column",
        "all_label_definition": "all valid national polygon-month medians",
    }


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
    bootstrap = base.block_curve_bootstrap(
        aligned, metadata, args.bootstrap_replicates, args.seed
    )
    curve = monthly.merge(
        bootstrap,
        on=["band", "relative_month"],
        validate="one_to_one",
    )

    fig, layout = v4.render_figure(
        national_values, events, table_summary, curve, args.seed
    )
    layout.update(apply_v6_layout(fig, curve))
    stem = output_dir / "figure_03_population_s1_signatures_v6"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v6"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 6,
        "workflow": "population_s1_signatures_three_panel_visual_v6",
        "statistics_unchanged_from": "signature_analysis_40tile_v1",
        "counts": counts,
        "protocol": {
            "pre_event": "relative months -12 through -3",
            "after_event": "relative months 0 through +2",
            "aligned_curve": "relative months -12 through +12",
            "uncertainty": "5,000 complete spatial-block bootstrap resamples",
            "seed": int(args.seed),
            "all_label_definition": "all valid national polygon-month medians",
        },
        "visual_changes": {
            "headings": "panel label and subtitle combined, left-aligned",
            "panel_c_legend": "line plus translucent interval composite handles",
        },
        "latex_modified": False,
        "layout": layout,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    (output_dir / "figure_03_population_s1_signature_v6_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("LaTeX was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

