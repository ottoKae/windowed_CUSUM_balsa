"""Render Fig. 3 v10 with aligned Panel C heading and legend labels.

This is a visual-only revision of v9.  The Panel C heading uses the same
complete-subfigure left edge as Panels A/B.  Its legend keeps only VV/VH
labels, with a taller nested uncertainty swatch and a thinner median line;
the interval definitions remain in the manuscript caption.  No LaTeX source
is modified.
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
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
from analysis.workflows import build_population_s1_signature_figure_v7 as v7
from analysis.workflows import build_population_s1_signature_figure_v9 as v9


DEFAULT_SEED = 20260822
HEADING_OFFSET = 0.010
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0


class HandlerCompositeBandV10(HandlerBase):
    """Draw a tall nested band swatch with a fine centre line."""

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        colour = orig_handle.colour
        outer = Rectangle(
            (xdescent, ydescent + 0.03 * height),
            width,
            0.94 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.28,
            transform=trans,
        )
        inner = Rectangle(
            (xdescent + 0.05 * width, ydescent + 0.18 * height),
            0.90 * width,
            0.64 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.56,
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + 0.50 * height, ydescent + 0.50 * height],
            color=colour,
            linewidth=0.95,
            transform=trans,
        )
        return [outer, inner, line]


def apply_v10_layout(
    fig: plt.Figure,
    curve: pd.DataFrame,
    events: pd.DataFrame,
    table_summary: pd.DataFrame,
) -> dict[str, object]:
    """Apply v9 layout, then correct Panel C heading and legend alignment."""

    layout = v9.apply_v9_layout(fig, curve, events, table_summary)
    axes = list(fig.axes)
    c_ax = axes[2]

    # Replace the v9 legend.  Text is vertically centred against the centre
    # line, while the two fills occupy more vertical area than that line.
    legend = c_ax.get_legend()
    if legend is not None:
        legend.remove()
    handles = [
        v7.CompositeBandHandle(v4.PALETTE[band])
        for band in base.BANDS
    ]
    legend = c_ax.legend(
        handles,
        ["VV", "VH"],
        handler_map={v7.CompositeBandHandle: HandlerCompositeBandV10()},
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=2.9,
        handleheight=1.65,
        handletextpad=0.50,
        labelspacing=0.28,
        borderaxespad=0.1,
    )
    legend._legend_box.align = "left"
    for text in legend.get_texts():
        text.set_verticalalignment("center")

    # Reposition only the Panel C combined heading to the complete tight
    # subfigure edge, matching Panels A/B.  The larger vertical offset is
    # retained from v9.
    for text in list(fig.texts):
        if text.get_text().startswith("(c) "):
            text.remove()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = c_ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    fig.text(
        tight.x0,
        tight.y1 + HEADING_OFFSET,
        "(c) Event-aligned trajectories",
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    layout.update({
        "panel_c_heading_alignment": "complete tight subfigure left edge",
        "panel_c_legend": "tall nested fills with a thin median line",
        "panel_c_legend_text_alignment": "vertically centred",
        "panel_c_legend_title": None,
    })
    return layout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    events = events_all.loc[
        events_all.event_id.astype(str).isin(set(event_ids))
    ].copy()
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
    layout.update(apply_v10_layout(fig, curve, events, table_summary))

    stem = output_dir / "figure_03_population_s1_signatures_v10"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v10"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 10,
        "workflow": "population_s1_signatures_three_panel_visual_v10",
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
            "panel_a_subtitle": "Backscatter distributions",
            "panel_a_grid": "rendered below distribution artists",
            "panel_a_separator": "black",
            "panel_b_title": "Event backscatter response",
            "panel_b_orientation": "horizontal; Pair difference (dB) on x-axis",
            "panel_c_y_label": "Event-relative change (dB)",
            "panel_c_heading": "aligned to complete tight subfigure edge",
            "panel_c_legend": "tall nested fills with thin median line; no title",
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
    (output_dir / "figure_03_population_s1_signature_v10_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("LaTeX was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
