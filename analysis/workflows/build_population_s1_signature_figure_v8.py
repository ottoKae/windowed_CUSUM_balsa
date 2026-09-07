"""Render Fig. 3 v8 with concise headings and a clearer composite legend.

This version keeps the frozen population statistics unchanged.  Panel B is
labelled ``Event backscatter response``; Panel C uses the shorter
``Event-relative change (dB)`` y-axis label.  The Panel C legend contains only
the VV/VH labels, while each symbol visibly overlays the wide uncertainty
band, the narrower IQR band, and the median line.  No LaTeX source is changed.
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

from analysis.workflows import build_population_s1_signature_figure_v4 as v4
from analysis.workflows import build_population_s1_signature_figure_v7 as v7


DEFAULT_SEED = 20260822
HEADING_OFFSET = 0.010
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0


class HandlerCompositeBandV8(HandlerBase):
    """Draw a high-contrast wide band, narrow band, and median line."""

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
            (xdescent, ydescent + 0.12 * height),
            width,
            0.76 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.20,
            transform=trans,
        )
        inner = Rectangle(
            (xdescent + 0.07 * width, ydescent + 0.22 * height),
            0.86 * width,
            0.56 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.48,
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + 0.50 * height, ydescent + 0.50 * height],
            color=colour,
            linewidth=2.0,
            transform=trans,
        )
        return [outer, inner, line]


def apply_v8_layout(fig: plt.Figure, curve: pd.DataFrame) -> dict[str, object]:
    """Apply concise panel text and the enlarged no-title composite legend."""

    axes = list(fig.axes)
    if len(axes) != 3:
        raise ValueError(f"Expected three panel axes, found {len(axes)}")

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
    axes[2].set_ylabel("Event-relative change (dB)", labelpad=2)

    # Recreate both plotted intervals explicitly so the legend's visual key
    # has the same two-layer semantics as the graph itself.
    for collection in list(axes[2].collections):
        collection.remove()
    for band in v7.base.BANDS:
        local = curve.loc[curve.band.eq(band)].sort_values("relative_month")
        x = local.relative_month.to_numpy(dtype=float)
        colour = v4.PALETTE[band]
        axes[2].fill_between(
            x,
            local["q25"],
            local["q75"],
            color=colour,
            alpha=0.18,
            linewidth=0,
        )
        axes[2].fill_between(
            x,
            local["bootstrap_lower"],
            local["bootstrap_upper"],
            color=colour,
            alpha=0.25,
            linewidth=0,
        )

    legend = axes[2].get_legend()
    if legend is not None:
        legend.remove()
    handles = [
        v7.CompositeBandHandle(v4.PALETTE[band])
        for band in v7.base.BANDS
    ]
    axes[2].legend(
        handles,
        ["VV", "VH"],
        handler_map={v7.CompositeBandHandle: HandlerCompositeBandV8()},
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=2.8,
        handletextpad=0.45,
        labelspacing=0.28,
        borderaxespad=0.1,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headings = [
        "(a) Absolute backscatter distributions",
        "(b) Event backscatter response",
        "(c) Event-aligned trajectories",
    ]
    for panel_index, (ax, heading) in enumerate(zip(axes, headings)):
        tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        heading_x = tight.x0 if panel_index < 2 else ax.get_position().x0
        fig.text(
            heading_x,
            tight.y1 + HEADING_OFFSET,
            heading,
            ha="left",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    return {
        "panel_headings": headings,
        "panel_heading_offset": HEADING_OFFSET,
        "panel_a_labels": [
            "VV all", "VV pre-event", "VV after-event",
            "VH all", "VH pre-event", "VH after-event",
        ],
        "panel_b_ylabel": "Pair difference (dB)",
        "panel_c_ylabel": "Event-relative change (dB)",
        "panel_c_legend": (
            "two nested fills and one median line per polarization"
        ),
        "panel_c_legend_title": None,
        "panel_c_legend_location": "lower-left, one column",
        "all_label_definition": "all valid national polygon-month medians",
    }


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
    event_ids = v7.base.strict_event_ids(events_all)
    events = events_all.loc[
        events_all.event_id.astype(str).isin(set(event_ids))
    ].copy()
    aligned = v7.base.filter_aligned(aligned_all, event_ids)
    metadata = v7.base.event_metadata(events, event_ids)
    counts = {
        "national_polygons": int(national_metadata.sample_id.nunique()),
        "strict_events": int(len(event_ids)),
        "strict_reference_polygons": int(metadata.sample_id.nunique()),
        "strict_spatial_blocks": int(metadata.spatial_block_id.nunique()),
        "strict_tiles": int(metadata.tile_id.nunique()),
        "aligned_rows": int(len(aligned)),
    }
    for key, expected in v7.base.EXPECTED.items():
        if counts[key] != expected:
            raise ValueError(f"Frozen count mismatch for {key}: {counts[key]} != {expected}")
    if national_values.shape[0] != counts["national_polygons"]:
        raise ValueError("National metadata and value arrays disagree on polygon count")

    monthly = v7.base.monthly_summary(aligned)
    bootstrap = v7.base.block_curve_bootstrap(
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
    layout.update(apply_v8_layout(fig, curve))

    stem = output_dir / "figure_03_population_s1_signatures_v8"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v8"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 8,
        "workflow": "population_s1_signatures_three_panel_visual_v8",
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
            "panel_b_heading": "Event backscatter response",
            "panel_c_y_label": "Event-relative change (dB)",
            "legend": (
                "title removed; enlarged nested wide/narrow bands and median line"
            ),
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
    (output_dir / "figure_03_population_s1_signature_v8_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("LaTeX was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

