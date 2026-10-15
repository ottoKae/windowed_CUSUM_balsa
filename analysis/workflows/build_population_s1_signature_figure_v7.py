"""Render Fig. 3 v7 with an overlaid band-and-line legend.

The frozen population statistics are reused unchanged from v6.  Panel C's
legend now mirrors the plot: a wide translucent rectangle (spatial-block
interval), a narrower rectangle (IQR), and the median line are drawn on top of
one another for each polarization.  The panel headings remain left aligned,
with a slightly larger gap above the axes.  No LaTeX source is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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


DEFAULT_SEED = 20260822
HEADING_OFFSET = 0.010
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0


@dataclass(frozen=True)
class CompositeBandHandle:
    """Legend handle containing two nested fills and a median line."""

    colour: str


class HandlerCompositeBand(HandlerBase):
    """Draw the wide band, narrow band, and line in one legend swatch."""

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
            (xdescent, ydescent + 0.14 * height),
            width,
            0.72 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.22,
            transform=trans,
        )
        inner = Rectangle(
            (xdescent + 0.08 * width, ydescent + 0.24 * height),
            0.84 * width,
            0.52 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.38,
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + 0.50 * height, ydescent + 0.50 * height],
            color=colour,
            linewidth=1.7,
            transform=trans,
        )
        return [outer, inner, line]


def apply_v7_layout(fig: plt.Figure, curve: pd.DataFrame) -> dict[str, object]:
    """Apply combined headings, explicit y-axis wording, and v7 legend."""

    axes = list(fig.axes)
    if len(axes) != 3:
        raise ValueError(f"Expected three panel axes, found {len(axes)}")

    # Remove v4 titles/tags before adding one complete heading per panel.
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
    axes[2].set_ylabel("Change relative to pre-event (dB)", labelpad=2)

    # Rebuild both visible uncertainty bands explicitly in the v7 layer.  The
    # wide band is the spatial-block interval; the narrower band is the IQR.
    # Using fill_between here keeps the plot and the legend auditably coupled.
    for collection in list(axes[2].collections):
        collection.remove()
    for band in base.BANDS:
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
        CompositeBandHandle(v4.PALETTE[band])
        for band in base.BANDS
    ]
    axes[2].legend(
        handles,
        ["VV", "VH"],
        handler_map={CompositeBandHandle: HandlerCompositeBand()},
        title="Median + uncertainty",
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=2.0,
        labelspacing=0.35,
        borderaxespad=0.1,
        title_fontsize=7.2,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headings = [
        "(a) Absolute backscatter distributions",
        "(b) Paired event responses",
        "(c) Event-aligned trajectories",
    ]
    for panel_index, (ax, heading) in enumerate(zip(axes, headings)):
        tight = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        # Panel C has a long vertical y-axis title.  Aligning its heading to
        # the plotting-area edge prevents the heading from touching that
        # rotated label, while Panels A/B retain the complete tight-edge
        # alignment used in the previous versions.
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
        "panel_heading_alignment": "left-aligned to each panel tight bounding box",
        "panel_c_heading_alignment": "plot-area left edge to clear the rotated y-axis title",
        "panel_heading_offset": HEADING_OFFSET,
        "panel_a_labels": [
            "VV all", "VV pre-event", "VV after-event",
            "VH all", "VH pre-event", "VH after-event",
        ],
        "panel_b_ylabel": "Pair difference (dB)",
        "panel_c_ylabel": "Change relative to pre-event (dB)",
        "panel_c_legend": (
            "nested wide spatial-block interval + narrow IQR + median line"
        ),
        "panel_c_legend_title": "Median + uncertainty",
        "panel_c_legend_location": "lower-left, one column",
        "all_label_definition": "all valid national polygon-month medians",
    }


def sha256(path: Path) -> str:
    import hashlib

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
    layout.update(apply_v7_layout(fig, curve))

    stem = output_dir / "figure_03_population_s1_signatures_v7"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v7"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 7,
        "workflow": "population_s1_signatures_three_panel_visual_v7",
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
            "headings": "panel label and subtitle combined, left-aligned, increased offset",
            "panel_c_y_label": "Change relative to pre-event (dB)",
            "panel_c_legend": (
                "one composite swatch per polarization: wide block interval, "
                "narrow IQR, and median line overlaid"
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
    (output_dir / "figure_03_population_s1_signature_v7_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("LaTeX was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
