"""Render Fig. 3 v9 with coordinated axes and a readable band legend.

The population statistics are frozen and reused unchanged.  Panel A places
grid lines behind the violin distributions and uses a black VV/VH separator.
Panel B is redrawn horizontally so its pair differences share the horizontal
dB orientation of Panel A.  Panel C uses an enlarged composite legend with
visibly thick uncertainty bands and a thinner median line.  No LaTeX source
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

from analysis.workflows import build_population_s1_signature_figure_v4 as v4
from analysis.workflows import build_population_s1_signature_figure_v2 as base
from analysis.workflows import build_population_s1_signature_figure_v7 as v7


DEFAULT_SEED = 20260822
HEADING_OFFSET = 0.010
FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0


class HandlerCompositeBandV9(HandlerBase):
    """Draw a wide/narrow fill pair with a fine median line."""

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
            (xdescent, ydescent + 0.06 * height),
            width,
            0.88 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.28,
            transform=trans,
        )
        inner = Rectangle(
            (xdescent + 0.06 * width, ydescent + 0.19 * height),
            0.88 * width,
            0.62 * height,
            facecolor=colour,
            edgecolor="none",
            alpha=0.56,
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + 0.50 * height, ydescent + 0.50 * height],
            color=colour,
            linewidth=1.1,
            transform=trans,
        )
        return [outer, inner, line]


def redraw_panel_b_horizontal(
    ax: plt.Axes,
    events: pd.DataFrame,
    table_summary: pd.DataFrame,
) -> None:
    """Redraw the paired differences with dB on the horizontal axis."""

    ax.clear()
    for position, band in enumerate(base.BANDS, start=1):
        values = events.loc[
            events.band.eq(band), "post_delta_db"
        ].to_numpy(dtype=float)
        violin = ax.violinplot(
            [values],
            positions=[position],
            orientation="horizontal",
            widths=0.66,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = violin["bodies"][0]
        body.set_facecolor(v4.PALETTE[band])
        body.set_edgecolor(v4.PALETTE[band])
        body.set_alpha(0.24)
        jitter = 0.16 * np.sin(np.arange(len(values), dtype=float) * 2.399963)
        ax.scatter(
            values,
            position + jitter,
            s=2.2,
            color=v4.PALETTE[band],
            alpha=0.20,
            linewidth=0,
            rasterized=True,
        )
        summary = table_summary.loc[table_summary.band.eq(band)].iloc[0]
        median = float(summary.paired_delta_median_db)
        low = float(summary.paired_delta_block_bootstrap_lower_db)
        high = float(summary.paired_delta_block_bootstrap_upper_db)
        ax.errorbar(
            [median],
            [position],
            xerr=[[median - low], [high - median]],
            fmt="D",
            color="black",
            markerfacecolor="white",
            markersize=3.6,
            linewidth=1.0,
            capsize=2.2,
            zorder=5,
        )
    ax.axvline(0, color=v4.PALETTE["zero"], linestyle="--", linewidth=0.8)
    ax.set_xlim(-7.2, 2.55)
    ax.set_ylim(0.45, 2.55)
    # Match Panel A's reading order: VV on top, VH below.
    ax.invert_yaxis()
    ax.set_xticks([-6, -3, 0, 2])
    ax.set_yticks([1, 2], base.BANDS)
    ax.set_xlabel("Pair difference (dB)", labelpad=3)
    ax.set_ylabel("Polarization", labelpad=2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dedede", linewidth=0.45, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def apply_v9_layout(
    fig: plt.Figure,
    curve: pd.DataFrame,
    events: pd.DataFrame,
    table_summary: pd.DataFrame,
) -> dict[str, object]:
    """Apply v9 layer ordering, axes, headings, and legend."""

    axes = list(fig.axes)
    if len(axes) != 3:
        raise ValueError(f"Expected three panel axes, found {len(axes)}")

    # Remove v4 titles and tags.  Panel B is redrawn after clearing its old
    # vertical-category artists so that the underlying statistics are exactly
    # the same but the visual orientation is horizontal.
    for ax in axes:
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
    for text in list(fig.texts):
        text.remove()
    redraw_panel_b_horizontal(axes[1], events, table_summary)

    # Panel A: grid lines are deliberately below all distribution artists;
    # the VV/VH separator is black and remains a visible group boundary.
    axes[0].set_axisbelow(True)
    for gridline in axes[0].get_xgridlines():
        gridline.set_zorder(0)
    for collection in axes[0].collections:
        collection.set_zorder(2)
    for line in axes[0].lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size == 2 and np.allclose(y, 2.0):
            line.set_color("#000000")
            line.set_linewidth(0.85)
            line.set_zorder(1)
        else:
            line.set_zorder(3)

    axes[0].set_yticklabels([
        "VV all", "VV pre-event", "VV after-event",
        "VH all", "VH pre-event", "VH after-event",
    ])
    axes[2].set_ylabel("Event-relative change (dB)", labelpad=2)

    # Rebuild Panel C's two plotted intervals so the composite legend below
    # has precisely the same wide/narrow semantics as the visible graph.
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
    legend = axes[2].get_legend()
    if legend is not None:
        legend.remove()
    handles = [
        v7.CompositeBandHandle(v4.PALETTE[band])
        for band in base.BANDS
    ]
    axes[2].legend(
        handles,
        ["VV", "VH"],
        handler_map={v7.CompositeBandHandle: HandlerCompositeBandV9()},
        loc="lower left",
        frameon=False,
        ncol=1,
        handlelength=2.8,
        handleheight=1.4,
        handletextpad=0.45,
        labelspacing=0.30,
        borderaxespad=0.1,
    )

    # Combined label + subtitle.  Panel C starts at the plotting-area edge
    # to leave the long rotated y-axis title unobstructed.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    headings = [
        "(a) Backscatter distributions",
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
        "panel_a_subtitle": "Backscatter distributions",
        "panel_a_grid": "below violin distributions",
        "panel_a_separator": "black VV/VH separator",
        "panel_b_orientation": "horizontal pair differences; dB on x-axis",
        "panel_b_order": "VV top, VH bottom",
        "panel_b_xlabel": "Pair difference (dB)",
        "panel_b_ylabel": "Polarization",
        "panel_c_ylabel": "Event-relative change (dB)",
        "panel_c_legend": "enlarged nested fills with thin median line",
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
    layout.update(apply_v9_layout(fig, curve, events, table_summary))

    stem = output_dir / "figure_03_population_s1_signatures_v9"
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Title": "Population Sentinel-1 stand-disturbance signatures v9"},
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(output_dir / "event_aligned_monthly_summary.csv", index=False)
    manifest = {
        "schema_version": 9,
        "workflow": "population_s1_signatures_three_panel_visual_v9",
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
            "panel_c_legend": "nested fills dominate a thinner median line",
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
    (output_dir / "figure_03_population_s1_signature_v9_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("LaTeX was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

