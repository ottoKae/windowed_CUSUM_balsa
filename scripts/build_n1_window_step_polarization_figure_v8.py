#!/usr/bin/env python3
"""Render Figure 6 v8 with compact legends and reduced upper whitespace.

This is a display-only revision of the frozen N1 figure. Point estimates,
spatial-block intervals, source rows, earlier figures, and LaTeX are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_n1_window_step_polarization_figure as base  # noqa: E402
import build_n1_window_step_polarization_figure_v2 as v2  # noqa: E402
import build_n1_window_step_polarization_figure_v3 as v3  # noqa: E402
import build_n1_window_step_polarization_figure_v5 as v5  # noqa: E402
import build_n1_window_step_polarization_figure_v6 as v6  # noqa: E402
import build_n1_window_step_polarization_figure_v7 as v7  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v8_compact_legend_grid"
)

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 72.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4

GREEN = v5.GREEN
YELLOW = v5.YELLOW
YELLOW_EDGE = v5.YELLOW_EDGE
DARK = v5.DARK
GRID = v5.GRID
POLARIZATION_ORDER = list(v5.POLARIZATION_ORDER)
POLARIZATION_LABELS = dict(v5.POLARIZATION_LABELS)
POLARIZATION_COLOURS = dict(v5.POLARIZATION_COLOURS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    """Declare final-size typography locally for auditable standalone rendering."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8.1,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def add_panel_letters(axes: list[plt.Axes]) -> None:
    """Align the left edge of each letter with its plotting box."""
    for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(
            0.0,
            1.0,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9.0,
            clip_on=False,
            zorder=20,
        )


def plot_panel_a(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    v7.plot_panel_a(axis, geometry)


def plot_panel_b(axis: plt.Axes, geometry_contrasts: pd.DataFrame) -> None:
    """Show all frozen contrasts without redundant numerical annotations."""
    contrasts = v3.prepare_f1_contrasts(geometry_contrasts)
    y_positions = np.arange(len(contrasts))[::-1]
    axis.axvline(0, color=DARK, linestyle="--", linewidth=0.85, zorder=1)

    for y, row in zip(y_positions, contrasts.itertuples(index=False)):
        estimate = float(row.estimate)
        low = float(row.ci_low)
        high = float(row.ci_high)
        is_twelve_month = str(row.left) == "w36_s12"
        colour = GREEN if is_twelve_month else YELLOW
        edge = GREEN if is_twelve_month else YELLOW_EDGE
        axis.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="D",
            markersize=4.5,
            markerfacecolor="none" if is_twelve_month else colour,
            markeredgecolor=edge,
            markeredgewidth=0.9,
            ecolor=edge,
            elinewidth=1.2,
            capsize=2.5,
            capthick=0.8,
            zorder=3,
        )

    axis.set(
        xlabel=r"Paired $\Delta$F1 relative to 48/12",
        xlim=(-0.087, 0.034),
        ylim=(-0.65, len(contrasts) - 0.35),
        yticks=y_positions,
        yticklabels=[v3.CONTRAST_LABELS[str(value)] for value in contrasts["left"]],
    )
    # Label every 0.04, but retain a visible 0.02 reference-grid interval.
    axis.set_xticks([-0.08, -0.04, 0.00])
    axis.set_xticks([-0.06, -0.02, 0.02], minor=True)
    axis.grid(axis="x", which="major", color=GRID, linewidth=0.45, zorder=0)
    axis.grid(axis="x", which="minor", color=GRID, linewidth=0.45, zorder=0)
    axis.tick_params(axis="x", which="minor", length=0)
    axis.tick_params(axis="y", length=0, pad=2)


def add_c_legend_container(axis: plt.Axes) -> None:
    """Use a centred compact frame instead of spanning the full plot width."""
    box = Rectangle(
        (0.145, 0.012),
        0.710,
        0.092,
        transform=axis.transAxes,
        facecolor="white",
        edgecolor=v5.LEGEND_EDGE,
        linewidth=0.65,
        zorder=8,
    )
    axis.add_patch(box)
    legend = axis.legend(
        handles=v5.metric_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.020),
        ncol=3,
        frameon=False,
        columnspacing=0.48,
        handlelength=1.15,
        handletextpad=0.18,
        borderaxespad=0,
    )
    legend.set_zorder(10)


def plot_panel_c(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    point = metrics.set_index("stage")
    interval = v3.interval_lookup(intervals)
    y_positions = np.array([3.40, 2.60, 1.80, 1.00])
    marker_styles = {
        "precision": ("s", 3),
        "recall": ("^", 4),
        "f1": ("D", 5),
    }

    for y, stage in zip(y_positions, POLARIZATION_ORDER):
        values = [float(point.loc[stage, metric]) for metric in marker_styles]
        colour = POLARIZATION_COLOURS[stage]
        edge = YELLOW_EDGE if stage == "VH only" else colour
        axis.plot(
            [min(values), max(values)],
            [y, y],
            color=edge,
            linewidth=1.05,
            alpha=0.63,
            zorder=1,
        )
        for metric, (marker, zorder) in marker_styles.items():
            estimate = float(point.loc[stage, metric])
            row = interval.loc[(stage, metric)]
            low = float(row.ci_low)
            high = float(row.ci_high)
            is_f1 = metric == "f1"
            axis.errorbar(
                estimate,
                y,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=marker,
                markersize=4.5 if is_f1 else 4.1,
                markerfacecolor="none" if is_f1 else colour,
                markeredgecolor=edge,
                markeredgewidth=0.85,
                ecolor=edge,
                elinewidth=0.8,
                capsize=1.7,
                capthick=0.65,
                alpha=1.0 if is_f1 else 0.80,
                zorder=zorder,
            )

    axis.set(
        xlabel="Metric value",
        xlim=(0.55, 0.925),
        ylim=(0.0, 4.10),
        yticks=y_positions,
        yticklabels=[POLARIZATION_LABELS[name] for name in POLARIZATION_ORDER],
    )
    axis.set_xticks(np.arange(0.55, 0.901, 0.10))
    axis.set_xticks(np.arange(0.60, 0.901, 0.10), minor=True)
    axis.grid(axis="x", which="major", color=GRID, linewidth=0.45, zorder=0)
    axis.grid(axis="x", which="minor", color=GRID, linewidth=0.45, zorder=0)
    axis.tick_params(axis="x", which="minor", length=0)
    axis.tick_params(axis="y", length=0, pad=2)
    add_c_legend_container(axis)


def render(
    geometry: pd.DataFrame,
    geometry_contrasts: pd.DataFrame,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> plt.Figure:
    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    plot_panel_a(axes[0], geometry)
    plot_panel_b(axes[1], geometry_contrasts)
    plot_panel_c(axes[2], metrics, intervals)
    figure.subplots_adjust(
        left=0.055,
        right=0.997,
        bottom=0.215,
        top=0.985,
        wspace=0.34,
    )
    add_panel_letters(list(axes))
    return figure


def save_bundle(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"bbox_inches": "tight", "pad_inches": 0.008}
    figure.savefig(
        stem.with_suffix(".pdf"),
        **save_options,
        metadata={"Title": "N1 window, step, and polarization sensitivity"},
    )
    figure.savefig(stem.with_suffix(".svg"), **save_options)
    figure.savefig(stem.with_suffix(".png"), dpi=600, **save_options)
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **save_options,
    )


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.out.resolve()
    paths = {
        "geometry_metrics": input_dir / "source_data/geometry_metrics.csv",
        "geometry_contrasts": input_dir / "source_data/geometry_contrasts.csv",
        "polarization_metrics": input_dir / "source_data/polarization_metrics.csv",
        "polarization_intervals": input_dir / "source_data/polarization_bootstrap_intervals.csv",
        "source_manifest": input_dir / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen Figure 6 inputs: " + ", ".join(missing))

    geometry = pd.read_csv(paths["geometry_metrics"])
    geometry_contrasts = pd.read_csv(paths["geometry_contrasts"])
    metrics = pd.read_csv(paths["polarization_metrics"])
    intervals = pd.read_csv(paths["polarization_intervals"])
    v2.assert_frozen_values(metrics)
    v3.prepare_f1_contrasts(geometry_contrasts)
    v3.interval_lookup(intervals)

    expected_geometries = {(24, 6), (36, 6), (48, 6), (36, 12), (48, 12)}
    observed_geometries = set(
        zip(geometry.window_months.astype(int), geometry.step_months.astype(int))
    )
    if observed_geometries != expected_geometries:
        raise ValueError(f"Frozen geometry grid changed: {sorted(observed_geometries)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    base.atomic_csv(geometry, source_dir / "geometry_metrics.csv")
    base.atomic_csv(geometry_contrasts, source_dir / "geometry_contrasts.csv")
    base.atomic_csv(metrics, source_dir / "polarization_metrics.csv")
    base.atomic_csv(intervals, source_dir / "polarization_bootstrap_intervals.csv")

    figure = render(geometry, geometry_contrasts, metrics, intervals)
    stem = output_dir / "figure_06_window_length_step_polarization_n1_v8"
    save_bundle(figure, stem)
    plt.close(figure)

    manifest = {
        "schema_version": 8,
        "workflow": "figure_06_n1_compact_legend_grid_revision",
        "statistics_unchanged_from": "window_length_step_polarization_n1_20260831_v1",
        "adaptation_level": "style-only inheritance",
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "layout": {
            "equal_width_ratios": [1, 1, 1],
            "subplot_wspace": 0.34,
            "panel_letters": "left edges aligned to each plotting box",
            "upper_export_padding_inches": 0.008,
            "panel_b_numeric_annotations": False,
            "panel_b_grid_interval": 0.02,
            "panel_c_legend_frame_axes_fraction": [0.145, 0.012, 0.710, 0.092],
        },
        "data_integrity": {
            "rows_excluded": 0,
            "statistics_recomputed": False,
            "all_prespecified_f1_contrasts_shown": True,
            "all_polarization_metrics_shown": ["precision", "recall", "f1"],
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": v7.sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "outputs": {
            suffix: str(stem.with_suffix(f".{suffix}"))
            for suffix in ("pdf", "svg", "png", "tiff")
        },
        "latex_modified": False,
    }
    v7.atomic_json(manifest, output_dir / "manifest.json")
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("Statistics and LaTeX were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
