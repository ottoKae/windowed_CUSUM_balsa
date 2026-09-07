#!/usr/bin/env python3
"""Render Figure 6 v5 with aligned panel headers and framed lower legends.

The frozen N1 point estimates and spatial-block intervals are reused without
recomputation. Earlier figure versions and manuscript files remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_n1_window_step_polarization_figure as base  # noqa: E402
import build_n1_window_step_polarization_figure_v2 as v2  # noqa: E402
import build_n1_window_step_polarization_figure_v3 as v3  # noqa: E402
import build_n1_window_step_polarization_figure_v4 as v4  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v5_framed_lower_legends"
)

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 77.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4

GREEN = v4.GREEN
YELLOW = v4.YELLOW
YELLOW_EDGE = v4.YELLOW_EDGE
BLUE_GREY = v4.BLUE_GREY
DARK = v4.DARK
GRID = v4.GRID
LEGEND_EDGE = "#AFAFAF"

STEP_COLOURS = {6: YELLOW, 12: GREEN}
STEP_EDGES = {6: YELLOW_EDGE, 12: GREEN}
METRIC_STYLES = {
    "precision": ("s", "--", 1.05, 3),
    "recall": ("^", ":", 1.05, 4),
    "f1": ("D", "-", 1.55, 5),
}

POLARIZATION_ORDER = list(v4.POLARIZATION_ORDER)
POLARIZATION_LABELS = dict(v4.POLARIZATION_LABELS)
POLARIZATION_COLOURS = dict(v4.POLARIZATION_COLOURS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8.1,
        "axes.titlesize": 8.7,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def add_panel_headers(figure: plt.Figure, axes: list[plt.Axes]) -> None:
    titles = [
        "Window length and step",
        "Paired F1 contrasts",
        "Polarization rules",
    ]
    labels = ["(a)", "(b)", "(c)"]
    for axis, label, title in zip(axes, labels, titles):
        position = axis.get_position()
        figure.text(
            position.x0,
            0.966,
            label,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9.0,
        )
        figure.text(
            (position.x0 + position.x1) / 2,
            0.966,
            title,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=8.7,
        )


def metric_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="s", color=DARK, lw=0,
            markerfacecolor=DARK, markersize=4.2, label="Precision"
        ),
        Line2D(
            [0], [0], marker="^", color=DARK, lw=0,
            markerfacecolor=DARK, markersize=4.4, label="Recall"
        ),
        Line2D(
            [0], [0], marker="D", color=DARK, lw=0,
            markerfacecolor="none", markeredgewidth=0.85,
            markersize=4.2, label="F1"
        ),
    ]


def add_a_legend_container(axis: plt.Axes) -> None:
    box = Rectangle(
        (0.008, 0.010),
        0.984,
        0.158,
        transform=axis.transAxes,
        facecolor="white",
        edgecolor=LEGEND_EDGE,
        linewidth=0.65,
        zorder=8,
    )
    axis.add_patch(box)

    step_handles = [
        Line2D([0], [0], color=YELLOW, lw=1.8, label="6-month step"),
        Line2D([0], [0], color=GREEN, lw=1.8, label="12-month step"),
    ]
    step_legend = axis.legend(
        handles=step_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.074),
        ncol=2,
        frameon=False,
        columnspacing=1.20,
        handlelength=1.5,
        handletextpad=0.38,
        borderaxespad=0,
    )
    step_legend.set_zorder(10)
    axis.add_artist(step_legend)

    metric_legend = axis.legend(
        handles=metric_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014),
        ncol=3,
        frameon=False,
        columnspacing=0.80,
        handletextpad=0.28,
        borderaxespad=0,
    )
    metric_legend.set_zorder(10)


def add_c_legend_container(axis: plt.Axes) -> None:
    box = Rectangle(
        (0.008, 0.012),
        0.984,
        0.092,
        transform=axis.transAxes,
        facecolor="white",
        edgecolor=LEGEND_EDGE,
        linewidth=0.65,
        zorder=8,
    )
    axis.add_patch(box)
    legend = axis.legend(
        handles=metric_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.020),
        ncol=3,
        frameon=False,
        columnspacing=0.82,
        handletextpad=0.28,
        borderaxespad=0,
    )
    legend.set_zorder(10)


def plot_panel_a(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    for step, local in geometry.groupby("step_months"):
        step = int(step)
        local = local.sort_values("window_months")
        colour = STEP_COLOURS[step]
        edge = STEP_EDGES[step]
        for metric, (marker, linestyle, linewidth, zorder) in METRIC_STYLES.items():
            axis.plot(
                local["window_months"],
                local[metric],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                markersize=4.5,
                markerfacecolor="none" if metric == "f1" else colour,
                markeredgecolor=edge,
                markeredgewidth=0.8,
                color=colour,
                alpha=1.0 if metric == "f1" else 0.88,
                zorder=zorder,
            )

    axis.set(
        xlabel="Window length (months)",
        ylabel="Event-level metric",
        xticks=[24, 36, 48],
        xlim=(22.5, 49.5),
        ylim=(0.60, 0.875),
    )
    axis.set_yticks(np.arange(0.60, 0.851, 0.05))
    axis.grid(axis="y", color=GRID, linewidth=0.45, zorder=0)
    add_a_legend_container(axis)


def plot_panel_c(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    point = metrics.set_index("stage")
    interval = v3.interval_lookup(intervals)
    y_positions = np.array([3.70, 2.80, 1.90, 1.00])
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
            [min(values), max(values)], [y, y],
            color=edge, linewidth=1.05, alpha=0.63, zorder=1,
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
        ylim=(0.0, 4.55),
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
    v4.plot_panel_b(axes[1], geometry_contrasts)
    plot_panel_c(axes[2], metrics, intervals)
    figure.subplots_adjust(
        left=0.060,
        right=0.996,
        bottom=0.205,
        top=0.835,
        wspace=0.46,
    )
    add_panel_headers(figure, list(axes))
    return figure


def save_bundle(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Title": "N1 window, step, and polarization sensitivity"},
    )
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025
    )
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
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
    stem = output_dir / "figure_06_window_length_step_polarization_n1_v5"
    save_bundle(figure, stem)
    plt.close(figure)

    manifest = {
        "schema_version": 5,
        "workflow": "figure_06_n1_framed_lower_legend_revision",
        "statistics_unchanged_from": "window_length_step_polarization_n1_20260831_v1",
        "adaptation_level": "style-only inheritance",
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "plot_box_width_ratios": [1, 1, 1],
        "visual_encoding": {
            "panel_labels": "separate, aligned to each axes upper-left",
            "panel_a_legend": "single white/grey container, two centered rows",
            "panel_c_legend": "single white/grey container, one row and three columns",
            "f1_fill": "transparent",
            "draw_order": {"precision": 3, "recall": 4, "f1": 5},
            "panel_c_grid_interval": 0.05,
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
                "sha256": sha256(path),
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
    atomic_json(manifest, output_dir / "manifest.json")
    print(f"Figure written to {stem}.pdf/.svg/.png/.tiff")
    print("Statistics and LaTeX were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

