#!/usr/bin/env python3
"""Render the Fig. 3-styled visual revision of N1 Figure 6.

This script performs no statistical recomputation. It reads the frozen source
tables produced by ``build_n1_window_step_polarization_figure.py``, verifies the
expected N1 values, and applies only typography, palette, spacing, and label
placement changes. The v1 figure and outputs remain untouched.
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_n1_window_step_polarization_figure as base  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v2_fig3_style"
)

FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 70.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4

PALETTE = {
    "primary": "#77964a",
    "secondary": "#fbc040",
    "comparison": "#6f7f89",
    "neutral": "#8f8f8f",
    "dark": "#333333",
    "grid": "#dedede",
}

EXPECTED_POLARIZATION = {
    "VV only": (0.7668085106382979, 0.8243366880146387, 0.7945326278659612),
    "VH only": (0.6817785527462947, 0.7154620311070449, 0.6982142857142859),
    "VV or VH": (0.6242424242424243, 0.8481244281793230, 0.7191621411947247),
    "VV and VH": (0.8664259927797834, 0.6587374199451053, 0.7484407484407485),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8.4,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.0,
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


def add_panel_headings(figure: plt.Figure, axes: list[plt.Axes]) -> None:
    """Align headings with each complete tight subfigure, as in Figure 3."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    headings = [
        "(a) Window length and step",
        "(b) Paired F1 difference",
        "(c) Polarization operating points",
    ]
    for axis, heading in zip(axes, headings):
        tight = axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
        figure.text(
            tight.x0,
            0.965,
            heading,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9.5,
        )


def save_bundle(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Title": "N1 window-length, step, and polarization sensitivity"},
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


def assert_frozen_values(metrics: pd.DataFrame) -> None:
    local = metrics.set_index("stage")
    if set(local.index) != set(EXPECTED_POLARIZATION):
        raise ValueError(f"Unexpected polarization strategies: {sorted(local.index)}")
    for name, expected in EXPECTED_POLARIZATION.items():
        observed = local.loc[name, ["precision", "recall", "f1"]].to_numpy(float)
        if not np.allclose(observed, np.asarray(expected), atol=1e-12, rtol=0):
            raise ValueError(f"Frozen N1 values changed for {name}: {observed}")


def plot_panel_a(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    step_colours = {6: PALETTE["comparison"], 12: PALETTE["primary"]}
    metric_styles = {
        "f1": ("o", "-", 1.55),
        "precision": ("s", "--", 1.0),
        "recall": ("^", ":", 1.0),
    }
    for step, local in geometry.groupby("step_months"):
        local = local.sort_values("window_months")
        for metric, (marker, linestyle, linewidth) in metric_styles.items():
            axis.plot(
                local.window_months,
                local[metric],
                marker=marker,
                linestyle=linestyle,
                linewidth=linewidth,
                markersize=4.4,
                markeredgewidth=0.45,
                markeredgecolor=PALETTE["dark"],
                color=step_colours[int(step)],
                alpha=1.0 if metric == "f1" else 0.84,
                zorder=3,
            )

    axis.set(
        xlabel="Window length (months)",
        ylabel="Event-level metric",
        xticks=[24, 36, 48],
        xlim=(22.5, 49.5),
        ylim=(0.62, 0.94),
    )
    axis.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, zorder=0)
    handles = [
        Line2D([0], [0], color=PALETTE["dark"], marker="o", lw=1.4, label="F1"),
        Line2D([0], [0], color=PALETTE["comparison"], lw=1.6, label="6-month step"),
        Line2D([0], [0], color=PALETTE["dark"], marker="s", lw=1.0, ls="--", label="Precision"),
        Line2D([0], [0], color=PALETTE["primary"], lw=1.6, label="12-month step"),
        Line2D([0], [0], color=PALETTE["dark"], marker="^", lw=1.0, ls=":", label="Recall"),
        Line2D([0], [0], color="none", lw=0, label=""),
    ]
    legend = axis.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.7,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.15,
    )
    legend._legend_box.align = "left"


def plot_panel_b(axis: plt.Axes, geometry_contrasts: pd.DataFrame) -> None:
    selected = geometry_contrasts.loc[
        geometry_contrasts.contrast.eq("w36_s12_minus_w48_s12")
        & geometry_contrasts.metric.eq("f1")
    ]
    if len(selected) != 1:
        raise ValueError("Expected one 36/12 minus 48/12 F1 contrast")
    row = selected.iloc[0]
    estimate = float(row.estimate)
    low = float(row.ci_low)
    high = float(row.ci_high)

    axis.axvline(0, color=PALETTE["dark"], linestyle="--", linewidth=0.85, zorder=1)
    axis.errorbar(
        estimate,
        0,
        xerr=np.array([[estimate - low], [high - estimate]]),
        fmt="D",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=PALETTE["dark"],
        markeredgewidth=0.9,
        ecolor=PALETTE["primary"],
        elinewidth=1.4,
        capsize=3.0,
        capthick=0.9,
        zorder=3,
    )
    axis.set(
        xlabel="Paired ΔF1",
        yticks=[],
        ylim=(-0.34, 0.34),
        xlim=(-0.025, 0.027),
    )
    axis.set_xticks([-0.02, 0.00, 0.02])
    axis.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, zorder=0)
    axis.text(
        0.57,
        0.73,
        "36/12 − 48/12",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.0,
    )
    axis.text(
        0.57,
        0.23,
        f"ΔF1 = {estimate:+.3f}\n95% CI\n[{low:+.3f}, {high:+.3f}]",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.1,
        color=PALETTE["dark"],
    )


def plot_panel_c(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    point = metrics.set_index("stage")
    interval = intervals.pivot(
        index="stage", columns="metric", values=["ci_low", "ci_high"]
    )
    styles = {
        "VV only": (PALETTE["primary"], "o", PALETTE["primary"]),
        "VH only": (PALETTE["secondary"], "o", PALETTE["secondary"]),
        "VV or VH": (PALETTE["comparison"], "^", PALETTE["comparison"]),
        "VV and VH": ("white", "D", PALETTE["dark"]),
    }
    labels = {
        "VV only": (0.905, 0.815, "right"),
        "VH only": (0.555, 0.745, "left"),
        "VV or VH": (0.905, 0.690, "right"),
        "VV and VH": (0.555, 0.790, "left"),
    }
    display = {
        "VV only": "VV only",
        "VH only": "VH only",
        "VV or VH": "VV or VH",
        "VV and VH": "VV–VH agreement",
    }

    for name in base.DISPLAY_ORDER:
        row = point.loc[name]
        recall = float(row.recall)
        precision = float(row.precision)
        x_low = float(interval.loc[name, ("ci_low", "recall")])
        x_high = float(interval.loc[name, ("ci_high", "recall")])
        y_low = float(interval.loc[name, ("ci_low", "precision")])
        y_high = float(interval.loc[name, ("ci_high", "precision")])
        face, marker, edge = styles[name]
        axis.errorbar(
            recall,
            precision,
            xerr=np.array([[recall - x_low], [x_high - recall]]),
            yerr=np.array([[precision - y_low], [y_high - precision]]),
            fmt=marker,
            markersize=5.3,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.85,
            ecolor=edge,
            elinewidth=1.0,
            capsize=2.2,
            capthick=0.75,
            zorder=3,
        )
        label_x, label_y, alignment = labels[name]
        axis.text(
            label_x,
            label_y,
            f"{display[name]}, F1={row.f1:.3f}",
            ha=alignment,
            va="center",
            fontsize=7.0,
            color=PALETTE["dark"],
        )

    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0.55, 0.92),
        ylim=(0.55, 0.92),
    )
    axis.set_xticks([0.60, 0.70, 0.80, 0.90])
    axis.set_yticks([0.60, 0.70, 0.80, 0.90])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color=PALETTE["grid"], linewidth=0.45, zorder=0)


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
        gridspec_kw={"width_ratios": [1.45, 0.85, 1.25]},
    )
    plot_panel_a(axes[0], geometry)
    plot_panel_b(axes[1], geometry_contrasts)
    plot_panel_c(axes[2], metrics, intervals)
    figure.subplots_adjust(
        left=0.070,
        right=0.992,
        bottom=0.205,
        top=0.875,
        wspace=0.43,
    )
    add_panel_headings(figure, list(axes))
    return figure


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
    assert_frozen_values(metrics)

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
    stem = output_dir / "figure_06_window_length_step_polarization_n1_v2"
    save_bundle(figure, stem)
    plt.close(figure)

    manifest = {
        "schema_version": 2,
        "workflow": "figure_06_n1_fig3_style_visual_revision",
        "statistics_unchanged_from": "window_length_step_polarization_n1_20260831_v1",
        "adaptation_level": "style-only inheritance",
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "panel_roles": {
            "a": "five admissible window-length/step operating points",
            "b": "decisive paired 36/12 minus 48/12 F1 comparison",
            "c": "polarization precision-recall operating points",
        },
        "visual_changes": {
            "panel_labels": "parenthesized labels with Fig. 3 typography",
            "palette": dict(PALETTE),
            "panel_a_legend": "single compact two-row legend above all data",
            "panel_b": "no y tick label; direct comparison and CI annotation",
            "panel_c": "equal precision-recall scales and direct labels clear of intervals",
            "uncertainty_note": "moved from plot body to caption",
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
