#!/usr/bin/env python3
"""Render the complete-contrast N1 revision of Figure 6.

This is a display-only revision. It reads and validates the frozen N1 source
tables produced by ``build_n1_window_step_polarization_figure.py``. It does not
recompute statistics, overwrite earlier figures, or modify the manuscript.

Panel roles
-----------
(a) Show all five admissible window-length/step operating points.
(b) Show every prespecified paired F1 contrast against frozen 48/12.
(c) Show Precision, Recall, and F1 (with spatial-block intervals) for each
    polarization decision rule.
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
import build_n1_window_step_polarization_figure_v2 as v2  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v3_full_contrasts_dotplot"
)

FIGURE_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 74.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4
PALETTE = dict(v2.PALETTE)

CONTRAST_ORDER = ["w36_s12", "w48_s06", "w36_s06", "w24_s06"]
CONTRAST_LABELS = {
    "w36_s12": "36/12",
    "w48_s06": "48/6",
    "w36_s06": "36/6",
    "w24_s06": "24/6",
}
POLARIZATION_ORDER = ["VV only", "VH only", "VV or VH", "VV and VH"]
POLARIZATION_LABELS = {
    "VV only": "VV only",
    "VH only": "VH only",
    "VV or VH": "VV or VH",
    "VV and VH": "VV–VH agreement",
}
POLARIZATION_COLOURS = {
    "VV only": PALETTE["primary"],
    "VH only": PALETTE["secondary"],
    "VV or VH": PALETTE["comparison"],
    "VV and VH": PALETTE["dark"],
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
    v2.configure_matplotlib()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.titlesize": 9.3,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 6.9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def add_panel_headings(figure: plt.Figure, axes: list[plt.Axes]) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    headings = [
        "(a) Window length and step",
        "(b) Paired F1 contrasts",
        "(c) Polarization rules",
    ]
    for axis, heading in zip(axes, headings):
        tight = axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
        figure.text(
            tight.x0,
            0.968,
            heading,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=9.3,
        )


def plot_panel_a(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    """Retain the validated v2 panel-a encoding."""
    v2.plot_panel_a(axis, geometry)


def prepare_f1_contrasts(geometry_contrasts: pd.DataFrame) -> pd.DataFrame:
    local = geometry_contrasts.loc[
        geometry_contrasts["metric"].eq("f1")
        & geometry_contrasts["right"].eq("w48_s12")
    ].copy()
    if set(local["left"]) != set(CONTRAST_ORDER) or len(local) != 4:
        raise ValueError(
            "Expected four frozen F1 contrasts against 48/12; found "
            f"{sorted(local['left'].tolist())}"
        )
    local["left"] = pd.Categorical(
        local["left"], categories=CONTRAST_ORDER, ordered=True
    )
    return local.sort_values("left").reset_index(drop=True)


def plot_panel_b(axis: plt.Axes, geometry_contrasts: pd.DataFrame) -> None:
    contrasts = prepare_f1_contrasts(geometry_contrasts)
    y_positions = np.arange(len(contrasts))[::-1]
    axis.axvline(0, color=PALETTE["dark"], linestyle="--", linewidth=0.85, zorder=1)

    for y, row in zip(y_positions, contrasts.itertuples(index=False)):
        estimate = float(row.estimate)
        low = float(row.ci_low)
        high = float(row.ci_high)
        is_decisive = str(row.left) == "w36_s12"
        colour = PALETTE["primary"] if is_decisive else PALETTE["comparison"]
        axis.errorbar(
            estimate,
            y,
            xerr=np.array([[estimate - low], [high - estimate]]),
            fmt="D",
            markersize=4.5,
            markerfacecolor="white" if is_decisive else colour,
            markeredgecolor=colour,
            markeredgewidth=0.85,
            ecolor=colour,
            elinewidth=1.15,
            capsize=2.5,
            capthick=0.8,
            zorder=3,
        )
        axis.text(
            high + 0.003,
            y,
            f"{estimate:+.3f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=PALETTE["dark"],
        )

    axis.set(
        xlabel=r"Paired $\Delta$F1 relative to 48/12",
        xlim=(-0.087, 0.034),
        ylim=(-0.65, len(contrasts) - 0.35),
        yticks=y_positions,
        yticklabels=[CONTRAST_LABELS[str(value)] for value in contrasts["left"]],
    )
    axis.set_xticks([-0.08, -0.04, 0.00])
    axis.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, zorder=0)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.text(
        0.01,
        1.015,
        "Window/step",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=PALETTE["dark"],
    )


def interval_lookup(intervals: pd.DataFrame) -> pd.DataFrame:
    expected = {
        (stage, metric)
        for stage in POLARIZATION_ORDER
        for metric in ("precision", "recall", "f1")
    }
    observed = set(zip(intervals["stage"], intervals["metric"]))
    if observed != expected:
        raise ValueError("Polarization interval table does not match frozen contract")
    return intervals.set_index(["stage", "metric"])


def plot_panel_c(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    point = metrics.set_index("stage")
    interval = interval_lookup(intervals)
    y_positions = np.arange(len(POLARIZATION_ORDER))[::-1]
    marker_styles = {
        "precision": "s",
        "recall": "^",
        "f1": "D",
    }

    for y, stage in zip(y_positions, POLARIZATION_ORDER):
        values = [float(point.loc[stage, metric]) for metric in marker_styles]
        colour = POLARIZATION_COLOURS[stage]
        axis.plot(
            [min(values), max(values)],
            [y, y],
            color=colour,
            linewidth=1.05,
            alpha=0.63,
            zorder=1,
        )
        for metric, marker in marker_styles.items():
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
                markerfacecolor="white" if is_f1 else colour,
                markeredgecolor=colour,
                markeredgewidth=0.85,
                ecolor=colour,
                elinewidth=0.8,
                capsize=1.7,
                capthick=0.65,
                alpha=1.0 if is_f1 else 0.78,
                zorder=3 if is_f1 else 2,
            )
        f1_value = float(point.loc[stage, "f1"])
        axis.text(
            f1_value + 0.012,
            y + 0.18,
            f"F1 {f1_value:.3f}",
            ha="left",
            va="bottom",
            fontsize=6.6,
            color=PALETTE["dark"],
        )

    axis.set(
        xlabel="Metric value",
        xlim=(0.54, 0.925),
        ylim=(-0.65, len(POLARIZATION_ORDER) - 0.35),
        yticks=y_positions,
        yticklabels=[POLARIZATION_LABELS[name] for name in POLARIZATION_ORDER],
    )
    axis.set_xticks([0.60, 0.70, 0.80, 0.90])
    axis.grid(axis="x", color=PALETTE["grid"], linewidth=0.45, zorder=0)
    axis.tick_params(axis="y", length=0, pad=2)
    handles = [
        Line2D(
            [0], [0], marker="s", color=PALETTE["dark"], lw=0,
            markerfacecolor=PALETTE["dark"], markersize=4.0, label="Precision"
        ),
        Line2D(
            [0], [0], marker="^", color=PALETTE["dark"], lw=0,
            markerfacecolor=PALETTE["dark"], markersize=4.2, label="Recall"
        ),
        Line2D(
            [0], [0], marker="D", color=PALETTE["dark"], lw=0,
            markerfacecolor="white", markeredgewidth=0.8,
            markersize=4.0, label="F1"
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.3,
        borderaxespad=0,
    )


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
        gridspec_kw={"width_ratios": [1.28, 1.02, 1.46]},
    )
    plot_panel_a(axes[0], geometry)
    plot_panel_b(axes[1], geometry_contrasts)
    plot_panel_c(axes[2], metrics, intervals)
    figure.subplots_adjust(
        left=0.065,
        right=0.993,
        bottom=0.205,
        top=0.835,
        wspace=0.47,
    )
    add_panel_headings(figure, list(axes))
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
    prepare_f1_contrasts(geometry_contrasts)
    interval_lookup(intervals)

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
    stem = output_dir / "figure_06_window_length_step_polarization_n1_v3"
    save_bundle(figure, stem)
    plt.close(figure)

    manifest = {
        "schema_version": 3,
        "workflow": "figure_06_n1_complete_contrasts_and_metric_dot_ranges",
        "statistics_unchanged_from": "window_length_step_polarization_n1_20260831_v1",
        "adaptation_level": "style-only inheritance with evidence-completeness revision",
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "figure_level_claim": (
            "Window displacement changes the precision--recall balance; 36/12 "
            "and frozen 48/12 have indistinguishable F1, while polarization "
            "rules provide distinct operating points."
        ),
        "panel_roles": {
            "a": "all five admissible window-length/step operating points",
            "b": "all paired F1 contrasts against the frozen 48/12 configuration",
            "c": "precision, recall, and F1 with spatial-block intervals for all polarization rules",
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
