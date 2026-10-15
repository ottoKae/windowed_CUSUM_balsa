#!/usr/bin/env python3
"""Build an additive five-panel Figure 5 (a--e).

The figure keeps the frozen VV/VH trajectories and stand-representation
comparison, removes the two paired-F1 panels from the display, and presents
the absolute window/step and polarization comparisons in a compact lower
layout.  Source statistics and all archived figures remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / "analysis/workflows"
SCRIPT_DIR = ROOT / "scripts"
for path in (WORKFLOW_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_stand_representation_ablation_figure as stand  # noqa: E402
import build_n1_window_step_polarization_figure_v8 as n1  # noqa: E402


DEFAULT_TRAJECTORIES = stand.DEFAULT_TRAJECTORIES
DEFAULT_REPRESENTATION_METRICS = stand.DEFAULT_METRICS
DEFAULT_N1_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_WINDOW_INPUT = ROOT / (
    "outputs/windowed_cusum_n1_window_length_extension_20260901_5x2/geometry"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "figure5_abcde_compact_20260901_v2"
)

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 188.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4

WINDOW_ORDER = [
    "w72_s06",
    "w72_s12",
    "w60_s06",
    "w60_s12",
    "w48_s06",
    "w48_s12",
    "w36_s06",
    "w36_s12",
    "w24_s06",
]
WINDOW_LABELS = {
    "w72_s06": "72/6",
    "w72_s12": "72/12",
    "w60_s06": "60/6",
    "w60_s12": "60/12",
    "w48_s06": "48/6",
    "w48_s12": "48/12",
    "w36_s06": "36/6",
    "w36_s12": "36/12",
    "w24_s06": "24/6",
}
BASELINE = "w48_s12"
METRICS = ("precision", "recall", "f1")
METRIC_MARKERS = {"precision": "s", "recall": "^", "f1": "D"}
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument(
        "--representation-metrics", type=Path, default=DEFAULT_REPRESENTATION_METRICS
    )
    parser.add_argument("--n1-input", type=Path, default=DEFAULT_N1_INPUT)
    parser.add_argument("--window-input", type=Path, default=DEFAULT_WINDOW_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required input(s) missing:\n" + "\n".join(missing))


def _configure() -> None:
    stand._style()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _add_heading(axis: plt.Axes, label: str, subtitle: str | None = None) -> None:
    axis.text(
        0.0,
        1.0,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=stand.INK,
        clip_on=False,
        zorder=20,
    )
    if subtitle:
        axis.text(
            0.14,
            1.0,
            subtitle,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color=stand.INK,
            clip_on=False,
            zorder=20,
        )


def _trajectory_legend(axis: plt.Axes) -> None:
    legend = axis.legend(
        handles=stand._trajectory_legend_handles_compact(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#B9C1C9",
        framealpha=0.96,
        fancybox=False,
        handlelength=1.9,
        columnspacing=0.85,
        borderaxespad=0.0,
        borderpad=0.25,
        labelspacing=0.22,
    )
    legend.get_frame().set_linewidth(0.45)
    legend.set_zorder(10)


def _representation_legend(axis: plt.Axes) -> None:
    legend = axis.legend(
        handles=stand._metric_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.015, 0.012),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#B9C1C9",
        framealpha=0.96,
        fancybox=False,
        handletextpad=0.28,
        labelspacing=0.20,
        borderaxespad=0.0,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.45)
    legend.set_zorder(10)


def _metric_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker=METRIC_MARKERS[metric],
            linestyle="none",
            markersize=4.8,
            markerfacecolor="white" if metric == "f1" else stand.METRIC_COLORS[METRIC_LABELS[metric]],
            markeredgecolor=stand.METRIC_COLORS[METRIC_LABELS[metric]],
            markeredgewidth=0.8,
            label=METRIC_LABELS[metric],
        )
        for metric in METRICS
    ]


def _window_legend(axis: plt.Axes) -> None:
    legend = axis.legend(
        handles=_metric_handles(),
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#B9C1C9",
        framealpha=0.96,
        fancybox=False,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.45)
    legend.set_zorder(10)


def _load_n1_polarization(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "polarization_metrics": input_dir / "source_data/polarization_metrics.csv",
        "polarization_intervals": input_dir / "source_data/polarization_bootstrap_intervals.csv",
        "source_manifest": input_dir / "manifest.json",
    }
    _require(paths.values())
    metrics = pd.read_csv(paths["polarization_metrics"])
    intervals = pd.read_csv(paths["polarization_intervals"])
    n1.v2.assert_frozen_values(metrics)
    n1.v3.interval_lookup(intervals)
    return metrics, intervals, paths


def _load_window_metrics(input_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    path = input_dir / "geometry_metrics_extended.csv"
    manifest = input_dir.parent / "manifest.json"
    _require([path, manifest])
    frame = pd.read_csv(path)
    if "estimable" in frame:
        frame = frame.loc[frame.estimable.astype(bool)].copy()
    else:
        frame = frame.loc[frame.central_decision_months.astype(float) > 0].copy()
    frame = frame.set_index("name").reindex(WINDOW_ORDER).reset_index()
    if frame["name"].isna().any() or frame["f1"].isna().any():
        raise ValueError("The window metrics do not contain all nine estimable configurations")
    frame["window_months"] = pd.to_numeric(frame["window_months"], errors="raise").astype(int)
    frame["step_months"] = pd.to_numeric(frame["step_months"], errors="raise").astype(int)
    expected = {(int(name[1:3]), int(name[-2:])) for name in WINDOW_ORDER}
    observed = set(zip(frame.window_months, frame.step_months))
    if observed != expected:
        raise ValueError(f"Unexpected estimable window grid: {sorted(observed)}")
    return frame, {"geometry_metrics_extended": path, "experiment_manifest": manifest}


def _plot_window_metrics(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    local = metrics.set_index("name").loc[WINDOW_ORDER]
    y = np.arange(len(WINDOW_ORDER))
    baseline_y = WINDOW_ORDER.index(BASELINE)
    axis.axhspan(
        baseline_y - 0.43,
        baseline_y + 0.43,
        color=stand.PALE_GREEN,
        alpha=0.95,
        zorder=0,
    )
    for index, name in enumerate(WINDOW_ORDER):
        values = [float(local.loc[name, metric]) for metric in METRICS]
        axis.plot(
            [min(values), max(values)],
            [index, index],
            color="#AAB4BE",
            linewidth=1.05,
            zorder=1,
        )
        for metric in METRICS:
            colour = stand.METRIC_COLORS[METRIC_LABELS[metric]]
            axis.scatter(
                float(local.loc[name, metric]),
                index,
                s=24 if metric != "f1" else 28,
                marker=METRIC_MARKERS[metric],
                facecolor="white" if metric == "f1" else colour,
                edgecolor=colour,
                linewidth=0.85,
                zorder=3,
            )
    axis.set(
        xlim=(0.62, 0.89),
        ylim=(-0.5, len(WINDOW_ORDER) - 0.5),
        yticks=y,
        yticklabels=[WINDOW_LABELS[name] for name in WINDOW_ORDER],
        xlabel="Event-level score",
    )
    axis.invert_yaxis()
    axis.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
    axis.grid(axis="x", color=stand.GRID, linewidth=0.45, zorder=0)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="both", labelsize=6.8, pad=1.4)
    axis.xaxis.label.set_size(7.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    # The common metric key is already shown in panels (c) and (e). Avoid a
    # third legend in this dense nine-row panel; the caption documents the
    # marker convention and the source data preserve the exact values.


def _save(figure: plt.Figure, output_dir: Path, dpi: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_05_representation_and_sensitivity_abcde"
    outputs = {
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "tiff": stem.with_suffix(".tiff"),
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s):\n" + "\n".join(existing))
    figure.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035, metadata={"Title": "Five-panel stand representation and detector sensitivity"})
    figure.savefig(outputs["png"], dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    figure.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.035, pil_kwargs={"compression": "tiff_lzw"})
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trajectories = args.trajectories.resolve()
    representation_metrics_path = args.representation_metrics.resolve()
    n1_input = args.n1_input.resolve()
    window_input = args.window_input.resolve()
    output_dir = args.output_dir.resolve()
    _require([trajectories, representation_metrics_path])

    calendar, trajectory_frame, trajectory_manifest = stand._load_trajectories(trajectories)
    trajectory_audit = stand._audit_trajectory(trajectory_frame)
    representation_metrics = stand._load_metrics(representation_metrics_path)
    polarization_metrics, polarization_intervals, polarization_paths = _load_n1_polarization(n1_input)
    window_metrics, window_paths = _load_window_metrics(window_input)

    _configure()
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    top = figure.add_gridspec(2, 1, left=0.245, right=0.985, bottom=0.625, top=0.965, hspace=0.15)
    lower = figure.add_gridspec(
        2,
        2,
        left=0.245,
        right=0.985,
        bottom=0.065,
        top=0.545,
        hspace=0.12,
        wspace=0.30,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
    )
    ax_a = figure.add_subplot(top[0, 0])
    ax_b = figure.add_subplot(top[1, 0], sharex=ax_a)
    ax_c = figure.add_subplot(lower[:, 0])
    ax_d = figure.add_subplot(lower[0, 1])
    ax_e = figure.add_subplot(lower[1, 1])

    stand._plot_band_trajectory(ax_a, calendar, trajectory_frame, trajectory_manifest, band="VV", show_x_labels=False)
    stand._plot_band_trajectory(ax_b, calendar, trajectory_frame, trajectory_manifest, band="VH", show_x_labels=True)
    _trajectory_legend(ax_b)
    stand._plot_metrics(ax_c, representation_metrics)
    _representation_legend(ax_c)
    _plot_window_metrics(ax_d, window_metrics)
    n1.plot_panel_c(ax_e, polarization_metrics, polarization_intervals)

    # Align the top time axes with the complete visible lower evidence extent.
    external_bounds = stand._align_time_axes(figure, [ax_a, ax_b], [ax_c, ax_d, ax_e])
    for axis, label in ((ax_a, "(a)"), (ax_b, "(b)")):
        lower_y, upper_y = sorted(axis.get_ylim())
        ticks = [tick for tick in axis.get_yticks() if lower_y <= tick <= upper_y]
        top_tick = max(ticks)
        display_y = axis.transData.transform((axis.get_xlim()[0], top_tick))[1]
        figure_y = figure.transFigure.inverted().transform((0.0, display_y))[1]
        figure.text(external_bounds.x0, figure_y, label, ha="left", va="center", fontsize=8.8, fontweight="bold", color=stand.INK)
    _add_heading(ax_c, "(c)", "Event-level score")
    _add_heading(ax_d, "(d)", "Event-level score")
    _add_heading(ax_e, "(e)", "Metric value")

    # Keep the lower evidence panels compact and aligned to the same visible
    # left and right extent without changing the trajectory data.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    lower_boxes = [axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()) for axis in (ax_c, ax_d, ax_e)]
    left = min(box.x0 for box in lower_boxes)
    right = max(box.x1 for box in lower_boxes)
    for axis in (ax_a, ax_b):
        position = axis.get_position()
        axis.set_position([left + 0.0, position.y0, right - left, position.height])

    outputs = _save(figure, output_dir, args.dpi)
    plt.close(figure)

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_tables = {
        "aggregation_trajectories.csv": trajectory_frame.assign(month=calendar.strftime("%Y-%m")),
        "representation_metrics.csv": representation_metrics,
        "window_metrics.csv": window_metrics,
        "polarization_metrics.csv": polarization_metrics,
        "polarization_intervals.csv": polarization_intervals,
    }
    source_paths: dict[str, Path] = {}
    for name, frame in source_tables.items():
        path = source_dir / name
        frame.to_csv(path, index=False, float_format="%.9f")
        source_paths[name] = path

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": str(Path(__file__).resolve()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "figure_contract": {
            "claim": "The stand trajectory, stand representation, and absolute detector settings can be compared without displaying redundant paired-F1 panels.",
            "panels": {
                "a": "Frozen illustrative VV trajectory with pixel-level interquartile envelope.",
                "b": "Frozen illustrative VH trajectory with the same stand and display semantics.",
                "c": "Absolute precision, recall, and F1 across seven stand representations.",
                "d": "Absolute event-level scores across nine estimable window/step configurations.",
                "e": "Absolute precision, recall, and F1 across four polarization decision rules.",
            },
            "omitted_display_panels": {
                "old_d": "Paired F1 representation contrasts; retained in archived source results.",
                "old_f": "Paired F1 window/step contrasts; retained in the complete 5x2 experiment output.",
            },
        },
        "data_integrity": {
            "window_configurations_displayed": WINDOW_ORDER,
            "window_configuration_omitted": {"w24_s12": "No central decision span (24 - 2*12 = 0); not estimable."},
            "representation_routes": int(len(representation_metrics)),
            "polarization_strategies": int(len(polarization_metrics)),
            "statistics_recomputed": False,
            "latex_modified": False,
            "red_used": False,
        },
        "trajectory_audit": trajectory_audit,
        "inputs": {
            "trajectories": stand._record(trajectories),
            "representation_metrics": stand._record(representation_metrics_path),
            **{name: stand._record(path) for name, path in polarization_paths.items()},
            **{name: stand._record(path) for name, path in window_paths.items()},
        },
        "source_tables": {name: stand._record(path) for name, path in source_paths.items()},
        "outputs": {name: stand._record(path) for name, path in outputs.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Figure 5 (a-e) compact layout\n\n"
        "This additive display keeps panels (a), (b), and (c) unchanged in scientific content, "
        "removes the redundant paired-F1 panels, and relabels the former window and polarization "
        "panels as (d) and (e). Panel (d) shows nine estimable window/step configurations; 24/12 "
        "is omitted because its central decision span is zero. The archived paired contrasts remain "
        "available in the experiment outputs. No LaTeX or archived figure was modified.\n",
        encoding="utf-8",
    )
    print(f"PNG: {outputs['png']}")
    print(f"PDF: {outputs['pdf']}")
    print(f"SVG: {outputs['svg']}")
    print(f"TIFF: {outputs['tiff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
