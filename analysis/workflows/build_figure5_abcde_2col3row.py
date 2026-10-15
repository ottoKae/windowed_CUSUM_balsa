#!/usr/bin/env python3
"""Build an additive two-column by three-row Figure 5 (a--e).

Layout (read row-wise): ``a | d``, ``b | d``, ``c | e``.  The first column
keeps the VV trajectory, VH trajectory, and stand-representation comparison;
the right column gives the window/step sensitivity and polarization
sensitivity.  The two paired-F1 panels and the non-estimable 24/12 setting
are deliberately not displayed, while their source results remain archived.

This is a new display-only renderer.  It reuses the frozen source tables and
plotting primitives, and never edits the earlier Figure 5 renderer, source
results, or LaTeX files.
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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / "analysis/workflows"
SCRIPT_DIR = ROOT / "scripts"
for path in (WORKFLOW_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_figure5_abcde_compact as compact  # noqa: E402

stand = compact.stand
n1 = compact.n1

DEFAULT_TRAJECTORIES = compact.DEFAULT_TRAJECTORIES
DEFAULT_REPRESENTATION_METRICS = compact.DEFAULT_REPRESENTATION_METRICS
DEFAULT_N1_INPUT = compact.DEFAULT_N1_INPUT
DEFAULT_WINDOW_INPUT = compact.DEFAULT_WINDOW_INPUT
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "figure5_abcde_2col3row_20260901_v18"
)

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 150.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4


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
    """Use the shared plotting style and declare final-size text explicitly."""
    compact._configure()
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
        # A previous invocation may have completed the four image files but
        # stopped before writing the manifest/source tables.  Reuse that
        # complete image bundle so the metadata stage can be resumed without
        # overwriting any binary output.  Partial bundles remain protected.
        if len(existing) != len(outputs):
            raise FileExistsError("Refusing to overwrite existing output(s):\n" + "\n".join(existing))
        return outputs
    figure.savefig(
        outputs["pdf"],
        bbox_inches="tight",
        pad_inches=0.035,
        metadata={"Title": "Figure 5: representation and detector sensitivity"},
    )
    figure.savefig(outputs["png"], dpi=dpi, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035)
    figure.savefig(
        outputs["tiff"],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return outputs


def _trajectory_legend_left(axis: plt.Axes) -> None:
    """Keep the long trajectory key inside the first-column panel."""
    handles = stand._trajectory_legend_handles_compact()
    # Use the gardenia-yellow accent consistently for the Polygon-first mean
    # in both the trajectory and legend, while preserving the other frozen
    # encodings.
    for handle in handles:
        if handle.get_label() == "Polygon-first mean":
            handle.set_color(stand.GARDENIA_YELLOW)
        elif handle.get_label() == "Polygon-first histogram mode":
            handle.set_label("Polygon-first mode")
    legend = axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#B9C1C9",
        framealpha=0.96,
        fancybox=False,
        handlelength=1.35,
        columnspacing=0.42,
        handletextpad=0.25,
        borderaxespad=0.0,
        borderpad=0.22,
        labelspacing=0.20,
        fontsize=5.8,
    )
    legend.get_frame().set_linewidth(0.45)
    legend.set_zorder(10)
    # Exclude the in-panel key from tight-box alignment; it is already
    # positioned inside the data axes and should not change column geometry.
    legend.set_in_layout(False)


def _set_trajectory_mean_colour(axis: plt.Axes) -> None:
    """Apply the legend's gardenia-yellow mean encoding to the plotted line."""
    # _plot_band_trajectory adds the four summaries in a fixed order:
    # median, mean, histogram mode, and spatial medoid; event markers follow.
    if len(axis.lines) < 2:
        raise RuntimeError("The trajectory plot did not create its mean line")
    axis.lines[1].set_color(stand.GARDENIA_YELLOW)


def _overlay_polarization_markers(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    """Redraw e-panel point markers above every interval line and cap."""
    point = metrics.set_index("stage")
    y_positions = np.array([3.40, 2.60, 1.80, 1.00])
    marker_styles = {"precision": "s", "recall": "^", "f1": "D"}
    for y, stage in zip(y_positions, n1.POLARIZATION_ORDER):
        colour = n1.POLARIZATION_COLOURS[stage]
        edge = n1.YELLOW_EDGE if stage == "VH only" else colour
        for metric, marker in marker_styles.items():
            estimate = float(point.loc[stage, metric])
            axis.plot(
                [estimate],
                [y],
                linestyle="none",
                marker=marker,
                markersize=4.5 if metric == "f1" else 4.1,
                markerfacecolor="white" if metric == "f1" else colour,
                markeredgecolor=edge,
                markeredgewidth=0.85,
                color=edge,
                alpha=1.0,
                zorder=20,
                clip_on=False,
            )


def _set_polarization_legend_f1_white(axis: plt.Axes) -> None:
    """Use a white F1 diamond in the e-panel metric legend."""
    legend = axis.get_legend()
    if legend is None:
        return
    for handle in getattr(legend, "legend_handles", []):
        if handle.get_label() == "F1":
            handle.set_markerfacecolor("white")
            handle.set_markeredgecolor(n1.DARK)
            handle.set_color(n1.DARK)


def _move_legend_up(axis: plt.Axes, points: float = 2.0) -> None:
    """Move an existing in-panel legend upward by a physical point amount."""
    legend = axis.get_legend()
    if legend is None:
        return
    box = legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted())
    x, y, width, height = box.bounds
    delta = (points / 72.0) / (axis.get_position().height * FIGURE_HEIGHT_IN)
    legend.set_bbox_to_anchor((x, y + delta), transform=axis.transAxes)
    legend.set_in_layout(False)


def _fit_manual_legend_frame(axis: plt.Axes, pad_points: float = 1.5) -> None:
    """Fit the e-panel manual legend frame around its rendered entries.

    ``plot_panel_c`` draws the frame as an axes-level white Rectangle while
    Matplotlib lays out the Legend independently.  After the requested
    two-point legend lift, the old fixed-height frame can cut through the
    entries.  We therefore derive the frame bounds from the final rendered
    legend and add a small physical-point padding on all sides.
    """
    legend = axis.get_legend()
    if legend is None:
        return
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer).transformed(axis.transAxes.inverted())
    pad_x = (pad_points / 72.0) / (axis.get_position().width * FIGURE_WIDTH_IN)
    pad_y = (pad_points / 72.0) / (axis.get_position().height * FIGURE_HEIGHT_IN)
    for patch in axis.patches:
        if not isinstance(patch, Rectangle):
            continue
        face = patch.get_facecolor()
        # The white rectangle is the legend container; the pale-green row
        # highlight is also a Rectangle but is intentionally excluded.
        if len(face) < 3 or not np.all(np.asarray(face[:3]) > 0.97):
            continue
        patch.set_transform(axis.transAxes)
        patch.set_bounds(
            legend_box.x0 - pad_x,
            legend_box.y0 - pad_y,
            legend_box.width + 2.0 * pad_x,
            legend_box.height + 2.0 * pad_y,
        )
        patch.set_zorder(8)
        break


C_DISPLAY_ORDER = [
    "frozen_mode",
    "frozen_mean",
    "frozen_median",
    "space_time_dbscan_r0.4",
    "connected_r0.2",
    "temporal_dbscan_r0.4",
    "frozen_center",
]


def _plot_representation_metrics(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    """Plot panel (c) with the requested fourth/last route exchange."""
    frame = metrics.set_index("name").loc[C_DISPLAY_ORDER].reset_index()
    y = np.arange(len(frame))
    median_y = C_DISPLAY_ORDER.index("frozen_median")
    axis.axhspan(median_y - 0.43, median_y + 0.43, color=stand.PALE_GREEN, alpha=0.95, zorder=0)
    for yi, (_, row) in zip(y, frame.iterrows()):
        lo = min(float(row["precision"]), float(row["recall"]), float(row["f1"]))
        hi = max(float(row["precision"]), float(row["recall"]), float(row["f1"]))
        axis.plot([lo, hi], [yi, yi], color="#AAB4BE", lw=1.15, zorder=1)
    axis.scatter(frame["precision"], y, s=25, marker="o", color=stand.METRIC_COLORS["Precision"], edgecolor="white", linewidth=0.4, zorder=3)
    axis.scatter(frame["recall"], y, s=25, marker="s", color=stand.METRIC_COLORS["Recall"], edgecolor="white", linewidth=0.4, zorder=3)
    axis.scatter(frame["f1"], y, s=27, marker="D", color=stand.METRIC_COLORS["F1"], edgecolor="white", linewidth=0.4, zorder=3)
    for boundary in (2.5, 3.5):
        axis.axhline(boundary, color=stand.GRID, lw=0.55, zorder=0)
    axis.set_yticks(y)
    axis.set_yticklabels([stand.ROUTE_LABELS[name] for name in frame["name"]])
    axis.invert_yaxis()
    axis.set_ylim(len(frame) - 0.5, -0.5)
    axis.set_xlim(0.30, 1.0)
    axis.set_xlabel("")
    axis.grid(axis="x", color=stand.GRID, lw=0.45, alpha=0.85)
    axis.grid(axis="y", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _add_panel_letters_at_left(figure: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    """Place letters at each rendered subfigure's upper-left corner."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    left_column_boxes = [
        axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
        for axis in axes[:3]
    ]
    shared_left = min(box.x0 for box in left_column_boxes)
    de_left = axes[3].get_tightbbox(renderer).transformed(figure.transFigure.inverted()).x0
    # Keep the two spanning/top panels aligned while applying the requested
    # small physical lifts.  The 0.003 baseline preserves the prior c/e
    # placement; only a/d (+5 pt) and b (+2 pt) are moved in this revision.
    label_lifts_pt = {0: 5.0, 1: 2.0, 3: 5.0}
    base_lift = 0.003
    for index, (axis, label) in enumerate(zip(axes, ("(a)", "(b)", "(c)", "(d)", "(e)"))):
        tight = axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
        position = axis.get_position()
        lift = base_lift + (label_lifts_pt.get(index, 0.0) / 72.0) / FIGURE_HEIGHT_IN
        figure.text(
            shared_left if index < 3 else de_left,
            position.y1 + lift,
            label,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
            color=stand.INK,
            clip_on=False,
        )


def _align_right_column_right_edges(figure: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    """Make the rendered right edges of (d) and (e) coincide."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()) for axis in axes]
    target_right = max(box.x1 for box in boxes)
    for axis, box in zip(axes, boxes):
        position = axis.get_position()
        shift = target_right - box.x1
        axis.set_position([position.x0 + shift, position.y0, position.width, position.height])
    figure.canvas.draw()


def _align_first_column_visual_edges(figure: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    """Align a/b y-label edges and right extent to c's rendered extent.

    The first-column axes share a common right edge with (c), while the longer
    route labels in (c) define the visible left edge.  The VV/VH y-axis titles
    in (a,b) are expanded leftward to that same edge, rather than aligning only
    the underlying Axes rectangles or only their tight boxes.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    anchor = axes[2].get_tightbbox(renderer).transformed(figure.transFigure.inverted())
    desired_left = []
    for axis in axes[:2]:
        ylabel = axis.yaxis.label.get_window_extent(renderer).transformed(figure.transFigure.inverted())
        desired_left.append(axis.get_position().x0 + (anchor.x0 - ylabel.x0))
    shared_left = min(desired_left)
    for axis in axes[:2]:
        tight = axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted())
        position = axis.get_position()
        new_x0 = shared_left
        new_x1 = position.x1 + (anchor.x1 - tight.x1)
        axis.set_position([new_x0, position.y0, new_x1 - new_x0, position.height])
    figure.canvas.draw()


def _tighten_lower_row_gap(
    figure: plt.Figure,
    upper_axes: Sequence[plt.Axes],
    lower_axes: Sequence[plt.Axes],
    gap_points: float = 5.0,
) -> None:
    """Place the lower panels within ``gap_points`` of the upper panels.

    The lower panels have already been height-compressed.  Their top edge is
    moved using rendered tight boxes (rather than grid-cell coordinates), so
    the distance includes tick labels and remains meaningful at final size.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    upper_bottoms = [
        axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).y0
        for axis in upper_axes
    ]
    if not upper_bottoms:
        return
    gap = (gap_points / 72.0) / FIGURE_HEIGHT_IN
    # Anchor to the lower of the two upper-panel bottoms.  This preserves a
    # visible gap from b while keeping the spanning d no farther away than the
    # previous five-point constraint (the top-pair axis shift is 3 pt).
    target_top = min(upper_bottoms) - gap
    for axis in lower_axes:
        position = axis.get_position()
        axis.set_position(
            [position.x0, target_top - position.height, position.width, position.height]
        )
    figure.canvas.draw()


def _copy_source_tables(
    output_dir: Path,
    trajectory_frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    representation_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    polarization_metrics: pd.DataFrame,
    polarization_intervals: pd.DataFrame,
) -> dict[str, Path]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "aggregation_trajectories.csv": trajectory_frame.assign(month=calendar.strftime("%Y-%m")),
        "representation_metrics.csv": representation_metrics,
        "window_metrics.csv": window_metrics,
        "polarization_metrics.csv": polarization_metrics,
        "polarization_intervals.csv": polarization_intervals,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = source_dir / name
        frame.to_csv(path, index=False, float_format="%.9f")
        paths[name] = path
    return paths


def _write_metadata(
    output_dir: Path,
    outputs: dict[str, Path],
    source_paths: dict[str, Path],
    trajectories: Path,
    representation_metrics_path: Path,
    n1_input_manifest: Path,
    window_input_metrics: Path,
    trajectory_audit: dict,
    representation_metrics: pd.DataFrame,
    window_metrics: pd.DataFrame,
    polarization_metrics: pd.DataFrame,
) -> None:
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
        "layout": {
            "grid": "2 columns by 3 rows",
            "row_1": "(a) | (d)",
            "row_2": "(b) | (d)",
            "row_3": "(c) | (e)",
            "right_panel_d_spans_rows": [1, 2],
            "top_panel_height_fraction_of_v11": 2.0 / 3.0,
            "metric_panel_height_fraction_of_grid_cell": 0.60,
            "legend_vertical_shift_pt": 2.0,
            "legend_frame_fitted_to_entries": True,
            "e_legend_frame_fit_padding_pt": 1.5,
            "top_label_lifts_pt": {"a": 5.0, "d": 5.0, "b": 2.0},
            "top_panel_axis_shift_pt": 3.0,
            "lower_row_gap_max_pt": 5.0,
            "lower_row_gap_target_pt": 2.0,
            "b_y_limits_db": [-24.0, -10.0],
            "ab_vertical_gap_reduction": "hspace 0.12 to 0.06",
            "trajectory_mean_colour": stand.GARDENIA_YELLOW,
            "b_legend_grid": "3 columns by 2 rows",
            "b_mode_label": "Polygon-first mode",
            "e_f1_legend_facecolor": "white",
            "e_marker_zorder": 20,
            "left_column_plot_boxes_equal": True,
            "left_column_plot_boxes_left_aligned": True,
            "bottom_x_axis_titles": "removed for (c), (d), and (e)",
            "panel_subtitles": "none",
            "e_spines": "left retained with ticks; top and right removed",
            "first_column_visual_alignment": "a/b y-label left edges and right extents aligned to c's rendered extent",
        },
        "figure_contract": {
            "claim": (
                "Frozen stand trajectories and complementary representation, "
                "window/step, and polarization sensitivities can be read as one "
                "non-redundant evidence chain."
            ),
            "panels": {
                "a": "Illustrative polygon-first VV trajectory with interior-cell IQR.",
                "b": "The same stand's VH trajectory with identical time semantics.",
                "c": "Absolute precision, recall, and F1 across seven stand representations.",
                "d": "Absolute event-level scores across nine estimable window/step settings.",
                "e": "Absolute precision, recall, and F1 across four polarization rules.",
            },
            "omitted_display": {
                "paired_f1_panels": "Retained in source experiment outputs; omitted to avoid repeating absolute scores.",
                "w24_s12": "Not estimable because 24 - 2*12 = 0 central-decision months.",
            },
        },
        "data_integrity": {
            "representation_routes": int(len(representation_metrics)),
            "window_configurations_displayed": list(compact.WINDOW_ORDER),
            "polarization_strategies": int(len(polarization_metrics)),
            "window_rows_available_after_estimability_filter": int(len(window_metrics)),
            "statistics_recomputed": False,
            "latex_modified": False,
            "red_used": False,
        },
        "trajectory_audit": trajectory_audit,
        "inputs": {
            "trajectories": stand._record(trajectories),
            "representation_metrics": stand._record(representation_metrics_path),
            "n1_input_manifest": stand._record(n1_input_manifest),
            "window_input_metrics": stand._record(window_input_metrics),
        },
        "source_tables": {name: stand._record(path) for name, path in source_paths.items()},
        "outputs": {name: stand._record(path) for name, path in outputs.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Figure 5 (a-e): two-column by three-row layout (v18)\n\n"
        "Panel order is (a) VV trajectory, (b) VH trajectory, (c) stand representation, "
        "(d) window/step sensitivity spanning the first two rows, and (e) polarization "
        "sensitivity. Panels (a,b) are horizontally expanded so their rendered y-axis "
        "label edges and right extent align with (c). The selected VV-only production "
        "rule is highlighted in pale green in (e). The two paired-F1 views are retained in the archived experiment "
        "tables but are not displayed. The 24/12 setting is excluded because its central "
        "decision span is zero. This renderer is additive: previous figures and LaTeX "
        "files are unchanged.\n",
        encoding="utf-8",
    )


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
    polarization_metrics, polarization_intervals, polarization_paths = compact._load_n1_polarization(n1_input)
    window_metrics, window_paths = compact._load_window_metrics(window_input)

    _configure()
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), constrained_layout=False)
    # The shared normalized coordinates intentionally leave enough room for
    # the long stand labels in (c) while keeping the two columns visually equal.
    # The 170-mm height is a compact A4 double-column target; the first two
    # rows (a,b) and the spanning (d) use the same normalized height.
    grid = figure.add_gridspec(
        3,
        2,
        left=0.245,
        right=0.985,
        bottom=0.075,
        top=0.955,
        hspace=0.06,
        wspace=0.30,
        # The first two grid rows are set to exactly two-thirds of their
        # previous physical height; the lower metric row is kept compact
        # independently below.
        height_ratios=[0.36, 0.36, 1.0],
        width_ratios=[1.0, 1.0],
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[1, 0], sharex=ax_a)
    ax_c = figure.add_subplot(grid[2, 0])
    ax_d = figure.add_subplot(grid[0:2, 1])
    ax_e = figure.add_subplot(grid[2, 1])

    stand._plot_band_trajectory(
        ax_a, calendar, trajectory_frame, trajectory_manifest, band="VV", show_x_labels=False
    )
    _set_trajectory_mean_colour(ax_a)
    stand._plot_band_trajectory(
        ax_b, calendar, trajectory_frame, trajectory_manifest, band="VH", show_x_labels=True
    )
    _set_trajectory_mean_colour(ax_b)
    # Extend the VH scale to -24 dB.  The extra lower range creates a clean
    # data-free pocket for the two-column trajectory legend.
    ax_b.set_ylim(-24.0, -10.0)
    ax_b.set_yticks(np.arange(-24.0, -8.0, 2.0))
    _trajectory_legend_left(ax_b)
    _plot_representation_metrics(ax_c, representation_metrics)
    compact._representation_legend(ax_c)
    compact._plot_window_metrics(ax_d, window_metrics)
    # The frozen production choice is the VV-only polarization rule; retain
    # that row as the selected configuration in the same green language used
    # for 48/12 in (d).
    ax_e.axhspan(3.00, 3.80, color=stand.PALE_GREEN, alpha=0.95, zorder=0)
    n1.plot_panel_c(ax_e, polarization_metrics, polarization_intervals)
    _overlay_polarization_markers(ax_e, polarization_metrics)
    _set_polarization_legend_f1_white(ax_e)

    # Keep the first-column plotting boxes exactly identical in width and
    # x-position (the labels are allowed to extend outside those boxes).
    left_x0 = ax_c.get_position().x0
    left_width = ax_c.get_position().width
    for axis in (ax_a, ax_b):
        position = axis.get_position()
        axis.set_position([left_x0, position.y0, left_width, position.height])

    # The metric title is carried by the compact heading beside each panel
    # letter; suppress only the bottom x-axis titles requested for c--e.
    for axis in (ax_c, ax_d, ax_e):
        axis.set_xlabel("")
    for axis in (ax_c, ax_e):
        legend = axis.get_legend()
        if legend is not None:
            legend.set_in_layout(False)
    # Match (e) to the clean panel treatment used for (c): retain the left
    # axis/ticks for the category labels, but remove the upper/right frame.
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["left"].set_visible(True)
    ax_e.spines["right"].set_visible(False)
    ax_e.tick_params(axis="y", left=True, length=3.0, width=0.7, pad=2)

    # Keep the compact lower row while preserving a small, symmetric breathing
    # space around the dense seven/four-row metric panels.
    for axis in (ax_c, ax_e):
        position = axis.get_position()
        new_height = position.height * 0.60
        axis.set_position([position.x0, position.y0 + 0.5 * (position.height - new_height), position.width, new_height])

    for axis in (ax_b, ax_c, ax_e):
        _move_legend_up(axis, points=2.0)

    # Preserve the existing 3-pt top-pair axis shift; the new 5-pt request
    # applies to the a/d labels themselves, not to the data panels.
    # Move the top pair upward by exactly 3 pt as a unit, retaining their
    # shared top edge and leaving the lower row independent.
    shift_up = (3.0 / 72.0) / FIGURE_HEIGHT_IN
    for axis in (ax_a, ax_d):
        position = axis.get_position()
        axis.set_position([position.x0, position.y0 + shift_up, position.width, position.height])

    # The centered height reduction above leaves a large gap below (b)/(d).
    # Re-anchor the compressed c/e row to the rendered upper-panel bottoms
    # after the top-pair lift so the inter-row distance is no more than five
    # physical points from either upper panel.
    _tighten_lower_row_gap(figure, [ax_b, ax_d], [ax_c, ax_e], gap_points=2.0)

    _align_right_column_right_edges(figure, [ax_d, ax_e])

    # Highlight the frozen proposed polarization configuration (VV only) in
    # the same pale-green language used for the selected 48/12 setting in (d).
    # Panel letters/subtitles are anchored to each plotting box, preserving
    # exact row/column alignment while keeping the source panel semantics.
    _add_panel_letters_at_left(figure, [ax_a, ax_b, ax_c, ax_d, ax_e])

    _align_first_column_visual_edges(figure, [ax_a, ax_b, ax_c])
    # The e-panel frame is fitted after all axis moves so it encloses the
    # lifted legend at final rendered coordinates without touching its text.
    _fit_manual_legend_frame(ax_e, pad_points=1.5)

    outputs = _save(figure, output_dir, args.dpi)
    plt.close(figure)

    source_paths = _copy_source_tables(
        output_dir,
        trajectory_frame,
        calendar,
        representation_metrics,
        window_metrics,
        polarization_metrics,
        polarization_intervals,
    )
    _write_metadata(
        output_dir,
        outputs,
        source_paths,
        trajectories,
        representation_metrics_path,
        n1_input / "manifest.json",
        window_input / "geometry_metrics_extended.csv",
        trajectory_audit,
        representation_metrics,
        window_metrics,
        polarization_metrics,
    )
    for key in ("png", "pdf", "svg", "tiff"):
        print(f"{key.upper()}: {outputs[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
