#!/usr/bin/env python3
"""Render an additive Figure 5 revision with one metric encoding.

This renderer reuses the frozen Figure 5 v18 workflow and source tables, but
standardises the metric glyphs and colours in panels (d) and (e) to the panel
(c) convention:

    Precision = green circle, Recall = gardenia-yellow square,
    F1 = blue diamond.

The v18 script, figures, source tables, and LaTeX are intentionally unchanged.
Panel (c)'s legend is the shared key for (d), while (e) carries an identical
copy because it is a separate metric panel. Polarization identity in (e) is
encoded by the row labels and the pale-green highlight of the retained row,
not by changing metric colours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep the publication contract explicit in this additive wrapper as well as
# in the inherited renderer.  The base workflow performs the actual four
# ``figure.savefig`` calls for PDF, SVG, PNG, and TIFF.
fig_width_mm = 183.0
dpi = 600
OUTPUT_FORMATS = (".pdf", ".svg", ".png", ".tiff")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / "analysis/workflows"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import build_figure5_abcde_2col3row as base  # noqa: E402


METRICS = ("precision", "recall", "f1")
METRIC_MARKERS = {"precision": "o", "recall": "s", "f1": "D"}
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}

DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "figure5_abcde_2col3row_20260902_v21_metric_encoding"
)


def _metric_legend(axis: plt.Axes) -> None:
    """Use exactly the panel-(c) metric legend in a second metric panel."""
    legend = axis.legend(
        handles=base.stand._metric_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.015, 0.012),
        ncol=3,
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


def _plot_window_metrics_unified(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    """Panel (d), retaining all values while using panel-(c) glyph semantics."""
    compact = base.compact
    stand = base.stand
    local = metrics.set_index("name").loc[compact.WINDOW_ORDER]
    y = np.arange(len(compact.WINDOW_ORDER))
    baseline_y = compact.WINDOW_ORDER.index(compact.BASELINE)
    axis.axhspan(
        baseline_y - 0.43,
        baseline_y + 0.43,
        color=stand.PALE_GREEN,
        alpha=0.95,
        zorder=0,
    )
    for index, name in enumerate(compact.WINDOW_ORDER):
        values = [float(local.loc[name, metric]) for metric in METRICS]
        axis.plot(
            [min(values), max(values)],
            [index, index],
            color="#AAB4BE",
            linewidth=1.15,
            zorder=1,
        )
        for metric in METRICS:
            colour = stand.METRIC_COLORS[METRIC_LABELS[metric]]
            axis.scatter(
                float(local.loc[name, metric]),
                index,
                s=25 if metric != "f1" else 27,
                marker=METRIC_MARKERS[metric],
                facecolor=colour,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    axis.set(
        xlim=(0.62, 0.89),
        ylim=(-0.5, len(compact.WINDOW_ORDER) - 0.5),
        yticks=y,
        yticklabels=[compact.WINDOW_LABELS[name] for name in compact.WINDOW_ORDER],
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


def _plot_polarization_metrics_unified(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
) -> None:
    """Panel (e) with metric colour/glyph semantics shared with panel (c)."""
    n1 = base.n1
    stand = base.stand
    point = metrics.set_index("stage")
    interval = n1.v3.interval_lookup(intervals)
    y_positions = np.array([3.40, 2.60, 1.80, 1.00])

    for y, stage in zip(y_positions, n1.POLARIZATION_ORDER):
        values = [float(point.loc[stage, metric]) for metric in METRICS]
        axis.plot(
            [min(values), max(values)],
            [y, y],
            color="#AAB4BE",
            linewidth=1.15,
            zorder=1,
        )
        for metric in METRICS:
            estimate = float(point.loc[stage, metric])
            row = interval.loc[(stage, metric)]
            low = float(row.ci_low)
            high = float(row.ci_high)
            colour = stand.METRIC_COLORS[METRIC_LABELS[metric]]
            axis.errorbar(
                estimate,
                y,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=METRIC_MARKERS[metric],
                markersize=4.5 if metric == "f1" else 4.1,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.65,
                ecolor=colour,
                elinewidth=0.8,
                capsize=1.7,
                capthick=0.65,
                alpha=1.0,
                zorder=5,
            )

    axis.set(
        xlabel="Metric value",
        xlim=(0.55, 0.925),
        ylim=(0.0, 4.10),
        yticks=y_positions,
        yticklabels=[n1.POLARIZATION_LABELS[name] for name in n1.POLARIZATION_ORDER],
    )
    axis.set_xticks(np.arange(0.55, 0.901, 0.10))
    axis.set_xticks(np.arange(0.60, 0.901, 0.10), minor=True)
    axis.grid(axis="x", which="major", color=n1.GRID, linewidth=0.45, zorder=0)
    axis.grid(axis="x", which="minor", color=n1.GRID, linewidth=0.45, zorder=0)
    axis.tick_params(axis="x", which="minor", length=0)
    axis.tick_params(axis="y", length=0, pad=2)
    _metric_legend(axis)


def _install_overrides() -> None:
    """Install display-only overrides in the imported v18 workflow."""
    base.compact._plot_window_metrics = _plot_window_metrics_unified
    base.n1.plot_panel_c = _plot_polarization_metrics_unified
    # The inherited v18 driver adds a second marker layer using the old
    # polarization-colour convention. Disable it so the c-style metric layer
    # is the only visible encoding in panel (e).
    base._overlay_polarization_markers = lambda axis, metrics: None
    # v18's helper changes F1 to an open diamond for the inherited N1 legend;
    # this revision uses the filled blue diamond from panel (c) throughout.
    base._set_polarization_legend_f1_white = lambda axis: None


def _patch_metadata(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["workflow"] = str(Path(__file__).resolve())
        payload["version"] = "v21_metric_encoding"
        payload["figure_contract"]["metric_encoding"] = {
            "Precision": "green circle",
            "Recall": "gardenia-yellow square",
            "F1": "blue diamond",
            "applies_to": ["c", "d", "e"],
        }
        payload["data_integrity"]["latex_modified"] = False
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    readme = output_dir / "README.md"
    readme.write_text(
        "# Figure 5 v19: consistent metric encoding\n\n"
        "This additive renderer preserves the frozen v18 values and layout. "
        "Panels (c), (d), and (e) use one metric key: Precision is a green "
        "circle, Recall is a gardenia-yellow square, and F1 is a blue diamond. "
        "Panel (c) provides the shared key for (d); panel (e) repeats the same "
        "key because it is a separate metric panel. Polarization identity in "
        "(e) is represented by row labels and the pale-green VV-only row.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    _install_overrides()
    arguments = list(argv or [])
    if "--output-dir" not in arguments:
        arguments.extend(["--output-dir", str(DEFAULT_OUTPUT)])
    result = base.main(arguments)
    if result == 0:
        output_dir = Path(arguments[arguments.index("--output-dir") + 1]).resolve()
        _patch_metadata(output_dir)
    return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
