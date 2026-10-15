#!/usr/bin/env python3
"""Render the manuscript workflow diagram without accessing study data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GREEN = "#2F6B4F"
PALE_GREEN = "#EEF4E8"
GARDENIA = "#D8A91B"
GRAY = "#555555"


def build_figure(output: Path) -> None:
    steps = [
        ("S1-GRiTS", "Orbit-specific monthly VV/VH\nmissing months retained"),
        ("Stand support", "Polygon-interior cells\nmonthly robust median"),
        ("Windowed CUSUM", "48-month windows, 12-month step\ncentral negative-change gate"),
        ("Statistical admission", "N1 permutation p <= 0.025\nfinite-sample correction"),
        ("Evaluation and products", "Spatially independent matching\nrecurrent-event chronology"),
    ]
    figure, axis = plt.subplots(figsize=(12.0, 3.4))
    axis.set_axis_off()
    positions = np.linspace(0.08, 0.92, len(steps))
    for index, ((title, body), x_position) in enumerate(zip(steps, positions)):
        axis.text(
            x_position,
            0.56,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=10.2,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": PALE_GREEN,
                "edgecolor": GREEN,
                "linewidth": 1.5,
            },
            transform=axis.transAxes,
        )
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.075, 0.56),
                xytext=(x_position + 0.075, 0.56),
                arrowprops={"arrowstyle": "->", "linewidth": 1.7, "color": GRAY},
                xycoords=axis.transAxes,
            )
    axis.text(
        0.5,
        0.08,
        "Detected events represent recurrent stand disturbances; causal interpretation requires independent evidence.",
        ha="center",
        color=GARDENIA,
        fontsize=10.0,
        fontweight="semibold",
        transform=axis.transAxes,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output PDF, SVG, or PNG path.")
    args = parser.parse_args()
    build_figure(args.output)


if __name__ == "__main__":
    main()
