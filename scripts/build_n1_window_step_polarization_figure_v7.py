#!/usr/bin/env python3
"""Render Figure 6 v7 with wider panels and exterior panel letters."""
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_n1_window_step_polarization_figure as base  # noqa: E402
import build_n1_window_step_polarization_figure_v2 as v2  # noqa: E402
import build_n1_window_step_polarization_figure_v3 as v3  # noqa: E402
import build_n1_window_step_polarization_figure_v6 as v6  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/jag_codex_20260821/method_figure/"
    "window_length_step_polarization_n1_20260831_v7_wider_tight_panels"
)

FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 72.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4


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


def add_panel_letters(axes: list[plt.Axes]) -> None:
    """Align letter tops with the top spine and place them left of the y-axis."""
    for axis, label in zip(axes, ("(a)", "(b)", "(c)")):
        axis.text(
            -0.018,
            1.0,
            label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            fontsize=9.0,
            clip_on=False,
            zorder=20,
        )


def plot_panel_a(axis: plt.Axes, geometry: pd.DataFrame) -> None:
    v6.plot_panel_a(axis, geometry)
    # Preserve physical panel height; extend only the data coordinate range.
    axis.set_ylim(0.58, 0.90)
    axis.set_yticks(np.arange(0.60, 0.851, 0.05))


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
    v6.plot_panel_b(axes[1], geometry_contrasts)
    v6.plot_panel_c(axes[2], metrics, intervals)
    figure.subplots_adjust(
        left=0.055,
        right=0.997,
        bottom=0.215,
        top=0.955,
        wspace=0.34,
    )
    add_panel_letters(list(axes))
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
    stem = output_dir / "figure_06_window_length_step_polarization_n1_v7"
    save_bundle(figure, stem)
    plt.close(figure)

    manifest = {
        "schema_version": 7,
        "workflow": "figure_06_n1_wider_tighter_panel_revision",
        "statistics_unchanged_from": "window_length_step_polarization_n1_20260831_v1",
        "adaptation_level": "style-only inheritance",
        "figure_dimensions_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
        "layout": {
            "equal_width_ratios": [1, 1, 1],
            "subplot_wspace": 0.34,
            "panel_letters": "left of y-axis; top aligned with top spine",
            "panel_a_ylim": [0.58, 0.90],
            "panel_a_labelled_ticks": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
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

