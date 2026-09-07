"""Build the refined, code-only Windowed CUSUM concept figure.

This workflow keeps the compact overlaid timeline design of the legacy
PowerPoint panel while using the frozen Polygon-first VV series, production
window configuration, production boundary semantics, and empirical
permutation test from ``build_windowed_cusum_concept``.  It never overwrites an existing
output and does not modify the archived manuscript figure.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

from analysis.workflows import build_windowed_cusum_concept as base


LAYOUT_VERSION = "classic-overlay-v4"


def _window_label(audit: base.WindowAudit) -> str:
    return rf"$W_{{{audit.logical_id}}}$"


def _map_db_to_window_axis(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(finite)
    if valid.sum() < 2:
        raise ValueError("At least two finite values are required for panel (a)")
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if maximum <= minimum:
        raise ValueError("Panel (a) requires a non-constant VV series")
    mapped = np.full_like(finite, np.nan, dtype=np.float64)
    mapped[valid] = lower + (finite[valid] - minimum) / (maximum - minimum) * (
        upper - lower
    )
    return mapped, minimum, maximum


def _plot_classic_timeline(
    series: base.SeriesData,
    events: pd.DatetimeIndex,
    audits: Sequence[base.WindowAudit],
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    focal_event: pd.Timestamp,
    figure: plt.Figure,
    subplot_spec: Any,
) -> tuple[plt.Axes, plt.Axes]:
    """Overlay the real series and sliding windows as in the legacy panel."""

    axis = figure.add_subplot(subplot_spec)
    window_count = len(audits)
    y_positions = np.arange(window_count, 0, -1, dtype=float)
    lower, upper = 0.45, window_count + 0.42
    mapped, db_minimum, db_maximum = _map_db_to_window_axis(
        series.values, lower=lower, upper=upper
    )

    axis.plot(
        series.calendar,
        mapped,
        color=base.BLUE,
        linewidth=1.75,
        alpha=0.88,
        marker="o",
        markersize=2.0,
        markeredgecolor="white",
        markeredgewidth=0.25,
        zorder=1,
    )

    context_face = mcolors.to_rgba("#D6EAF8", 0.42)
    context_edge = mcolors.to_rgba("#5DADE2", 0.75)
    decision_face = mcolors.to_rgba(base.RED, 0.16)
    decision_edge = mcolors.to_rgba(base.RED, 0.76)
    bar_height = 0.56
    for audit, y_position in zip(audits, y_positions):
        selected = audit.logical_id in {outside.logical_id, inside.logical_id}
        context_linewidth = 1.05 if selected else 0.65
        context_edge_selected = (
            mcolors.to_rgba(base.ORANGE, 0.95)
            if audit.logical_id == outside.logical_id
            else (
                mcolors.to_rgba(base.BLUE, 0.95)
                if audit.logical_id == inside.logical_id
                else context_edge
            )
        )
        axis.add_patch(
            Rectangle(
                (mdates.date2num(audit.window_start), y_position - bar_height / 2),
                mdates.date2num(base._window_end_boundary(audit))
                - mdates.date2num(audit.window_start),
                bar_height,
                facecolor=context_face,
                edgecolor=context_edge_selected,
                linewidth=context_linewidth,
                zorder=2,
            )
        )
        axis.add_patch(
            Rectangle(
                (mdates.date2num(audit.decision_start), y_position - bar_height / 2),
                mdates.date2num(base._decision_end_boundary(audit))
                - mdates.date2num(audit.decision_start),
                bar_height,
                facecolor=decision_face,
                edgecolor=decision_edge,
                linewidth=0.75,
                zorder=3,
            )
        )
        label_color = (
            base.ORANGE
            if audit.logical_id == outside.logical_id
            else base.BLUE if audit.logical_id == inside.logical_id else base.INK
        )
        axis.text(
            audit.window_start - pd.DateOffset(months=1),
            y_position,
            _window_label(audit),
            ha="right",
            va="center",
            fontsize=6.8,
            color=label_color,
            fontweight="bold" if selected else "normal",
            clip_on=False,
            zorder=5,
        )

    focal_period = pd.Period(focal_event, freq="M")
    for event in events:
        event_index = int(np.argmin(np.abs(series.calendar - event)))
        event_y = float(mapped[event_index])
        is_focal = pd.Period(event, freq="M") == focal_period
        marker_color = base.RED if is_focal else base.GREY
        axis.scatter(
            event,
            event_y + 0.20,
            marker="v",
            s=32 if is_focal else 24,
            color=marker_color,
            edgecolor="white",
            linewidth=0.55,
            zorder=7,
            clip_on=False,
        )
        axis.text(
            event,
            event_y + 0.47,
            event.strftime("%Y-%m"),
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=marker_color,
            fontweight="bold" if is_focal else "normal",
            zorder=7,
        )

    axis.set_xlim(
        series.calendar.min() - pd.DateOffset(months=6),
        series.calendar.max() + pd.DateOffset(months=6),
    )
    # Explicitly mark the extra months in the expanded archive as context;
    # they are not part of the formal 2017--2025 detection interval.
    if series.calendar.min().year <= 2016:
        axis.text(
            pd.Timestamp("2016-07-01"), window_count + 0.88,
            "2016 context", ha="center", va="top", fontsize=6.0,
            color=base.GREY, style="italic", clip_on=False,
        )
    if series.calendar.max() >= pd.Timestamp("2026-06-01"):
        axis.text(
            pd.Timestamp("2026-03-01"), window_count + 0.88,
            "2026 H1 context", ha="center", va="top", fontsize=6.0,
            color=base.GREY, style="italic", clip_on=False,
        )
    axis.set_ylim(0.0, window_count + 1.08)
    axis.set_yticks([])
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlabel("Year")
    axis.grid(axis="x", color=base.GRID, linewidth=0.45, linestyle="--", alpha=0.7)
    axis.spines[["top", "left", "right"]].set_visible(False)

    db_axis = axis.twinx()
    db_axis.set_ylim(axis.get_ylim())
    first_tick = np.ceil(db_minimum / 2.0) * 2.0
    last_tick = np.floor(db_maximum / 2.0) * 2.0
    db_ticks = np.arange(first_tick, last_tick + 0.1, 2.0)
    mapped_ticks = lower + (db_ticks - db_minimum) / (db_maximum - db_minimum) * (
        upper - lower
    )
    db_axis.set_yticks(mapped_ticks)
    db_axis.set_yticklabels([f"{value:.0f}" for value in db_ticks])
    db_axis.set_ylabel("VV (dB)")
    db_axis.spines[["top", "left", "right"]].set_visible(False)
    db_axis.tick_params(axis="y", length=0, pad=2)

    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=base.BLUE,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="Polygon-first VV median",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color=base.RED,
                linewidth=0,
                markersize=5,
                label="Reference disturbance",
            ),
            Patch(
                facecolor=context_face,
                edgecolor=context_edge,
                label="48-month analysis window",
            ),
            Patch(
                facecolor=decision_face,
                edgecolor=decision_edge,
                label="24-month central decision interval",
            ),
        ],
        loc="lower left",
        ncol=1,
        fontsize=6.2,
        frameon=True,
        framealpha=0.88,
        edgecolor="#C8CDD0",
    )
    axis.text(
        -0.075,
        1.08,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    return axis, db_axis


def _plot_refined_contrast(
    series: base.SeriesData,
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    figure: plt.Figure,
    subplot_spec: Any,
) -> tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    axes = base._plot_contrast(series, outside, inside, figure, subplot_spec)
    out_series, in_series, out_cusum, in_cusum = axes
    out_series.set_title(
        rf"$W_{{{outside.logical_id}}}$: outside the central decision interval",
        fontsize=7.7,
        fontweight="bold",
        color=base.ORANGE,
        pad=4,
    )
    in_series.set_title(
        rf"$W_{{{inside.logical_id}}}$: inside the central decision interval",
        fontsize=7.7,
        fontweight="bold",
        color=base.BLUE,
        pad=4,
    )
    for axis in (out_cusum, in_cusum):
        for annotation in axis.texts:
            annotation.set_text(
                annotation.get_text()
                .replace("CUSUM maximum:", "CUSUM maximum sample:")
                .replace("reported transition:", "reported disturbance:")
            )
    return axes


def _plot_clean_permutation(
    audit: base.WindowAudit,
    config: base.DetectorConfig,
    axis: plt.Axes,
) -> None:
    null = audit.null_amplitudes
    axis.hist(
        null,
        bins=32,
        density=True,
        color="#8DBBDD",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
        label="Independent-permutation null",
    )
    threshold = float(
        np.quantile(null, config.confidence_threshold, method="higher")
    )
    axis.axvline(
        threshold,
        color=base.GREY,
        linestyle=":",
        linewidth=1.2,
        label=f"{config.confidence_threshold:.3f} null quantile",
    )
    axis.axvline(
        audit.cusum_amplitude,
        color=base.RED,
        linewidth=1.8,
        label=rf"Observed $Q={audit.cusum_amplitude:.2f}$",
    )
    axis.set_xlabel("CUSUM amplitude, $Q$")
    axis.set_ylabel("Density")
    axis.grid(axis="y", color=base.GRID, linewidth=0.45, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    right = max(float(audit.cusum_amplitude) * 1.42, float(np.max(null)) * 1.25)
    axis.set_xlim(max(0.0, float(np.min(null)) * 0.75), right)
    axis.legend(loc="upper right", fontsize=6.8, frameon=False)
    axis.text(
        -0.075,
        1.10,
        "(c)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def render_refined_figure(
    series: base.SeriesData,
    events: pd.DatetimeIndex,
    audits: Sequence[base.WindowAudit],
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    config: base.DetectorConfig,
    focal_event: pd.Timestamp,
) -> tuple[plt.Figure, dict[str, Any]]:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Keep every rendered parent text glyph at or above the publication
    # minimum.  Matplotlib mathtext scales subscripts/superscripts below the
    # parent size, so the compact base panel labels are promoted before export.
    matplotlib.rcParams["svg.fonttype"] = "none"
    figure = plt.figure(figsize=(7.15, 7.0), constrained_layout=False)
    outer = GridSpec(
        3,
        1,
        figure=figure,
        height_ratios=[1.82, 3.05, 1.45],
        hspace=0.42,
        left=0.095,
        right=0.955,
        top=0.965,
        bottom=0.07,
    )
    timeline_axis, db_axis = _plot_classic_timeline(
        series, events, audits, outside, inside, focal_event, figure, outer[0, 0]
    )
    contrast_axes = _plot_refined_contrast(
        series, outside, inside, figure, outer[1, 0]
    )
    permutation_axis = figure.add_subplot(outer[2, 0])
    _plot_clean_permutation(inside, config, permutation_axis)
    for text in figure.findobj(match=Text):
        if text.get_fontsize() < 7.2:
            text.set_fontsize(7.2)
    diagnostics = {
        "layout_version": LAYOUT_VERSION,
        "figure_size_inches": [float(value) for value in figure.get_size_inches()],
        "panel_a_window_labels": [_window_label(audit) for audit in audits],
        "panel_a_decision_fill_alpha": 0.16,
        "panel_b_titles": [
            contrast_axes[0].get_title(),
            contrast_axes[1].get_title(),
        ],
        "panel_c_gate_box_present": False,
        "panel_c_legend_location": "upper right",
        "axes_count": len(figure.axes),
        "minimum_parent_text_size_pt": min(
            float(text.get_fontsize()) for text in figure.findobj(match=Text)
        ),
        "timeline_axis": timeline_axis.get_position().bounds,
        "db_axis": db_axis.get_position().bounds,
    }
    return figure, diagnostics


def _publication_caption(
    series: base.SeriesData,
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    config: base.DetectorConfig,
) -> str:
    below = int(np.sum(inside.null_amplitudes < inside.cusum_amplitude))
    return (
        "Windowed CUSUM workflow illustrated with the Polygon-first VV median "
        f"for reference polygon {series.historical_id}. (a) The monthly series "
        f"and overlapping {config.window_length}-month analysis windows; W_i denotes "
        f"the i-th window and the red transparent segment denotes its "
        f"{config.window_length - 2 * config.step}-month central decision interval. "
        f"(b) The same CUSUM boundary is outside the central interval of "
        f"{outside.label} but inside that of {inside.label}. The CUSUM maximum sample "
        f"({inside.candidate_sample_month:%b %Y}) is the implementation-defined "
        f"CUSUM boundary sample and is also the month reported by the production "
        f"detector; that sample is included in the post-change slice used for "
        f"the directional gate. (c) Empirical null distribution from {config.permutations} independent "
        f"within-window permutations; {below}/{config.permutations} null amplitudes "
        f"were below the observed Q (plus-one p={inside.plus_one_p_value:.4f})."
    )


def _write_refined_readme(
    path: Path,
    *,
    output_dir: Path,
    series: base.SeriesData,
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    config: base.DetectorConfig,
) -> None:
    try:
        target = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        target = output_dir
    command = (
        "python analysis/workflows/build_windowed_cusum_concept_refined.py "
        f'--output-dir "{target}"'
    )
    caption = _publication_caption(series, outside, inside, config)
    text = f"""# Refined reproducible Windowed CUSUM concept figure

This code-only figure restores the compact overlaid timeline design of the
legacy panel while retaining the frozen production configuration and production
month semantics. The archived Figure 3 and the v3 reconstruction are untouched.

## Reproduction command

```powershell
{command}
```

The workflow refuses to overwrite existing files. Use a new output directory
for a later revision.

## Band choice

VV is shown because it is the frozen primary detector. On the fixed reference
cohort, VV achieved precision 0.752, recall 0.831, and F1 0.790, whereas the
VH-only F1 was 0.697. VH remains useful as corroborating evidence rather than
as the primary series in this method illustration.

## Suggested caption

{caption}
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = base.parse_args(argv)
    inputs = [args.metadata, args.series, args.reference_events, args.config]
    base._require_paths(inputs)
    output_dir = args.output_dir.resolve()
    pdf_path = output_dir / "windowed_cusum_concept_refined.pdf"
    png_path = output_dir / "windowed_cusum_concept_refined.png"
    svg_path = output_dir / "windowed_cusum_concept_refined.svg"
    audit_path = output_dir / "windowed_cusum_window_audit.csv"
    manifest_path = output_dir / "windowed_cusum_concept_refined_manifest.json"
    readme_path = output_dir / "README.md"
    outputs = [pdf_path, png_path, svg_path, audit_path, manifest_path, readme_path]
    base._refuse_existing(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = base.load_detector_config(args.config)
    series = base.load_polygon_series(
        args.metadata, args.series, args.sample_id, args.historical_id
    )
    events = base.load_reference_events(
        args.reference_events, args.sample_id, args.historical_id
    )
    focal_event = pd.Period(args.focal_event, freq="M").to_timestamp()
    if pd.Period(focal_event, freq="M") not in {
        pd.Period(event, freq="M") for event in events
    }:
        raise ValueError(f"Focal event {args.focal_event} is not a reference event")
    all_windows = list(
        base.generate_windows(len(series.values), config.window_length, config.step)
    )
    window_coverage = []
    for index, window in enumerate(all_windows, start=1):
        length = int(window[1] - window[0])
        window_coverage.append(
            {
                "logical_id": index,
                "start_index": int(window[0]),
                "end_index": int(window[1]),
                "length": length,
                "kind": "full" if length == config.window_length else "edge_context",
            }
        )
    if getattr(args, "full_windows_only", False):
        full_windows = [
            window for window in all_windows
            if int(window[1] - window[0]) == config.window_length
        ]
        audits = [
            base.audit_window(series, config, logical_id, window)
            for logical_id, window in enumerate(full_windows, start=1)
        ]
    else:
        audits = base.build_window_audits(series, config)
    outside, inside = base.choose_contrast_pair(audits, focal_event)

    figure, diagnostics = render_refined_figure(
        series, events, audits, outside, inside, config, focal_event
    )
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    base.write_audit_csv(audit_path, audits)
    _write_refined_readme(
        readme_path,
        output_dir=output_dir,
        series=series,
        outside=outside,
        inside=inside,
        config=config,
    )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layout_version": LAYOUT_VERSION,
        "purpose": "Code-only refined replacement for the legacy Figure 3.",
        "workflow": base._file_record(Path(__file__)),
        "workflow_dependencies": {
            "base_workflow": base._file_record(Path(base.__file__)),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "dpi_png": int(args.dpi),
        },
        "input_semantics": {
            "representation": "Polygon-first monthly median VV",
            "sample_id": series.sample_id,
            "historical_polygon_id": series.historical_id,
            "calendar": [
                series.calendar.min().strftime("%Y-%m"),
                series.calendar.max().strftime("%Y-%m"),
            ],
            "reference_events": [event.strftime("%Y-%m") for event in events],
            "focal_reference_event": focal_event.strftime("%Y-%m"),
        },
        "detector_config": asdict(config),
        "month_semantics": {
            "cusum_maximum_sample": "CUSUM maximum sample at the implementation-defined change boundary.",
            "reported_disturbance_month": "Calendar month of the CUSUM maximum sample, matching production core.lr_cusum.",
            "directional_split": "window[:local_index] versus window[local_index:]; the boundary sample belongs to post.",
        },
        "selected_contrast": {
            "outside": base._audit_record(outside),
            "inside": base._audit_record(inside),
        },
        "window_coverage": {
            "all_generated": window_coverage,
            "full_windows_only": bool(getattr(args, "full_windows_only", False)),
            "edge_windows_are_context_only": bool(getattr(args, "full_windows_only", False)),
        },
        "layout_diagnostics": diagnostics,
        "publication_caption": _publication_caption(
            series, outside, inside, config
        ),
        "inputs": {path.name: base._file_record(path) for path in inputs},
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"] = {
        path.name: base._file_record(path)
        for path in (pdf_path, png_path, svg_path, audit_path, readme_path)
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    print(f"Audit: {audit_path}")
    print(f"Manifest: {manifest_path}")
    print(f"README: {readme_path}")
    print(
        f"Contrast: {outside.label} outside vs {inside.label} inside; "
        f"reported disturbance={inside.reported_transition_month:%Y-%m}; "
        f"p+1={inside.plus_one_p_value:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
