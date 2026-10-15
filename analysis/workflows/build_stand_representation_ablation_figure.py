"""Build a compact, publication-ready stand-representation ablation figure.

The figure combines four panels that answer different parts of the same
question:

* panels (a) and (b): the same polygon's VV and VH trajectories under the
  available pre-detection single-trajectory rules, including the pixel-level
  interquartile envelopes;
* panel (c): precision, recall, and F1 for all representation routes
  under the frozen detector;
* panel (d): paired spatial-block bootstrap differences in F1 relative to the
  production Polygon-first median.

The representation metrics deliberately come from the frozen route-comparison
archive.  This is a sensitivity experiment, not a second model-selection
run.  The nested component-ablation results in the manuscript remain the
primary detector experiment and are not overwritten by this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAJECTORIES = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "method_figure"
    / "windowed_cusum_concept_20260830_v8_all_aggregation_trajectories_qa"
    / "aggregation_trajectories.csv"
)
DEFAULT_METRICS = (
    REPO_ROOT
    / "outputs"
    / "notebook_complete_comparison_40tiles_validated_v2_prospective_lock"
    / "complete_fixed_metrics.csv"
)
DEFAULT_BOOTSTRAP = (
    REPO_ROOT
    / "outputs"
    / "notebook_complete_comparison_40tiles_validated_v2_prospective_lock"
    / "complete_paired_bootstrap.csv"
)

INK = "#17212B"
GRID = "#D9E0E5"
BISHAN_GREEN = "#77964A"
GARDENIA_YELLOW = "#F2B134"
SUPPORT_BLUE = "#3B6FB6"
SUPPORT_PURPLE = "#7562A8"
SUPPORT_TEAL = "#2F6F89"
GREY = "#667085"
IQR = "#AAB4BE"
EVENT_GREY = "#555B61"
PALE_GREEN = "#EEF2E5"
METRIC_COLORS = {
    "Precision": BISHAN_GREEN,
    "Recall": GARDENIA_YELLOW,
    "F1": SUPPORT_BLUE,
}

ROUTE_ORDER = [
    "frozen_median",
    "frozen_mean",
    "frozen_mode",
    "frozen_center",
    "connected_r0.2",
    "temporal_dbscan_r0.4",
    "space_time_dbscan_r0.4",
]
PRODUCTION_DISPLAY_ORDER = [
    "frozen_mode",
    "frozen_mean",
    "frozen_median",
    "frozen_center",
    "connected_r0.2",
    "temporal_dbscan_r0.4",
    "space_time_dbscan_r0.4",
]
DISPLAY_ORDER = list(PRODUCTION_DISPLAY_ORDER)
DISPLAY_BOUNDARIES = (2.5, 3.5)
ROUTE_LABELS = {
    "frozen_median": "Polygon-first median",
    "frozen_mean": "Polygon-first mean",
    "frozen_mode": "Polygon-first mode",
    "frozen_center": "Single-cell spatial medoid",
    "connected_r0.2": "Pixel-first connected support",
    "temporal_dbscan_r0.4": "Pixel-first temporal DBSCAN",
    "space_time_dbscan_r0.4": "Pixel-first space-time DBSCAN",
}
PAIR_RIGHT = {
    "frozen_mean": "frozen_mean",
    "frozen_mode": "frozen_mode",
    "frozen_center": "frozen_center",
    "connected_r0.2": "fixed_connected_component",
    "temporal_dbscan_r0.4": "fixed_temporal_dbscan",
    "space_time_dbscan_r0.4": "fixed_space_time_dbscan",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": _sha256(path)}


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required input(s) missing:\n" + "\n".join(missing))


def _refuse(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing output(s):\n" + "\n".join(existing))


def _load_trajectories(path: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    if "month" not in frame:
        raise ValueError(f"Trajectory source has no month column: {path}")
    calendar = pd.DatetimeIndex(pd.to_datetime(frame.pop("month"), format="%Y-%m"))
    required = [
        f"{band}_{suffix}"
        for band in ("VV", "VH")
        for suffix in (
            "polygon_median",
            "polygon_mean",
            "polygon_mode",
            "representative_pixel",
            "pixel_median_diagnostic",
            "pixel_q25",
            "pixel_q75",
        )
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Trajectory source is missing columns: {missing}")
    manifest_path = path.with_name("aggregation_trajectories_manifest.json")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return calendar, frame, source_manifest


def _load_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    mask = (
        frame["protocol"].astype(str).eq("frozen_lr")
        & frame["role"].astype(str).eq("all")
        & pd.to_numeric(frame["tolerance_months"], errors="coerce").eq(2)
        & frame["name"].isin(ROUTE_ORDER)
    )
    selected = frame.loc[mask].copy()
    if len(selected) != len(ROUTE_ORDER) or set(selected["name"]) != set(ROUTE_ORDER):
        raise ValueError(
            "The frozen representation archive must contain exactly one row for each route; "
            f"found {sorted(selected['name'].unique())}"
        )
    selected = selected.set_index("name").loc[ROUTE_ORDER].reset_index()
    numeric = ["precision", "recall", "f1", "temporal_mae_months"]
    selected[numeric] = selected[numeric].apply(pd.to_numeric, errors="raise")
    expected_f1 = 2.0 * selected["precision"] * selected["recall"] / (
        selected["precision"] + selected["recall"]
    )
    if not np.allclose(selected["f1"], expected_f1, rtol=0.0, atol=5e-9):
        raise ValueError("Stored F1 values do not match the harmonic mean of precision and recall")
    return selected


def _load_deltas(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["comparison_group"].astype(str).eq("frozen_lr_spatial_ablation")
        & frame["left"].astype(str).eq("frozen_median")
    ].copy()
    rows: list[dict[str, Any]] = []
    for route in ROUTE_ORDER[1:]:
        right = PAIR_RIGHT[route]
        match = selected.loc[selected["right"].astype(str).eq(right)]
        if len(match) != 1:
            raise ValueError(f"Expected one paired bootstrap row for {route}; found {len(match)}")
        row = match.iloc[0]
        # The archive stores median-minus-alternative.  The figure uses the
        # manuscript convention alternative-minus-median.
        rows.append(
            {
                "name": route,
                "estimate": -float(row["median"]),
                "lower": -float(row["q975"]),
                "upper": -float(row["q025"]),
            }
        )
    result = pd.DataFrame(rows)
    invalid = result.loc[
        (result["lower"] > result["estimate"]) | (result["estimate"] > result["upper"])
    ]
    if not invalid.empty:
        raise ValueError(f"Invalid paired-bootstrap interval(s): {invalid['name'].tolist()}")
    return result


def _audit_trajectory(frame: pd.DataFrame) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for band in ("VV", "VH"):
        polygon = frame[f"{band}_polygon_median"].to_numpy(dtype=float)
        diagnostic = frame[f"{band}_pixel_median_diagnostic"].to_numpy(dtype=float)
        joint = np.isfinite(polygon) & np.isfinite(diagnostic)
        if not np.any(joint):
            raise ValueError(
                f"{band} Polygon-first and diagnostic medians have no jointly finite months"
            )
        differences = np.abs(polygon[joint] - diagnostic[joint])
        band_audit = {
            "joint_finite_months": int(joint.sum()),
            "max_abs_difference_db": float(np.max(differences)),
            "mean_abs_difference_db": float(np.mean(differences)),
            "diagnostic_duplicates_polygon_median": bool(
                np.allclose(polygon[joint], diagnostic[joint], rtol=0.0, atol=1e-12)
            ),
        }
        if not band_audit["diagnostic_duplicates_polygon_median"]:
            raise ValueError(
                f"The diagnostic {band} median may be omitted only when it duplicates "
                "the Polygon-first median; the supplied trajectories differ"
            )
        audit[band] = band_audit
    return audit


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "lines.solid_capstyle": "round",
        }
    )


def _plot_trajectory(ax: plt.Axes, calendar: pd.DatetimeIndex, frame: pd.DataFrame, source_manifest: dict[str, Any]) -> None:
    x = calendar.to_pydatetime()
    ax.fill_between(x, frame["VV_pixel_q25"], frame["VV_pixel_q75"], color=IQR, alpha=0.24, linewidth=0, zorder=1)
    ax.plot(x, frame["VV_polygon_median"], color=BISHAN_GREEN, lw=1.75, zorder=5)
    ax.plot(x, frame["VV_polygon_mean"], color=SUPPORT_BLUE, lw=1.05, zorder=4)
    ax.plot(x, frame["VV_polygon_mode"], color=SUPPORT_PURPLE, lw=1.05, ls=(0, (3, 1.5)), zorder=4)
    ax.plot(x, frame["VV_representative_pixel"], color=SUPPORT_TEAL, lw=1.05, ls=(0, (5, 1.5)), zorder=4)

    # The source trajectory contains reference-event months in its manifest;
    # plotting them here keeps the trajectory panel auditable.
    event_months = source_manifest.get("semantics", {}).get("reference_events", [])
    events = [pd.Timestamp(value) for value in event_months]
    for event in events:
        ax.axvline(event, color=EVENT_GREY, lw=0.7, ls=(0, (3, 2)), alpha=0.72, zorder=2)

    ax.set_title("(a) Alternative summaries of the same plantation stand", loc="left", pad=24.0, color=INK, fontweight="bold")
    ax.set_ylabel("VV backscatter (dB)")
    ax.set_xlabel("Observation month")
    ax.set_xlim(calendar.min(), calendar.max())
    plotted_columns = [
        "VV_polygon_median",
        "VV_polygon_mean",
        "VV_polygon_mode",
        "VV_representative_pixel",
        "VV_pixel_q25",
        "VV_pixel_q75",
    ]
    finite = frame[plotted_columns].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite):
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max(0.7, 0.05 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Line2D([], [], color=BISHAN_GREEN, lw=1.75, label="Median"),
        Line2D([], [], color=SUPPORT_BLUE, lw=1.05, label="Mean"),
        Line2D([], [], color=SUPPORT_PURPLE, lw=1.05, ls=(0, (3, 1.5)), label="Histogram mode"),
        Line2D([], [], color=SUPPORT_TEAL, lw=1.05, ls=(0, (5, 1.5)), label="Representative cell"),
        Patch(facecolor=IQR, edgecolor="none", alpha=0.36, label="Pixel IQR (P25–P75)"),
    ]
    if events:
        handles.append(Line2D([], [], color=EVENT_GREY, lw=0.7, ls=(0, (3, 2)), label="Reference event"))
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005, 1.0, 0.08),
        mode="expand",
        ncol=len(handles),
        frameon=False,
        handlelength=1.7,
        columnspacing=0.8,
        borderaxespad=0.0,
        labelspacing=0.2,
    )


def _plot_band_trajectory(
    ax: plt.Axes,
    calendar: pd.DatetimeIndex,
    frame: pd.DataFrame,
    source_manifest: dict[str, Any],
    *,
    band: str,
    show_x_labels: bool,
) -> None:
    """Plot one polarization while keeping method encodings identical."""

    if band not in {"VV", "VH"}:
        raise ValueError(f"Unsupported polarization: {band}")
    x = calendar.to_pydatetime()
    ax.fill_between(
        x,
        frame[f"{band}_pixel_q25"],
        frame[f"{band}_pixel_q75"],
        color=IQR,
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        x,
        frame[f"{band}_polygon_median"],
        color=BISHAN_GREEN,
        lw=1.75,
        marker="o",
        markersize=1.8,
        markevery=3,
        zorder=5,
    )
    ax.plot(
        x,
        frame[f"{band}_polygon_mean"],
        color=SUPPORT_BLUE,
        lw=1.05,
        ls=(0, (5, 2)),
        zorder=4,
    )
    ax.plot(
        x,
        frame[f"{band}_polygon_mode"],
        color=SUPPORT_PURPLE,
        lw=1.05,
        ls=(0, (2.4, 1.5)),
        zorder=4,
    )
    ax.plot(
        x,
        frame[f"{band}_representative_pixel"],
        color=SUPPORT_TEAL,
        lw=1.05,
        ls=(0, (7, 2, 1.5, 2)),
        zorder=4,
    )

    event_months = source_manifest.get("semantics", {}).get("reference_events", [])
    events = [pd.Timestamp(value) for value in event_months]
    for event in events:
        ax.axvline(
            event,
            ymin=0.0,
            ymax=1.0,
            color=EVENT_GREY,
            lw=0.7,
            ls=(0, (3, 2)),
            alpha=0.72,
            zorder=2,
        )
        if band == "VV":
            ax.annotate(
                event.strftime("%b. %Y"),
                xy=(event, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=EVENT_GREY,
                clip_on=False,
            )

    ax.set_ylabel(f"{band} backscatter\n(dB)")
    ax.set_xlabel("")
    ax.set_xlim(calendar.min(), calendar.max())
    plotted_columns = [
        f"{band}_polygon_median",
        f"{band}_polygon_mean",
        f"{band}_polygon_mode",
        f"{band}_representative_pixel",
        f"{band}_pixel_q25",
        f"{band}_pixel_q75",
    ]
    finite = frame[plotted_columns].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if band == "VV":
        ax.set_ylim(-12.0, -2.0)
        ax.set_yticks(np.arange(-12.0, 0.0, 2.0))
    else:
        ax.set_ylim(-23.0, -10.0)
        ax.set_yticks(np.arange(-22.0, -8.0, 2.0))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelbottom=show_x_labels)
    ax.grid(axis="y", color=GRID, lw=0.45, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def _trajectory_legend_handles() -> list[Line2D | Patch]:
    """Return the shared legend for the two single-trajectory panels."""

    return [
        Line2D([], [], color=BISHAN_GREEN, lw=1.75, marker="o", markersize=2.5, label="Polygon-first median"),
        Line2D([], [], color=SUPPORT_BLUE, lw=1.05, ls=(0, (5, 2)), label="Polygon-first mean"),
        Line2D([], [], color=SUPPORT_PURPLE, lw=1.05, ls=(0, (2.4, 1.5)), label="Polygon-first histogram mode"),
        Line2D([], [], color=SUPPORT_TEAL, lw=1.05, ls=(0, (7, 2, 1.5, 2)), label="Single-cell spatial medoid"),
        Patch(facecolor=IQR, edgecolor="none", alpha=0.34, label="Interior-cell IQR (P25–P75)"),
        Line2D([], [], color=EVENT_GREY, lw=0.7, ls=(0, (3, 2)), label="Reference event"),
    ]


def _trajectory_legend_handles_compact() -> list[Line2D | Patch]:
    """Return the compact in-panel legend used by the final layout."""

    return [
        Line2D([], [], color=BISHAN_GREEN, lw=1.75, marker="o", markersize=2.5, label="Polygon-first median"),
        Line2D([], [], color=SUPPORT_BLUE, lw=1.05, ls=(0, (5, 2)), label="Polygon-first mean"),
        Line2D([], [], color=SUPPORT_PURPLE, lw=1.05, ls=(0, (2.4, 1.5)), label="Polygon-first histogram mode"),
        Line2D([], [], color=SUPPORT_TEAL, lw=1.05, ls=(0, (7, 2, 1.5, 2)), label="Single-cell spatial medoid"),
        Patch(facecolor=IQR, edgecolor="none", alpha=0.34, label="Interior-cell IQR (P25–P75)"),
    ]


def _metric_legend_handles() -> list[Line2D]:
    """Return the three vertically stacked metric legend handles."""

    return [
        Line2D([], [], marker="o", ls="none", markersize=5.2, markerfacecolor=METRIC_COLORS["Precision"], markeredgecolor="white", markeredgewidth=0.4, label="Precision"),
        Line2D([], [], marker="s", ls="none", markersize=5.2, markerfacecolor=METRIC_COLORS["Recall"], markeredgecolor="white", markeredgewidth=0.4, label="Recall"),
        Line2D([], [], marker="D", ls="none", markersize=5.2, markerfacecolor=METRIC_COLORS["F1"], markeredgecolor="white", markeredgewidth=0.4, label="F1"),
    ]


def _align_time_axes(
    fig: plt.Figure,
    top_axes: Sequence[plt.Axes],
    bottom_axes: Sequence[plt.Axes],
) -> Bbox:
    """Give both time panels identical x-axis endpoints within the full figure."""

    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        target = Bbox.union([axis.get_tightbbox(renderer) for axis in bottom_axes])
        target = target.transformed(fig.transFigure.inverted())
        left_decorations = []
        right_decorations = []
        for axis in top_axes:
            position = axis.get_position()
            tight = axis.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            left_decorations.append(position.x0 - tight.x0)
            right_decorations.append(tight.x1 - position.x1)
        new_left = target.x0 + max(left_decorations)
        new_right = target.x1 - max(right_decorations)
        if new_right <= new_left:
            raise RuntimeError("Cannot align the common time-axis span to the bottom-panel union.")
        for axis in top_axes:
            position = axis.get_position()
            axis.set_position([new_left, position.y0, new_right - new_left, position.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return Bbox.union([axis.get_tightbbox(renderer) for axis in bottom_axes]).transformed(fig.transFigure.inverted())


def _plot_metrics(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    metrics = metrics.set_index("name").loc[DISPLAY_ORDER].reset_index()
    y = np.arange(len(metrics))
    median_y = DISPLAY_ORDER.index("frozen_median")
    ax.axhspan(median_y - 0.43, median_y + 0.43, color=PALE_GREEN, alpha=0.95, zorder=0)
    for yi, (_, row) in zip(y, metrics.iterrows()):
        lo = min(float(row["precision"]), float(row["recall"]), float(row["f1"]))
        hi = max(float(row["precision"]), float(row["recall"]), float(row["f1"]))
        ax.plot([lo, hi], [yi, yi], color="#AAB4BE", lw=1.15, zorder=1)
    ax.scatter(
        metrics["precision"],
        y,
        s=25,
        marker="o",
        color=METRIC_COLORS["Precision"],
        edgecolor="white",
        linewidth=0.4,
        label="Precision",
        zorder=3,
    )
    ax.scatter(
        metrics["recall"],
        y,
        s=25,
        marker="s",
        color=METRIC_COLORS["Recall"],
        edgecolor="white",
        linewidth=0.4,
        label="Recall",
        zorder=3,
    )
    ax.scatter(
        metrics["f1"],
        y,
        s=27,
        marker="D",
        color=METRIC_COLORS["F1"],
        edgecolor="white",
        linewidth=0.4,
        label="F1",
        zorder=3,
    )
    for boundary in DISPLAY_BOUNDARIES:
        ax.axhline(boundary, color=GRID, lw=0.55, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([ROUTE_LABELS[name] for name in metrics["name"]])
    ax.invert_yaxis()
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    ax.set_xlim(0.30, 1.0)
    ax.set_xlabel("Event-level score")
    ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_deltas(ax: plt.Axes, metrics: pd.DataFrame, deltas: pd.DataFrame) -> None:
    y = np.arange(len(DISPLAY_ORDER))
    lookup = deltas.set_index("name")
    median_y = DISPLAY_ORDER.index("frozen_median")
    ax.axhspan(median_y - 0.43, median_y + 0.43, color=PALE_GREEN, alpha=0.95, zorder=0)
    for boundary in DISPLAY_BOUNDARIES:
        ax.axhline(boundary, color=GRID, lw=0.55, zorder=0)
    ax.axvline(0.0, color="#89939E", lw=0.8, zorder=1)
    ax.scatter(
        [0.0],
        [median_y],
        s=27,
        marker="D",
        facecolor="white",
        edgecolor=BISHAN_GREEN,
        linewidth=0.9,
        zorder=4,
    )
    median_f1 = float(metrics.loc[metrics["name"].eq("frozen_median"), "f1"].iloc[0])
    ax.text(-0.012, median_y, f"F1 = {median_f1:.3f}", va="center", ha="right", fontsize=6.4, color=INK)
    for yi, name in enumerate(DISPLAY_ORDER):
        if name == "frozen_median":
            continue
        row = lookup.loc[name]
        overlaps_zero = float(row["lower"]) <= 0.0 <= float(row["upper"])
        color = SUPPORT_BLUE if overlaps_zero else SUPPORT_PURPLE
        ax.plot([row["lower"], row["upper"]], [yi, yi], color=color, lw=1.15, zorder=2)
        ax.scatter(
            row["estimate"],
            yi,
            s=23,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(len(DISPLAY_ORDER) - 0.5, -0.5)
    ax.set_xlim(-0.40, 0.085)
    ax.set_xlabel("Paired $\\Delta$F1 (alternative $-$ median)")
    ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _write_readme(
    path: Path,
    trajectory_manifest: dict[str, Any],
    trajectory_audit: dict[str, Any],
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    sample = trajectory_manifest.get("sample", {})
    tile_ids = sample.get("tile_ids", [])
    tile_label = ", ".join(str(value) for value in tile_ids) if tile_ids else "unknown tile"
    pixel_count = sample.get("interior_pixels_total", "unknown")
    rows = [
        "# Stand-representation ablation figure",
        "",
        "This figure is a transparent sensitivity analysis, not a new model-selection run.",
        "",
        "- Panels (a) and (b) use the existing full-interior cache-derived VV and VH trajectories for the illustrative stand. Duplicate Pixel-first median diagnostics are omitted only after numerical identity checks for both polarizations.",
        "- Panel (c) reports precision, recall, and F1; Panel (d) separately owns the paired F1 inference.",
        "- Panels (c) and (d) use the frozen detector route-comparison archive (`protocol=frozen_lr`, 48-month window, 12-month displacement, +/-2-month matching tolerance).",
        "- Panel (d) converts the archive's median-minus-alternative bootstrap contrast to the manuscript convention alternative-minus-median.",
        "- The nested component-ablation results in the manuscript are unchanged.",
        "",
        "## Routes included",
        "",
    ]
    for _, row in metrics.iterrows():
        rows.append(f"- {ROUTE_LABELS[row['name']]}: P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}, MAE={row['temporal_mae_months']:.3f} months.")
    rows.extend(["", "## Paired F1 contrasts", ""])
    for _, row in deltas.iterrows():
        rows.append(f"- {ROUTE_LABELS[row['name']]} minus Polygon-first median: {row['estimate']:+.3f} [{row['lower']:+.3f}, {row['upper']:+.3f}].")
    rows.extend(
        [
            "",
            "## Trajectory identity audit",
            "",
            f"- VV jointly finite months: {trajectory_audit['VV']['joint_finite_months']}.",
            f"- VV maximum absolute difference between the Polygon-first median and the omitted diagnostic median: {trajectory_audit['VV']['max_abs_difference_db']:.12g} dB.",
            f"- VH jointly finite months: {trajectory_audit['VH']['joint_finite_months']}.",
            f"- VH maximum absolute difference between the Polygon-first median and the omitted diagnostic median: {trajectory_audit['VH']['max_abs_difference_db']:.12g} dB.",
            "",
            "## Suggested manuscript caption",
            "",
            "Sensitivity of disturbance detection to stand representation under the frozen Windowed CUSUM configuration. "
            f"(a,b) Alternative monthly VV and VH summaries for an illustrative {tile_label} plantation stand containing {pixel_count} valid polygon-interior grid cells; shading denotes the cell-level P25--P75 range. Vertical dashed lines mark independently interpreted reference events in April 2019, February 2022, and March 2024; their dates are labelled in panel (a), and the shared legend occupies the data-free lower portion of panel (b). "
            "Pixel-first fusion routes retain individual cell detections and therefore do not define a single stand-level input trajectory. "
            "(c) Precision, recall, and F1 under the common 48-month window, 12-month displacement, p_perm <= 0.025, and +/-2-month event-matching tolerance. "
            "(d) Paired F1 differences relative to the Polygon-first median; intervals are 95% complete-spatial-block bootstrap intervals from 5,000 resamples.",
            "",
            "## Provenance",
            "",
            f"Trajectory manifest sample: `{trajectory_manifest.get('sample', {}).get('sample_id', 'unknown')}`.",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument(
        "--reverse-display-order",
        action="store_true",
        help="Reverse the seven rows in panels (c) and (d) and place the metric legend at lower left.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global DISPLAY_ORDER
    args = parse_args(argv)
    DISPLAY_ORDER = (
        list(reversed(PRODUCTION_DISPLAY_ORDER))
        if args.reverse_display_order
        else list(PRODUCTION_DISPLAY_ORDER)
    )
    _require([args.trajectories, args.metrics, args.bootstrap])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "stand_representation_ablation.pdf",
        output_dir / "stand_representation_ablation.png",
        output_dir / "stand_representation_ablation.svg",
        output_dir / "stand_representation_metrics.csv",
        output_dir / "stand_representation_deltas.csv",
        output_dir / "stand_representation_ablation_manifest.json",
        output_dir / "README.md",
        output_dir / "stand_representation_ablation.tiff",
    ]
    _refuse(outputs)

    calendar, trajectory_frame, trajectory_manifest = _load_trajectories(args.trajectories)
    trajectory_audit = _audit_trajectory(trajectory_frame)
    metrics = _load_metrics(args.metrics)
    deltas = _load_deltas(args.bootstrap)
    _style()

    fig = plt.figure(figsize=(7.16, 4.30), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.36, 0.18, 0.72],
        hspace=0.0,
    )
    top = outer[0, 0].subgridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.14,
    )
    bottom = outer[2, 0].subgridspec(
        1,
        12,
        wspace=0.55,
    )
    ax_a_vv = fig.add_subplot(top[0, 0])
    ax_a_vh = fig.add_subplot(top[1, 0], sharex=ax_a_vv)
    ax_c = fig.add_subplot(bottom[0, :7])
    ax_d = fig.add_subplot(bottom[0, 7:], sharey=ax_c)
    _plot_band_trajectory(
        ax_a_vv,
        calendar,
        trajectory_frame,
        trajectory_manifest,
        band="VV",
        show_x_labels=False,
    )
    _plot_band_trajectory(
        ax_a_vh,
        calendar,
        trajectory_frame,
        trajectory_manifest,
        band="VH",
        show_x_labels=True,
    )
    vh_legend_band_ceiling = -20.0
    vh_display_columns = [
        "VH_polygon_median",
        "VH_polygon_mean",
        "VH_polygon_mode",
        "VH_representative_pixel",
        "VH_pixel_q25",
        "VH_pixel_q75",
    ]
    vh_minimum = float(np.nanmin(trajectory_frame[vh_display_columns].to_numpy(dtype=float)))
    if vh_minimum <= vh_legend_band_ceiling:
        raise ValueError(
            "The reserved VH legend band (-22 to -20 dB) is not data-free: "
            f"minimum displayed value is {vh_minimum:.3f} dB"
        )
    trajectory_legend = ax_a_vh.legend(
        handles=_trajectory_legend_handles_compact(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
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
    trajectory_legend.get_frame().set_linewidth(0.45)
    for label in trajectory_legend.get_texts():
        label.set_path_effects(
            [path_effects.withStroke(linewidth=1.25, foreground="white")]
        )
    trajectory_legend.set_zorder(10)
    _plot_metrics(ax_c, metrics)
    _plot_deltas(ax_d, metrics, deltas)
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.09, top=0.94)
    external_bounds = _align_time_axes(fig, [ax_a_vv, ax_a_vh], [ax_c, ax_d])
    label_x = external_bounds.x0
    fig.canvas.draw()
    for axis, label in ((ax_a_vv, "(a)"), (ax_a_vh, "(b)")):
        lower, upper = sorted(axis.get_ylim())
        visible_ticks = [tick for tick in axis.get_yticks() if lower <= tick <= upper]
        top_tick = max(visible_ticks)
        display_y = axis.transData.transform((axis.get_xlim()[0], top_tick))[1]
        figure_y = fig.transFigure.inverted().transform((0.0, display_y))[1]
        fig.text(label_x, figure_y, label, ha="left", va="center", fontsize=8.8, fontweight="bold", color=INK)

    # The lower-panel letters are anchored to the true plotting rectangles,
    # not to tight bounding boxes that also include tick and method labels.
    c_panel_label = ax_c.annotate(
        "(c)",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(3.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
        zorder=11,
    )
    ax_d.annotate(
        "(d)",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(3.0, 0.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=INK,
        zorder=11,
    )
    if args.reverse_display_order:
        metric_legend_loc = "lower left"
        metric_legend_anchor = (0.015, 0.015)
    else:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_right_display = (
            c_panel_label.get_window_extent(renderer).x1 + 4.0 * fig.dpi / 72.0
        )
        legend_anchor_x = ax_c.transAxes.inverted().transform(
            (label_right_display, ax_c.bbox.y1)
        )[0]
        metric_legend_loc = "upper left"
        metric_legend_anchor = (legend_anchor_x, 1.0)
    metric_legend = ax_c.legend(
        handles=_metric_legend_handles(),
        loc=metric_legend_loc,
        bbox_to_anchor=metric_legend_anchor,
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
    metric_legend.get_frame().set_linewidth(0.45)
    metric_legend.set_zorder(10)

    pdf_path, png_path, svg_path = outputs[:3]
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(outputs[7], dpi=600, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    metrics_path, deltas_path = outputs[3:5]
    metrics.to_csv(metrics_path, index=False, float_format="%.9f")
    deltas.to_csv(deltas_path, index=False, float_format="%.9f")
    _write_readme(outputs[6], trajectory_manifest, trajectory_audit, metrics, deltas)
    manifest = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": _record(Path(__file__).resolve()),
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "png_dpi": args.dpi},
        "inputs": {path.name: _record(path) for path in [args.trajectories, args.metrics, args.bootstrap]},
        "protocol": {"trajectory": "Existing cache-derived illustrative VV and VH stand trajectories; duplicate Pixel-first median diagnostics are omitted after band-specific identity audits.", "metrics": "Frozen detector route comparison; 48-month window, 12-month displacement, ±2-month event matching.", "delta": "Alternative minus Polygon-first median; 95% paired complete-spatial-block bootstrap interval."},
        "figure_contract": {
            "claim": "The Polygon-first median preserves a stable stand signal and performs comparably to the strongest alternatives without pixel-support or clustering parameters.",
            "panel_a": "Interpretive VV example of how alternative pre-detection stand summaries change the same polygon trajectory.",
            "panel_b": "Interpretive VH example using the same stand and display semantics.",
            "panel_c": "Operational precision, recall, and F1 comparison across all seven representation routes.",
            "panel_d": "Paired F1 inference relative to the production median.",
        },
        "trajectory_audit": trajectory_audit,
        "routes": ROUTE_LABELS,
        "outputs": {},
    }
    manifest_path = outputs[5]
    manifest["outputs"] = {path.name: _record(path) for path in outputs if path != manifest_path}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in outputs[:3]:
        print(f"{path.suffix.upper().lstrip('.')}: {path}")
    print(f"TIFF: {outputs[7]}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
