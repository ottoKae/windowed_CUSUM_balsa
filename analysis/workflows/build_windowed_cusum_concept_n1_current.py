"""Build the current-data Windowed CUSUM concept figure.

This workflow updates the archived PowerPoint-assembled concept figure using
the frozen 40-tile Polygon-first cache and the N1 temporal-null protocol.  It
keeps the three inferential roles of the original figure:

1. overlapping local windows and their central decision intervals;
2. the same CUSUM boundary rejected/retained by adjacent windows; and
3. finite-sample-corrected independent-permutation screening.

The source series spans January 2016--June 2026.  The formal historical
reporting interval remains 2017--2025; 2016 and 2026 H1 are temporal context.
Outputs are versioned and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.workflows import build_windowed_cusum_concept as base  # noqa: E402
from analysis.workflows import build_windowed_cusum_concept_refined as refined  # noqa: E402


DEFAULT_SAMPLE_ID = None
DEFAULT_HISTORICAL_ID = 1499
DEFAULT_FOCAL_EVENT = "2022-02"
DEFAULT_CACHE = (
    REPO_ROOT
    / "outputs"
    / "balsa_validation_closure_40tiles_validated_v2_strict_decision"
)
DEFAULT_METADATA = DEFAULT_CACHE / "polygon_series_metadata.parquet"
DEFAULT_SERIES = DEFAULT_CACHE / "polygon_series.npz"
DEFAULT_REFERENCE_EVENTS = DEFAULT_CACHE / "reference_events.parquet"
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "outputs"
    / "windowed_cusum_component_ablation_40tile_v1"
    / "candidate_audit"
    / "sliding_candidates.parquet"
)
DEFAULT_N1_SCORES = (
    REPO_ROOT
    / "outputs"
    / "windowed_cusum_temporal_null_audit_40tile_v1"
    / "candidate_scores"
    / "N1_IID_P025.parquet"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT / "configs" / "windowed_cusum_temporal_null_audit_40tile_v1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "jag_20260821"
    / "method_figure"
    / "windowed_cusum_concept_20260903_n1_current_v22_terminology_final"
)

LAYOUT_VERSION = "n1-current-data-bishan-v22"
N1_MODEL_ID = "N1_IID_P025"

# Bishan Gardenia palette. Green carries the observed VV trajectory, yellow
# carries the moving-window geometry, and black is reserved for independent
# reference dates and axes.
BISHAN_GREEN = "#597D23"
BISHAN_GREEN_MID = "#77964A"
BISHAN_GREEN_LIGHT = "#D3EEAC"
BISHAN_YELLOW = "#FBC040"
BISHAN_YELLOW_DARK = "#B37A00"
BISHAN_YELLOW_LIGHT = "#FFF0B8"
BISHAN_BLUE = "#2A6F97"
BISHAN_BLUE_LIGHT = "#DCEAF4"
BISHAN_RED = "#C83E2B"
BISHAN_BLACK = "#1B1B1B"
BISHAN_GREY = "#7B8175"
WINDOW_BLUE_ALPHA = 0.42
DECISION_YELLOW_ALPHA = 0.62


@dataclass(frozen=True)
class N1Protocol:
    window_length: int
    step: int
    minimum_observations: int
    alpha: float
    replicates: int
    near_threshold_low: float
    near_threshold_high: float
    near_threshold_replicates: int
    seed: int
    context_start: str
    context_end: str
    reporting_start_year: int
    reporting_end_year: int


@dataclass(frozen=True)
class PanelWindow:
    """Window geometry shown in panel (a), independent of candidate validity."""

    logical_id: int
    window_start_index: int
    window_end_index: int
    decision_start_index: int
    decision_end_index: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    decision_start: pd.Timestamp
    decision_end: pd.Timestamp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }


def _require_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))


def _refuse_existing(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(existing)
        )


def load_n1_protocol(path: Path) -> N1Protocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    context = payload["context_calendar"]
    reporting = payload["reporting_years"]
    refine = payload["near_threshold_interval"]
    protocol = N1Protocol(
        window_length=int(payload["window_length"]),
        step=int(payload["step"]),
        minimum_observations=int(payload["minimum_observations"]),
        alpha=float(payload["alpha"]),
        replicates=int(payload["replicates"]),
        near_threshold_low=float(refine[0]),
        near_threshold_high=float(refine[1]),
        near_threshold_replicates=int(payload["near_threshold_replicates"]),
        seed=int(payload["seed"]),
        context_start=str(context[0]),
        context_end=str(context[1]),
        reporting_start_year=int(reporting[0]),
        reporting_end_year=int(reporting[1]),
    )
    if protocol.window_length <= 2 * protocol.step:
        raise ValueError("The central decision interval must be non-empty")
    if not 0.0 < protocol.alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return protocol


def build_panel_windows(
    series: base.SeriesData,
    protocol: N1Protocol,
) -> list[PanelWindow]:
    """Return W1--W8 whose central intervals cover 2017--2025.

    W1--W7 have complete 48-month context. W8 starts in January 2023 and
    uses the available record through June 2026; its central interval remains
    the complete January 2024--December 2025 reporting interval. Candidate
    validity is deliberately irrelevant here because panel (a) explains the
    deterministic window geometry, not which windows produce a valid split.
    """

    calendar = series.calendar
    reporting_end = pd.Timestamp(f"{protocol.reporting_end_year}-12-01")
    windows: list[PanelWindow] = []
    for logical_id, start in enumerate(
        range(0, len(calendar), protocol.step), start=1
    ):
        end = min(len(calendar), start + protocol.window_length)
        decision_start = start + protocol.step
        decision_end = min(len(calendar), start + protocol.window_length - protocol.step)
        if decision_start >= decision_end or decision_start >= len(calendar):
            break
        if (
            pd.Timestamp(calendar[decision_start]) > reporting_end
            or pd.Timestamp(calendar[decision_end - 1]) > reporting_end
        ):
            break
        windows.append(
            PanelWindow(
                logical_id=logical_id,
                window_start_index=start,
                window_end_index=end,
                decision_start_index=decision_start,
                decision_end_index=decision_end,
                window_start=pd.Timestamp(calendar[start]),
                window_end=pd.Timestamp(calendar[end - 1]),
                decision_start=pd.Timestamp(calendar[decision_start]),
                decision_end=pd.Timestamp(calendar[decision_end - 1]),
            )
        )
    if [window.logical_id for window in windows] != list(range(1, 9)):
        raise AssertionError("Expected W1--W8 for the frozen 2017--2025 protocol")
    return windows


def load_reference_events(
    path: Path, sample_id: str, historical_id: int
) -> pd.DatetimeIndex:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    mask = frame["sample_id"].astype(str).eq(str(sample_id))
    if "historical_id" in frame.columns:
        mask &= pd.to_numeric(frame["historical_id"], errors="coerce").eq(
            int(historical_id)
        )
    events = pd.DatetimeIndex(
        pd.to_datetime(frame.loc[mask, "event_month"], errors="raise")
    ).sort_values()
    if events.empty:
        raise ValueError(f"No reference events found for {sample_id}")
    return events


def _candidate_row(
    candidates: pd.DataFrame,
    *,
    sample_id: str,
    logical_window_id: int,
) -> pd.Series:
    selected = candidates.loc[
        candidates["sample_id"].astype(str).eq(str(sample_id))
        & candidates["Band"].astype(str).eq("VV_dB")
        & candidates["candidate_scope"].astype(str).eq("sliding")
        & pd.to_numeric(candidates["logical_window_id"], errors="coerce").eq(
            int(logical_window_id)
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one VV candidate for {sample_id}, W{logical_window_id}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def _attach_frozen_n1_score(
    audit: base.WindowAudit,
    candidate: pd.Series,
    scores: pd.DataFrame,
    protocol: N1Protocol,
) -> tuple[base.WindowAudit, pd.Series]:
    candidate_id = str(candidate["candidate_id"])
    selected = scores.loc[scores["candidate_id"].astype(str).eq(candidate_id)]
    if len(selected) != 1:
        raise ValueError(f"Expected one frozen N1 score for {candidate_id}")
    score = selected.iloc[0]

    checks = {
        "window_start_index": audit.window_start_index,
        "window_end_index": audit.window_end_index,
        "decision_start_index": audit.decision_start_index,
        "decision_end_index": audit.decision_end_index,
        "change_index": audit.candidate_sample_index,
    }
    for column, expected in checks.items():
        observed = int(candidate[column])
        if observed != int(expected):
            raise AssertionError(
                f"Current candidate ledger disagrees for {column}: "
                f"{observed} != {expected}"
            )
    if not np.isclose(
        float(candidate["cusum_amplitude"]), audit.cusum_amplitude, atol=1e-9
    ):
        raise AssertionError("CUSUM amplitude differs from the frozen candidate ledger")

    final_replicates = int(score["replicates"])
    seed = int(score["null_seed"])
    return replace(
        audit,
        permutation_seed=seed,
        passes_permutation=bool(score["admitted"]),
        accepted=bool(
            audit.passes_central
            and audit.passes_decrease
            and bool(score["admitted"])
        ),
    ), score


def _exact_null_for_audit(
    series: base.SeriesData,
    audit: base.WindowAudit,
    score: pd.Series,
) -> base.WindowAudit:
    local = series.values[audit.window_start_index : audit.window_end_index]
    clean = np.asarray(local[np.isfinite(local)], dtype=np.float64)
    null = base.permutation_null_amplitudes(
        clean,
        iterations=int(score["replicates"]),
        seed=int(score["null_seed"]),
    )
    exceedances = int(np.sum(null >= audit.cusum_amplitude))
    plus_one_p = float((exceedances + 1) / (len(null) + 1))
    if exceedances != int(score["exceedance_count"]):
        raise AssertionError("Reconstructed null differs from the frozen N1 score")
    if not np.isclose(plus_one_p, float(score["null_p_value"]), atol=1e-15):
        raise AssertionError("Reconstructed p-value differs from the frozen N1 score")
    return replace(
        audit,
        null_amplitudes=null,
        permutation_rank=float(np.mean(null < audit.cusum_amplitude)),
        plus_one_p_value=plus_one_p,
        permutation_seed=int(score["null_seed"]),
        passes_permutation=bool(score["admitted"]),
        accepted=bool(
            audit.passes_central
            and audit.passes_decrease
            and bool(score["admitted"])
        ),
    )


def _plot_n1_null(
    audit: base.WindowAudit,
    protocol: N1Protocol,
    axis: plt.Axes,
) -> None:
    null = audit.null_amplitudes
    axis.hist(
        null,
        bins=34,
        density=True,
        color=BISHAN_GREEN,
        edgecolor="white",
        linewidth=0.35,
        alpha=1.0,
        label="Permutation null distribution",
    )
    threshold = float(np.quantile(null, 1.0 - protocol.alpha, method="higher"))
    axis.axvline(
        threshold,
        color=base.GREY,
        linestyle=":",
        linewidth=1.2,
        label="Null quantile (97.5th percentile)",
    )
    axis.axvline(
        audit.cusum_amplitude,
        color=BISHAN_RED,
        linewidth=1.8,
        label="Observed CUSUM amplitude",
    )
    axis.set_xlabel("CUSUM amplitude, $Q$")
    axis.set_ylabel("Density")
    axis.grid(axis="y", color=base.GRID, linewidth=0.45, alpha=0.75, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(BISHAN_BLACK)
        spine.set_linewidth(0.78)
    axis.set_xlim(0.0, 18.0)
    axis.xaxis.set_major_locator(MultipleLocator(1.0))
    axis.set_ylim(0.0, 0.42)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.002, 0.985),
        fontsize=7.2,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor=BISHAN_BLACK,
        fancybox=False,
        borderpad=0.55,
    )
    axis.text(
        -0.075,
        1.0,
        "(f)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )


def _plot_bishan_timeline(
    series: base.SeriesData,
    events: pd.DatetimeIndex,
    windows: Sequence[PanelWindow],
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    figure: plt.Figure,
    subplot_spec: Any,
) -> tuple[plt.Axes, plt.Axes]:
    """Draw panel (a) with a left dB axis and Bishan Gardenia colors."""

    axis = figure.add_subplot(subplot_spec)
    window_count = len(windows)
    y_positions = np.arange(window_count, 0, -1, dtype=float)
    lower, upper = 0.45, window_count + 0.42
    mapped, db_minimum, db_maximum = refined._map_db_to_window_axis(
        series.values, lower=lower, upper=upper
    )

    axis.plot(
        series.calendar,
        mapped,
        color=BISHAN_GREEN,
        linewidth=1.85,
        alpha=0.95,
        marker="o",
        markersize=2.0,
        markeredgecolor="white",
        markeredgewidth=0.25,
        zorder=4,
    )

    # In panel (a), windows are explanatory background fills only.  They have
    # no outlines and sit below the observed trajectory.
    context_face = mcolors.to_rgba(BISHAN_BLUE_LIGHT, WINDOW_BLUE_ALPHA)
    decision_face = mcolors.to_rgba(BISHAN_YELLOW_LIGHT, DECISION_YELLOW_ALPHA)
    bar_height = 0.92
    for window, y_position in zip(windows, y_positions):
        observed_window_end = pd.Timestamp(window.window_end) + pd.offsets.MonthBegin(1)
        nominal_window_end = (
            pd.Timestamp("2027-01-01")
            if window.logical_id == 8
            else observed_window_end
        )
        # W8 is displayed to its nominal December 2026 extent so that the
        # 2016--2027 time axis shows the complete window geometry.  The green
        # trajectory itself remains limited to the observed record through
        # June 2026.
        axis.add_patch(
            Rectangle(
                (mdates.date2num(window.window_start), y_position - bar_height / 2),
                mdates.date2num(nominal_window_end)
                - mdates.date2num(window.window_start),
                bar_height,
                facecolor=context_face,
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
        )
        axis.add_patch(
            Rectangle(
                (mdates.date2num(window.decision_start), y_position - bar_height / 2),
                mdates.date2num(pd.Timestamp(window.decision_end) + pd.offsets.MonthBegin(1))
                - mdates.date2num(window.decision_start),
                bar_height,
                facecolor=decision_face,
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
        )
    # Reference evidence is visually independent of the detector: all three
    # dates therefore use the same black dashed line and top-aligned label.
    for event in events:
        axis.axvline(
            event,
            color=BISHAN_BLACK,
            linestyle=(0, (3.0, 2.0)),
            linewidth=0.85,
            alpha=0.92,
            zorder=5,
        )
        axis.text(
            event - pd.DateOffset(days=7),
            1.012,
            event.strftime("%Y-%m"),
            transform=axis.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=BISHAN_BLACK,
            fontweight="normal",
            zorder=7,
        )

    axis.set_xlim(
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2027-01-01"),
    )
    axis.set_ylim(0.0, window_count + 1.08)
    first_tick = np.ceil(db_minimum / 2.0) * 2.0
    last_tick = np.floor(db_maximum / 2.0) * 2.0
    db_ticks = np.arange(first_tick, last_tick + 0.1, 1.0)
    mapped_ticks = lower + (db_ticks - db_minimum) / (db_maximum - db_minimum) * (
        upper - lower
    )
    axis.set_yticks(mapped_ticks)
    axis.set_yticklabels([f"{value:.0f}" for value in db_ticks])
    axis.set_ylabel("VV (dB)")
    axis.yaxis.set_label_position("left")
    axis.yaxis.tick_left()
    axis.tick_params(axis="y", length=3.0, width=0.75, colors=BISHAN_BLACK, pad=2)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlabel("")
    axis.grid(
        axis="x",
        color="#D8D9D2",
        linewidth=0.45,
        linestyle="--",
        alpha=0.72,
        zorder=0,
    )
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(BISHAN_BLACK)
        spine.set_linewidth(0.78)

    axis.legend(
        handles=[
            Line2D(
                [0], [0], color=BISHAN_GREEN, marker="o", markersize=3,
                linewidth=1.6, label="VV polygon median",
            ),
            Line2D(
                [0], [0], color=BISHAN_BLACK, linestyle=(0, (3.0, 2.0)),
                linewidth=0.9, label="Reference disturbance",
            ),
            Patch(
                facecolor=context_face,
                edgecolor="none",
                label="48-month context",
            ),
            Patch(
                facecolor=decision_face,
                edgecolor="none",
                label="24-month decision window",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.008, 0.025),
        ncol=1,
        borderaxespad=0.0,
        labelspacing=0.38,
        handlelength=1.55,
        handletextpad=0.55,
        fontsize=7.2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#9A9A8C",
    )
    axis.text(
        -0.075,
        1.0,
        "(a)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    return axis, axis


def _draw_mean_difference_bracket(
    axis: plt.Axes,
    audit: base.WindowAudit,
    *,
    x_fraction: float,
    label_side: str,
) -> None:
    """Show the post-minus-pre mean difference without an opaque text box."""

    transform = blended_transform_factory(axis.transAxes, axis.transData)
    y_pre = float(audit.pre_mean_corrected)
    y_post = float(audit.post_mean_corrected)
    cap_half_width = 0.012
    axis.plot(
        [x_fraction, x_fraction],
        [y_post, y_pre],
        transform=transform,
        color=BISHAN_BLACK,
        linewidth=0.9,
        solid_capstyle="butt",
        zorder=7,
        clip_on=False,
    )
    for y_value in (y_post, y_pre):
        axis.plot(
            [x_fraction - cap_half_width, x_fraction + cap_half_width],
            [y_value, y_value],
            transform=transform,
            color=BISHAN_BLACK,
            linewidth=0.9,
            solid_capstyle="butt",
            zorder=7,
            clip_on=False,
        )
    direction = 1.0 if label_side == "right" else -1.0
    axis.text(
        x_fraction + direction * 0.022,
        0.5 * (y_pre + y_post),
        f"{audit.post_minus_pre_corrected:.1f} dB",
        transform=transform,
        ha="left" if label_side == "right" else "right",
        va="center",
        fontsize=7.2,
        color=BISHAN_BLACK,
        zorder=8,
    )


def _event_left_x_fraction(
    axis: plt.Axes,
    audit: base.WindowAudit,
    *,
    offset: float = 0.025,
) -> float:
    """Return a stable axes fraction immediately left of the detected event."""

    x_minimum, x_maximum = axis.get_xlim()
    event_number = mdates.date2num(audit.reported_transition_month)
    event_fraction = (event_number - x_minimum) / (x_maximum - x_minimum)
    return float(np.clip(event_fraction - offset, 0.16, 0.82))


def _style_cusum_extrema(
    axis: plt.Axes,
    audit: base.WindowAudit,
    *,
    is_left_panel: bool,
) -> None:
    """Simplify Q and directly label the two extrema used to compute it."""

    x_minimum, x_maximum = axis.get_xlim()
    event_number = mdates.date2num(audit.reported_transition_month)
    event_fraction = (event_number - x_minimum) / (x_maximum - x_minimum)
    for annotation in axis.texts:
        if annotation.get_text().startswith("$Q ="):
            annotation.set_text(f"Q = {audit.cusum_amplitude:.2f}")
            annotation.set_position((event_fraction - 0.022, 0.44))
            annotation.set_horizontalalignment("right")
            annotation.set_verticalalignment("center")

    for collection in axis.collections:
        collection.set_facecolor(BISHAN_RED)
        collection.set_edgecolor("white")

    maximum_index = int(np.argmax(audit.cumulative_values))
    minimum_index = int(np.argmin(audit.cumulative_values))
    maximum_number = mdates.date2num(audit.cumulative_dates[maximum_index])
    minimum_number = mdates.date2num(audit.cumulative_dates[minimum_index])
    maximum_fraction = (maximum_number - x_minimum) / (x_maximum - x_minimum)
    minimum_fraction = (minimum_number - x_minimum) / (x_maximum - x_minimum)
    if is_left_panel:
        maximum_label_position = (max(maximum_fraction - 0.10, 0.10), 0.78)
        minimum_label_position = (minimum_fraction + 0.025, 0.27)
        maximum_alignment = "right"
        minimum_alignment = "center"
    else:
        maximum_label_position = (min(maximum_fraction + 0.11, 0.90), 0.91)
        minimum_label_position = (min(minimum_fraction + 0.07, 0.90), 0.075)
        maximum_alignment = "left"
        minimum_alignment = "left"
    annotations = (
        (
            "max(S)",
            audit.cumulative_dates[maximum_index],
            float(audit.cumulative_values[maximum_index]),
            maximum_label_position,
            maximum_alignment,
        ),
        (
            "min(S)",
            audit.cumulative_dates[minimum_index],
            float(audit.cumulative_values[minimum_index]),
            minimum_label_position,
            minimum_alignment,
        ),
    )
    for label, x_value, y_value, label_position, horizontal_alignment in annotations:
        axis.annotate(
            label,
            xy=(x_value, y_value),
            xytext=label_position,
            textcoords=axis.transAxes,
            ha=horizontal_alignment,
            va="center",
            fontsize=7.2,
            color=BISHAN_RED,
            arrowprops={
                "arrowstyle": "-",
                "color": BISHAN_RED,
                "linewidth": 0.65,
                "shrinkA": 2,
                "shrinkB": 3,
            },
            zorder=9,
        )


def _tighten_shared_y_columns(
    axes: Sequence[plt.Axes],
    *,
    gap: float = 0.012,
) -> None:
    """Reduce the column gutter while preserving the two-row geometry."""

    for left_axis, right_axis in ((axes[0], axes[1]), (axes[2], axes[3])):
        left_position = left_axis.get_position()
        right_position = right_axis.get_position()
        midpoint = 0.5 * (left_position.x1 + right_position.x0)
        left_axis.set_position(
            [
                left_position.x0,
                left_position.y0,
                midpoint - gap / 2.0 - left_position.x0,
                left_position.height,
            ]
        )
        right_axis.set_position(
            [
                midpoint + gap / 2.0,
                right_position.y0,
                right_position.x1 - midpoint - gap / 2.0,
                right_position.height,
            ]
        )


def render_figure(
    series: base.SeriesData,
    events: pd.DatetimeIndex,
    display_windows: Sequence[PanelWindow],
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    protocol: N1Protocol,
    focal_event: pd.Timestamp,
) -> tuple[plt.Figure, dict[str, Any]]:
    # Apply one coherent Bishan Gardenia vocabulary to the inherited panel-(b)
    # drawing helpers. No statistical or geometric transformation is changed.
    base.BLUE = BISHAN_GREEN
    base.BLUE_LIGHT = BISHAN_BLUE_LIGHT
    base.ORANGE = BISHAN_BLUE
    base.ORANGE_LIGHT = BISHAN_YELLOW_LIGHT
    base.RED = BISHAN_RED
    base.GREY = BISHAN_GREY
    base.GRID = "#D8D9D2"
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
    figure = plt.figure(figsize=(7.15, 5.28), constrained_layout=False)
    outer = GridSpec(
        3,
        1,
        figure=figure,
        height_ratios=[1.75, 3.0, 1.25],
        hspace=0.17,
        left=0.095,
        right=0.955,
        top=0.965,
        bottom=0.07,
    )
    timeline_axis, db_axis = _plot_bishan_timeline(
        series, events, display_windows, outside, inside, figure, outer[0, 0]
    )
    contrast_axes = refined._plot_refined_contrast(
        series, outside, inside, figure, outer[1, 0]
    )
    _tighten_shared_y_columns(contrast_axes)
    # The inherited contrast helper uses opaque-looking spans by default.
    # Reduce only the blue context opacity so the signal remains primary while
    # retaining a clearly visible yellow decision interval.
    blue_rgb = np.asarray(mcolors.to_rgb(base.BLUE_LIGHT))
    yellow_rgb = np.asarray(mcolors.to_rgb(base.ORANGE_LIGHT))
    for axis in contrast_axes:
        for patch in axis.patches:
            face_rgb = np.asarray(patch.get_facecolor())[:3]
            if np.allclose(face_rgb, blue_rgb):
                patch.set_alpha(WINDOW_BLUE_ALPHA)
            elif np.allclose(face_rgb, yellow_rgb):
                patch.set_alpha(DECISION_YELLOW_ALPHA)
    # The contrast grid is a four-subfigure panel: (b,c) show VV backscatter
    # and (d,e) show the corresponding cumulative deviations.  The inherited
    # helper labels only its first axis, so make the panel grammar explicit
    # here and keep the y-axis semantics visible on every subfigure.
    panel_labels = ("(b)", "(c)", "(d)", "(e)")
    panel_ylabels = (
        "VV (dB)",
        "",
        "Cumulative deviation, S",
        "",
    )
    for index, (axis, panel_label, ylabel) in enumerate(
        zip(contrast_axes, panel_labels, panel_ylabels)
    ):
        for annotation in list(axis.texts):
            if annotation.get_text() in panel_labels:
                annotation.remove()
        axis.text(
            0.018,
            0.025,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=BISHAN_BLACK,
            clip_on=False,
            zorder=10,
        )
        axis.set_ylabel(ylabel)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(BISHAN_BLACK)
            spine.set_linewidth(0.78)
        axis.tick_params(colors=BISHAN_BLACK)
        for line in axis.lines:
            if mcolors.to_hex(line.get_color()).lower() == BISHAN_RED.lower():
                line.set_linewidth(1.55)
                line.set_zorder(6)
        if index in (1, 3):
            # The right column shares the left-column scale and therefore does
            # not repeat y-axis ticks or labels.
            axis.set_ylabel("")
            axis.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
                labelright=False,
            )
        else:
            axis.yaxis.set_label_position("left")
            axis.yaxis.tick_left()
            axis.tick_params(axis="y", labelleft=True, labelright=False)
    contrast_axes[0].set_title("")
    contrast_axes[1].set_title("")
    for axis in contrast_axes:
        for annotation in list(axis.texts):
            annotation_text = annotation.get_text()
            if (
                annotation_text.startswith("post - pre =")
                or annotation_text.startswith("CUSUM boundary / event:")
                or annotation_text.startswith("CUSUM boundary:")
                or annotation_text.startswith("reported disturbance:")
            ):
                annotation.remove()
        axis.set_xlabel("")
    _draw_mean_difference_bracket(
        contrast_axes[0],
        outside,
        x_fraction=_event_left_x_fraction(contrast_axes[0], outside),
        label_side="left",
    )
    _draw_mean_difference_bracket(
        contrast_axes[1],
        inside,
        x_fraction=_event_left_x_fraction(contrast_axes[1], inside),
        label_side="left",
    )
    _style_cusum_extrema(contrast_axes[2], outside, is_left_panel=True)
    _style_cusum_extrema(contrast_axes[3], inside, is_left_panel=False)
    # The interpreted reference month is independent of the detector and is
    # kept distinct from the implementation-defined CUSUM boundary sample.
    for axis in (contrast_axes[0], contrast_axes[1]):
        axis.axvline(
            focal_event,
            color=base.GREY,
            linestyle=":",
            linewidth=0.9,
            zorder=2,
        )
    permutation_axis = figure.add_subplot(outer[2, 0])
    _plot_n1_null(inside, protocol, permutation_axis)
    for text in figure.findobj(match=Text):
        if text.get_fontsize() < 7.2:
            text.set_fontsize(7.2)
    diagnostics = {
        "layout_version": LAYOUT_VERSION,
        "figure_size_inches": [float(v) for v in figure.get_size_inches()],
        "axes_count": len(figure.axes),
        "minimum_parent_text_size_pt": min(
            float(text.get_fontsize()) for text in figure.findobj(match=Text)
        ),
        "panel_a_window_count": len(display_windows),
        "panel_a_window_labels": [],
        "panel_b_labels": list(panel_labels),
        "panel_b_ylabels": list(panel_ylabels),
        "panel_b_titles": [axis.get_title() for axis in contrast_axes],
        "panel_f_permutations": int(len(inside.null_amplitudes)),
        "panel_f_plus_one_p_value": float(inside.plus_one_p_value),
        # Backward-compatible aliases for scripts that consumed the v7 schema.
        "panel_c_permutations": int(len(inside.null_amplitudes)),
        "panel_c_plus_one_p_value": float(inside.plus_one_p_value),
        "timeline_axis": timeline_axis.get_position().bounds,
        "db_axis": db_axis.get_position().bounds,
    }
    return figure, diagnostics


def publication_caption(
    series: base.SeriesData,
    events: pd.DatetimeIndex,
    outside: base.WindowAudit,
    inside: base.WindowAudit,
    protocol: N1Protocol,
) -> str:
    event_text = ", ".join(event.strftime("%b %Y") for event in events)
    return (
        "Conceptual illustration of the Windowed CUSUM procedure using the "
        f"Polygon-first VV monthly median for reference BALSA polygon "
        f"{series.historical_id}. (a) The January 2016--June 2026 trajectory "
        f"and overlapping nominal {protocol.window_length}-month analysis windows; "
        "The final window is drawn to its nominal December 2026 extent; the observed trajectory "
        "ends in June 2026, so the remaining 2026 context is geometric only. "
        "Its 2024--2025 central interval remains complete. Blue shading denotes the analysis context and yellow "
        "shading denotes the "
        f"{protocol.window_length - 2 * protocol.step}-month central decision "
        f"intervals, and black dashed lines mark independently interpreted "
        f"disturbances ({event_text}). The formal historical reporting interval is "
        f"{protocol.reporting_start_year}--{protocol.reporting_end_year}; 2016 and "
        "2026 H1 provide temporal context. (b--e) The January 2022 CUSUM boundary is "
        "outside the central interval in the left-hand window but inside it in the "
        "right-hand window; the independently interpreted reference month is February "
        "2022. The boundary sample belongs to the post-boundary slice used by the "
        "decrease constraint and is marked by the red dashed line. The two upper "
        "subfigures (b,c) show VV backscatter "
        "and the two lower subfigures (d,e) show cumulative deviation for the rejected "
        "and retained windows. Red points in (d,e) identify max(S) and min(S), whose "
        "range defines Q; brackets in (b,c) show the post-minus-pre mean "
        "differences. (f) The independent-permutation null distribution for the "
        f"retained right-hand window ({len(inside.null_amplitudes):,} permutations; "
        f"$Q={inside.cusum_amplitude:.2f}$; finite-sample-corrected "
        f"$p_{{\\mathrm{{perm}}}}={inside.plus_one_p_value:.5f}$; "
        f"$\\alpha={protocol.alpha:.3f}$)."
    )


def _audit_record(audit: base.WindowAudit) -> dict[str, Any]:
    record = base._audit_record(audit)
    record["null_replicates"] = int(len(audit.null_amplitudes))
    return record


def _write_readme(
    path: Path,
    *,
    output_dir: Path,
    caption: str,
) -> None:
    try:
        relative_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative_output = output_dir
    text = f"""# Current-data Windowed CUSUM concept figure

This figure replaces the PowerPoint-assembled legacy concept graphic with a
single reproducible Python workflow. It uses the frozen 40-tile Polygon-first
cache (January 2016--June 2026) and the frozen N1 finite-sample permutation
screen. The archived figure is retained unchanged.

## Reproduce

```powershell
python analysis/workflows/build_windowed_cusum_concept_n1_current.py `
  --output-dir "{relative_output}"
```

The workflow refuses to overwrite existing files. Select a new output
directory for a later revision.

## Figure contract

- Core conclusion: overlapping local context, a central decision interval,
  a decrease constraint, and an empirical permutation screen convert recurrent
  stand-level departures into auditable disturbance candidates.
- Panel (a): data and window geometry.
- Panels (b)--(e): central-window decision and month semantics; (b,c) show
  VV backscatter and (d,e) show cumulative deviation.
- Panel (f): finite-sample-corrected independent-permutation evidence.
- Formal reporting interval: 2017--2025.
- Context retained for window support: 2016 and January--June 2026.

## Manuscript caption

{caption}
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--reference-events", type=Path, default=DEFAULT_REFERENCE_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--n1-scores", type=Path, default=DEFAULT_N1_SCORES)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID, required=True)
    parser.add_argument("--historical-id", type=int, default=DEFAULT_HISTORICAL_ID)
    parser.add_argument("--focal-event", default=DEFAULT_FOCAL_EVENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [
        args.metadata,
        args.series,
        args.reference_events,
        args.candidates,
        args.n1_scores,
        args.protocol,
    ]
    _require_paths(inputs)
    output_dir = args.output_dir.resolve()
    pdf_path = output_dir / "windowed_cusum_concept_n1_current.pdf"
    png_path = output_dir / "windowed_cusum_concept_n1_current.png"
    svg_path = output_dir / "windowed_cusum_concept_n1_current.svg"
    tiff_path = output_dir / "windowed_cusum_concept_n1_current.tiff"
    audit_path = output_dir / "windowed_cusum_concept_n1_current_audit.csv"
    manifest_path = output_dir / "windowed_cusum_concept_n1_current_manifest.json"
    readme_path = output_dir / "README.md"
    outputs = [
        pdf_path,
        png_path,
        svg_path,
        tiff_path,
        audit_path,
        manifest_path,
        readme_path,
    ]
    _refuse_existing(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_n1_protocol(args.protocol)
    series = base.load_polygon_series(
        args.metadata, args.series, args.sample_id, args.historical_id
    )
    if series.calendar.min().strftime("%Y-%m") != protocol.context_start:
        raise AssertionError("Series start differs from the frozen N1 context calendar")
    if series.calendar.max().strftime("%Y-%m") != protocol.context_end:
        raise AssertionError("Series end differs from the frozen N1 context calendar")
    events = load_reference_events(
        args.reference_events, args.sample_id, args.historical_id
    )
    focal_event = pd.Period(args.focal_event, freq="M").to_timestamp()
    if pd.Period(focal_event, freq="M") not in {
        pd.Period(event, freq="M") for event in events
    }:
        raise ValueError(f"Focal event {args.focal_event} is not a reference event")

    detector_config = base.DetectorConfig(
        window_length=protocol.window_length,
        step=protocol.step,
        confidence_threshold=1.0 - protocol.alpha,
        permutations=1,
        minimum_observations=protocol.minimum_observations,
        seed=protocol.seed,
    )
    full_windows = [
        window
        for window in base.generate_windows(
            len(series.values), protocol.window_length, protocol.step
        )
        if int(window[1] - window[0]) == protocol.window_length
    ]
    audits: list[base.WindowAudit] = []
    for logical_id, window in enumerate(full_windows, start=1):
        try:
            audits.append(
                base.audit_window(series, detector_config, logical_id, window)
            )
        except ValueError as error:
            # A CUSUM maximum at the first sample cannot define non-empty pre-
            # and post-boundary slices. Production rejects that window before
            # the central/decrease/null gates, so it is omitted here as well.
            if "cannot form a valid split" not in str(error):
                raise
    outside, inside = base.choose_contrast_pair(audits, focal_event)

    candidates = pd.read_parquet(args.candidates)
    scores = pd.read_parquet(args.n1_scores)
    outside_candidate = _candidate_row(
        candidates, sample_id=args.sample_id, logical_window_id=outside.logical_id
    )
    inside_candidate = _candidate_row(
        candidates, sample_id=args.sample_id, logical_window_id=inside.logical_id
    )
    # The outside window is rejected before N1 because its boundary is not in
    # the central interval.  Only the inside F3 candidate has a frozen N1 row.
    if bool(outside_candidate["passes_central"]):
        raise AssertionError("The outside example unexpectedly passes the central gate")
    inside, frozen_score = _attach_frozen_n1_score(
        inside, inside_candidate, scores, protocol
    )
    inside = _exact_null_for_audit(series, inside, frozen_score)
    audits = [inside if audit.logical_id == inside.logical_id else audit for audit in audits]
    display_windows = build_panel_windows(series, protocol)

    figure, diagnostics = render_figure(
        series, events, display_windows, outside, inside, protocol, focal_event
    )
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        tiff_path,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    audit_rows = [_audit_record(audit) for audit in audits]
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    caption = publication_caption(series, events, outside, inside, protocol)
    _write_readme(readme_path, output_dir=output_dir, caption=caption)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layout_version": LAYOUT_VERSION,
        "purpose": "Current-data reproducible replacement for the legacy LR-CuSum concept figure.",
        "figure_contract": {
            "core_conclusion": (
                "Overlapping local context, the central decision interval, the "
                "decrease constraint, and finite-sample N1 screening form an "
                "auditable recurrent-disturbance detector."
            ),
            "archetype": "schematic-led quantitative composite",
            "target_output": "JAG A4 double-column manuscript figure",
            "backend": "Python/matplotlib",
            "panel_map": {
                "a": "Current Polygon-first series and window geometry",
                "b-e": "Central-window decision and boundary-month semantics",
                "f": "Independent-permutation null screen",
            },
        },
        "workflow": _file_record(Path(__file__)),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "dpi_raster": int(args.dpi),
        },
        "input_semantics": {
            "representation": "Polygon-first monthly median VV",
            "sample_id": series.sample_id,
            "historical_polygon_id": series.historical_id,
            "polygon_track_id": str(series.metadata.get("polygon_track_id", "")),
            "tile_id": str(series.metadata.get("tile_id", "")),
            "track_key": str(series.metadata.get("track_key", "")),
            "interior_pixels": int(series.metadata.get("interior_pixels", 0)),
            "calendar": [
                series.calendar.min().strftime("%Y-%m"),
                series.calendar.max().strftime("%Y-%m"),
            ],
            "formal_reporting_years": [
                protocol.reporting_start_year,
                protocol.reporting_end_year,
            ],
            "reference_events": [event.strftime("%Y-%m") for event in events],
            "focal_reference_event": focal_event.strftime("%Y-%m"),
        },
        "n1_protocol": asdict(protocol),
        "month_semantics": {
            "cusum_boundary": (
                "Calendar month of the CUSUM maximum sample; the boundary "
                "sample belongs to the post-boundary slice."
            ),
            "reference_event": (
                "Independently interpreted reference month, kept distinct from "
                "the detector boundary and compared using the frozen tolerance."
            ),
        },
        "selected_contrast": {
            "outside": _audit_record(outside),
            "inside": _audit_record(inside),
            "outside_candidate_id": str(outside_candidate["candidate_id"]),
            "inside_candidate_id": str(inside_candidate["candidate_id"]),
        },
        "layout_diagnostics": diagnostics,
        "publication_caption": caption,
        "inputs": {path.name: _file_record(path) for path in inputs},
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"] = {
        path.name: _file_record(path)
        for path in (pdf_path, png_path, svg_path, tiff_path, audit_path, readme_path)
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    print(f"TIFF: {tiff_path}")
    print(f"Audit: {audit_path}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Current example: {series.sample_id}; {len(series.calendar)} months; "
        f"W{inside.logical_id}; Q={inside.cusum_amplitude:.4f}; "
        f"p_perm={inside.plus_one_p_value:.5f}; B={len(inside.null_amplitudes):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
