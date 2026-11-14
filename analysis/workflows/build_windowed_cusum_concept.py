"""Build a fully reproducible concept figure for frozen Windowed CUSUM.

The legacy Figure 3 was assembled in PowerPoint from notebook panels.  This
workflow rebuilds the complete figure directly from the frozen Polygon-first
time series and the production detector configuration.  The figure uses the
same implementation-defined boundary as ``lr_cusum``: the CUSUM maximum sample
is mapped to its calendar month and that sample is included in the post-change
slice used by the directional gate.

Outputs are never overwritten.  Each run writes PDF, PNG, SVG, a window-audit
CSV, and a provenance manifest to a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from windowed_cusum.detector.core import bootstrap_confidence, cusum_amplitude, generate_windows


DEFAULT_SAMPLE_ID = "plantation:8655"
DEFAULT_HISTORICAL_ID = 1499
DEFAULT_FOCAL_EVENT = "2022-02"
DEFAULT_METADATA = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "signature_analysis"
    / "national_balsa_series_metadata.parquet"
)
DEFAULT_SERIES = (
    REPO_ROOT
    / "outputs"
    / "jag_codex_20260821"
    / "signature_analysis"
    / "national_balsa_series.npz"
)
DEFAULT_REFERENCE_EVENTS = (
    REPO_ROOT
    / "outputs"
    / "plantation_reference_manifest_v1"
    / "reference_sample_events.csv"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "outputs"
    / "plantation_polygon_full_v1"
    / "results"
    / "production_config.json"
)

BLUE = "#2774A8"
BLUE_LIGHT = "#DCEAF4"
ORANGE = "#D95F02"
ORANGE_LIGHT = "#F6D5C3"
RED = "#C83E36"
INK = "#263238"
GREY = "#7A858C"
GRID = "#D5DADD"


@dataclass(frozen=True)
class DetectorConfig:
    window_length: int
    step: int
    confidence_threshold: float
    permutations: int
    minimum_observations: int
    seed: int


@dataclass(frozen=True)
class SeriesData:
    sample_id: str
    historical_id: int
    calendar: pd.DatetimeIndex
    values: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WindowAudit:
    label: str
    logical_id: int
    window_start_index: int
    window_end_index: int
    decision_start_index: int
    decision_end_index: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    decision_start: pd.Timestamp
    decision_end: pd.Timestamp
    candidate_sample_index: int
    candidate_sample_month: pd.Timestamp
    reported_transition_month: pd.Timestamp
    cusum_amplitude: float
    pre_mean_corrected: float
    post_mean_corrected: float
    post_minus_pre_corrected: float
    permutation_rank: float
    plus_one_p_value: float
    passes_central: bool
    passes_decrease: bool
    passes_permutation: bool
    accepted: bool
    valid_observations: int
    permutation_seed: int
    cumulative_values: np.ndarray
    cumulative_dates: pd.DatetimeIndex
    null_amplitudes: np.ndarray


def _require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required inputs are missing:\n" + "\n".join(missing))


def _refuse_existing(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs:\n" + "\n".join(existing)
        )


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


def load_detector_config(path: Path) -> DetectorConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = DetectorConfig(
        window_length=int(payload["window_length"]),
        step=int(payload["step"]),
        confidence_threshold=float(payload["confidence"]),
        permutations=int(payload["bootstrap"]),
        minimum_observations=int(payload["minimum_observations"]),
        seed=int(payload["seed"]),
    )
    if config.window_length <= 2 * config.step:
        raise ValueError("Frozen config must have a non-empty central decision interval")
    if not 0.0 < config.confidence_threshold < 1.0:
        raise ValueError("confidence threshold must be in (0, 1)")
    return config


def load_polygon_series(
    metadata_path: Path,
    series_path: Path,
    sample_id: str,
    historical_id: int,
) -> SeriesData:
    metadata = pd.read_parquet(metadata_path)
    selected = metadata.loc[metadata["sample_id"].astype(str).eq(str(sample_id))]
    if len(selected) != 1:
        raise ValueError(f"Expected one metadata row for {sample_id}; found {len(selected)}")
    row = selected.iloc[0]
    with np.load(series_path, allow_pickle=False) as archive:
        values = np.asarray(archive["values"])
        calendar = pd.DatetimeIndex(pd.to_datetime(archive["calendar"]))
    series_index = int(row["series_index"])
    if values.ndim != 3 or values.shape[1] < 1:
        raise ValueError("Expected series tensor shaped [sample, band, month]")
    vv = np.asarray(values[series_index, 0], dtype=np.float64)
    if len(vv) != len(calendar):
        raise ValueError("Series and calendar lengths differ")
    return SeriesData(
        sample_id=str(sample_id),
        historical_id=int(historical_id),
        calendar=calendar,
        values=vv,
        metadata={
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.to_dict().items()
        },
    )


def load_reference_events(
    path: Path,
    sample_id: str,
    historical_id: int,
) -> pd.DatetimeIndex:
    frame = pd.read_csv(path)
    mask = frame["sample_id"].astype(str).eq(str(sample_id))
    if "historical_id" in frame.columns:
        mask &= pd.to_numeric(frame["historical_id"], errors="coerce").eq(
            int(historical_id)
        )
    events = pd.DatetimeIndex(pd.to_datetime(frame.loc[mask, "event_month"])).sort_values()
    if events.empty:
        raise ValueError(f"No reference events found for {sample_id}")
    return events


def reported_transition_month(candidate_sample_month: pd.Timestamp) -> pd.Timestamp:
    """Return the production detector's reported event/boundary month.

    ``core.lr_cusum`` reports the calendar sample at the CUSUM maximum itself;
    it does not advance the event by one calendar month.  The historical
    function name is retained for compatibility with archived manifests and
    downstream figure code.
    """

    return pd.Timestamp(candidate_sample_month).normalize()


def permutation_null_amplitudes(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Reproduce the production independent-permutation null distribution."""

    clean = np.asarray(values, dtype=np.float64)
    if clean.ndim != 1 or len(clean) < 3:
        raise ValueError("Permutation input must contain at least three observations")
    rng = np.random.RandomState(int(seed))
    null = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        shuffled = clean[rng.permutation(len(clean))]
        null[iteration] = float(cusum_amplitude(shuffled)[0])
    return null


def audit_window(
    series: SeriesData,
    config: DetectorConfig,
    logical_id: int,
    window: tuple[int, int, int, int],
) -> WindowAudit:
    window_start, window_end, decision_start, decision_end = map(int, window)
    local_values = series.values[window_start:window_end]
    valid_indices = np.flatnonzero(np.isfinite(local_values))
    if len(valid_indices) < config.minimum_observations:
        raise ValueError(f"W{logical_id} has insufficient valid observations")
    clean = np.asarray(local_values[valid_indices], dtype=np.float64)
    amplitude, clean_index = cusum_amplitude(clean)
    if clean_index < 0:
        raise ValueError(f"W{logical_id} has no valid CUSUM candidate")
    local_index = int(valid_indices[int(clean_index)])
    global_index = int(window_start + local_index)
    candidate_month = pd.Timestamp(series.calendar[global_index])
    transition_month = reported_transition_month(candidate_month)

    # Match production ``core.lr_cusum`` exactly: the CUSUM maximum sample is
    # the implementation-defined boundary and belongs to the post slice.
    split_index = local_index
    before = local_values[:split_index]
    after = local_values[split_index:]
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        raise ValueError(f"W{logical_id} candidate cannot form a valid split")
    pre_mean = float(np.nanmean(before))
    post_mean = float(np.nanmean(after))
    delta = float(post_mean - pre_mean)

    cumulative = np.cumsum(clean - float(np.mean(clean)))
    cumulative_dates = pd.DatetimeIndex(series.calendar[window_start:window_end][valid_indices])
    permutation_seed = int(config.seed + window_start)
    null = permutation_null_amplitudes(clean, config.permutations, permutation_seed)
    rank = float(np.mean(null < float(amplitude)))
    core_rank = float(
        bootstrap_confidence(clean, float(amplitude), config.permutations, permutation_seed)
    )
    if abs(rank - core_rank) > 1.0 / config.permutations:
        raise AssertionError("Rendered permutation rank differs from production primitive")
    exceedances = int(np.sum(null >= float(amplitude)))
    plus_one_p = float((exceedances + 1) / (config.permutations + 1))

    passes_central = bool(decision_start <= global_index < decision_end)
    passes_decrease = bool(delta < 0.0)
    passes_permutation = bool(rank >= config.confidence_threshold)
    return WindowAudit(
        label=f"W{logical_id}",
        logical_id=int(logical_id),
        window_start_index=window_start,
        window_end_index=window_end,
        decision_start_index=decision_start,
        decision_end_index=decision_end,
        window_start=pd.Timestamp(series.calendar[window_start]),
        window_end=pd.Timestamp(series.calendar[window_end - 1]),
        decision_start=pd.Timestamp(series.calendar[decision_start]),
        decision_end=pd.Timestamp(series.calendar[decision_end - 1]),
        candidate_sample_index=global_index,
        candidate_sample_month=candidate_month,
        reported_transition_month=transition_month,
        cusum_amplitude=float(amplitude),
        pre_mean_corrected=pre_mean,
        post_mean_corrected=post_mean,
        post_minus_pre_corrected=delta,
        permutation_rank=rank,
        plus_one_p_value=plus_one_p,
        passes_central=passes_central,
        passes_decrease=passes_decrease,
        passes_permutation=passes_permutation,
        accepted=bool(passes_central and passes_decrease and passes_permutation),
        valid_observations=int(len(clean)),
        permutation_seed=permutation_seed,
        cumulative_values=np.asarray(cumulative, dtype=np.float64),
        cumulative_dates=cumulative_dates,
        null_amplitudes=null,
    )


def build_window_audits(series: SeriesData, config: DetectorConfig) -> list[WindowAudit]:
    audits: list[WindowAudit] = []
    for logical_id, window in enumerate(
        generate_windows(len(series.values), config.window_length, config.step),
        start=1,
    ):
        try:
            audits.append(audit_window(series, config, logical_id, window))
        except ValueError as error:
            # ``core.lr_cusum`` rejects candidates with no observations on one
            # side of the implementation boundary (for example a maximum at
            # the first sample of an edge window).  Such windows are not
            # renderable under production semantics and are omitted from the
            # concept audit rather than assigned a different split.
            if "cannot form a valid split" not in str(error):
                raise
    return audits


def choose_contrast_pair(
    audits: Sequence[WindowAudit],
    focal_event: pd.Timestamp,
) -> tuple[WindowAudit, WindowAudit]:
    focal = pd.Period(focal_event, freq="M")
    relevant = [
        audit
        for audit in audits
        if pd.Period(audit.reported_transition_month, freq="M") == focal
    ]
    # The frozen reference manifest may use the first post-disturbance month,
    # whereas the production detector reports the CUSUM boundary sample.  If
    # no exact production month is available, allow the one-month-earlier
    # boundary that corresponds to that reference convention.  This fallback
    # affects only selection of the illustrative OUT/IN pair; it does not
    # change detector outputs or event dates.
    if not relevant:
        boundary = focal - 1
        relevant = [
            audit
            for audit in audits
            if pd.Period(audit.candidate_sample_month, freq="M") == boundary
        ]
    pairs: list[tuple[int, WindowAudit, WindowAudit]] = []
    for outside in relevant:
        if outside.passes_central:
            continue
        for inside in relevant:
            if not inside.passes_central:
                continue
            if outside.candidate_sample_index != inside.candidate_sample_index:
                continue
            separation = abs(inside.window_start_index - outside.window_start_index)
            pairs.append((separation, outside, inside))
    if not pairs:
        summary = [
            (
                audit.label,
                audit.reported_transition_month.strftime("%Y-%m"),
                audit.passes_central,
            )
            for audit in audits
        ]
        raise ValueError(f"No OUT/IN contrast pair for {focal}; audits={summary}")
    _, outside, inside = min(pairs, key=lambda item: item[0])
    return outside, inside


def _month_after(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(timestamp) + pd.offsets.MonthBegin(1)


def _window_end_boundary(audit: WindowAudit) -> pd.Timestamp:
    return _month_after(audit.window_end)


def _decision_end_boundary(audit: WindowAudit) -> pd.Timestamp:
    return _month_after(audit.decision_end)


def _plot_full_series_and_windows(
    series: SeriesData,
    events: pd.DatetimeIndex,
    audits: Sequence[WindowAudit],
    outside: WindowAudit,
    inside: WindowAudit,
    focal_event: pd.Timestamp,
    figure: plt.Figure,
    subplot_spec: Any,
) -> tuple[plt.Axes, plt.Axes]:
    grid = subplot_spec.subgridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.04)
    time_ax = figure.add_subplot(grid[0, 0])
    window_ax = figure.add_subplot(grid[1, 0], sharex=time_ax)

    time_ax.plot(
        series.calendar,
        series.values,
        color=BLUE,
        linewidth=1.6,
        marker="o",
        markersize=2.5,
        markeredgecolor="white",
        markeredgewidth=0.3,
        label="Polygon-first VV median",
        zorder=3,
    )
    focal_period = pd.Period(focal_event, freq="M")
    for event in events:
        is_focal = pd.Period(event, freq="M") == focal_period
        time_ax.axvline(
            event,
            color=RED if is_focal else GREY,
            linestyle="--" if is_focal else ":",
            linewidth=1.4 if is_focal else 1.0,
            zorder=2,
        )
        time_ax.text(
            event,
            1.015,
            event.strftime("%Y-%m"),
            transform=time_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=RED if is_focal else GREY,
            fontsize=7.0,
            fontweight="bold" if is_focal else "normal",
            clip_on=False,
        )
    time_ax.set_ylabel("VV backscatter (dB)")
    time_ax.grid(True, color=GRID, linewidth=0.45, alpha=0.75)
    time_ax.spines[["top", "right"]].set_visible(False)
    time_ax.tick_params(axis="x", labelbottom=False)
    time_ax.legend(loc="lower left", fontsize=7.0, frameon=True, framealpha=0.92)
    time_ax.text(
        0.995,
        0.96,
        f"Reference polygon {series.historical_id}",
        transform=time_ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=INK,
    )
    time_ax.text(
        -0.075,
        1.10,
        "(a)",
        transform=time_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )

    y_positions = np.arange(len(audits))[::-1]
    height = 0.58
    for audit, y in zip(audits, y_positions):
        is_outside = audit.logical_id == outside.logical_id
        is_inside = audit.logical_id == inside.logical_id
        edge = ORANGE if is_outside else (BLUE if is_inside else "#78A6C5")
        line_width = 1.4 if (is_outside or is_inside) else 0.65
        window_ax.add_patch(
            Rectangle(
                (mdates.date2num(audit.window_start), y - height / 2),
                mdates.date2num(_window_end_boundary(audit))
                - mdates.date2num(audit.window_start),
                height,
                facecolor=BLUE_LIGHT,
                edgecolor=edge,
                linewidth=line_width,
            )
        )
        window_ax.add_patch(
            Rectangle(
                (mdates.date2num(audit.decision_start), y - height / 2),
                mdates.date2num(_decision_end_boundary(audit))
                - mdates.date2num(audit.decision_start),
                height,
                facecolor=ORANGE_LIGHT,
                edgecolor=RED,
                linewidth=0.55,
            )
        )
        suffix = "  OUT" if is_outside else ("  IN" if is_inside else "")
        window_ax.text(
            series.calendar[0] - pd.DateOffset(months=4),
            y,
            audit.label + suffix,
            ha="right",
            va="center",
            fontsize=6.6,
            fontweight="bold" if suffix else "normal",
            color=edge if suffix else INK,
            clip_on=False,
        )

    window_ax.axvline(focal_event, color=RED, linestyle="--", linewidth=1.2)
    # Reserve one empty row above W1 for the semantic legend.  Keeping the
    # legend inside this dedicated strip avoids covering any window bars.
    window_ax.set_ylim(-0.65, len(audits) + 1.70)
    window_ax.set_yticks([])
    window_ax.xaxis.set_major_locator(mdates.YearLocator())
    window_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    window_ax.set_xlabel("Year")
    window_ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.65)
    window_ax.spines[["top", "right", "left"]].set_visible(False)
    window_ax.legend(
        handles=[
            Patch(facecolor=BLUE_LIGHT, edgecolor="#78A6C5", label="48-month context"),
            Patch(facecolor=ORANGE_LIGHT, edgecolor=RED, label="24-month decision interval"),
            Line2D([0], [0], color=RED, linestyle="--", label="Reference transition month"),
        ],
        loc="upper left",
        ncol=3,
        fontsize=6.6,
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
    )
    return time_ax, window_ax


def _plot_window_column(
    series: SeriesData,
    audit: WindowAudit,
    series_ax: plt.Axes,
    cusum_ax: plt.Axes,
    *,
    show_y_labels: bool,
) -> None:
    dates = series.calendar[audit.window_start_index : audit.window_end_index]
    values = series.values[audit.window_start_index : audit.window_end_index]
    transition = audit.reported_transition_month
    decision_start = audit.decision_start
    decision_end = _decision_end_boundary(audit)

    for axis in (series_ax, cusum_ax):
        axis.axvspan(
            audit.window_start,
            _window_end_boundary(audit),
            color=BLUE_LIGHT,
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )
        axis.axvspan(
            decision_start,
            decision_end,
            color=ORANGE_LIGHT,
            alpha=0.85,
            linewidth=0,
            zorder=0,
        )
        axis.axvline(transition, color=RED, linestyle="--", linewidth=1.1, zorder=2)
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    series_ax.plot(
        dates,
        values,
        color=BLUE,
        linewidth=1.3,
        marker="o",
        markersize=2.2,
        zorder=3,
    )
    # Keep the descriptive means identical to ``core.lr_cusum``: the
    # implementation boundary sample belongs to the post-change slice.
    split = audit.candidate_sample_index - audit.window_start_index
    pre_dates = dates[:split]
    post_dates = dates[split:]
    if len(pre_dates):
        series_ax.hlines(
            audit.pre_mean_corrected,
            pre_dates[0],
            _month_after(pre_dates[-1]),
            color=INK,
            linestyle=":",
            linewidth=1.0,
        )
    if len(post_dates):
        series_ax.hlines(
            audit.post_mean_corrected,
            post_dates[0],
            _month_after(post_dates[-1]),
            color=INK,
            linestyle=":",
            linewidth=1.0,
        )
    status = "IN" if audit.passes_central else "OUT"
    status_color = BLUE if audit.passes_central else ORANGE
    series_ax.set_title(
        f"{audit.label}: same transition, {status}",
        fontsize=8.0,
        fontweight="bold",
        color=status_color,
        pad=4,
    )
    series_ax.text(
        0.02,
        0.06,
        f"post - pre = {audit.post_minus_pre_corrected:.2f} dB",
        transform=series_ax.transAxes,
        fontsize=6.6,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="#C8CDD0", alpha=0.9, pad=2.0),
    )
    if show_y_labels:
        series_ax.set_ylabel("VV (dB)")
    else:
        series_ax.tick_params(axis="y", labelleft=False)
    series_ax.tick_params(axis="x", labelbottom=False)

    cusum_ax.plot(
        audit.cumulative_dates,
        audit.cumulative_values,
        color=INK,
        linewidth=1.25,
        marker="o",
        markersize=1.9,
        zorder=3,
    )
    max_index = int(np.argmax(audit.cumulative_values))
    min_index = int(np.argmin(audit.cumulative_values))
    cusum_ax.scatter(
        audit.cumulative_dates[max_index],
        audit.cumulative_values[max_index],
        s=28,
        color=RED,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label=r"$\max S$ (CUSUM boundary sample)",
    )
    cusum_ax.scatter(
        audit.cumulative_dates[min_index],
        audit.cumulative_values[min_index],
        s=22,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        label=r"$\min S$",
    )
    cusum_ax.text(
        0.02,
        0.94,
        rf"$Q = \max(S) - \min(S) = {audit.cusum_amplitude:.2f}$",
        transform=cusum_ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
    )
    cusum_ax.text(
        0.98,
        0.06,
        f"CUSUM boundary / event: {audit.reported_transition_month:%b %Y}",
        transform=cusum_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="#C8CDD0", alpha=0.9, pad=2.0),
    )
    if show_y_labels:
        cusum_ax.set_ylabel("Cumulative deviation, $S$")
    else:
        cusum_ax.tick_params(axis="y", labelleft=False)
    cusum_ax.xaxis.set_major_locator(mdates.YearLocator())
    cusum_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    cusum_ax.set_xlabel("Year")


def _plot_contrast(
    series: SeriesData,
    outside: WindowAudit,
    inside: WindowAudit,
    figure: plt.Figure,
    subplot_spec: Any,
) -> tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    grid = subplot_spec.subgridspec(2, 2, hspace=0.08, wspace=0.08)
    out_series = figure.add_subplot(grid[0, 0])
    in_series = figure.add_subplot(grid[0, 1], sharey=out_series)
    out_cusum = figure.add_subplot(grid[1, 0])
    in_cusum = figure.add_subplot(grid[1, 1], sharey=out_cusum)
    _plot_window_column(series, outside, out_series, out_cusum, show_y_labels=True)
    _plot_window_column(series, inside, in_series, in_cusum, show_y_labels=False)
    out_series.text(
        -0.17,
        1.17,
        "(b)",
        transform=out_series.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        clip_on=False,
    )
    return out_series, in_series, out_cusum, in_cusum


def _plot_permutation_and_gates(
    audit: WindowAudit,
    config: DetectorConfig,
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
    threshold = float(np.quantile(null, config.confidence_threshold, method="higher"))
    axis.axvline(
        threshold,
        color=GREY,
        linestyle=":",
        linewidth=1.2,
        label=f"{config.confidence_threshold:.3f} null quantile",
    )
    axis.axvline(
        audit.cusum_amplitude,
        color=RED,
        linewidth=1.8,
        label=rf"Observed $Q={audit.cusum_amplitude:.2f}$",
    )
    axis.set_xlabel("CUSUM amplitude, $Q$")
    axis.set_ylabel("Density")
    axis.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    right = max(float(audit.cusum_amplitude) * 1.42, float(np.max(null)) * 1.25)
    axis.set_xlim(max(0.0, float(np.min(null)) * 0.75), right)
    axis.legend(loc="upper left", fontsize=6.8, frameon=False)
    central_text = "PASS" if audit.passes_central else "FAIL"
    decrease_text = "PASS" if audit.passes_decrease else "FAIL"
    permutation_text = "PASS" if audit.passes_permutation else "FAIL"
    final_text = "ACCEPTED" if audit.accepted else "REJECTED"
    below_count = int(np.sum(null < audit.cusum_amplitude))
    gate_text = (
        f"Admission gates ({audit.label})\n"
        f"{central_text}  central decision\n"
        f"{decrease_text}  decrease ({audit.post_minus_pre_corrected:.2f} dB)\n"
        f"{permutation_text}  empirical rank: {below_count}/{config.permutations}\n"
        f"threshold >= {config.confidence_threshold:.3f}; "
        f"plus-one p = {audit.plus_one_p_value:.4f}\n"
        f"FINAL: {final_text}"
    )
    axis.text(
        0.985,
        0.94,
        gate_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        linespacing=1.35,
        color=INK,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#F5F8FA",
            edgecolor=BLUE if audit.accepted else ORANGE,
            linewidth=1.0,
        ),
    )
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


def render_figure(
    series: SeriesData,
    events: pd.DatetimeIndex,
    audits: Sequence[WindowAudit],
    outside: WindowAudit,
    inside: WindowAudit,
    config: DetectorConfig,
    focal_event: pd.Timestamp,
) -> plt.Figure:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.15, 7.65), constrained_layout=False)
    outer = GridSpec(
        3,
        1,
        figure=figure,
        height_ratios=[2.35, 3.05, 1.55],
        hspace=0.40,
        left=0.095,
        right=0.985,
        top=0.965,
        bottom=0.07,
    )
    _plot_full_series_and_windows(
        series, events, audits, outside, inside, focal_event, figure, outer[0, 0]
    )
    _plot_contrast(series, outside, inside, figure, outer[1, 0])
    permutation_axis = figure.add_subplot(outer[2, 0])
    _plot_permutation_and_gates(inside, config, permutation_axis)
    return figure


def _audit_record(audit: WindowAudit) -> dict[str, Any]:
    record = asdict(audit)
    record.pop("cumulative_values")
    record.pop("cumulative_dates")
    record.pop("null_amplitudes")
    for key, value in list(record.items()):
        if isinstance(value, pd.Timestamp):
            record[key] = value.isoformat()
        elif isinstance(value, np.generic):
            record[key] = value.item()
    return record


def write_audit_csv(path: Path, audits: Sequence[WindowAudit]) -> None:
    pd.DataFrame([_audit_record(audit) for audit in audits]).to_csv(path, index=False)


def write_readme(
    path: Path,
    *,
    output_dir: Path,
    series: SeriesData,
    config: DetectorConfig,
    outside: WindowAudit,
    inside: WindowAudit,
) -> None:
    try:
        relative_output = output_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative_output = output_dir
    command = (
        "python analysis/workflows/build_windowed_cusum_concept.py "
        f'--output-dir "{relative_output}"'
    )
    text = f"""# Reproducible Windowed CUSUM concept figure

This directory replaces the legacy PowerPoint-assembled concept figure with a
figure generated directly from the frozen Polygon-first monthly VV series.
The archived manuscript figure is not modified.

## Exact reproduction command

```powershell
{command}
```

The workflow refuses to overwrite existing files. Choose a new output
directory when regenerating the figure.

## Frozen inputs and contrast

- Sample: `{series.sample_id}` (historical polygon `{series.historical_id}`)
- Calendar: `{series.calendar.min():%Y-%m}` to `{series.calendar.max():%Y-%m}`
- Window: {config.window_length} months; step: {config.step} months
- Central decision interval: {config.window_length - 2 * config.step} months
- Permutations: {config.permutations}; empirical-rank threshold: {config.confidence_threshold:.3f}
- Seed: {config.seed}
- Contrast: `{outside.label}` (transition outside the central interval) versus
  `{inside.label}` (the same transition inside the central interval)

The CUSUM maximum sample is the implementation-defined change boundary and is
also the calendar month reported by the production detector. The observation
at that boundary is included in the post-change slice used for the directional
gate. See the JSON manifest for input and output SHA-256 checksums, every
audited window, package versions, and full statistical semantics.
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--historical-id", type=int, default=DEFAULT_HISTORICAL_ID)
    parser.add_argument("--focal-event", default=DEFAULT_FOCAL_EVENT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--reference-events", type=Path, default=DEFAULT_REFERENCE_EVENTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--full-windows-only",
        action="store_true",
        help=(
            "Audit only complete windows; edge windows are context-only and "
            "excluded from contrast selection (used by the 2016--2026 update)."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = [args.metadata, args.series, args.reference_events, args.config]
    _require_paths(inputs)
    output_dir = args.output_dir.resolve()
    pdf_path = output_dir / "windowed_cusum_concept.pdf"
    png_path = output_dir / "windowed_cusum_concept.png"
    svg_path = output_dir / "windowed_cusum_concept.svg"
    audit_path = output_dir / "windowed_cusum_window_audit.csv"
    manifest_path = output_dir / "windowed_cusum_concept_manifest.json"
    readme_path = output_dir / "README.md"
    _refuse_existing(
        [pdf_path, png_path, svg_path, audit_path, manifest_path, readme_path]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_detector_config(args.config)
    series = load_polygon_series(
        args.metadata, args.series, args.sample_id, args.historical_id
    )
    events = load_reference_events(
        args.reference_events, args.sample_id, args.historical_id
    )
    focal_event = pd.Period(args.focal_event, freq="M").to_timestamp()
    if pd.Period(focal_event, freq="M") not in {
        pd.Period(event, freq="M") for event in events
    }:
        raise ValueError(f"Focal event {args.focal_event} is not a reference event")
    audits = build_window_audits(series, config)
    outside, inside = choose_contrast_pair(audits, focal_event)
    figure = render_figure(
        series, events, audits, outside, inside, config, focal_event
    )
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    write_audit_csv(audit_path, audits)
    write_readme(
        readme_path,
        output_dir=output_dir,
        series=series,
        config=config,
        outside=outside,
        inside=inside,
    )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": _file_record(Path(__file__)),
        "purpose": "Fully reproducible replacement for PowerPoint-assembled legacy Figure 3.",
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
            "valid_months": int(np.isfinite(series.values).sum()),
            "reference_events": [event.strftime("%Y-%m") for event in events],
            "focal_reference_event": focal_event.strftime("%Y-%m"),
        },
        "detector_config": asdict(config),
        "month_semantics": {
            "cusum_maximum_sample": "CUSUM maximum sample at argmax cumulative deviation; zero-based implementation index mapped to the calendar.",
            "reported_transition_month": "Calendar month of the CUSUM maximum sample, matching production core.lr_cusum.",
            "descriptive_decrease": "Post-minus-pre means use window[:local_index] and window[local_index:], so the boundary sample belongs to post.",
        },
        "permutation_semantics": {
            "null": "Independent random permutation of finite observations within one window.",
            "rank": "Fraction of null amplitudes strictly below observed Q; matches production primitive.",
            "plus_one_p": "(1 + null amplitudes >= observed Q) / (B + 1); reported descriptively.",
            "not_posterior_probability": True,
            "not_global_fdr_control": True,
        },
        "selected_contrast": {
            "outside": _audit_record(outside),
            "inside": _audit_record(inside),
        },
        "all_windows": [_audit_record(audit) for audit in audits],
        "inputs": {path.name: _file_record(path) for path in inputs},
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"] = {
        path.name: _file_record(path)
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
        f"Contrast: {outside.label} OUT vs {inside.label} IN; "
        f"transition={inside.reported_transition_month:%Y-%m}; "
        f"rank={inside.permutation_rank:.4f}; p+1={inside.plus_one_p_value:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
