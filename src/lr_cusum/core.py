"""Core sliding-window LR-CuSum algorithm extracted from notebook cell 39."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - functional fallback for light environments
    def njit(function):
        return function


@dataclass(frozen=True)
class ChangeEvent:
    change_index: int
    change_time: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    cusum_amplitude: float
    bootstrap_confidence: float

    def as_dict(self) -> dict:
        return {
            "change_index": self.change_index,
            "change_time": self.change_time,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "cusum_amplitude": self.cusum_amplitude,
            "bootstrap_confidence": self.bootstrap_confidence,
        }


@njit
def cusum_amplitude(series):
    """Return CUSUM range and the index of its maximum cumulative residual."""
    length = len(series)
    if length < 3:
        return 0.0, -1
    mean = 0.0
    for index in range(length):
        mean += series[index]
    mean /= length
    accumulator = 0.0
    minimum = 1e15
    maximum = -1e15
    change_index = 0
    for index in range(length):
        accumulator += series[index] - mean
        if accumulator < minimum:
            minimum = accumulator
        if accumulator > maximum:
            maximum = accumulator
            change_index = index
    return maximum - minimum, change_index


@njit
def bootstrap_confidence(series, observed_amplitude, iterations, seed):
    """Permutation confidence used by the original notebook implementation."""
    np.random.seed(seed)
    length = len(series)
    if length < 3:
        return 0.0
    mean = 0.0
    for index in range(length):
        mean += series[index]
    mean /= length
    below_observed = 0
    for _ in range(iterations):
        permutation = np.random.permutation(length)
        accumulator = 0.0
        minimum = 1e15
        maximum = -1e15
        for index in range(length):
            accumulator += series[permutation[index]] - mean
            if accumulator < minimum:
                minimum = accumulator
            if accumulator > maximum:
                maximum = accumulator
        if maximum - minimum < observed_amplitude:
            below_observed += 1
    return below_observed / iterations


def generate_windows(length: int, window_length: int, step: int) -> Iterable[tuple[int, int, int, int]]:
    """Yield window and central decision ranges with complete edge coverage."""
    if length < 1:
        return
    if window_length < 3 or step < 1:
        raise ValueError("window_length must be >= 3 and step must be >= 1")
    for start in range(-step, length, step):
        window_start = max(0, start)
        window_end = min(length, start + window_length)
        decision_start = max(0, start + step)
        decision_end = min(length, start + window_length - step)
        if decision_start < decision_end:
            yield window_start, window_end, decision_start, decision_end


def _calendar_align(series, dates):
    dates = pd.to_datetime(np.asarray(dates))
    if len(series) != len(dates):
        raise ValueError("series and dates must have equal lengths")
    if len(dates) == 0:
        return np.asarray(series, np.float64), dates
    values = np.asarray(series, np.float64)
    if dates[0].month != 1 and len(values) > 12:
        count = dates[0].month - 1
        prefix_dates = pd.DatetimeIndex([
            dates[0] - pd.DateOffset(months=count - index) for index in range(count)
        ])
        values = np.concatenate((np.full(count, np.nan), values))
        dates = prefix_dates.append(pd.DatetimeIndex(dates))
    return values, pd.DatetimeIndex(dates)


def lr_cusum(series, dates, *, window_length=36, step=12, confidence_threshold=.95,
             bootstrap_iterations=800, seed=20, start_year=None, end_year=None,
             minimum_observations=12, require_decrease=True) -> list[ChangeEvent]:
    """Detect significant directional changes with calendar-aligned sliding windows."""
    values, dates = _calendar_align(series, dates)
    events: list[ChangeEvent] = []
    for window_start, window_end, decision_start, decision_end in generate_windows(
        len(values), window_length, step
    ):
        if start_year is not None and dates[decision_end - 1].year < start_year:
            continue
        if end_year is not None and dates[decision_start].year > end_year:
            continue
        window = values[window_start:window_end]
        valid_indices = np.flatnonzero(np.isfinite(window))
        if len(valid_indices) < minimum_observations:
            continue
        clean = window[valid_indices]
        amplitude, clean_index = cusum_amplitude(clean)
        if clean_index < 0:
            continue
        local_index = int(valid_indices[clean_index])
        global_index = window_start + local_index
        if not decision_start <= global_index < decision_end:
            continue
        event_time = pd.Timestamp(dates[global_index])
        if start_year is not None and event_time.year < start_year:
            continue
        if end_year is not None and event_time.year > end_year:
            continue
        if local_index == 0 or local_index >= len(window):
            continue
        before_window = window[:local_index]
        after_window = window[local_index:]
        if not np.isfinite(before_window).any() or not np.isfinite(after_window).any():
            continue
        before = np.nanmean(before_window)
        after = np.nanmean(after_window)
        if not np.isfinite(before) or not np.isfinite(after):
            continue
        if require_decrease and after >= before:
            continue
        confidence = bootstrap_confidence(
            clean, amplitude, bootstrap_iterations, seed + window_start
        )
        if confidence < confidence_threshold:
            continue
        events.append(ChangeEvent(
            change_index=global_index,
            change_time=event_time,
            window_start=pd.Timestamp(dates[window_start]),
            window_end=pd.Timestamp(dates[window_end - 1]),
            cusum_amplitude=float(amplitude),
            bootstrap_confidence=float(confidence),
        ))
    return events


# Backward-compatible notebook spelling.
LR_CuSum = lr_cusum
