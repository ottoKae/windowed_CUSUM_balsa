"""Run the frozen N1 primitives with independently specified context and stride.

The production API uses ``step`` for both quantities. This small experimental
adapter changes only the window schedule, not CUSUM, missing-value handling,
direction gating, permutation scoring, or random seeds. Its A/B parity is
checked against the archived detector before C is run.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lr_cusum.core import cusum_amplitude  # noqa: E402
from lr_cusum.detection import stable_seed  # noqa: E402
from lr_cusum.paper import PaperDetectionConfig  # noqa: E402
from lr_cusum.temporal_nulls import NullModelConfig, iid_permutation_score  # noqa: E402


@dataclass(frozen=True)
class Geometry:
    window_months: int
    decision_months: int
    stride_months: int
    phase_index: int = 12  # January 2017 in the January 2016 calendar

    @property
    def margin_months(self) -> int:
        extra = self.window_months - self.decision_months
        if extra < 2 or extra % 2 or self.stride_months < 1:
            raise ValueError("W-D must be a positive even number; stride must be positive")
        return extra // 2


ARMS = {
    "W48_D24_S12": Geometry(48, 24, 12),
    "W36_D12_S12": Geometry(36, 12, 12),
    "W48_D24_S24": Geometry(48, 24, 24),
    "W48_D24_S24_OFFSET12": Geometry(48, 24, 24, 24),
}


def windows(length: int, geometry: Geometry, *, report_start: int = 12,
            report_end: int = 120):
    """Yield half-open W/D index ranges, including useful clipped edge windows.

    ``report_end=120`` is January 2026, hence the reported years are
    2017--2025. A negative logical decision start is allowed only for an
    offset phase; its clipped portion can still cover 2017.
    """
    margin = geometry.margin_months
    first = geometry.phase_index
    while first - geometry.stride_months + geometry.decision_months > report_start:
        first -= geometry.stride_months
    decision = first
    while decision < min(length, report_end):
        ds = max(0, decision)
        de = min(length, decision + geometry.decision_months)
        ws = max(0, decision - margin)
        we = min(length, decision + geometry.decision_months + margin)
        if ds < de and ws < we and de > report_start:
            yield ws, we, ds, de
        decision += geometry.stride_months


COLUMNS = [
    "Band", "change_time", "window_start", "window_end", "decision_start",
    "decision_end", "change_index", "cusum_amplitude", "pre_mean", "post_mean",
    "post_minus_pre", "null_p_value", "exceedance_count", "permutations",
    "near_threshold_refined", "candidate_id",
]


def detect_series(series: np.ndarray, calendar, *, pixel_track_id: str,
                  geometry: Geometry, config: PaperDetectionConfig) -> pd.DataFrame:
    """Return accepted VV candidates using production mathematical primitives."""
    dates = pd.DatetimeIndex(calendar)
    expected = pd.date_range("2016-01-01", periods=len(dates), freq="MS")
    if len(series) != len(dates) or not dates.equals(expected):
        raise ValueError("Expected an explicit, contiguous calendar from January 2016")
    values = np.asarray(series, dtype=np.float64)
    key = f"{pixel_track_id}:VV_dB"
    series_seed = stable_seed(key, config.candidate_seed)
    base = NullModelConfig(method="N1_IID_P025", replicates=config.permutations,
                           alpha=config.alpha)
    refined = NullModelConfig(method="N1_IID_P025", replicates=config.refined_permutations,
                              alpha=config.alpha)
    result = []
    for logical_id, (ws, we, ds, de) in enumerate(windows(len(values), geometry)):
        if dates[de - 1].year < config.start_year or dates[ds].year > config.end_year:
            continue
        window = values[ws:we]
        valid = np.flatnonzero(np.isfinite(window))
        if len(valid) < config.minimum_observations:
            continue
        amplitude, clean_index = cusum_amplitude(window[valid])
        if clean_index < 0:
            continue
        local = int(valid[int(clean_index)])
        global_index = ws + local
        if local == 0 or not (ds <= global_index < de):
            continue
        date = dates[global_index]
        if not (config.start_year <= date.year <= config.end_year):
            continue
        before, after = window[:local], window[local:]
        if not (np.isfinite(before).any() and np.isfinite(after).any()):
            continue
        pre_mean, post_mean = float(np.nanmean(before)), float(np.nanmean(after))
        if not (post_mean < pre_mean):
            continue
        seed = int(series_seed + ws)
        score = iid_permutation_score(window, float(amplitude), config=base, seed=seed)
        refine = config.refine_low <= score.null_p_value <= config.refine_high
        if refine:
            score = iid_permutation_score(window, float(amplitude), config=refined, seed=seed)
        if not score.admitted:
            continue
        result.append(dict(
            Band="VV_dB", change_time=date, window_start=dates[ws],
            window_end=dates[we - 1], decision_start=dates[ds],
            decision_end=dates[de - 1], change_index=global_index,
            cusum_amplitude=float(amplitude), pre_mean=pre_mean,
            post_mean=post_mean, post_minus_pre=post_mean - pre_mean,
            null_p_value=float(score.null_p_value),
            exceedance_count=int(score.exceedance_count),
            permutations=int(score.replicates), near_threshold_refined=bool(refine),
            candidate_id=f"{key}|sliding|w{logical_id}",
        ))
    return pd.DataFrame(result, columns=COLUMNS)
