"""Temporal null models for Windowed CUSUM candidate screening.

This module is deliberately additive.  The production implementation in
``core.py`` remains the immutable legacy path used by the archived F4 result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .core import bootstrap_confidence, cusum_amplitude

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@dataclass(frozen=True)
class NullModelConfig:
    """Configuration for one candidate-level Monte Carlo null."""

    method: str
    replicates: int
    alpha: float = 0.025
    block_length: int | None = None
    seed: int = 20260829

    def __post_init__(self) -> None:
        if self.replicates < 1:
            raise ValueError("replicates must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.block_length is not None and self.block_length < 1:
            raise ValueError("block_length must be positive")


@dataclass(frozen=True)
class NullScore:
    """Auditable result from one candidate-level surrogate null."""

    observed_statistic: float
    exceedance_count: int
    replicates: int
    null_p_value: float
    admitted: bool
    method: str
    block_length: int | None
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_statistic": self.observed_statistic,
            "exceedance_count": self.exceedance_count,
            "replicates": self.replicates,
            "null_p_value": self.null_p_value,
            "admitted": self.admitted,
            "null_model_id": self.method,
            "block_length": self.block_length,
            "null_seed": self.seed,
        }


def monte_carlo_pvalue(exceedance_count: int, replicates: int) -> float:
    """Return the finite-sample corrected Monte Carlo p-value."""

    exceedance_count = int(exceedance_count)
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0 <= exceedance_count <= replicates:
        raise ValueError("exceedance_count must lie in [0, replicates]")
    return float((1 + exceedance_count) / (replicates + 1))


def derive_candidate_seed(candidate_id: str, method: str, base_seed: int) -> int:
    """Derive a stable platform-independent positive 31-bit seed."""

    payload = f"{int(base_seed)}|{method}|{candidate_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return int(value % (2**31 - 1))


def _validate_window(window) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("window must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("window contains infinite values")
    if np.isfinite(values).sum() < 3:
        raise ValueError("window must contain at least three finite observations")
    return values


def _score(
    observed_statistic: float,
    exceedance_count: int,
    config: NullModelConfig,
    seed: int,
) -> NullScore:
    p_value = monte_carlo_pvalue(exceedance_count, config.replicates)
    return NullScore(
        observed_statistic=float(observed_statistic),
        exceedance_count=int(exceedance_count),
        replicates=int(config.replicates),
        null_p_value=p_value,
        admitted=bool(p_value <= config.alpha),
        method=str(config.method),
        block_length=config.block_length,
        seed=int(seed),
    )


def iid_permutation_score(
    window,
    observed_statistic: float,
    *,
    config: NullModelConfig,
    seed: int | None = None,
) -> NullScore:
    """Score a candidate against independent permutations of finite values.

    The legacy primitive is reused so the randomization semantics are exactly
    those of the archived detector.  The admission expression changes to the
    finite-sample corrected p-value requested for the new experiment.
    """

    values = _validate_window(window)
    clean = np.ascontiguousarray(values[np.isfinite(values)], dtype=np.float64)
    local_seed = int(config.seed if seed is None else seed)
    below_fraction = float(
        bootstrap_confidence(
            clean,
            float(observed_statistic),
            int(config.replicates),
            local_seed,
        )
    )
    below = int(round(below_fraction * config.replicates))
    exceedance_count = int(config.replicates - below)
    return _score(observed_statistic, exceedance_count, config, local_seed)


@njit(cache=True)
def _cusum_range_finite(values, length):
    mean = 0.0
    for index in range(length):
        mean += values[index]
    mean /= length
    accumulator = 0.0
    minimum = 1e15
    maximum = -1e15
    for index in range(length):
        accumulator += values[index] - mean
        if accumulator < minimum:
            minimum = accumulator
        if accumulator > maximum:
            maximum = accumulator
    return maximum - minimum


@njit(cache=True)
def _block_permutation_exceedances(
    window,
    observed_statistic,
    replicates,
    seed,
    block_length,
):
    np.random.seed(seed)
    length = len(window)
    work = np.empty(length, dtype=np.float64)
    starts = np.empty(length, dtype=np.int64)
    ends = np.empty(length, dtype=np.int64)
    exceedances = 0

    for _ in range(replicates):
        offset = np.random.randint(0, min(block_length, length))
        block_count = 0
        if offset > 0:
            starts[block_count] = 0
            ends[block_count] = offset
            block_count += 1
        position = offset
        while position < length:
            starts[block_count] = position
            ends[block_count] = min(length, position + block_length)
            block_count += 1
            position += block_length

        order = np.random.permutation(block_count)
        finite_count = 0
        for order_index in range(block_count):
            block = order[order_index]
            for source_index in range(starts[block], ends[block]):
                value = window[source_index]
                if np.isfinite(value):
                    work[finite_count] = value
                    finite_count += 1
        statistic = _cusum_range_finite(work, finite_count)
        if statistic >= observed_statistic:
            exceedances += 1
    return exceedances


def calendar_block_permutation_score(
    window,
    observed_statistic: float,
    *,
    config: NullModelConfig,
    seed: int | None = None,
) -> NullScore:
    """Reorder contiguous calendar blocks while retaining within-block order."""

    values = _validate_window(window)
    if config.block_length is None:
        raise ValueError("calendar block permutation requires block_length")
    if config.block_length >= len(values):
        raise ValueError("block_length must be smaller than the calendar window")
    local_seed = int(config.seed if seed is None else seed)
    exceedances = int(
        _block_permutation_exceedances(
            np.ascontiguousarray(values),
            float(observed_statistic),
            int(config.replicates),
            local_seed,
            int(config.block_length),
        )
    )
    return _score(observed_statistic, exceedances, config, local_seed)


@njit(cache=True)
def _residual_block_bootstrap_exceedances(
    residuals,
    observed_statistic,
    replicates,
    seed,
    block_length,
):
    np.random.seed(seed)
    length = len(residuals)
    finite_target = 0
    for index in range(length):
        if np.isfinite(residuals[index]):
            finite_target += 1
    work = np.empty(finite_target, dtype=np.float64)
    exceedances = 0
    maximum_start = max(1, length - block_length + 1)

    for _ in range(replicates):
        filled = 0
        attempts = 0
        while filled < finite_target:
            start = np.random.randint(0, maximum_start)
            end = min(length, start + block_length)
            for source_index in range(start, end):
                value = residuals[source_index]
                if np.isfinite(value):
                    work[filled] = value
                    filled += 1
                    if filled == finite_target:
                        break
            attempts += 1
            if attempts > length * 100:
                return -1
        statistic = _cusum_range_finite(work, finite_target)
        if statistic >= observed_statistic:
            exceedances += 1
    return exceedances


def residual_moving_block_bootstrap_score(
    window,
    observed_statistic: float,
    change_index: int,
    *,
    config: NullModelConfig,
    seed: int | None = None,
) -> NullScore:
    """Score against a no-step residual moving-block bootstrap null."""

    values = _validate_window(window)
    if config.block_length is None:
        raise ValueError("residual block bootstrap requires block_length")
    change_index = int(change_index)
    if not 0 < change_index < len(values):
        raise ValueError("change_index must split the calendar window")
    before = values[:change_index]
    after = values[change_index:]
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        raise ValueError("candidate split must have finite data on both sides")
    residuals = values.copy()
    residuals[:change_index] -= float(np.nanmean(before))
    residuals[change_index:] -= float(np.nanmean(after))
    residuals[np.isfinite(residuals)] -= float(np.nanmean(residuals))
    local_seed = int(config.seed if seed is None else seed)
    exceedances = int(
        _residual_block_bootstrap_exceedances(
            np.ascontiguousarray(residuals),
            float(observed_statistic),
            int(config.replicates),
            local_seed,
            int(config.block_length),
        )
    )
    if exceedances < 0:
        raise RuntimeError("residual block bootstrap failed to draw finite blocks")
    return _score(observed_statistic, exceedances, config, local_seed)


def circular_shift_ranges(window) -> np.ndarray:
    """Return all cyclic-shift CUSUM ranges for a complete finite sequence.

    This diagnostic exists to document why circular shifts are not a useful
    null for the max-minus-min CUSUM range statistic.
    """

    values = _validate_window(window)
    if not np.isfinite(values).all():
        raise ValueError("circular-shift invariance diagnostic requires no gaps")
    return np.asarray(
        [float(cusum_amplitude(np.roll(values, shift))[0]) for shift in range(len(values))]
    )


def block_permutation_once(window, block_length: int, seed: int) -> np.ndarray:
    """Reference block permutation used by tests and audit illustrations."""

    values = np.asarray(window)
    if values.ndim != 1:
        raise ValueError("window must be one-dimensional")
    if not 0 < block_length < len(values):
        raise ValueError("block_length must lie in [1, len(window) - 1]")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, min(block_length, len(values))))
    blocks = []
    if offset:
        blocks.append(values[:offset])
    blocks.extend(
        values[start : min(len(values), start + block_length)]
        for start in range(offset, len(values), block_length)
    )
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[int(index)] for index in order])


@njit(cache=True)
def _cusum_range_and_max_index(values, length):
    mean = 0.0
    for index in range(length):
        mean += values[index]
    mean /= length
    accumulator = 0.0
    minimum = 1e15
    maximum = -1e15
    maximum_index = -1
    for index in range(length):
        accumulator += values[index] - mean
        if accumulator < minimum:
            minimum = accumulator
        if accumulator > maximum:
            maximum = accumulator
            maximum_index = index
    return maximum - minimum, maximum_index


@njit(cache=True)
def _trackwise_block_maxima(
    series,
    replicates,
    seed,
    block_length,
    window_starts,
    window_ends,
    decision_starts,
    decision_ends,
    minimum_observations,
    reporting_start_index,
    reporting_end_index,
):
    """Generate one maximum eligible-window statistic per surrogate track."""

    np.random.seed(seed)
    length = len(series)
    surrogate = np.empty(length, dtype=np.float64)
    clean = np.empty(length, dtype=np.float64)
    clean_positions = np.empty(length, dtype=np.int64)
    starts = np.empty(length, dtype=np.int64)
    ends = np.empty(length, dtype=np.int64)
    maxima = np.zeros(replicates, dtype=np.float64)

    for replicate in range(replicates):
        offset = np.random.randint(0, min(block_length, length))
        block_count = 0
        if offset > 0:
            starts[block_count] = 0
            ends[block_count] = offset
            block_count += 1
        position = offset
        while position < length:
            starts[block_count] = position
            ends[block_count] = min(length, position + block_length)
            block_count += 1
            position += block_length
        order = np.random.permutation(block_count)
        output_index = 0
        for order_index in range(block_count):
            block = order[order_index]
            for source_index in range(starts[block], ends[block]):
                surrogate[output_index] = series[source_index]
                output_index += 1

        largest = 0.0
        for window_number in range(len(window_starts)):
            window_start = int(window_starts[window_number])
            window_end = int(window_ends[window_number])
            finite_count = 0
            for local_index in range(window_end - window_start):
                value = surrogate[window_start + local_index]
                if np.isfinite(value):
                    clean[finite_count] = value
                    clean_positions[finite_count] = local_index
                    finite_count += 1
            if finite_count < minimum_observations:
                continue
            statistic, clean_index = _cusum_range_and_max_index(clean, finite_count)
            if clean_index < 0:
                continue
            local_change = int(clean_positions[clean_index])
            global_change = window_start + local_change
            if global_change < reporting_start_index or global_change >= reporting_end_index:
                continue
            if (
                global_change < decision_starts[window_number]
                or global_change >= decision_ends[window_number]
            ):
                continue
            if local_change == 0:
                continue
            pre_sum = 0.0
            post_sum = 0.0
            pre_count = 0
            post_count = 0
            for local_index in range(window_end - window_start):
                value = surrogate[window_start + local_index]
                if not np.isfinite(value):
                    continue
                if local_index < local_change:
                    pre_sum += value
                    pre_count += 1
                else:
                    post_sum += value
                    post_count += 1
            if pre_count == 0 or post_count == 0:
                continue
            if post_sum / post_count >= pre_sum / pre_count:
                continue
            if statistic > largest:
                largest = statistic
        maxima[replicate] = largest
    return maxima


def trackwise_max_block_scores(
    series,
    candidate_statistics,
    *,
    config: NullModelConfig,
    seed: int,
    window_starts,
    window_ends,
    decision_starts,
    decision_ends,
    minimum_observations: int,
    reporting_start_index: int,
    reporting_end_index: int,
) -> list[NullScore]:
    """Score candidates against a track-wise maximum block-null distribution."""

    values = _validate_window(series)
    if config.block_length is None:
        raise ValueError("track-wise block maximum requires block_length")
    statistics = np.asarray(candidate_statistics, dtype=np.float64)
    maxima = _trackwise_block_maxima(
        np.ascontiguousarray(values),
        int(config.replicates),
        int(seed),
        int(config.block_length),
        np.asarray(window_starts, dtype=np.int64),
        np.asarray(window_ends, dtype=np.int64),
        np.asarray(decision_starts, dtype=np.int64),
        np.asarray(decision_ends, dtype=np.int64),
        int(minimum_observations),
        int(reporting_start_index),
        int(reporting_end_index),
    )
    return [
        _score(
            float(statistic),
            int(np.count_nonzero(maxima >= float(statistic))),
            config,
            int(seed),
        )
        for statistic in statistics
    ]
