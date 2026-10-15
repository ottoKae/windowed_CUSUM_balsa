"""Window-local gap filling and fixed-mask, whole-pipeline randomization.

Experiments only: no production detector or source observation is modified.
Observed values remain exact; interpolation never creates acquisition counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve
from numba import njit

from .core import cusum_amplitude, generate_windows

METHODS = ("OBS", "LIN-G1", "LIN-G2", "PCHIP-G2", "WHIT-G2")


def gap_runs(observed):
    """Half-open missing runs, including unfillable calendar endpoints."""
    observed = np.asarray(observed, bool)
    edges = np.diff(np.r_[False, ~observed, False].astype(np.int8))
    return [(int(a), int(b), a > 0 and b < len(observed))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


class GapFiller:
    """One fixed-mask operator, reused across observed and surrogate windows."""

    def __init__(self, observed, method, whittaker_lambda=10.0):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.observed = np.asarray(observed, bool).copy()
        if self.observed.ndim != 1 or not len(self.observed):
            raise ValueError("A nonempty one-dimensional observation mask is required")
        if not np.isfinite(whittaker_lambda) or whittaker_lambda <= 0:
            raise ValueError("Whittaker lambda must be finite and positive")
        self.method = method
        self.positions = np.flatnonzero(self.observed)
        self.imputed = np.zeros(len(self.observed), bool)
        limit = 0 if method == "OBS" else (1 if method == "LIN-G1" else 2)
        for a, b, inside in gap_runs(self.observed):
            if inside and b - a <= limit:
                self.imputed[a:b] = True
        self.targets = np.flatnonzero(self.imputed)
        self.weights = None
        if len(self.targets) and method.startswith("LIN"):
            weights = np.zeros((len(self.targets), len(self.positions)))
            for row, month in enumerate(self.targets):
                right = int(np.searchsorted(self.positions, month))
                left = right - 1
                fraction = (month - self.positions[left]) / (self.positions[right] - self.positions[left])
                weights[row, left], weights[row, right] = 1 - fraction, fraction
            self.weights = weights
        elif len(self.targets) and method == "WHIT-G2":
            # Fit with zero weight for absent months. Only missing predictions
            # are retained; fitted values never replace real observations.
            difference = np.diff(np.eye(len(self.observed)), n=2, axis=0)
            system = np.diag(self.observed.astype(float)) + whittaker_lambda * difference.T @ difference
            self.weights = solve(system, np.eye(len(self.observed))[:, self.positions], assume_a="pos")[self.targets]

    def apply(self, observed_values):
        """Accept [..., n_real] values; return [..., calendar_length]."""
        values = np.asarray(observed_values, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[-1] != len(self.positions) or not np.isfinite(values).all():
            raise ValueError("Finite values for exactly the observed positions are required")
        filled = np.full(values.shape[:-1] + (len(self.observed),), np.nan)
        filled[..., self.positions] = values
        if len(self.targets):
            if self.method == "PCHIP-G2":
                filled[..., self.targets] = PchipInterpolator(self.positions, values, axis=-1, extrapolate=False)(self.targets)
            else:
                filled[..., self.targets] = values @ self.weights.T
        return filled


@njit(cache=True)
def amplitudes(windows):
    """Exactly reuse the production CUSUM primitive, not a new statistic."""
    scores = np.empty(len(windows))
    for row in range(len(windows)):
        clean = windows[row][np.isfinite(windows[row])]
        scores[row] = cusum_amplitude(clean)[0]
    return scores


def permutation_pvalue(window, method, *, replicates, seed, whittaker_lambda=10.0, naive=False):
    """Permute real values, then refill every surrogate at the fixed mask.

IID exchangeability remains an assumption. `naive` is a diagnostic negative
control: fill once and shuffle the enlarged sequence. Never use it in maps.
"""
    x = np.asarray(window, float)
    if x.ndim != 1 or np.isinf(x).any() or np.isfinite(x).sum() < 3 or replicates < 1:
        raise ValueError("Need >=3 finite observations and positive replicates")
    operator = GapFiller(np.isfinite(x), method, whittaker_lambda)
    real = x[operator.observed]
    filled = operator.apply(real)
    statistic = float(cusum_amplitude(filled[np.isfinite(filled)])[0])
    rng = np.random.RandomState(seed)
    exceedances = 0
    values = filled[np.isfinite(filled)] if naive else real
    for start in range(0, replicates, 256):
        permutations = np.stack([rng.permutation(values) for _ in range(min(256, replicates - start))])
        surrogates = permutations if naive else operator.apply(permutations)
        exceedances += int(np.count_nonzero(amplitudes(surrogates) >= statistic))
    return (1 + exceedances) / (replicates + 1)


def candidate(window, method, whittaker_lambda=10.0):
    """Preserve production split semantics; additionally identify imputed dates."""
    x = np.asarray(window, float)
    if x.ndim != 1 or np.isinf(x).any():
        raise ValueError("A one-dimensional finite/NaN window is required")
    operator = GapFiller(np.isfinite(x), method, whittaker_lambda)
    filled = operator.apply(x[operator.observed])
    positions = np.flatnonzero(np.isfinite(filled))
    amplitude, clean_index = cusum_amplitude(filled[positions])
    if clean_index < 0:
        return None
    index = int(positions[clean_index])
    # The number of real observations, not interpolated values, controls support.
    before = x[:index]; after = x[index:]
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        return None
    previous = np.flatnonzero(operator.observed & (np.arange(len(x)) < index))
    following = np.flatnonzero(operator.observed & (np.arange(len(x)) > index))
    return dict(local_index=index, cusum_amplitude=float(amplitude),
                pre_mean=float(np.nanmean(filled[:index])), post_mean=float(np.nanmean(filled[index:])),
                real_observations=int(operator.observed.sum()), imputed_months=int(operator.imputed.sum()),
                event_month_is_imputed=bool(operator.imputed[index]),
                previous_observed_index=int(previous[-1]) if len(previous) else None,
                next_observed_index=int(following[0]) if len(following) else None,
                gap_length_months=next((b-a for a,b,_ in gap_runs(operator.observed) if a <= index < b), 0))


def scan_candidates(values, dates, method, *, config, seed):
    """Experimental full scan; no production files or candidate dates are reused."""
    values = np.asarray(values, float)
    dates = pd.DatetimeIndex(dates)
    if values.shape != (len(dates),) or not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="MS")):
        raise ValueError("Use a contiguous monthly calendar with explicit NaN rows")
    rows = []
    for a, b, left, right in generate_windows(len(values), config["window_length"], config["step"]):
        window = values[a:b]
        if np.isfinite(window).sum() < config["minimum_observations"]:
            continue
        event = candidate(window, method, config["whittaker_lambda"])
        if event is None:
            continue
        index = a + event["local_index"]
        if not left <= index < right or not 2017 <= dates[index].year <= 2025 or event["post_mean"] >= event["pre_mean"]:
            continue
        p = permutation_pvalue(window, method, replicates=config["calibration_permutations"], seed=seed+a,
                               whittaker_lambda=config["whittaker_lambda"])
        rows.append(dict(**event, change_time=dates[index], window_start=dates[a], window_end=dates[b-1],
                         null_p_value=p, admitted=p <= config["alpha"]))
    return pd.DataFrame(rows)
