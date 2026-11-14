"""Monthly stand adaptation of Ygorra's CuSum NRT numerical core.

Source: INRAE cusum-deforestation_monitoring, commit
8ed46c036d7947633b4e7eedf495d295f660d603, CuSum_NRT_example.py:529-563.
Author core copyright (c) 2023 Bertrand Ygorra, MIT licensed;
see docs/third_party/CuSum_NRT_LICENSE.txt for the retained notice.
No spatial cross-Tc or permanent forest exclusion is implied by this adapter.
Missing values are removed with their calendar indices retained, never filled.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from numba import njit


@lru_cache(maxsize=128)
def permutations(n, count, seed):
    """Match the author's initial permutation followed by in-place shuffles."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    out = np.empty((count, n), dtype=np.int64)
    for b in range(count):
        out[b] = order
        rng.shuffle(order)
    return out


@njit(cache=True)
def score_windows(windows, orders):
    """Return Q, first-maximum index, native Tc and corrected p for each row."""
    result = np.empty((len(windows), 4), dtype=np.float64)
    for r in range(len(windows)):
        x = windows[r]
        avg = np.mean(x)
        c = np.cumsum(x - avg)
        q = np.max(c) - np.min(c)
        ci = 1.0
        exceed = 0
        for order in orders:
            shuffled = np.cumsum(x[order] - avg)
            if np.max(shuffled) - np.min(shuffled) >= q:
                ci -= 1.0 / len(orders)
                exceed += 1
        result[r, 0] = q
        result[r, 1] = np.argmax(c)
        result[r, 2] = ci
        result[r, 3] = (exceed + 1.0) / (len(orders) + 1.0)
    return result


def scan_nrt(values, dates, *, before=11, after=3, samples=500,
             seed=20260905, window_unit="observations"):
    """Keep EVERY scanned candidate, including unsupported/downstream rejects.

    values: (trajectory, month). Caller attaches immutable trajectory identities.
    The earliest available alert is the last REAL month used by the window.
    No central-window admission is borrowed from Windowed CUSUM.
    """
    values = np.asarray(values, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)
    if values.ndim != 2 or values.shape[1] != len(dates):
        raise ValueError("expected trajectories x dates")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be sorted and unique")
    if before < 1 or after < 0 or samples < 1:
        raise ValueError("invalid NRT window/permutation count")
    if window_unit not in ("observations", "calendar"):
        raise ValueError(window_unit)
    groups = {}
    for row, x in enumerate(values):
        finite = np.flatnonzero(np.isfinite(x))
        if window_unit == "observations":
            candidates = ((int(finite[j]), finite[j-before:j+after+1])
                          for j in range(before, len(finite)-after))
        else:
            candidates = ((j, finite[(finite >= j-before) & (finite <= j+after)])
                          for j in range(before, len(dates)-after))
        for target, idx in candidates:
            if len(idx) < 3:
                continue
            # Calendar windows become available at their calendar end, even if missing.
            available = int(idx[-1]) if window_unit == "observations" else target+after
            groups.setdefault(len(idx), []).append((row, target, available, idx))
    records = []
    for n, items in groups.items():
        order = permutations(n, samples, seed)
        for offset in range(0, len(items), 2048):
            chunk = items[offset:offset+2048]
            windows = np.stack([values[row, idx] for row, _, _, idx in chunk])
            scores = score_windows(windows, order)
            for item, x, (q, k, tc, p) in zip(chunk, windows, scores):
                row, target, available, idx = item
                k = int(k)
                pre = float(np.mean(x[:k])) if k else np.nan
                post = float(np.mean(x[k:]))
                records.append({
                    "trajectory_index": row, "target_month": dates[target],
                    "change_time": dates[int(idx[k])],
                    "available_through_month": dates[available],
                    "context_start": dates[int(idx[0])], "context_end": dates[int(idx[-1])],
                    "raw_boundary_index": k, "n_obs": n,
                    "q": q, "native_confidence": tc, "corrected_p": p,
                    "pre_mean": pre, "post_mean": post,
                    "decrease": bool(k > 0 and post < pre),
                    "before": before, "after": after, "window_unit": window_unit,
                    "permutations": samples, "seed": seed,
                })
    return pd.DataFrame(records)


def admitted(raw, confidence=0.92, *, corrected=False, decrease=False, cutoff=None):
    if raw.empty:
        return raw.copy()
    mask = raw.corrected_p.le(1-confidence) if corrected else raw.native_confidence.ge(confidence)
    if decrease:
        mask &= raw.decrease
    if cutoff is not None:
        mask &= raw.available_through_month.le(pd.Timestamp(cutoff))
    return raw.loc[mask].copy()
