"""Locked BALSA event-validation helpers and transparent detector baselines."""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


def month_number(values) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    return np.asarray(dates.year * 12 + dates.month, dtype=np.int32)


def _separated_candidates(times, scores, minimum_separation: int):
    """Retain highest-scoring candidates first, enforcing a month separation."""
    times = np.asarray(times, dtype=np.int32)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    retained: list[int] = []
    for index in order:
        if not np.isfinite(scores[index]):
            continue
        if all(abs(int(times[index] - times[other])) > minimum_separation for other in retained):
            retained.append(int(index))
    return sorted(retained, key=lambda index: int(times[index]))


def maximum_drop_events(
    values,
    dates,
    *,
    threshold_db: float,
    minimum_separation: int = 3,
    start_year: int = 2017,
    end_year: int = 2025,
) -> list[dict]:
    """Detect local negative month-to-month drops above a fixed dB threshold."""
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    months = month_number(dates)
    candidates, scores = [], []
    for index in range(1, len(values)):
        if not (np.isfinite(values[index - 1]) and np.isfinite(values[index])):
            continue
        if months[index] - months[index - 1] != 1:
            continue
        score = float(values[index - 1] - values[index])
        if score >= threshold_db and start_year <= dates[index].year <= end_year:
            candidates.append(index); scores.append(score)
    if not candidates:
        return []
    candidate_months = months[np.asarray(candidates)]
    selected = _separated_candidates(candidate_months, scores, minimum_separation)
    return [
        {
            "change_time": pd.Timestamp(dates[candidates[index]]),
            "score": float(scores[index]),
            "threshold": float(threshold_db),
        }
        for index in selected
    ]


def classical_lower_cusum_events(
    values,
    dates,
    *,
    allowance: float,
    decision_interval: float,
    reset_gap_months: int = 2,
    minimum_separation: int = 3,
    start_year: int = 2017,
    end_year: int = 2025,
) -> list[dict]:
    """Classical standardized lower one-sided CUSUM with an explicit gap reset."""
    values = np.asarray(values, dtype=float)
    dates = pd.DatetimeIndex(dates)
    valid = np.isfinite(values)
    if valid.sum() < 12:
        return []
    centre = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - centre)))
    scale = max(1.4826 * mad, 0.25)
    z = (values - centre) / scale
    months = month_number(dates)
    cumulative = 0.0; previous_valid = None; candidates = []; scores = []
    for index, value in enumerate(z):
        if not np.isfinite(value):
            continue
        if previous_valid is not None and months[index] - months[previous_valid] > reset_gap_months:
            cumulative = 0.0
        previous_valid = index
        cumulative = min(0.0, cumulative + float(value) + allowance)
        score = -cumulative
        if score >= decision_interval:
            if start_year <= dates[index].year <= end_year:
                candidates.append(index); scores.append(score)
            cumulative = 0.0
    if not candidates:
        return []
    selected = _separated_candidates(months[np.asarray(candidates)], scores, minimum_separation)
    return [
        {
            "change_time": pd.Timestamp(dates[candidates[index]]),
            "score": float(scores[index]),
            "threshold": float(decision_interval),
        }
        for index in selected
    ]


EVENT_MATCHER_GREEDY_V1 = "greedy_v1"
EVENT_MATCHER_OPTIMAL_V2 = "optimal_v2"


def _event_match_metrics(refs, det, rows) -> tuple[dict, pd.DataFrame]:
    """Build the shared event-matching metric contract."""
    true_positive = len(rows)
    fp = len(det) - true_positive
    fn = len(refs) - true_positive
    precision = true_positive / max(1, true_positive + fp)
    recall = true_positive / max(1, true_positive + fn)
    matched = pd.DataFrame(rows)
    metrics = {
        "tp": int(true_positive), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
        "temporal_mae_months": float(matched.absolute_error_months.mean()) if len(matched) else np.nan,
        "median_signed_error_months": float(matched.signed_error_months.median()) if len(matched) else np.nan,
        "reference_events": int(len(refs)), "detected_events": int(len(det)),
        "objects": int(refs.sample_id.nunique()),
    }
    return metrics, matched


def match_events_detailed_greedy_v1(
    references,
    detections,
    tolerance_months: int = 2,
) -> tuple[dict, pd.DataFrame]:
    """Legacy greedy one-to-one event matcher.

    This function preserves the original ``match_events_detailed`` behavior for
    reproducibility.  It considers candidate pairs in increasing absolute-lag
    order, but it is not guaranteed to maximize the number of matches.
    """
    refs = references[["sample_id", "event_month"]].drop_duplicates().copy()
    det = detections[["sample_id", "change_time"]].drop_duplicates().copy()
    rows = []
    identifiers = sorted(set(refs.sample_id.astype(str)) | set(det.sample_id.astype(str)))
    for identifier in identifiers:
        local_r = pd.to_datetime(refs.loc[refs.sample_id.astype(str).eq(identifier), "event_month"])
        local_d = pd.to_datetime(det.loc[det.sample_id.astype(str).eq(identifier), "change_time"])
        rm = month_number(local_r); dm = month_number(local_d)
        candidates = sorted(
            (abs(int(dv - rv)), int(dv - rv), ri, di)
            for ri, rv in enumerate(rm) for di, dv in enumerate(dm)
            if abs(int(dv - rv)) <= tolerance_months
        )
        used_r, used_d = set(), set()
        for absolute, signed, ri, di in candidates:
            if ri in used_r or di in used_d:
                continue
            used_r.add(ri); used_d.add(di)
            rows.append({
                "sample_id": identifier,
                "reference_time": pd.Timestamp(local_r.iloc[ri]),
                "detected_time": pd.Timestamp(local_d.iloc[di]),
                "signed_error_months": signed,
                "absolute_error_months": absolute,
            })
    return _event_match_metrics(refs, det, rows)


def _optimal_ordered_pairs(
    reference_months: np.ndarray,
    detected_months: np.ndarray,
    tolerance_months: int,
) -> tuple[tuple[int, int], ...]:
    """Return a deterministic cardinality-first, error-second event matching.

    Both input arrays must be sorted.  For one-dimensional calendar times with
    absolute-lag costs, an optimal non-crossing matching exists.  The dynamic
    program therefore maximizes cardinality first, minimizes total absolute
    month error second, and finally selects the lexicographically earliest pair
    sequence to make exact ties deterministic.
    """
    reference_months = np.asarray(reference_months, dtype=np.int64)
    detected_months = np.asarray(detected_months, dtype=np.int64)
    reference_count = len(reference_months)
    detected_count = len(detected_months)

    @lru_cache(maxsize=None)
    def solve(reference_index: int, detected_index: int):
        if reference_index >= reference_count or detected_index >= detected_count:
            return 0, 0, ()

        candidates = [
            solve(reference_index + 1, detected_index),
            solve(reference_index, detected_index + 1),
        ]
        absolute_error = abs(int(
            detected_months[detected_index] - reference_months[reference_index]
        ))
        if absolute_error <= tolerance_months:
            count, total_error, pairs = solve(reference_index + 1, detected_index + 1)
            candidates.append((
                count + 1,
                total_error + absolute_error,
                ((reference_index, detected_index),) + pairs,
            ))

        return min(
            candidates,
            key=lambda result: (-result[0], result[1], result[2]),
        )

    return solve(0, 0)[2]


def match_events_detailed_optimal_v2(
    references,
    detections,
    tolerance_months: int = 2,
) -> tuple[dict, pd.DataFrame]:
    """Match events by maximum cardinality, then minimum total month error.

    Exact objective ties are resolved chronologically, so results do not depend
    on input row order.  Matching remains one-to-one within each ``sample_id``.
    """
    if tolerance_months < 0:
        raise ValueError("tolerance_months must be non-negative")

    refs = references[["sample_id", "event_month"]].drop_duplicates().copy()
    det = detections[["sample_id", "change_time"]].drop_duplicates().copy()
    rows = []
    identifiers = sorted(set(refs.sample_id.astype(str)) | set(det.sample_id.astype(str)))
    for identifier in identifiers:
        local_r = refs.loc[
            refs.sample_id.astype(str).eq(identifier), ["event_month"]
        ].copy()
        local_d = det.loc[
            det.sample_id.astype(str).eq(identifier), ["change_time"]
        ].copy()
        local_r["event_month"] = pd.to_datetime(local_r.event_month)
        local_d["change_time"] = pd.to_datetime(local_d.change_time)
        local_r = local_r.sort_values("event_month", kind="mergesort").reset_index(drop=True)
        local_d = local_d.sort_values("change_time", kind="mergesort").reset_index(drop=True)
        reference_months = month_number(local_r.event_month)
        detected_months = month_number(local_d.change_time)

        for reference_index, detected_index in _optimal_ordered_pairs(
            reference_months, detected_months, tolerance_months
        ):
            signed_error = int(
                detected_months[detected_index] - reference_months[reference_index]
            )
            rows.append({
                "sample_id": identifier,
                "reference_time": pd.Timestamp(local_r.event_month.iloc[reference_index]),
                "detected_time": pd.Timestamp(local_d.change_time.iloc[detected_index]),
                "signed_error_months": signed_error,
                "absolute_error_months": abs(signed_error),
            })
    return _event_match_metrics(refs, det, rows)


def match_events_detailed(
    references,
    detections,
    tolerance_months: int = 2,
    *,
    matcher_version: str = EVENT_MATCHER_GREEDY_V1,
) -> tuple[dict, pd.DataFrame]:
    """Versioned one-to-one event matching with signed timing residuals.

    The default remains ``greedy_v1`` for backward compatibility.  New analyses
    can explicitly request ``optimal_v2`` to maximize true-positive cardinality
    before minimizing total absolute temporal error.
    """
    matchers = {
        EVENT_MATCHER_GREEDY_V1: match_events_detailed_greedy_v1,
        EVENT_MATCHER_OPTIMAL_V2: match_events_detailed_optimal_v2,
    }
    try:
        matcher = matchers[matcher_version]
    except KeyError as error:
        supported = ", ".join(sorted(matchers))
        raise ValueError(
            f"Unknown matcher_version {matcher_version!r}; expected one of: {supported}"
        ) from error
    return matcher(references, detections, tolerance_months)


def balanced_spatial_folds(references, folds: int = 5, seed: int = 20260820) -> pd.DataFrame:
    """Assign complete spatial blocks to deterministic event-balanced outer folds."""
    unique = references[["sample_id", "spatial_block_id", "event_month"]].drop_duplicates()
    counts = unique.groupby("spatial_block_id").agg(
        events=("event_month", "size"), samples=("sample_id", "nunique")
    ).reset_index()
    counts["tie"] = counts.spatial_block_id.astype(str).map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    )
    counts = counts.sort_values(["events", "samples", "tie"], ascending=[False, False, True])
    loads = np.zeros((folds, 2), dtype=int); assignments = []
    for row in counts.itertuples(index=False):
        fold = min(range(folds), key=lambda value: (loads[value, 0], loads[value, 1], value))
        loads[fold] += (int(row.events), int(row.samples))
        assignments.append({"spatial_block_id": row.spatial_block_id, "outer_fold": fold})
    return pd.DataFrame(assignments)


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-12, precision + recall)


def paired_block_bootstrap(
    references,
    left_events,
    right_events,
    *,
    tolerance_months: int = 2,
    replicates: int = 5000,
    seed: int = 20260820,
) -> tuple[pd.DataFrame, dict]:
    """Paired resampling of complete spatial blocks for a delta-F1 endpoint."""
    blocks = sorted(references.spatial_block_id.dropna().unique())
    per_method = []
    for events in (left_events, right_events):
        rows = []
        for block in blocks:
            local_refs = references.loc[references.spatial_block_id.eq(block)]
            allowed = set(local_refs.sample_id.astype(str))
            local_events = events.loc[events.sample_id.astype(str).isin(allowed)]
            metrics, _ = match_events_detailed(local_refs, local_events, tolerance_months)
            rows.append((metrics["tp"], metrics["fp"], metrics["fn"]))
        per_method.append(np.asarray(rows, dtype=int))
    rng = np.random.default_rng(seed); rows = []
    for replicate in range(replicates):
        selected = rng.integers(0, len(blocks), len(blocks))
        scores = []
        for counts in per_method:
            tp, fp, fn = counts[selected].sum(axis=0)
            scores.append(f1_from_counts(int(tp), int(fp), int(fn)))
        rows.append({"replicate": replicate, "left_f1": scores[0], "right_f1": scores[1],
                     "delta_left_minus_right": scores[0] - scores[1]})
    frame = pd.DataFrame(rows); delta = frame.delta_left_minus_right
    summary = {
        "median": float(delta.median()), "q025": float(delta.quantile(.025)),
        "q975": float(delta.quantile(.975)),
        "probability_gt_zero": float((delta > 0).mean()),
    }
    return frame, summary
