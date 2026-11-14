"""Research prototype for phase boundaries and STS response-shape constraints.

This module does not run CUSUM, infer physical harvest identity, or publish
episodes. Callers supply consecutive calendar months and one already-clipped
local trajectory per unique phase member, using its earliest accepted node.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


OFFSETS = np.arange(-6, 7)
QUIET = "observed_no_candidate"
UNKNOWN_STATES = {"unobserved_or_ineligible", "outside_detection_scope"}
B1_RELATIONS = {"temporal_join", "temporal_separate", "boundary_unknown"}


def active_phases(counts):
    """Return inclusive index intervals of consecutive months with any node.

    Each vector position must represent one calendar month, including gaps.
    """
    counts = np.asarray(counts)
    if (counts.ndim != 1 or not np.issubdtype(counts.dtype, np.number)
            or np.iscomplexobj(counts) or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("counts must be a finite nonnegative integer vector")
    transitions = np.diff(np.r_[False, counts > 0, False].astype(int))
    return list(zip(np.flatnonzero(transitions == 1).tolist(),
                    (np.flatnonzero(transitions == -1) - 1).tolist()))


def classify_gap(states, max_gap):
    """Classify a nonempty gap between distinct maximal active phases.

    A sufficiently long observed quiet run remains separating evidence even
    when another part of the gap is unknown; it is not proof of no harvesting.
    """
    states = tuple(states)
    if (not isinstance(max_gap, Integral) or isinstance(max_gap, bool)
            or max_gap < 0):
        raise ValueError("max_gap must be a nonnegative integer")
    if not states or any(s not in UNKNOWN_STATES | {QUIET} for s in states):
        raise ValueError("gap must contain only non-active monthly states")
    longest = run = 0
    quiet_runs = []
    for index, state in enumerate(states):
        if state == QUIET:
            run += 1
        else:
            if run:
                quiet_runs.append((index - run, index - 1))
            run = 0
        longest = max(longest, run)
    if run:
        quiet_runs.append((len(states) - run, len(states) - 1))
    has_unknown = any(state != QUIET for state in states)
    if longest >= max_gap + 1:
        relation = "temporal_separate"
    elif not has_unknown and len(states) <= max_gap:
        relation = "temporal_join"
    else:
        relation = "boundary_unknown"
    return {"relation": relation, "gap_months": len(states),
            "longest_quiet_run": longest, "has_unknown": has_unknown,
            "quiet_runs": quiet_runs,
            "unknown_offsets": [i for i, state in enumerate(states) if state != QUIET]}


def build_phase_template(unit_ids, local_values):
    """Build the normalized phase median and retain its actual contributors.

    Columns are calendar offsets -6..+6 from each member's candidate month.
    The caller must mask observations outside the original detection context
    or archive as NaN, and deduplicate members before entering this function.
    """
    unit_ids = tuple(unit_ids)
    values = np.asarray(local_values, dtype=float)
    if (not unit_ids or any(not isinstance(u, str) or not u for u in unit_ids)
            or len(set(unit_ids)) != len(unit_ids)):
        raise ValueError("unit_ids must be nonempty unique strings")
    if values.shape != (len(unit_ids), len(OFFSETS)) or np.isinf(values).any():
        raise ValueError("local_values must have shape (members, 13), with NaN for missing values")

    normalized = np.full(values.shape, np.nan)
    baseline = np.full(len(unit_ids), np.nan)
    scale = np.full(len(unit_ids), np.nan)
    eligible = np.zeros(len(unit_ids), dtype=bool)
    for i, row in enumerate(values):
        finite = np.isfinite(row)
        if (finite.sum() < 8 or finite[OFFSETS < 0].sum() < 3
                or finite[OFFSETS >= 0].sum() < 3):
            continue
        baseline[i] = np.median(row[finite & (OFFSETS < 0)])
        low, high = np.quantile(row[finite], [.1, .9], method="linear")
        scale[i] = high - low
        if scale[i] > 0:
            normalized[i] = (row - baseline[i]) / scale[i]
            eligible[i] = True

    eligible_count = int(eligible.sum())
    computable = eligible_count >= 2 and 2 * eligible_count >= len(unit_ids)
    counts = np.isfinite(normalized).sum(axis=0)
    template = np.full(len(OFFSETS), np.nan)
    lower = template.copy()
    upper = template.copy()
    contributors = []
    for j in range(len(OFFSETS)):
        present = np.isfinite(normalized[:, j])
        contributors.append(tuple(u for u, ok in zip(unit_ids, present) if ok))
        if computable and 2 * counts[j] >= eligible_count:
            lower[j], template[j], upper[j] = np.quantile(
                normalized[present, j], [.25, .5, .75], method="linear")

    # shortcut: contributor changes are diagnostic only; new gates need independent calibration.
    changes = [()]
    changes.extend(tuple(sorted(set(current) ^ set(previous)))
                   for previous, current in zip(contributors[:-1], contributors[1:]))
    return {"computable": computable, "unit_ids": unit_ids,
            "eligible": eligible, "eligible_count": eligible_count,
            "baseline": baseline, "scale": scale, "normalized": normalized,
            "template": template, "q25": lower, "q75": upper,
            "contributor_ids": contributors, "contributor_count": counts,
            "contributor_fraction": counts / len(unit_ids),
            "contributor_changes": changes}


def response_distance(template_a, template_b):
    """Mean absolute difference on sufficiently covered common real offsets."""
    a, b = np.asarray(template_a, dtype=float), np.asarray(template_b, dtype=float)
    if (a.shape != (13,) or b.shape != (13,)
            or np.isinf(a).any() or np.isinf(b).any()):
        raise ValueError("templates must contain 13 values, with NaN for missing offsets")
    common = np.isfinite(a) & np.isfinite(b)
    available = (common.sum() >= 8 and common[OFFSETS < 0].sum() >= 3
                 and common[OFFSETS >= 0].sum() >= 3)
    distance = float(np.mean(np.abs(a[common] - b[common]))) if available else np.nan
    return {"available": bool(available), "distance": distance,
            "common_offsets": OFFSETS[common].copy()}


def apply_sts_veto(relation, distance, threshold):
    """Only remove a B1 join; unavailable STS never creates a boundary."""
    if relation not in B1_RELATIONS:
        raise ValueError("relation must be a B1 boundary classification")
    if np.isnan(threshold) or threshold < 0 or distance < 0:
        raise ValueError("threshold and available distance must be nonnegative")
    if relation != "temporal_join":
        status = "not_applicable"
    elif np.isinf(threshold):
        status = "disabled"
    elif not np.isfinite(distance):
        status = "sts_unavailable"
    elif distance > threshold:
        relation, status = "sts_separate", "vetoed"
    else:
        status = "retained"
    return {"relation": relation, "sts_status": status}
