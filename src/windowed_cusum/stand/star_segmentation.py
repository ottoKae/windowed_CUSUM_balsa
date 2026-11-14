"""STARS: space-time asynchronous round segmentation over frozen report components.

This module does not detect change, does not redefine response months, and does
not create new pixel evidence. It consumes already-accepted pixel nodes grouped
by a frozen report, tests whether their first-response dates are spatially
organized beyond a permutation null, and splits a parent report into zones and
rounds only when that organization survives the test and the support gate.

The null preserves pixel positions, the rook adjacency structure and the
multiset of first-response dates. It destroys only the assignment of dates to
positions, which is exactly the hypothesis under test.

Every randomized statistic is calibrated jointly over the declared tolerance
set, so scanning several tolerances cannot manufacture significance.
"""
from __future__ import annotations

from dataclasses import dataclass

import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


EPOCH = pd.Period("2016-01", freq="M").ordinal
DESTINATIONS = ("zone_rounds", "no_split_evidence", "below_detectability")
ROUND_COLUMNS = [
    "method", "sample_id", "component_id", "parent_prediction_id", "round_id",
    "round_index", "zone_count", "first_response_month", "last_response_month",
    "representative_month", "response_pixels", "parent_pixels", "parent_support_fraction",
    "round_support_fraction", "source_node_count", "source_unit_ids", "response_months",
    "zone_ids",
]
ZONE_COLUMNS = [
    "method", "sample_id", "component_id", "round_id", "zone_id", "first_response_month",
    "last_response_month", "median_response_month", "median_month_index", "response_pixels",
    "parent_pixels", "parent_support_fraction", "source_node_count", "source_unit_ids",
    "native_rows", "native_cols",
]
STAND_COLUMNS = [
    "method", "sample_id", "component_id", "parent_prediction_id", "destination_type",
    "reason", "admissible", "parent_pixels", "response_pixels", "tolerance_months",
    "observed_join_count", "null_join_mean", "null_join_sd", "max_standardized_join",
    "p_value", "q_value", "zone_count", "round_count",
]


@dataclass(frozen=True)
class Config:
    """Frozen operating parameters.

    ``mode`` exists so the implementation can be audited against the frozen
    time-only baseline. ``disabled`` never partitions and must reproduce the
    parent reports exactly; it is an invariant probe, not a method variant.
    """

    method: str = "STARS_V1"
    mode: str = "full"
    min_stand_pixels: int = 25
    min_response_pixels: int = 15
    tolerances: tuple = (0, 1, 2)
    min_zone_pixels: int = 3
    round_gap_months: int = 2
    parent_support: float = 0.20
    permutations: int = 999
    alpha: float = 0.05
    seed: int = 20261102
    force_significant: bool = False
    skip_support_gate: bool = False

    def __post_init__(self):
        if self.mode not in ("full", "disabled"):
            raise ValueError("mode must be 'full' or 'disabled'")
        if not self.tolerances or min(self.tolerances) < 0:
            raise ValueError("tolerances must be a nonempty nonnegative sequence")
        if tuple(sorted(set(self.tolerances))) != tuple(self.tolerances):
            raise ValueError("tolerances must be sorted and unique")
        if self.min_stand_pixels < 2 or self.min_response_pixels < 2:
            raise ValueError("admissibility floors must admit at least two pixels")
        if not 0 < self.parent_support <= 1:
            raise ValueError("parent_support must lie in (0, 1]")
        if self.permutations < 99:
            raise ValueError("permutations must be substantial")
        if self.min_zone_pixels < 1 or self.round_gap_months < 0:
            raise ValueError("zone floor must be positive and gap nonnegative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly inside (0, 1)")


def month_label(index):
    """Return the frozen ``YYYY-MM`` label for an index counted from 2016-01."""
    return str(pd.Period(ordinal=EPOCH + int(index), freq="M"))


def rook_edges(rows, cols):
    """Return index pairs of 4-connected pixels sharing one native grid.

    Adjacency is built per call, so callers must pass pixels from a single
    ``track_key``. Coordinates are native integer row/column positions; pixels a
    diagonal apart are not neighbours.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.ndim != 1 or cols.shape != rows.shape:
        raise ValueError("rows and cols must be equally sized one-dimensional arrays")
    if len(rows) and len(np.unique(np.column_stack([rows, cols]), axis=0)) != len(rows):
        raise ValueError("native grid coordinates must be unique")
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    left, right = [], []
    for index, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in ((1, 0), (0, 1)):
            other = lookup.get((int(r) + dr, int(c) + dc))
            if other is not None:
                left.append(index)
                right.append(other)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def join_count(labels, left, right):
    """Count neighbour pairs whose labels differ.

    A high count means the same response period is scattered across many
    separate patches; a low count means periods form compact patches. This is
    the multi-colour join-count statistic, not a Moran statistic and not a local
    indicator of spatial association.
    """
    if len(left) == 0:
        return 0
    labels = np.asarray(labels)
    return int(np.count_nonzero(labels[left] != labels[right]))


def null_same_pair_probability(labels):
    """Exact per-pair probability that two random pixels share a label.

    Under a free permutation of the observed label multiset this is
    ``sum_v n_v(n_v-1) / (n(n-1))``, so the expected join count over ``E`` pairs
    is ``E * (1 - p)``. Only the expectation is analytic here; the variance is
    taken from permutations because neighbour pairs are not independent.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n < 2:
        return 0.0
    counts = np.unique(labels, return_counts=True)[1]
    return float((counts * (counts - 1)).sum()) / (n * (n - 1))


def zone_labels(first, left, right, tolerance):
    """Union pixels through neighbour edges whose first-response dates agree.

    Two neighbouring pixels share a zone when their first-response months differ
    by at most ``tolerance``. Zones are connected, disjoint and cover every
    response pixel.
    """
    n = len(first)
    if tolerance < 0 or n == 0:
        raise ValueError("tolerance must be nonnegative and pixels nonempty")
    if len(left) == 0:
        return np.arange(n)
    first = np.asarray(first, np.int64)
    near = np.abs(first[left] - first[right]) <= tolerance
    if not near.any():
        return np.arange(n)
    graph = coo_matrix((np.ones(int(near.sum())), (left[near], right[near])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def merge_small_zones(labels, first, min_pixels):
    """Absorb zones below the pixel floor into the nearest period in time.

    Fragments a few pixels wide are not independent rounds. Folding them into the
    closest period keeps the round count from being inflated by boundary noise.
    Ties resolve to the lower original label so results stay deterministic.
    """
    labels = np.asarray(labels, dtype=np.int64)
    first = np.asarray(first, dtype=np.int64)
    if min_pixels <= 1 or len(labels) == 0:
        return np.unique(labels, return_inverse=True)[1]
    sizes = pd.Series(labels).value_counts()
    small = set(sizes.index[sizes < min_pixels].tolist())
    if not small:
        return np.unique(labels, return_inverse=True)[1]
    centres = {z: float(np.median(first[labels == z])) for z in sizes.index if z not in small}
    if not centres:
        return np.zeros(len(labels), dtype=np.int64)
    out = labels.copy()
    for zone in sorted(small):
        here = float(np.median(first[labels == zone]))
        target = min(centres, key=lambda z: (abs(centres[z] - here), z))
        out[labels == zone] = target
    return np.unique(out, return_inverse=True)[1]


def organization_test(first, left, right, config, rng):
    """Jointly calibrate the join-count statistic over all declared tolerances.

    Compact patches of equal first-response date produce FEW neighbour pairs with
    differing dates, so the alternative is the lower tail: the statistic is the
    most negative standardized join count across tolerances. Every permutation
    contributes its own minimum, so scanning tolerances cannot make the observed
    value look extreme without shifting the null the same way; the p-value stays
    valid under the multi-scale search.
    """
    first = np.asarray(first, dtype=np.int64)
    if len(first) < 2:
        raise ValueError("at least two pixels are required")
    if len(left) < 1:
        raise ValueError("at least one neighbour pair is required")
    tolerances = list(config.tolerances)
    draws = {t: np.empty(config.permutations, dtype=np.int64) for t in tolerances}
    for draw in range(config.permutations):
        shuffled = rng.permutation(first)
        for tolerance in tolerances:
            draws[tolerance][draw] = join_count(
                zone_labels(shuffled, left, right, tolerance), left, right)
    observed, means, sds = {}, {}, {}
    for tolerance in tolerances:
        observed[tolerance] = join_count(zone_labels(first, left, right, tolerance), left, right)
        sample = draws[tolerance].astype(float)
        means[tolerance] = float(sample.mean())
        sds[tolerance] = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    empty = dict(observed=observed, means=means, sds=sds, tolerance=None,
                 statistic=np.nan, p_value=np.nan, permutations=config.permutations)
    if any(not np.isfinite(sds[t]) or sds[t] <= 0 for t in tolerances):
        return empty
    standardized = {t: (observed[t] - means[t]) / sds[t] for t in tolerances}
    statistic = min(standardized.values())
    if not np.isfinite(statistic) or statistic >= 0:
        return dict(empty, statistic=float(statistic), p_value=1.0)
    null_min = np.full(config.permutations, np.inf)
    for tolerance in tolerances:
        null_min = np.minimum(null_min, (draws[tolerance] - means[tolerance]) / sds[tolerance])
    p_value = (1 + int(np.count_nonzero(null_min <= statistic))) / (config.permutations + 1)
    selected = min(tolerances, key=lambda t: (standardized[t], t))
    return dict(observed=observed, means=means, sds=sds, tolerance=int(selected),
                statistic=float(statistic), p_value=float(p_value),
                permutations=config.permutations)


def _zone_records(frame, config, parent_pixels):
    """Turn a labelled pixel table into zone records and an id map."""
    records, order = [], 0
    for label, part in frame.groupby("zone_label", sort=True):
        order += 1
        zone_id = f"{part.component_id.iloc[0]}#z{order - 1:02d}"
        records.append(dict(
            method=config.method, sample_id=part.sample_id.iloc[0],
            component_id=part.component_id.iloc[0], round_id="", zone_id=zone_id,
            first_response_month=month_label(int(part.first_month.min())),
            last_response_month=month_label(int(part.last_month.max())),
            median_response_month=month_label(int(np.floor(np.median(part.first_month) + 0.5))),
            median_month_index=int(np.floor(np.median(part.first_month) + 0.5)),
            response_pixels=int(len(part)), parent_pixels=int(parent_pixels),
            parent_support_fraction=len(part) / int(parent_pixels),
            source_node_count=int(part.nodes.sum()),
            source_unit_ids=";".join(sorted(part.unit_id)),
            native_rows=";".join(map(str, part.native_row.tolist())),
            native_cols=";".join(map(str, part.native_col.tolist()))))
    return pd.DataFrame(records, columns=ZONE_COLUMNS)


def _round_records(zones, members, config, parent_pixels, parent_id):
    """Group zones into rounds by the gap between consecutive response periods.

    Zones are ordered by median first-response month and join the same round
    while the gap to the previous zone stays within ``round_gap_months``. This
    reuses the frozen aggregator's one-quiet-interval convention without
    re-reading pixel states.
    """
    ordered = zones.sort_values(["median_month_index", "zone_id"]).reset_index(drop=True)
    index = np.zeros(len(ordered), dtype=np.int64)
    if len(ordered) > 1:
        index = np.concatenate([[0], np.cumsum(
            np.diff(ordered.median_month_index.to_numpy()) > config.round_gap_months)])
    ordered["round_index"] = index
    component = ordered.component_id.iloc[0]
    rounds = []
    for step, part in ordered.groupby("round_index", sort=True):
        round_id = f"{component}#r{int(step):02d}"
        units = sorted({u for text in part.source_unit_ids for u in text.split(";")})
        local = members.loc[members.unit_id.isin(units), "month_index"]
        rounds.append(dict(
            method=config.method, sample_id=part.sample_id.iloc[0], component_id=component,
            parent_prediction_id=parent_id, round_id=round_id, round_index=int(step),
            zone_count=int(len(part)),
            first_response_month=pd.Period(part.first_response_month.min(), freq="M").strftime("%Y-%m"),
            last_response_month=pd.Period(part.last_response_month.max(), freq="M").strftime("%Y-%m"),
            representative_month=month_label(int(np.floor(local.median() + 0.5))),
            response_pixels=int(part.response_pixels.sum()), parent_pixels=int(parent_pixels),
            parent_support_fraction=part.response_pixels.sum() / int(parent_pixels),
            round_support_fraction=part.response_pixels.sum() / int(ordered.response_pixels.sum()),
            source_node_count=int(part.source_node_count.sum()),
            source_unit_ids=";".join(units),
            response_months=json.dumps([month_label(m) for m in sorted(local.unique().tolist())]),
            zone_ids=";".join(part.zone_id)))
    result = pd.DataFrame(rounds, columns=ROUND_COLUMNS)
    lookup = dict(zip(ordered.zone_id, ordered.round_index.map(
        lambda i: f"{component}#r{int(i):02d}")))
    zones = zones.assign(round_id=zones.zone_id.map(lookup))
    return zones, result


def segment_component(component_id, sample_id, parent_id, members, units, total_pixels,
                      config, rng, cached=None):
    """Test and segment one frozen report.

    Returns ``(stand_row, zone_frame, round_frame, ownership, test)``. ``members``
    holds accepted nodes of this report with ``unit_id``, ``node_id`` and
    ``month_index``. Nothing outside this report is read, so reports never
    exchange pixels. ``cached`` may carry a previously computed test for the same
    report so ablations can separate the calibration from the post-test rules.
    """
    pixels = (members.groupby("unit_id", as_index=False)
              .agg(first_month=("month_index", "min"),
                   last_month=("month_index", "max"),
                   nodes=("node_id", "size")))
    joined = pixels.merge(units[["unit_id", "native_row", "native_col", "track_key"]],
                          on="unit_id", how="left", validate="one_to_one")
    if joined[["native_row", "native_col", "track_key"]].isna().any().any():
        raise ValueError("every response pixel requires native coordinates and a track")
    joined = (joined.assign(sample_id=sample_id, component_id=component_id)
              .sort_values(["track_key", "native_row", "native_col"]).reset_index(drop=True))
    response_pixels, parent_pixels = int(len(joined)), int(total_pixels)
    if parent_pixels < 1:
        raise ValueError("parent pixel denominator must be positive")

    def base_row(**extra):
        row = dict(method=config.method, sample_id=sample_id, component_id=component_id,
                   parent_prediction_id=parent_id, parent_pixels=parent_pixels,
                   response_pixels=response_pixels, destination_type=None, reason=None,
                   admissible=True, tolerance_months=np.nan, observed_join_count=np.nan,
                   null_join_mean=np.nan, null_join_sd=np.nan, max_standardized_join=np.nan,
                   p_value=np.nan, q_value=np.nan, zone_count=1, round_count=1)
        row.update(extra)
        return row

    def finish(destination, reason, frame, assigned, test=None, **extra):
        flat = _flatten_test(test, config)
        zones = _zone_records(frame, config, parent_pixels)
        zones, rounds = _round_records(zones, members, config, parent_pixels, parent_id)
        row = base_row(destination_type=destination, reason=reason, **flat, **extra)
        row["zone_count"], row["round_count"] = int(len(zones)), int(len(rounds))
        if assigned is None:
            ownership = members.assign(zone_id=zones.zone_id.iloc[0],
                                       round_id=rounds.round_id.iloc[0])
        else:
            pairs = {}
            for record in zones.itertuples(index=False):
                for unit in record.source_unit_ids.split(";"):
                    pairs[unit] = (record.zone_id, record.round_id)
            ownership = members.assign(
                zone_id=members.unit_id.map(lambda u: pairs.get(u, (np.nan, np.nan))[0]),
                round_id=members.unit_id.map(lambda u: pairs.get(u, (np.nan, np.nan))[1]))
        return row, zones, rounds, ownership, flat

    def _flatten_test(test, conf):
        test = test if test else dict(observed={}, means={}, sds={}, tolerance=None,
                                      statistic=np.nan, p_value=np.nan,
                                      permutations=conf.permutations)
        chosen = test.get("tolerance")
        return dict(tolerance_months=chosen, statistic=test.get("statistic", np.nan),
                    p_value=test.get("p_value", np.nan),
                    observed_join_count=test.get("observed", {}).get(chosen, np.nan),
                    null_join_mean=test.get("means", {}).get(chosen, np.nan),
                    null_join_sd=test.get("sds", {}).get(chosen, np.nan),
                    permutations=test.get("permutations", conf.permutations))

    single = joined.assign(zone_label=0)
    if config.mode == "disabled":
        return finish("zone_rounds", "invariant_single_zone", single, None)

    if parent_pixels < config.min_stand_pixels or response_pixels < config.min_response_pixels:
        return finish("below_detectability", "insufficient_pixels", single, None, admissible=False)

    # Test the largest track group only: cross-track pixels share no native grid,
    # so a combined adjacency would invent neighbour pairs that do not exist.
    track = joined.groupby("track_key").size().idxmax()
    local = joined.loc[joined.track_key.eq(track)].reset_index(drop=True)
    left, right = rook_edges(local.native_row.to_numpy(), local.native_col.to_numpy())
    if len(local) < 4 or len(left) < 3:
        return finish("below_detectability", "insufficient_adjacency", single, None,
                      admissible=False)

    test = cached
    if test is None:
        test = organization_test(local.first_month.to_numpy(), left, right, config, rng)
    if config.force_significant:
        # Ablation A2: bypass the test and treat every admissible report as
        # organized. Isolates how much of the result the test itself supplies.
        # A cached test may carry NaN rather than None when no scale was usable.
        cached_scale = test.get("tolerance")
        usable = cached_scale is not None and np.isfinite(cached_scale)
        chosen = cached_scale if usable else min(config.tolerances)
        test = dict(test, tolerance=int(chosen),
                    p_value=0.0 if test.get("p_value") is None else test["p_value"],
                    statistic=test["statistic"] if np.isfinite(test.get("statistic", np.nan)) else 0.0)
    chosen = test["tolerance"]
    extra = dict(observed_join_count=test["observed"].get(chosen, np.nan),
                 null_join_mean=test["means"].get(chosen, np.nan),
                 null_join_sd=test["sds"].get(chosen, np.nan),
                 max_standardized_join=test["statistic"],
                 p_value=test["p_value"], tolerance_months=chosen)
    if not config.force_significant:
        if chosen is None or not np.isfinite(test["p_value"]) or test["p_value"] > config.alpha:
            return finish("no_split_evidence", "spatial_organization_not_significant",
                          single, None, test=test)

    labels = zone_labels(local.first_month.to_numpy(), left, right, chosen)
    if config.min_zone_pixels > 1:
        labels = merge_small_zones(labels, local.first_month.to_numpy(),
                                   config.min_zone_pixels)
    assigned = joined.copy()
    assigned["zone_label"] = -1
    assigned.loc[assigned.track_key.eq(track), "zone_label"] = labels
    stray = assigned.zone_label < 0
    if stray.any():
        # Pixels outside the tested track stay with the earliest contiguous zone
        # instead of forming their own round, which would mix grid identities.
        assigned.loc[stray, "zone_label"] = 0
    if assigned.zone_label.nunique() < 2:
        return finish("no_split_evidence", "single_zone_after_merge", single, None, test=test)

    provisional = _zone_records(assigned, config, parent_pixels)
    _, rounds = _round_records(provisional, members, config, parent_pixels, parent_id)
    if len(rounds) < 2:
        # A smooth ramp splits into many single-column zones that all fall inside
        # one interleaved period. That is one continuous operation, not a
        # rotation, so it must not be reported as a partition.
        return finish("no_split_evidence", "single_round_no_partition",
                      single, None, test=test)
    # Every round must independently clear the frozen parent-support gate. A zone
    # that only clears it because the denominator shrank would change the product
    # definition rather than improve it.
    if not config.skip_support_gate and (
            rounds.parent_support_fraction < config.parent_support).any():
        return finish("no_split_evidence", "round_support_below_parent_gate",
                      single, None, test=test)
    return finish("zone_rounds", "spatially_supported_partition", assigned, assigned, test=test)


def segment(reports, members, units, config, test_provider=None):
    """Segment every frozen report; return stands, zones, rounds and node ownership.

    ``test_provider`` optionally supplies a previously computed organization test
    keyed by ``component_id``. The permutation test is the expensive step, so
    ablations that only change post-test rules reuse one calibration instead of
    re-randomizing. A provider must return a dict shaped like
    :func:`organization_test`, or ``None`` to fall back to a fresh test.
    """
    required_reports = {"component_id", "sample_id", "prediction_id", "total_pixels"}
    required_members = {"component_id", "unit_id", "node_id", "month_index"}
    missing = required_reports - set(reports.columns)
    if missing:
        raise ValueError(f"reports missing {sorted(missing)}")
    missing = required_members - set(members.columns)
    if missing:
        raise ValueError(f"members missing {sorted(missing)}")
    if not reports.component_id.is_unique:
        raise ValueError("report components must be unique")
    if members.duplicated(["component_id", "unit_id", "month_index"]).any():
        raise ValueError("accepted nodes must be deduplicated before segmentation")
    orphans = set(members.component_id) - set(reports.component_id)
    if orphans:
        raise ValueError(f"{len(orphans)} member components have no frozen report")

    # Group once: filtering the full node table per report is quadratic and was
    # the dominant cost of a whole-domain run.
    grouped = {key: part for key, part in members.groupby("component_id", sort=False)}
    rng = np.random.default_rng(config.seed)
    stands, zones, rounds, ownership, tests = [], [], [], [], []
    for record in reports.itertuples(index=False):
        local = grouped.get(record.component_id)
        if local is None or local.empty:
            raise ValueError(f"report {record.component_id} has no accepted nodes")
        cached = None if test_provider is None else test_provider(record.component_id)
        row, zone, rnd, owned, test = segment_component(
            record.component_id, record.sample_id, record.prediction_id, local, units,
            int(record.total_pixels), config, rng, cached=cached)
        stands.append(row)
        zones.append(zone)
        rounds.append(rnd)
        ownership.append(owned)
        tests.append(dict(component_id=record.component_id, sample_id=record.sample_id,
                          total_pixels=int(record.total_pixels), **test))
    stands = pd.DataFrame(stands, columns=STAND_COLUMNS)
    stands["q_value"] = benjamini_hochberg(stands.p_value)
    return dict(stands=stands, zones=pd.concat(zones, ignore_index=True),
                rounds=pd.concat(rounds, ignore_index=True),
                node_ownership=pd.concat(ownership, ignore_index=True),
                tests=pd.DataFrame(tests))


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted q-values, preserving missing entries."""
    values = pd.Series(p_values, dtype="float64").to_numpy()
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    ordered = values[finite]
    order = np.argsort(ordered, kind="stable")
    ranked = ordered[order]
    m = len(ranked)
    monotone = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    result[finite] = adjusted
    return result
