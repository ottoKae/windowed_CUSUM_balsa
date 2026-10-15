"""Prespecified pixel-first temporal fusion and descriptive response summaries.

These helpers do not inspect reference dates or alter the detector. P1 uses a
geometric acceptance denominator; P2 and P3 use mask-defined opportunity. P3
splits provisional P2 components, including those failing P2's final gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .core import _calendar_align, generate_windows
from .paper import _validated_month_index


def _calendar(calendar):
    dates = _validated_month_index(calendar)
    if dates.hasnans:
        raise ValueError("calendar cannot contain missing dates")
    return dates


def _mask(values, shape, name):
    result = np.asarray(values)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if result.dtype != bool and not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must contain only boolean mask values")
    return result.astype(bool, copy=False)


def eligibility_mask(valid, calendar, window_length=36, step=12,
                     minimum_observations=12, start_year=2017, end_year=2025):
    """Return ``[pixel, original calendar position]`` detection opportunity.

    ``valid`` is the frozen shared VV/VH/observation-count mask, not the mask
    of detections. Missing months must remain explicit calendar positions.
    January padding follows the production scanner and is removed on return.
    Overlapping decision windows, if requested, contribute the union of their
    opportunities; the frozen W36/S12 windows have one owner per month.
    """
    dates = _calendar(calendar)
    raw = np.asarray(valid)
    if raw.ndim != 2:
        raise ValueError("valid must have dimensions [pixel, calendar position]")
    raw = _mask(raw, (raw.shape[0], len(dates)), "valid")
    if window_length <= 2 * step or step < 1 or minimum_observations < 1:
        raise ValueError("require window_length > 2*step, step >= 1 and minimum_observations >= 1")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    _, aligned_dates = _calendar_align(np.zeros(len(dates)), dates)
    prefix = len(aligned_dates) - len(dates)
    mask = np.pad(raw, ((0, 0), (prefix, 0)), constant_values=False)
    result = np.zeros_like(mask)
    cumulative = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
    reporting = np.ones(len(aligned_dates), dtype=bool)
    if start_year is not None:
        reporting &= aligned_dates.year >= start_year
    if end_year is not None:
        reporting &= aligned_dates.year <= end_year
    for left, right, begin, end in generate_windows(len(aligned_dates), window_length, step):
        full_count = cumulative[:, right] - cumulative[:, left]
        before = cumulative[:, begin:end] - cumulative[:, left, None]
        after = cumulative[:, right, None] - cumulative[:, begin:end]
        result[:, begin:end] |= (
            mask[:, begin:end] & reporting[None, begin:end]
            & (full_count[:, None] >= minimum_observations)
            & (before >= 1) & (after >= 1)
        )
    return result[:, prefix:]


def empirical_date_quantile(dates, probability):
    """Inverse empirical CDF: an observed date, with the lower even median."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    ordered = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if ordered.hasnans:
        raise ValueError("response dates cannot be missing")
    if not len(ordered):
        return pd.NaT
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def spatial_components(coordinates):
    """Return eight-neighbour component count and largest responder fraction."""
    points = {tuple(map(int, coordinate)) for coordinate in coordinates}
    if not points:
        return 0, np.nan
    remaining = set(points)
    sizes = []
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            row, col = pending.pop()
            size += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbour = (row + dr, col + dc)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        pending.append(neighbour)
        sizes.append(size)
    return len(sizes), max(sizes) / len(points)


def _stable_id(prefix, *values):
    encoded = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return prefix + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


_EPISODE_COLUMNS = [
    "episode_id", "track_event_id", "parent_component_id", "route", "ratio", "selected_ratio",
    "Band", "band", "status", "accepted", "rejection_reason", "change_time",
    "representative_month", "peak_month", "q10_month", "q25_month", "q50_month",
    "q75_month", "q90_month", "min_detected_month", "max_detected_month",
    "envelope_start", "envelope_end", "episode_centre", "envelope_definition",
    "N_geometry", "N_observable", "N_response", "acceptance_denominator",
    "support_threshold", "support_pixels", "support_fraction", "response_fraction",
    "response_fraction_reason", "observable_coverage", "component_count",
    "largest_component_fraction", "calendar_clipped", "fallback_extraction",
    "long_chain_parent", "parent_span_days", "assigned_span_days", "cross_year",
    "date_assignment_ambiguity", "node_count", "source_row_count", "multi_track_support",
]
_MEMBERSHIP_COLUMNS = [
    "source_event_id", "node_id", "episode_id", "track_event_id", "parent_component_id",
    "pixel_track_id", "Band", "band", "change_time", "event_month", "native_row", "native_col",
    "route", "ratio", "pixel_vote_date", "is_core", "membership_type", "status",
    "accepted", "rejection_reason", "noise_reason", "eligible_at_event",
    "fallback_extraction", "N_geometry",
]
_NODE_COLUMNS = [
    "node_id", "pixel_track_id", "Band", "band", "event_month", "eligible_at_event",
    "density_response_pixels", "density_denominator", "density_threshold", "is_core",
    "parent_component_id", "episode_id", "pixel_vote_date", "membership_type", "status",
    "accepted", "rejection_reason", "noise_reason", "source_row_count", "route", "ratio",
    "fallback_extraction", "N_geometry",
]


def fuse_group(events, pixels, eligible, calendar, *, route, ratio, metadata,
               eps_days=60, fallback=False):
    """Fuse one polygon-track/band and retain every source row and disposition.

    ``pixels`` contains unique pixel IDs and native coordinates in ``eligible``
    row order. ``events`` accepts ``Band`` or ``band`` and preserves all original
    columns. IDs are content-derived and independent of source row ordering.
    Exact duplicate rows share their source ID but retain separate ledger rows.
    ``response_fraction`` always uses the fixed observable cohort;
    ``support_fraction`` uses the arm's declared acceptance denominator.
    Metadata ``geometric_native_pixel_centres`` may exceed the cached pixel
    count when extraction omitted never-observed cells. Those cells remain in
    geometry, but receive no invented observations or native coordinates.
    Representative-pixel fallback has zero genuine geometry and no spatial map.
    """
    route = str(route).upper()
    if route not in {"P1", "P2", "P3"}:
        raise ValueError("route must be P1, P2 or P3")
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must lie in (0, 1]")
    if not np.isfinite(eps_days) or eps_days <= 0:
        raise ValueError("eps_days must be positive")
    metadata = dict(metadata) if isinstance(metadata, Mapping) else dict(metadata.to_dict())
    if not metadata.get("polygon_track_id"):
        raise ValueError("metadata must identify polygon_track_id")
    dates = _calendar(calendar)
    if not len(dates):
        raise ValueError("fusion requires a nonempty calendar")
    pixels = pixels.copy().reset_index(drop=True)
    for column in ("pixel_track_id", "native_row", "native_col"):
        if column not in pixels:
            raise ValueError(f"pixels missing {column}")
    if pixels.pixel_track_id.isna().any() or pixels.pixel_track_id.duplicated().any():
        raise ValueError("pixels must have unique, nonmissing pixel_track_id values")
    pixels["pixel_track_id"] = pixels.pixel_track_id.astype(str)
    if pixels.pixel_track_id.duplicated().any():
        raise ValueError("pixel IDs must remain unique as strings")
    coords = pixels[["native_row", "native_col"]].to_numpy(float)
    if not np.isfinite(coords).all() or not np.equal(coords, np.floor(coords)).all():
        raise ValueError("native pixel coordinates must be finite integers")
    if pixels.duplicated(["native_row", "native_col"]).any():
        raise ValueError("one native cell cannot have multiple pixel IDs within a track")
    mask = _mask(eligible, (len(pixels), len(dates)), "eligible")
    fallback_flag = metadata.get("is_fallback", False)
    fallback = bool(fallback or metadata.get("support_mode") == "representative_fallback"
                    or (isinstance(fallback_flag, (bool, np.bool_)) and fallback_flag))
    geometry = metadata.get("geometric_native_pixel_centres", len(pixels))
    if (isinstance(geometry, (bool, np.bool_))
            or not isinstance(geometry, (int, float, np.integer, np.floating))
            or not np.isfinite(geometry) or geometry < 0 or int(geometry) != geometry):
        raise ValueError("geometric_native_pixel_centres must be a nonnegative integer")
    n_geometry = 0 if fallback else int(geometry)
    if not fallback and n_geometry < len(pixels):
        raise ValueError("Genuine geometric pixel count cannot be smaller than the cached membership")
    pixel_indexes = dict(zip(pixels.pixel_track_id, range(len(pixels))))
    coordinate_lookup = pixels.set_index("pixel_track_id")[["native_row", "native_col"]]
    source = events.copy().reset_index(drop=True)
    episode_columns = list(dict.fromkeys(list(metadata) + _EPISODE_COLUMNS))
    member_columns = list(dict.fromkeys(list(source.columns) + list(metadata) + _MEMBERSHIP_COLUMNS))
    node_columns = list(dict.fromkeys(list(metadata) + _NODE_COLUMNS))
    if source.empty:
        return (pd.DataFrame(columns=episode_columns), pd.DataFrame(columns=member_columns),
                pd.DataFrame(columns=node_columns))
    if not {"pixel_track_id", "change_time"}.issubset(source):
        raise ValueError("events require pixel_track_id and change_time")
    source["pixel_track_id"] = source.pixel_track_id.astype(str)
    unknown = sorted(set(source.pixel_track_id) - set(pixel_indexes))
    if unknown:
        raise ValueError(f"events contain pixels outside the polygon-track: {unknown[:3]}")
    source["Band"] = source.get("Band", source.get("band", metadata.get("Band", metadata.get("band", "VV_dB"))))
    if source.Band.isna().any() or source.Band.nunique() != 1:
        raise ValueError("fuse_group requires exactly one nonmissing band")
    band = str(source.Band.iloc[0])
    source["band"] = band
    event_dates = pd.to_datetime(source.change_time, errors="raise")
    if event_dates.isna().any():
        raise ValueError("event dates cannot be missing")
    source["event_month"] = event_dates.dt.to_period("M").dt.to_timestamp()
    if "source_event_id" not in source:
        source["source_event_id"] = [
            _stable_id("source_", record) for record in source.to_dict("records")
        ]
    elif source.source_event_id.isna().any():
        raise ValueError("source_event_id cannot be missing")
    source["node_id"] = [
        _stable_id("node_", pixel, band, month.isoformat())
        for pixel, month in zip(source.pixel_track_id, source.event_month)
    ]
    if source.groupby("source_event_id").node_id.nunique().gt(1).any():
        raise ValueError("a source_event_id cannot identify different pixel-date nodes")
    nodes = source[["node_id", "pixel_track_id", "event_month"]].drop_duplicates("node_id")
    nodes = nodes.sort_values(["event_month", "pixel_track_id", "node_id"]).reset_index(drop=True)
    positions = dates.get_indexer(nodes.event_month)
    pixel_positions = nodes.pixel_track_id.map(pixel_indexes).to_numpy(int)
    at_event = np.zeros(len(nodes), bool)
    present = positions >= 0
    at_event[present] = mask[pixel_positions[present], positions[present]]
    nodes["eligible_at_event"] = at_event
    reporting_start = pd.Timestamp(metadata.get("reporting_start", "2017-01-01"))
    reporting_end = pd.Timestamp(metadata.get("reporting_end", "2025-12-31"))
    in_reporting = nodes.event_month.between(reporting_start, reporting_end).to_numpy()
    # P1 preserves geometric support; P2/P3 require eligible event contributors.
    contributing = (present if route == "P1" else at_event) & in_reporting
    times = pd.DatetimeIndex(nodes.event_month).asi8
    radius = pd.Timedelta(days=float(eps_days)).value
    scope = [metadata["polygon_track_id"], band, float(ratio), float(eps_days)]

    def observable(left, right):
        begin = dates.searchsorted(pd.Timestamp(left), side="left")
        end = dates.searchsorted(pd.Timestamp(right), side="right")
        return int(mask[:, begin:end].any(axis=1).sum()) if begin < end else 0

    densities, denominators, thresholds = {}, {}, {}
    for centre in np.unique(times):
        nearby = (np.abs(times - centre) <= radius) & contributing
        densities[centre] = int(nodes.loc[nearby, "pixel_track_id"].nunique())
        denominators[centre] = n_geometry if route == "P1" else observable(centre - radius, centre + radius)
        thresholds[centre] = max(3, math.ceil(float(ratio) * denominators[centre]))
    nodes["density_response_pixels"] = [densities[time] for time in times]
    nodes["density_denominator"] = [denominators[time] for time in times]
    nodes["density_threshold"] = [thresholds[time] for time in times]
    core = contributing & (nodes.density_response_pixels.to_numpy() >= nodes.density_threshold.to_numpy())
    nodes["is_core"] = core
    labels = np.full(len(nodes), -1, dtype=int)
    core_indices = np.flatnonzero(core)
    label = -1
    previous = None
    for index in core_indices:
        if previous is None or times[index] - times[previous] > radius:
            label += 1
        labels[index] = label
        previous = index
    for index in np.flatnonzero(contributing & ~core):
        nearby = core_indices[np.abs(times[core_indices] - times[index]) <= radius]
        if len(nearby):
            closest = min(nearby, key=lambda j: (abs(times[j] - times[index]), times[j], nodes.node_id.iloc[j]))
            labels[index] = labels[closest]
    nodes["membership_type"] = np.where(core, "core", np.where(labels >= 0, "border", "noise"))
    nodes["noise_reason"] = np.where(
        ~present, "event_month_outside_calendar",
        np.where(~in_reporting, "event_month_outside_reporting_calendar",
                 np.where((route != "P1") & ~at_event, "ineligible_event_month",
                          np.where(labels < 0, "no_core_neighbour", ""))),
    )
    nodes["parent_component_id"] = ""
    nodes["episode_id"] = ""
    nodes["pixel_vote_date"] = pd.NaT
    nodes["status"] = "noise"
    nodes["accepted"] = False
    nodes["rejection_reason"] = ""
    nodes["source_row_count"] = nodes.node_id.map(source.node_id.value_counts()).astype(int)
    episode_rows = []
    available_start = max(dates[0], reporting_start)
    available_end = min(dates[-1] + pd.offsets.MonthEnd(0), reporting_end)

    def emit(indices, parent_id, parent_span, left, right, centre=pd.NaT):
        assigned = nodes.loc[indices]
        votes = assigned.groupby("pixel_track_id", sort=True).event_month.agg(
            lambda values: empirical_date_quantile(values, .5)
        )
        n_response = len(votes)
        n_observable = observable(left, right)
        denominator = n_geometry if route == "P1" else n_observable
        threshold = max(3, math.ceil(float(ratio) * denominator))
        if fallback:
            reason = "representative_pixel_fallback"
        elif denominator == 0:
            reason = "zero_geometry" if route == "P1" else "zero_observable"
        elif n_response < threshold:
            reason = "insufficient_distinct_responders"
        else:
            reason = ""
        accepted = not reason
        episode_id = _stable_id("episode_", scope, route, sorted(assigned.node_id), str(left), str(right))
        vote_dates = pd.DatetimeIndex(votes)
        month_counts = Counter(vote_dates)
        peak = min(month_counts, key=lambda date: (-month_counts[date], date))
        components, largest = spatial_components(coordinate_lookup.loc[votes.index].to_numpy())
        quantiles = {f"q{int(q * 100)}_month": empirical_date_quantile(vote_dates, q)
                     for q in (.1, .25, .5, .75, .9)}
        low, high = assigned.event_month.min(), assigned.event_month.max()
        row = {
            **metadata, "episode_id": episode_id, "track_event_id": episode_id,
            "parent_component_id": parent_id, "route": route, "ratio": float(ratio),
            "selected_ratio": float(ratio), "Band": band, "band": band,
            "status": "accepted" if accepted else "rejected", "accepted": accepted,
            "rejection_reason": reason, "change_time": quantiles["q50_month"],
            "representative_month": quantiles["q50_month"], "peak_month": peak, **quantiles,
            "min_detected_month": low, "max_detected_month": high,
            "envelope_start": pd.Timestamp(left), "envelope_end": pd.Timestamp(right),
            "episode_centre": centre,
            "envelope_definition": "fixed_centre_plus_minus_radius" if route == "P3" else "original_component_date_span",
            "N_geometry": n_geometry, "N_observable": n_observable, "N_response": n_response,
            "acceptance_denominator": denominator, "support_threshold": threshold,
            "support_pixels": n_response, "support_fraction": n_response / denominator if denominator else np.nan,
            "response_fraction": n_response / n_observable if n_observable else np.nan,
            "response_fraction_reason": "" if n_observable else "zero_observable",
            "observable_coverage": n_observable / n_geometry if n_geometry else np.nan,
            "component_count": np.nan if fallback else components,
            "largest_component_fraction": np.nan if fallback else largest,
            "calendar_clipped": bool(pd.Timestamp(left) < available_start or pd.Timestamp(right) > available_end),
            "fallback_extraction": fallback, "long_chain_parent": bool(parent_span > 2 * eps_days),
            "parent_span_days": parent_span, "assigned_span_days": (high - low).total_seconds() / 86400,
            "cross_year": bool(low.year != high.year),
            "date_assignment_ambiguity": bool(assigned.groupby("pixel_track_id").event_month.nunique().gt(1).any()),
            "node_count": len(assigned), "source_row_count": int(assigned.source_row_count.sum()),
            "multi_track_support": 1,
        }
        episode_rows.append(row)
        nodes.loc[indices, "episode_id"] = episode_id
        nodes.loc[indices, "pixel_vote_date"] = assigned.pixel_track_id.map(votes)
        nodes.loc[indices, "status"] = row["status"]
        nodes.loc[indices, "accepted"] = accepted
        nodes.loc[indices, "rejection_reason"] = reason

    for component_label in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == component_label)
        component = nodes.loc[indices]
        parent_id = _stable_id("component_", scope, "P1" if route == "P1" else "P2", sorted(component.node_id))
        nodes.loc[indices, "parent_component_id"] = parent_id
        left, right = component.event_month.min(), component.event_month.max()
        parent_span = (right - left).total_seconds() / 86400
        if route != "P3":
            emit(indices, parent_id, parent_span, left, right)
            continue
        # Consume every assigned record, even if the fixed-envelope gate fails.
        remaining = indices.copy()
        while len(remaining):
            choices = []
            for centre in np.unique(times[remaining]):
                assigned = remaining[np.abs(times[remaining] - centre) <= radius]
                n_near = nodes.loc[assigned, "pixel_track_id"].nunique()
                n_centre = nodes.loc[remaining[times[remaining] == centre], "pixel_track_id"].nunique()
                choices.append((-n_near, -n_centre, centre, assigned))
            _, _, centre, assigned = min(choices, key=lambda choice: choice[:3])
            emit(assigned, parent_id, parent_span, centre - radius, centre + radius, pd.Timestamp(centre))
            remaining = remaining[~np.isin(remaining, assigned)]
    nodes["Band"] = band
    nodes["band"] = band
    nodes["route"] = route
    nodes["ratio"] = float(ratio)
    nodes["fallback_extraction"] = fallback
    nodes["N_geometry"] = n_geometry
    for key, value in metadata.items():
        if key not in nodes:
            nodes[key] = [value] * len(nodes)
    lookup = nodes.set_index("node_id")
    members = source.copy()
    for column in _MEMBERSHIP_COLUMNS:
        if column in lookup and column not in {"pixel_track_id", "Band", "band", "event_month"}:
            members[column] = members.node_id.map(lookup[column])
    members["track_event_id"] = members.episode_id
    for coordinate in ("native_row", "native_col"):
        expected = members.pixel_track_id.map(coordinate_lookup[coordinate])
        if coordinate in members and not members[coordinate].eq(expected).all():
            raise ValueError(f"source events disagree with native coordinate {coordinate}")
        members[coordinate] = expected
    for key, value in metadata.items():
        if key not in members:
            members[key] = [value] * len(members)
    members = members.sort_values(["event_month", "pixel_track_id", "node_id", "source_event_id"]).reset_index(drop=True)
    episodes = pd.DataFrame(episode_rows, columns=episode_columns)
    if len(episodes):
        episodes = episodes.sort_values(["envelope_start", "episode_id"]).reset_index(drop=True)
    return episodes, members, nodes.reindex(columns=node_columns)


def temporal_profiles(episodes, memberships, calendar):
    """Histogram and cumulative assigned responses with each episode's fixed N.

    One row per calendar month in each accepted or rejected episode envelope.
    No reclustering or reference-date matching occurs in this summary.
    """
    dates = _calendar(calendar)
    columns = ["episode_id", "track_event_id", "month", "response_pixels", "cumulative_response_pixels",
               "N_observable", "response_fraction", "cumulative_response_fraction", "status", "route", "ratio"]
    rows = []
    for episode in episodes.itertuples(index=False):
        local = memberships.loc[memberships.episode_id.eq(episode.episode_id)]
        votes = local[["pixel_track_id", "pixel_vote_date"]].drop_duplicates()
        if votes.pixel_track_id.duplicated().any():
            raise ValueError("an episode must assign exactly one vote date per pixel")
        counts = votes.pixel_vote_date.value_counts()
        selected = dates[(dates >= episode.envelope_start) & (dates <= episode.envelope_end)]
        cumulative = 0
        for month in selected:
            count = int(counts.get(month, 0))
            cumulative += count
            rows.append({
                "episode_id": episode.episode_id, "track_event_id": episode.track_event_id,
                "month": month, "response_pixels": count, "cumulative_response_pixels": cumulative,
                "N_observable": episode.N_observable,
                "response_fraction": count / episode.N_observable if episode.N_observable else np.nan,
                "cumulative_response_fraction": cumulative / episode.N_observable if episode.N_observable else np.nan,
                "status": episode.status, "route": episode.route, "ratio": episode.ratio,
            })
    return pd.DataFrame(rows, columns=columns)
