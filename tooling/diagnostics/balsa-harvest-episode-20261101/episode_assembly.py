"""Assemble frozen pixel responses using the v3 phase-boundary contract.

No labels enter assembly. Unknown identities and subthreshold components keep
their source nodes, and record-edge censoring does not remove a dated report.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd

from episode_relations import B1_RELATIONS, QUIET, UNKNOWN_STATES, apply_sts_veto


EPOCH = pd.Period("2016-01", freq="M").ordinal
COMPONENT_COLUMNS = [
    "method", "sample_id", "component_id", "prediction_id", "destination_type", "reason",
    "first_response_month", "last_response_month", "span_months", "representative_month",
    "progress_midpoint_month", "response_pixels", "total_pixels", "support_fraction",
    "source_event_count", "phase_ids", "source_node_ids", "source_unit_ids", "response_months",
    "left_boundary_status", "right_boundary_status", "left_censored", "right_censored",
    "history_has_unknown", "interval_time_basis",
]
PROGRESS_COLUMNS = ["method", "sample_id", "component_id", "prediction_id", "response_month",
                    "new_response_pixels", "monthly_response_pixels", "cumulative_response_pixels",
                    "total_pixels", "cumulative_support_fraction"]


def assemble_episodes(phases, edges, nodes, monthly, *, max_gap, max_span, threshold, method):
    """Return reports, all components, unique node destinations, edges, progress.

    Indices are calendar months since 2016-01. Caller supplies the full report
    calendar and B1 relations for every pair of temporally adjacent phases.
    """
    _validate_inputs(phases, edges, nodes, monthly, max_gap, max_span, threshold, method)
    decisions = edges.copy()
    results = [apply_sts_veto(row.relation, row.distance, threshold)
               for row in decisions.itertuples(index=False)]
    decisions["b1_relation"] = decisions.relation
    decisions["relation"] = [result["relation"] for result in results]
    decisions["sts_status"] = [result["sts_status"] for result in results]
    decisions["method"] = method
    edge_lookup = {(r.left_phase_id, r.right_phase_id): r for r in decisions.itertuples(index=False)}
    month_groups = {key: group.sort_values("month_index") for key, group in monthly.groupby("sample_id")}
    node_groups = {key: group for key, group in nodes.groupby("sample_id")}
    component_rows, member_rows, progress_rows = [], [], []

    for sample_id, stand_phases in phases.groupby("sample_id", sort=True):
        ordered = stand_phases.sort_values("phase_order").to_dict("records")
        stand_nodes = node_groups[sample_id]
        stand_months = month_groups[sample_id]
        links = [edge_lookup[a["phase_id"], b["phase_id"]] for a, b in zip(ordered[:-1], ordered[1:])]
        boundaries = [0] + [i + 1 for i, link in enumerate(links) if link.relation != "temporal_join"] + [len(ordered)]

        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            phase_ids = [p["phase_id"] for p in ordered[begin:end]]
            sources = stand_nodes.loc[stand_nodes.phase_id.isin(phase_ids)].sort_values(
                ["month_index", "unit_id", "node_id"], kind="stable")
            first, last = int(sources.month_index.min()), int(sources.month_index.max())
            span = last - first + 1
            total = int(ordered[begin]["total_pixels"])
            response_pixels = int(sources.unit_id.nunique())
            left = links[begin - 1].relation if begin else _outer_boundary(
                stand_months.loc[stand_months.month_index.lt(first), "state"], max_gap, "left")
            right = links[end - 1].relation if end < len(ordered) else _outer_boundary(
                stand_months.loc[stand_months.month_index.gt(last), "state"], max_gap, "right")
            left_limit = int(ordered[begin - 1]["end_index"]) + 1 if begin else int(stand_months.month_index.min())
            right_limit = int(ordered[end]["start_index"]) - 1 if end < len(ordered) else int(stand_months.month_index.max())
            history_unknown = bool(stand_months.loc[
                stand_months.month_index.between(left_limit, right_limit), "state"].isin(UNKNOWN_STATES).any())
            if span > max_span:
                destination, reason = "unresolved", "long_chain_unresolved"
            elif "boundary_unknown" in (left, right):
                destination, reason = "unresolved", "boundary_unknown"
            elif 5 * response_pixels < total:
                destination, reason = "residual", "support_below_20_percent"
            else:
                destination, reason = "episode", "supported_model_episode"

            source_ids = sorted(sources.node_id.tolist())
            identity = json.dumps([method, sample_id, source_ids], separators=(",", ":"))
            component_id = "episode_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            prediction_id = component_id if destination == "episode" else ""
            first_per_pixel = sources.groupby("unit_id").month_index.min()
            record = dict(method=method, sample_id=sample_id, component_id=component_id,
                prediction_id=prediction_id, destination_type=destination, reason=reason,
                first_response_month=_month(first), last_response_month=_month(last), span_months=span,
                representative_month=_month(int(np.floor(sources.month_index.median() + .5))),
                progress_midpoint_month=_month(int(np.floor(first_per_pixel.median() + .5))),
                response_pixels=response_pixels, total_pixels=total, support_fraction=response_pixels / total,
                source_event_count=len(sources), phase_ids=json.dumps(phase_ids), source_node_ids=json.dumps(source_ids),
                source_unit_ids=json.dumps(sorted(sources.unit_id.unique().tolist())),
                response_months=json.dumps(sorted(sources.response_month.unique().tolist())),
                left_boundary_status=left, right_boundary_status=right,
                left_censored=left == "left_censored", right_censored=right == "right_censored",
                history_has_unknown=history_unknown, interval_time_basis="actual_pixel_response_months")
            component_rows.append(record)
            for source in sources.to_dict("records"):
                member_rows.append(dict(source, method=method, component_id=component_id,
                    prediction_id=prediction_id, destination_type=destination, reason=reason))
            increments = first_per_pixel.value_counts()
            monthly_counts = sources.groupby("month_index").unit_id.nunique()
            cumulative = 0
            for month in range(first, last + 1):
                added = int(increments.get(month, 0))
                cumulative += added
                progress_rows.append(dict(method=method, sample_id=sample_id, component_id=component_id,
                    prediction_id=prediction_id, response_month=_month(month), new_response_pixels=added,
                    monthly_response_pixels=int(monthly_counts.get(month, 0)), cumulative_response_pixels=cumulative,
                    total_pixels=total, cumulative_support_fraction=cumulative / total))

    components = pd.DataFrame(component_rows, columns=COMPONENT_COLUMNS)
    members = pd.DataFrame(member_rows, columns=list(nodes.columns) + [c for c in
        ["method", "component_id", "prediction_id", "destination_type", "reason"] if c not in nodes.columns])
    assert len(members) == len(nodes) and members.node_id.is_unique
    assert set(members.node_id) == set(nodes.node_id)
    return {"reports": components.loc[components.destination_type.eq("episode")].reset_index(drop=True),
            "components": components, "source_members": members, "edge_decisions": decisions,
            "progress": pd.DataFrame(progress_rows, columns=PROGRESS_COLUMNS)}


def _month(index):
    return str(pd.Period(ordinal=EPOCH + index, freq="M"))


def _outer_boundary(states, max_gap, side):
    longest = run = 0
    for state in states:
        run = run + 1 if state == QUIET else 0
        longest = max(longest, run)
    return "observed_separator" if longest >= max_gap + 1 else side + "_censored"


def _validate_inputs(phases, edges, nodes, monthly, max_gap, max_span, threshold, method):
    if (not isinstance(max_gap, Integral) or isinstance(max_gap, bool) or max_gap < 0
            or not isinstance(max_span, Integral) or isinstance(max_span, bool) or max_span < 1):
        raise ValueError("max_gap and max_span must be nonnegative and positive integers")
    if not isinstance(method, str) or not method or np.isnan(threshold) or threshold < 0:
        raise ValueError("method and nonnegative threshold are required")
    required = [
        (phases, ["sample_id", "phase_id", "phase_order", "start_index", "end_index", "total_pixels"]),
        (edges, ["sample_id", "left_phase_id", "right_phase_id", "relation", "distance"]),
        (nodes, ["sample_id", "phase_id", "node_id", "unit_id", "month_index", "response_month"]),
        (monthly, ["sample_id", "month_index", "state", "total_pixels"]),
    ]
    for frame, columns in required:
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"Missing input columns: {set(columns) - set(frame.columns)}")
        if frame[[c for c in columns if c != "distance"]].isna().any().any():
            raise ValueError("Input identities, dates and states cannot be missing")
    if not phases.phase_id.is_unique or not nodes.node_id.is_unique:
        raise ValueError("Phase and source node IDs must be unique")
    if nodes.duplicated(["sample_id", "unit_id", "month_index"]).any():
        raise ValueError("Pixel-month aliases must be deduplicated before assembly")
    if not edges.relation.isin(B1_RELATIONS).all() or not monthly.state.isin(UNKNOWN_STATES | {QUIET, "active"}).all():
        raise ValueError("Unknown phase relation or monthly state")
    if not set(phases.sample_id).issubset(monthly.sample_id) or set(nodes.phase_id) != set(phases.phase_id):
        raise ValueError("Phase members and monthly observation records must be complete")
    if monthly.duplicated(["sample_id", "month_index"]).any():
        raise ValueError("Monthly observations must be unique")
    for _, group in monthly.groupby("sample_id"):
        if (not np.all(np.diff(np.sort(group.month_index)) == 1)
                or group.total_pixels.nunique() != 1 or int(group.total_pixels.iloc[0]) < 1):
            raise ValueError("Monthly records require a consecutive calendar and fixed positive denominator")
    observed = nodes.merge(monthly[["sample_id", "month_index", "state"]],
                           on=["sample_id", "month_index"], how="left", validate="many_to_one")
    if not observed.state.eq("active").all():
        raise ValueError("Every source month must occur as active in the report calendar")
    phase_lookup = phases.set_index("phase_id")
    joined = nodes.join(phase_lookup[["sample_id", "start_index", "end_index", "total_pixels"]], on="phase_id", rsuffix="_phase")
    if (not joined.sample_id.eq(joined.sample_id_phase).all()
            or not joined.month_index.between(joined.start_index, joined.end_index).all()
            or not nodes.response_month.eq(nodes.month_index.map(_month)).all()):
        raise ValueError("Node phase ownership or original calendar months disagree")
    actual = nodes.groupby("phase_id").month_index.agg(["min", "max"])
    if (not actual["min"].eq(phase_lookup.start_index).all()
            or not actual["max"].eq(phase_lookup.end_index).all()):
        raise ValueError("Phase bounds must equal their first and last source months")
    expected = set()
    denominators = monthly.groupby("sample_id").total_pixels.first()
    if not phases.total_pixels.eq(phases.sample_id.map(denominators)).all():
        raise ValueError("Phase denominators differ from monthly records")
    unique_members = nodes.groupby("sample_id").unit_id.nunique()
    if (unique_members > denominators.reindex(unique_members.index)).any():
        raise ValueError("Response membership exceeds the fixed stand denominator")
    for sample_id, group in phases.groupby("sample_id"):
        group = group.sort_values("phase_order")
        if (group.phase_order.duplicated().any()
                or np.any(group.start_index.to_numpy()[1:] - group.end_index.to_numpy()[:-1] < 2)):
            raise ValueError("Phases must be ordered, disjoint maximal active runs")
        expected.update((sample_id, a, b) for a, b in zip(group.phase_id.iloc[:-1], group.phase_id.iloc[1:]))
    actual_edges = list(edges[["sample_id", "left_phase_id", "right_phase_id"]].itertuples(index=False, name=None))
    if len(actual_edges) != len(expected) or set(actual_edges) != expected:
        raise ValueError("Exactly one relation per adjacent phase pair is required")
