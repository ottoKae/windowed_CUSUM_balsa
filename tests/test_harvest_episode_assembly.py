"""Real assembly contracts for source conservation, support, and boundaries."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101"))
import episode_assembly as assembly
import episode_relations as relations


def inputs(records, total=10, missing=(), gap=1):
    months = np.arange(12, 120)
    counts = pd.Series([month for _, month in records], dtype=int).value_counts()
    monthly = pd.DataFrame({"sample_id": "s", "month_index": months,
                            "state": relations.QUIET, "total_pixels": total})
    monthly.loc[monthly.month_index.isin(missing), "state"] = "unobserved_or_ineligible"
    monthly.loc[monthly.month_index.isin(counts.index), "state"] = "active"
    phases, nodes, edges = [], [], []
    runs = relations.active_phases([counts.get(m, 0) for m in months])
    for order, (begin, end) in enumerate(runs):
        first, last, phase_id = int(months[begin]), int(months[end]), f"p{order}"
        phases.append(dict(sample_id="s", phase_id=phase_id, phase_order=order,
                           start_index=first, end_index=last, total_pixels=total))
        for unit, month in records:
            if first <= month <= last:
                nodes.append(dict(sample_id="s", phase_id=phase_id, node_id=f"{unit}:{month}",
                                  unit_id=unit, month_index=month, response_month=assembly._month(month)))
    for a, b in zip(phases[:-1], phases[1:]):
        states = monthly.loc[monthly.month_index.between(a["end_index"] + 1, b["start_index"] - 1), "state"]
        edges.append(dict(sample_id="s", left_phase_id=a["phase_id"], right_phase_id=b["phase_id"],
                          relation=relations.classify_gap(states, gap)["relation"], distance=.8))
    return (pd.DataFrame(phases, columns=["sample_id", "phase_id", "phase_order", "start_index", "end_index", "total_pixels"]),
            pd.DataFrame(edges, columns=["sample_id", "left_phase_id", "right_phase_id", "relation", "distance"]),
            pd.DataFrame(nodes, columns=["sample_id", "phase_id", "node_id", "unit_id", "month_index", "response_month"]), monthly)


def run(args, gap=1, span=12, threshold=np.inf, method="B1"):
    return assembly.assemble_episodes(*args, max_gap=gap, max_span=span, threshold=threshold, method=method)


def test_cross_year_distinct_support_and_legacy_median_round_up():
    result = run(inputs([("a", 59), ("b", 60)]))
    row = result["reports"].iloc[0]
    assert row.first_response_month == "2020-12" and row.last_response_month == "2021-01"
    assert row.representative_month == "2021-01" and row.span_months == 2
    assert row.response_pixels == 2 and row.support_fraction == .2
    assert row.left_boundary_status == row.right_boundary_status == "observed_separator"
    assert result["progress"].new_response_pixels.tolist() == [1, 1]


def test_repeated_pixel_does_not_create_support_or_progress():
    result = run(inputs([("a", 50), ("a", 51)], total=10))
    assert result["reports"].empty
    component = result["components"].iloc[0]
    assert component.response_pixels == 1 and component.support_fraction == .1
    assert component.destination_type == "residual"
    assert result["progress"].new_response_pixels.tolist() == [1, 0]
    assert result["source_members"].destination_type.eq("residual").all()


def test_source_weighted_legacy_date_differs_from_unique_pixel_progress_date():
    result = run(inputs([("a", 50), ("a", 51), ("a", 52), ("b", 52)], total=5))
    row = result["reports"].iloc[0]
    assert row.representative_month == assembly._month(52)
    assert row.progress_midpoint_month == assembly._month(51)
    assert row.source_event_count == 4 and row.response_pixels == 2
    assert result["progress"].new_response_pixels.tolist() == [1, 0, 1]


def test_long_chain_remains_one_unresolved_group():
    records = [("a", m) for m in range(40, 48)] + [("b", 44)]
    result = run(inputs(records), span=6)
    assert result["reports"].empty and len(result["components"]) == 1
    assert result["components"].iloc[0].reason == "long_chain_unresolved"
    assert len(result["source_members"]) == 9


def test_sts_can_resolve_a_long_chain_but_cannot_add_connections():
    args = inputs([("a", 40), ("b", 41), ("c", 43), ("d", 44)])
    b1 = run(args, span=3)
    b2 = run(args, span=3, threshold=.2)
    assert b1["reports"].empty and len(b1["components"]) == 1
    assert len(b2["reports"]) == 2
    assert b2["source_members"].node_id.is_unique
    zero_gap = run(inputs([("a", 40), ("b", 40), ("c", 42), ("d", 42)], gap=0),
                   gap=0, threshold=0)
    assert zero_gap["edge_decisions"].relation.eq("temporal_separate").all()
    assert zero_gap["edge_decisions"].sts_status.eq("not_applicable").all()


def test_unknown_internal_boundary_marks_both_components_unresolved():
    result = run(inputs([("a", 50), ("b", 50), ("c", 52), ("d", 52)], missing=[51]))
    assert result["reports"].empty and len(result["components"]) == 2
    assert result["components"].reason.eq("boundary_unknown").all()
    assert result["components"].history_has_unknown.all()
    assert result["source_members"].destination_type.eq("unresolved").all()


def test_sts_internal_cut_needs_no_second_quiet_period_and_keeps_all_sources():
    args = inputs([("a", 50), ("b", 50), ("c", 52), ("d", 52)])
    result = run(args, threshold=.2, method="B2")
    assert len(result["reports"]) == 2
    assert result["reports"].iloc[0].right_boundary_status == "sts_separate"
    assert result["reports"].iloc[1].left_boundary_status == "sts_separate"
    assert result["edge_decisions"].iloc[0].sts_status == "vetoed"
    assert result["source_members"].node_id.is_unique
    assert set(result["source_members"].node_id) == set(args[2].node_id)
    assert result["progress"].groupby("component_id").new_response_pixels.sum().eq(2).all()


def test_infinite_threshold_preserves_b1_scientific_contents():
    args = inputs([("a", 50), ("b", 50), ("c", 52), ("d", 52)])
    b1 = run(args, method="B1")
    b2 = run(args, threshold=np.inf, method="B2_off")
    stable = [c for c in assembly.COMPONENT_COLUMNS if c not in ("method", "prediction_id", "component_id")]
    pd.testing.assert_frame_equal(b1["components"][stable], b2["components"][stable])
    assert len(b2["reports"]) == 1 and b2["reports"].iloc[0].response_pixels == 4


def test_outer_record_censoring_is_retained_in_dated_reports():
    result = run(inputs([("a", 12), ("b", 12), ("c", 119), ("d", 119)]))
    assert len(result["reports"]) == 2
    assert bool(result["reports"].iloc[0].left_censored)
    assert bool(result["reports"].iloc[-1].right_censored)
    assert result["reports"].representative_month.tolist() == ["2017-01", "2025-12"]


def test_known_separator_and_missing_history_are_both_retained():
    result = run(inputs([("a", 50), ("b", 50), ("c", 56), ("d", 56)], missing=[53]), threshold=.1)
    assert len(result["reports"]) == 2 and result["reports"].history_has_unknown.all()
    assert result["edge_decisions"].iloc[0].relation == "temporal_separate"
    assert result["edge_decisions"].iloc[0].sts_status == "not_applicable"


def test_missing_sts_retains_join_and_single_phase_has_no_edge():
    args = inputs([("a", 50), ("b", 52)])
    args[1]["distance"] = np.nan
    result = run(args, threshold=.1)
    assert len(result["reports"]) == 1
    assert result["edge_decisions"].iloc[0].sts_status == "sts_unavailable"
    single = run(inputs([("a", 50), ("b", 51)]), threshold=0)
    assert single["edge_decisions"].empty and len(single["reports"]) == 1


def test_empty_node_cohort_has_schema_and_no_false_stability_report():
    result = run(inputs([]))
    assert all(frame.empty for frame in result.values())
    assert "representative_month" in result["reports"]
    assert "destination_type" in result["source_members"]


def test_aliases_or_missing_adjacency_are_rejected():
    args = list(inputs([("a", 50), ("b", 52)]))
    args[2] = pd.concat([args[2], args[2].iloc[[0]].assign(node_id="alias")])
    with pytest.raises(ValueError, match="deduplicated"):
        run(args)
    args = list(inputs([("a", 50), ("b", 52)]))
    args[1] = args[1].iloc[:0]
    with pytest.raises(ValueError, match="adjacent"):
        run(args)
    args = inputs([("a", 50), ("b", 52)])
    args[3].loc[args[3].month_index.eq(50), "state"] = relations.QUIET
    with pytest.raises(ValueError, match="active"):
        run(args)
