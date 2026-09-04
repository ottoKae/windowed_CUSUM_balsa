import pandas as pd
import pytest

from lr_cusum.validation_closure import (
    EVENT_MATCHER_GREEDY_V1,
    EVENT_MATCHER_OPTIMAL_V2,
    match_events_detailed,
    match_events_detailed_greedy_v1,
    match_events_detailed_optimal_v2,
)


def _events(reference_months, detected_months, sample_id="stand-1"):
    references = pd.DataFrame({
        "sample_id": [sample_id] * len(reference_months),
        "event_month": reference_months,
    })
    detections = pd.DataFrame({
        "sample_id": [sample_id] * len(detected_months),
        "change_time": detected_months,
    })
    return references, detections


def test_default_dispatch_preserves_greedy_v1_contract():
    references, detections = _events(
        ["2020-01", "2020-06"],
        ["2020-02", "2020-09"],
    )

    default_metrics, default_pairs = match_events_detailed(
        references, detections, tolerance_months=2
    )
    legacy_metrics, legacy_pairs = match_events_detailed_greedy_v1(
        references, detections, tolerance_months=2
    )

    assert default_metrics == legacy_metrics
    pd.testing.assert_frame_equal(default_pairs, legacy_pairs)


def test_optimal_v2_maximizes_cardinality_when_greedy_v1_does_not():
    # Greedy first selects 2020-03 -> 2020-02 because its signed lag is -1.
    # That blocks both feasible chronological pairs.  The cardinality optimum is
    # 2020-01 -> 2020-02 and 2020-03 -> 2020-04.
    references, detections = _events(
        ["2020-01", "2020-03"],
        ["2020-02", "2020-04"],
    )

    greedy_metrics, _ = match_events_detailed(
        references,
        detections,
        tolerance_months=1,
        matcher_version=EVENT_MATCHER_GREEDY_V1,
    )
    optimal_metrics, optimal_pairs = match_events_detailed(
        references,
        detections,
        tolerance_months=1,
        matcher_version=EVENT_MATCHER_OPTIMAL_V2,
    )

    assert greedy_metrics["tp"] == 1
    assert optimal_metrics["tp"] == 2
    assert optimal_metrics["fp"] == 0
    assert optimal_metrics["fn"] == 0
    assert optimal_pairs.signed_error_months.tolist() == [1, 1]


def test_optimal_v2_minimizes_total_absolute_error_after_cardinality():
    references, detections = _events(
        ["2020-01", "2020-04"],
        ["2020-02", "2020-03", "2020-04"],
    )

    metrics, pairs = match_events_detailed_optimal_v2(
        references, detections, tolerance_months=3
    )

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert pairs.detected_time.tolist() == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-04-01"),
    ]
    assert pairs.absolute_error_months.sum() == 1


def test_optimal_v2_breaks_exact_objective_ties_chronologically():
    references, detections = _events(
        ["2020-03", "2020-01"],
        ["2020-02"],
    )

    metrics, pairs = match_events_detailed_optimal_v2(
        references, detections, tolerance_months=1
    )

    assert metrics["tp"] == 1
    assert pairs.reference_time.tolist() == [pd.Timestamp("2020-01-01")]
    assert pairs.detected_time.tolist() == [pd.Timestamp("2020-02-01")]


def test_optimal_v2_is_invariant_to_input_row_order():
    references, detections = _events(
        ["2020-07", "2020-01", "2020-04"],
        ["2020-05", "2020-08", "2020-02"],
    )
    shuffled_references = references.sample(frac=1, random_state=3).reset_index(drop=True)
    shuffled_detections = detections.sample(frac=1, random_state=7).reset_index(drop=True)

    metrics_a, pairs_a = match_events_detailed_optimal_v2(
        references, detections, tolerance_months=1
    )
    metrics_b, pairs_b = match_events_detailed_optimal_v2(
        shuffled_references, shuffled_detections, tolerance_months=1
    )

    assert metrics_a == metrics_b
    pd.testing.assert_frame_equal(pairs_a, pairs_b)


def test_versioned_api_rejects_unknown_matcher():
    references, detections = _events(["2020-01"], ["2020-01"])

    with pytest.raises(ValueError, match="Unknown matcher_version"):
        match_events_detailed(
            references,
            detections,
            matcher_version="not-a-matcher",
        )

