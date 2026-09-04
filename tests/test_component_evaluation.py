import numpy as np
import pandas as pd
import pytest

from lr_cusum.component_evaluation import (
    common_index_paired_block_bootstrap,
    evaluate_event_stages,
    holm_adjust,
    paired_block_label_swap_test,
)
from lr_cusum.validation_closure import (
    EVENT_MATCHER_GREEDY_V1,
    EVENT_MATCHER_OPTIMAL_V2,
)


def _reference_table():
    return pd.DataFrame({
        "sample_id": ["a", "a", "b"],
        "event_month": ["2020-01", "2020-03", "2020-06"],
        "spatial_block_id": ["block-x", "block-x", "block-y"],
        "outer_fold": [0, 0, 1],
    })


def _stage_tables():
    complete = pd.DataFrame({
        "sample_id": ["a", "a", "b"],
        "change_time": ["2020-02", "2020-04", "2020-06"],
    })
    reduced = pd.DataFrame({
        "sample_id": ["a"],
        "change_time": ["2020-02"],
    })
    return {"A0": reduced, "A1": complete, "A1_clone": complete.copy()}


def test_stage_evaluation_reports_pooled_fold_and_block_counts():
    result = evaluate_event_stages(
        _reference_table(),
        _stage_tables(),
        matcher_version=EVENT_MATCHER_OPTIMAL_V2,
        tolerances=(0, 1),
        fold_column="outer_fold",
    )

    complete = result.stage_metrics.loc[
        result.stage_metrics.stage.eq("A1")
        & result.stage_metrics.tolerance_months.eq(1)
    ].iloc[0]
    assert (complete.tp, complete.fp, complete.fn) == (3, 0, 0)

    blocks = result.block_metrics.loc[
        result.block_metrics.stage.eq("A1")
        & result.block_metrics.tolerance_months.eq(1)
    ].set_index("spatial_block_id")
    assert (blocks.loc["block-x", "tp"], blocks.loc["block-y", "tp"]) == (2, 1)
    assert set(result.fold_metrics.outer_fold) == {0, 1}
    assert len(result.matches.loc[
        result.matches.stage.eq("A1")
        & result.matches.tolerance_months.eq(1)
    ]) == 3

    exact_month = result.stage_metrics.loc[
        result.stage_metrics.stage.eq("A1")
        & result.stage_metrics.tolerance_months.eq(0)
    ].iloc[0]
    assert (exact_month.tp, exact_month.fp, exact_month.fn) == (1, 2, 2)


def test_matcher_and_fold_selection_are_explicit():
    optimal = evaluate_event_stages(
        _reference_table(),
        {"complete": _stage_tables()["A1"]},
        matcher_version=EVENT_MATCHER_OPTIMAL_V2,
        tolerances=1,
    )
    greedy = evaluate_event_stages(
        _reference_table(),
        {"complete": _stage_tables()["A1"]},
        matcher_version=EVENT_MATCHER_GREEDY_V1,
        tolerances=1,
    )
    selected_fold = evaluate_event_stages(
        _reference_table(),
        {"complete": _stage_tables()["A1"]},
        matcher_version=EVENT_MATCHER_OPTIMAL_V2,
        tolerances=1,
        fold_column="outer_fold",
        folds=[0],
    )

    assert optimal.stage_metrics.iloc[0].tp == 3
    assert greedy.stage_metrics.iloc[0].tp == 2
    assert selected_fold.stage_metrics.iloc[0].tp == 2
    assert selected_fold.stage_metrics.iloc[0].reference_events == 2
    assert selected_fold.block_metrics.spatial_block_id.tolist() == ["block-x"]


def test_common_index_bootstrap_is_paired_and_deterministic():
    evaluation = evaluate_event_stages(
        _reference_table(),
        _stage_tables(),
        tolerances=1,
    )
    contrasts = {
        "A1_minus_A0": ("A1", "A0"),
        "clone_minus_A1": ("A1_clone", "A1"),
    }
    first = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=1,
        replicates=100,
        seed=17,
    )
    second = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=1,
        replicates=100,
        seed=17,
    )

    pd.testing.assert_frame_equal(first.stage_replicates, second.stage_replicates)
    pd.testing.assert_frame_equal(first.contrast_replicates, second.contrast_replicates)

    stage_f1 = first.stage_replicates.pivot(
        index="replicate", columns="stage", values="f1"
    )
    contrast_f1 = first.contrast_replicates.loc[
        first.contrast_replicates.contrast.eq("A1_minus_A0")
    ].set_index("replicate").delta_f1
    np.testing.assert_allclose(contrast_f1, stage_f1.A1 - stage_f1.A0)

    clone_delta = first.contrast_replicates.loc[
        first.contrast_replicates.contrast.eq("clone_minus_A1"), "delta_f1"
    ]
    assert np.allclose(clone_delta, 0)


def test_label_swap_randomization_is_paired_deterministic_and_holm_adjusted():
    evaluation = evaluate_event_stages(
        _reference_table(),
        _stage_tables(),
        tolerances=1,
    )
    contrasts = {
        "A1_minus_A0": ("A1", "A0"),
        "clone_minus_A1": ("A1_clone", "A1"),
    }
    first = paired_block_label_swap_test(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=1,
        permutations=255,
        seed=23,
    )
    second = paired_block_label_swap_test(
        evaluation.block_metrics,
        contrasts,
        tolerance_months=1,
        permutations=255,
        seed=23,
    )

    pd.testing.assert_frame_equal(first.replicates, second.replicates)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    assert first.summary.p_value.between(0, 1).all()
    assert (first.summary.p_value_holm >= first.summary.p_value).all()
    clone = first.summary.loc[first.summary.contrast.eq("clone_minus_A1")].iloc[0]
    assert clone.observed_delta == 0
    assert clone.p_value == 1


def test_holm_adjustment_is_monotone_in_sorted_p_values_and_preserves_nan():
    adjusted = holm_adjust([0.01, 0.03, 0.20, 0.04, np.nan])

    np.testing.assert_allclose(adjusted[:4], [0.04, 0.09, 0.20, 0.09])
    assert np.isnan(adjusted[4])


def test_evaluation_rejects_events_without_a_reference_block():
    events = pd.DataFrame({
        "sample_id": ["unknown"],
        "change_time": ["2020-01"],
    })

    with pytest.raises(ValueError, match="absent from references"):
        evaluate_event_stages(_reference_table(), {"A0": events})


def test_bootstrap_rejects_an_incomplete_stage_block_grid():
    evaluation = evaluate_event_stages(
        _reference_table(),
        _stage_tables(),
        tolerances=1,
    )
    incomplete = evaluation.block_metrics.loc[
        ~(
            evaluation.block_metrics.stage.eq("A0")
            & evaluation.block_metrics.spatial_block_id.eq("block-y")
        )
    ]

    with pytest.raises(ValueError, match="missing paired blocks"):
        common_index_paired_block_bootstrap(
            incomplete,
            {"A1_minus_A0": ("A1", "A0")},
            tolerance_months=1,
            replicates=5,
        )

