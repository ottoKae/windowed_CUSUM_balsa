"""Independent design calculations and S3 completion gates for B20 S4."""

import importlib.util
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/b20_probability_estimation.py"
spec = importlib.util.spec_from_file_location("b20_probability_estimation", SCRIPT)
est = importlib.util.module_from_spec(spec)
spec.loader.exec_module(est)


def cell(h, block, category, M, m, N, n, labels):
    assert len(labels) == n
    pi = (m / M) * (n / N)
    return [
        {"prediction_id": f"{h}_{block}_{category}_{i}", "selected": True,
         "geo_stratum": h, "block_id": block, "candidate_class": category,
         "M_h": M, "m_h": m, "N_bc": N, "n_bc": n,
         "inclusion_probability": pi, "sampling_weight": 1 / pi, "y": y}
        for i, y in enumerate(labels)
    ]


def example():
    return pd.DataFrame(
        cell("A", "A1", "C1", 4, 2, 4, 2, [1, 0])
        + cell("A", "A1", "C2", 4, 2, 1, 1, [1])
        + cell("A", "A2", "C1", 4, 2, 6, 2, [0, 0])
        + cell("B", "B1", "C1", 1, 1, 3, 3, [1, 0, 1])
    )


def reviews(frame, labels=None):
    sample = frame.loc[frame.selected]
    if labels is None:
        labels = ["T" if value else "F" for value in sample.y]
    return pd.DataFrame([
        {"prediction_id": prediction, "tolerance_quarters": delta,
         "review_status": "reviewed", "judgement": label}
        for delta in (0, 1)
        for prediction, label in zip(sample.prediction_id, labels)
    ])


def full_frame():
    rows = []
    for b in range(3):
        block = cell("H", f"B{b}", "C1", 3, 2, 3, 2, [1, 0])
        block.append({**block[0], "prediction_id": f"H_B{b}_C1_2", "selected": False, "y": 1})
        if b == 2:
            for row in block:
                row["selected"] = False
        rows.extend(block)
    return pd.DataFrame(rows)


def test_total_variance_matches_hand_calculation_with_both_stages():
    frame = example()
    actual = est.estimate_two_stage_total(frame, frame.y)
    # Block totals 3, 0, 2; A has expansion 2. Within A1: 4^2 * .5 * .5 / 2 = 2.
    assert actual == pytest.approx({
        "total": 8, "variance": 22, "variance_between": 18, "variance_within": 4,
    })


def test_ratio_linearizes_block_totals_and_uses_all_weights():
    frame = example()
    actual = est.estimate_two_stage_ratio(frame, frame.y)
    assert frame.sampling_weight.sum() == 25
    assert actual["estimate"] == pytest.approx(8 / 25)
    # Residual totals in A: 3 - .32*5 = 1.4, 0 - .32*6 = -1.92.
    assert actual["variance_between"] == pytest.approx(22.0448 / 625)
    assert actual["variance_within"] == pytest.approx(4 / 625)
    assert actual["variance"] == pytest.approx(26.0448 / 625)
    assert actual["se"] == pytest.approx(np.sqrt(26.0448) / 25)
    result = est.summarize_probability_results(frame, reviews(frame), population_size=24)
    assert (result.precision_hajek == 8 / 25).all()
    assert (result.precision_ht == 8 / 24).all()


def test_enumerating_every_two_stage_sample_recovers_true_total_and_variance():
    population = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    choices = list(combinations(range(3), 2))
    estimates, variances = [], []
    # All 27 equiprobable samples; no Monte Carlo tolerance or implementation-derived oracle.
    for blocks in combinations(range(3), 2):
        for selected in product(choices, repeat=2):
            rows = []
            for b, indices in zip(blocks, selected):
                rows += cell("H", f"B{b}", "C1", 3, 2, 3, 2, [population[b][i] for i in indices])
            frame = pd.DataFrame(rows)
            result = est.estimate_two_stage_total(frame, frame.y)
            estimates.append(result["total"])
            variances.append(result["variance"])
    assert len(estimates) == 27
    assert np.mean(estimates) == pytest.approx(6)
    assert np.mean(variances) == pytest.approx(np.var(estimates, ddof=0))


def test_first_stage_census_retains_second_stage_variance():
    frame = pd.DataFrame(
        cell("H", "A", "C1", 2, 2, 4, 2, [1, 0])
        + cell("H", "B", "C1", 2, 2, 4, 2, [1, 0])
    )
    assert est.estimate_two_stage_total(frame, frame.y) == pytest.approx({
        "total": 4, "variance": 4, "variance_between": 0, "variance_within": 4,
    })
    result = est.summarize_probability_results(frame, reviews(frame), population_size=8)
    assert (result.variance_lower == .0625).all()
    assert (result.degrees_of_freedom == 0).all()
    assert result.ci95_lower.isna().all()
    assert result.ci_status.eq("insufficient_degrees_of_freedom").all()


def test_singleton_population_census_has_valid_zero_variance_interval():
    frame = pd.DataFrame(cell("H", "B", "C1", 1, 1, 1, 1, [1]))
    result = est.summarize_probability_results(frame, reviews(frame), population_size=1)
    assert result.is_population_census.all()
    assert result.ci_status.eq("population_census").all()
    assert result.variance_lower.eq(0).all()
    assert result.ci95_lower.eq(1).all() and result.ci95_upper.eq(1).all()


def test_unknowns_retain_denominator_and_have_their_own_standard_error():
    frame = example()
    labels = ["T", "U", "F", "F", "U", "T", "F", "U"]
    result = est.summarize_probability_results(frame, reviews(frame, labels), population_size=24).iloc[1]
    assert np.isnan(result.precision_hajek)  # No midpoint or T/(T+F).
    assert np.isnan(result.precision_ht)
    assert result.lower_hajek == pytest.approx(5 / 25)
    assert result.upper_hajek == pytest.approx(16 / 25)
    assert result.unknown_hajek == pytest.approx(11 / 25)
    assert result.upper_hajek - result.lower_hajek == pytest.approx(result.unknown_hajek)
    unknown = est.estimate_two_stage_ratio(frame, np.array(labels) == "U")
    assert result.se_unknown == pytest.approx(unknown["se"])
    assert result.se_unknown != pytest.approx(result.se_lower + result.se_upper)
    # Large t multiplier at df=1 must be clipped only for the physical CI.
    assert result.ci95_lower == 0 and result.ci95_upper == 1


@pytest.mark.parametrize("pending_label", [None, "", "U"])
def test_any_pending_review_blocks_both_tolerances_and_is_not_reviewed_unknown(pending_label):
    frame = example()
    result_rows = reviews(frame)
    result_rows.loc[0, ["review_status", "judgement"]] = ["pending", pending_label]
    result = est.summarize_probability_results(frame, result_rows, population_size=24)
    assert result.review_status.eq("pending").all()
    assert result.ci_status.eq("pending").all()
    assert result.loc[:, est.ESTIMATE_COLUMNS].isna().all().all()
    assert result.n_pending.tolist() == [1, 0]
    assert result.n_U.eq(0).all()
    assert result.n_reviewed.tolist() == [7, 8]


def test_raw_ht_can_exceed_one_and_sample_degeneracy_does_not_claim_certainty():
    frame = example()
    result = est.summarize_probability_results(frame, reviews(frame, ["T"] * len(frame)), population_size=24)
    assert result.precision_hajek.eq(1).all()
    assert result.precision_ht.eq(25 / 24).all()
    assert result.ci_status.eq("degenerate_variance").all()
    assert result.ci95_lower.isna().all() and result.ci95_upper.isna().all()
    assert result.n_selected.eq(len(frame)).all()


def test_one_constant_sample_cell_does_not_invalidate_nonzero_overall_variance():
    frame = example()  # A2 is all F and A1/C2 all T, with nonzero overall variance.
    result = est.summarize_probability_results(frame, reviews(frame), population_size=24)
    assert result.ci_status.eq("approximate_95_percent").all()
    assert result.ci95_lower.notna().all() and result.ci95_upper.notna().all()


def test_floating_weights_do_not_turn_constant_judgements_into_sampling_certainty():
    rows = []
    for b in range(22):
        rows += cell("H", f"B{b}", "C1", 24, 22, 7 + b, 3, [1, 1, 1])
    frame = pd.DataFrame(rows)
    result = est.summarize_probability_results(frame, reviews(frame), population_size=600)
    assert result.precision_hajek.eq(1).all()
    assert result.variance_lower.eq(0).all()
    assert result.ci_status.eq("degenerate_variance").all()


def test_roundoff_in_equal_block_ratios_is_not_a_nonzero_sampling_variance():
    frame = pd.DataFrame(
        cell("H", "A", "C1", 3, 2, 3, 3, [1, 0, 0])
        + cell("H", "B", "C1", 3, 2, 9, 9, [1, 0, 0] * 3)
    )
    result = est.summarize_probability_results(frame, reviews(frame), population_size=15)
    assert result.precision_hajek.to_numpy() == pytest.approx([1 / 3] * 2)
    assert result.ci_status.eq("degenerate_variance").all()
    assert result.ci95_lower.isna().all()


def test_degenerate_endpoint_blocks_interval_even_if_the_other_endpoint_varies():
    frame = example()
    labels = ["U" if value else "F" for value in frame.y]
    result = est.summarize_probability_results(frame, reviews(frame, labels), population_size=24)
    assert result.variance_lower.eq(0).all()
    assert result.variance_upper.gt(0).all()
    assert result.ci_status.eq("degenerate_variance").all()
    assert result.ci95_upper.isna().all()


def test_frozen_allocation_contributes_26_degrees_of_freedom():
    quotas = [(7, 2), (24, 22), (12, 12), (5, 5), (5, 3), (5, 2), (7, 2), (2, 2)]
    rows = []
    for h, (M, m) in enumerate(quotas, 1):
        for b in range(m):
            rows += cell(f"H{h:02d}", f"H{h:02d}_B{b}", "C1", M, m, 2, 2, [b % 2] * 2)
    frame = pd.DataFrame(rows)
    result = est.summarize_probability_results(frame, reviews(frame), population_size=134)
    assert result.degrees_of_freedom.eq(26).all()
    assert result.t_critical.to_numpy() == pytest.approx([2.0555294386428713] * 2)
    assert result.is_primary_result.tolist() == [False, True]


def test_full_frame_and_reordered_reviews_preserve_estimates_and_inputs():
    frame = full_frame()
    result_rows = reviews(frame)
    before_frame, before_results = frame.copy(deep=True), result_rows.copy(deep=True)
    expected = est.summarize_probability_results(frame, result_rows, population_size=9)
    actual = est.summarize_probability_results(
        frame.sample(frac=1, random_state=5), result_rows.sample(frac=1, random_state=9), population_size=9,
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, before_frame)
    pd.testing.assert_frame_equal(result_rows, before_results)


@pytest.mark.parametrize("change, message", [
    (lambda f: f.iloc[:-1], "selected cell count"),
    (lambda f: pd.concat([f, f.iloc[[0]]]), "duplicate prediction_id"),
    (lambda f: f.assign(sampling_weight=f.sampling_weight * 2), "sampling_weight"),
    (lambda f: f.assign(inclusion_probability=.5), "inclusion_probability"),
    (lambda f: f.assign(N_bc=f.N_bc.where(f.index != 0, 5)), "inconsistent N_bc"),
    (lambda f: f.assign(m_h=f.m_h.where(f.index != 0, 3)), "inconsistent m_h"),
    (lambda f: f.loc[f.block_id != "A2"], "selected block count"),
    (lambda f: f.assign(selected="True"), "explicit booleans"),
    (lambda f: f.assign(n_bc=.5), "integers"),
    (lambda f: f.assign(inclusion_probability=np.nan), "finite and positive"),
])
def test_corrupt_sample_design_is_rejected_before_estimation(change, message):
    frame = change(example())
    with pytest.raises(ValueError, match=message):
        est.estimate_two_stage_total(frame, np.zeros(len(frame)))


@pytest.mark.parametrize("change, message", [
    (lambda r: r.iloc[1:], "cover all and only"),
    (lambda r: pd.concat([r, r.iloc[[0]]]), "duplicate prediction_id"),
    (lambda r: r.assign(prediction_id=r.prediction_id.where(r.index != 0, "context_only")), "cover all and only"),
    (lambda r: r.assign(judgement=r.judgement.where(r.index != 0, None)), "reviewed results require"),
    (lambda r: r.assign(review_status="done"), "review_status"),
    (lambda r: r.assign(tolerance_quarters=2), "tolerance_quarters"),
])
def test_review_join_and_status_errors_cannot_silently_drop_or_score_rows(change, message):
    frame = example()
    with pytest.raises(ValueError, match=message):
        est.summarize_probability_results(frame, change(reviews(frame)), population_size=24)


def test_full_frame_detects_missing_unselected_population_row():
    frame = full_frame().iloc[:-1]
    with pytest.raises(ValueError, match="complete frame cell count"):
        est.summarize_probability_results(frame, reviews(frame), population_size=9)


def test_complete_frame_requires_a_sample_from_each_nonempty_cell_in_selected_block():
    frame = full_frame()
    extra = pd.DataFrame(cell("H", "B0", "C2", 3, 2, 1, 1, [1])).assign(selected=False)
    frame = pd.concat([frame, extra], ignore_index=True)
    with pytest.raises(ValueError, match="selected cell count"):
        est.summarize_probability_results(frame, reviews(frame), population_size=10)


def test_ratio_rejects_a_series_with_mismatched_row_order():
    frame = example()
    with pytest.raises(ValueError, match="values index"):
        est.estimate_two_stage_ratio(frame, frame.y.iloc[::-1])


def test_population_size_is_verified_when_all_blocks_are_observed():
    frame = pd.DataFrame(cell("H", "B", "C1", 1, 1, 1, 1, [1]))
    with pytest.raises(ValueError, match="all-block population total"):
        est.summarize_probability_results(frame, reviews(frame), population_size=2)


def month_reviews(frame):
    rows = reviews(frame)
    rows = rows.loc[rows.tolerance_quarters.eq(0)].drop(columns="tolerance_quarters")
    return pd.concat([rows.assign(tolerance_months=delta, tolerance_unit="month")
                      for delta in (1, 2, 3)], ignore_index=True)


def test_monthly_summaries_reuse_two_stage_weights_variance_and_keep_unknowns():
    frame = example()
    rows = month_reviews(frame)
    rows.loc[rows.tolerance_months.ge(2) & rows.prediction_id.eq(frame.prediction_id.iloc[1]), "judgement"] = "T"
    rows.loc[rows.tolerance_months.eq(3) & rows.prediction_id.eq(frame.prediction_id.iloc[4]), "judgement"] = "U"
    result = est.summarize_probability_results(frame, rows, population_size=24, tolerance_unit="month")
    assert "tolerance_quarters" not in result
    assert result.tolerance_months.tolist() == [1, 2, 3]
    assert result.tolerance_unit.eq("month").all()
    assert not result.is_primary_result.any()  # No primary tolerance chosen after seeing results.
    assert result.lower_hajek.to_numpy() == pytest.approx([8 / 25, 12 / 25, 12 / 25])
    assert result.upper_hajek.to_numpy() == pytest.approx([8 / 25, 12 / 25, 18 / 25])
    assert result.loc[0, "variance_lower"] == pytest.approx(26.0448 / 625)
    assert result.loc[1, "precision_hajek"] == pytest.approx(12 / 25)
    assert np.isnan(result.loc[2, "precision_hajek"])
    assert result.loc[2, "unknown_hajek"] == pytest.approx(6 / 25)
    assert result.n_U.tolist() == [0, 0, 1]
    assert result.n_selected.eq(8).all()


def test_monthly_pending_is_not_unknown_and_blocks_all_three_estimates():
    frame = example()
    rows = month_reviews(frame)
    rows.loc[0, ["review_status", "judgement"]] = ["pending", ""]
    result = est.summarize_probability_results(frame, rows, population_size=24, tolerance_unit="month")
    assert result.review_status.eq("pending").all()
    assert result.loc[:, est.ESTIMATE_COLUMNS].isna().all().all()
    assert result.n_pending.tolist() == [1, 0, 0]
    assert result.n_U.eq(0).all()


def test_monthly_units_and_complete_selected_rows_cannot_be_silently_relabelled():
    frame = example()
    rows = month_reviews(frame)
    with pytest.raises(ValueError, match="tolerance_quarters"):
        est.summarize_probability_results(frame, rows, population_size=24)
    with pytest.raises(ValueError, match="tolerance_unit"):
        est.summarize_probability_results(frame, rows.assign(tolerance_unit="quarter"),
                                           population_size=24, tolerance_unit="month")
    with pytest.raises(ValueError, match="cover all and only"):
        est.summarize_probability_results(frame, rows.loc[rows.tolerance_months.ne(3)],
                                           population_size=24, tolerance_unit="month")
    with pytest.raises(ValueError, match="duplicate prediction_id"):
        est.summarize_probability_results(frame, pd.concat([rows, rows.iloc[[0]]]),
                                           population_size=24, tolerance_unit="month")


def test_monthly_explicit_subset_requires_exactly_the_requested_tolerances():
    frame = example()
    rows = month_reviews(frame)
    subset = rows.loc[rows.tolerance_months.isin((1, 3))]
    result = est.summarize_probability_results(frame, subset, population_size=24,
                                              tolerance_unit="month", tolerances=(1, 3))
    assert result.tolerance_months.tolist() == [1, 3]
    assert result.precision_hajek.to_numpy() == pytest.approx([8 / 25] * 2)


@pytest.mark.parametrize("unit, values", [
    ("day", None), ("month", ()), ("month", (1, 1)), ("month", (1, -1)),
    ("month", (1.5,)), ("month", (True,)), ("month", 2),
])
def test_invalid_tolerance_specification_is_rejected(unit, values):
    with pytest.raises(ValueError, match="tolerance"):
        est.tolerance_spec(unit, values)
