import numpy as np

from lr_cusum.core import bootstrap_confidence, cusum_amplitude
from lr_cusum.temporal_nulls import (
    NullModelConfig,
    block_permutation_once,
    calendar_block_permutation_score,
    circular_shift_ranges,
    derive_candidate_seed,
    iid_permutation_score,
    monte_carlo_pvalue,
    residual_moving_block_bootstrap_score,
    trackwise_max_block_scores,
)


def test_finite_sample_pvalue_has_plus_one_correction():
    assert monte_carlo_pvalue(0, 3999) == 1 / 4000
    assert monte_carlo_pvalue(99, 3999) == 0.025


def test_iid_score_matches_legacy_exceedance_count_exactly():
    values = np.r_[np.zeros(20), np.full(20, -3.0)]
    observed = float(cusum_amplitude(values)[0])
    replicates = 200
    seed = 17
    legacy = float(bootstrap_confidence(values, observed, replicates, seed))
    expected_exceedances = replicates - int(round(legacy * replicates))
    score = iid_permutation_score(
        values,
        observed,
        config=NullModelConfig("iid_permutation", replicates, seed=seed),
        seed=seed,
    )
    assert score.exceedance_count == expected_exceedances
    assert score.null_p_value == (1 + expected_exceedances) / (replicates + 1)


def test_block_permutation_preserves_values_and_contiguous_block_order():
    values = np.arange(12)
    permuted = block_permutation_once(values, 3, seed=8)
    np.testing.assert_array_equal(np.sort(permuted), values)
    rng = np.random.default_rng(8)
    offset = int(rng.integers(0, 3))
    blocks = []
    if offset:
        blocks.append(values[:offset])
    blocks.extend(values[start : min(len(values), start + 3)] for start in range(offset, 12, 3))
    # Every block selected by the randomized boundary is present contiguously
    # and keeps its original internal order.
    as_list = permuted.tolist()
    for block in blocks:
        expected = block.tolist()
        assert any(
            as_list[start : start + len(expected)] == expected
            for start in range(len(as_list) - len(expected) + 1)
        )


def test_block_null_preserves_missing_values_without_zero_fill():
    values = np.r_[np.zeros(18), np.full(18, -2.0)].astype(float)
    values[[2, 7, 25]] = np.nan
    observed = float(cusum_amplitude(values[np.isfinite(values)])[0])
    score = calendar_block_permutation_score(
        values,
        observed,
        config=NullModelConfig("block_perm_l3", 50, block_length=3, seed=4),
    )
    assert 0 <= score.exceedance_count <= 50
    assert np.isnan(values).sum() == 3
    assert not np.any(values[np.isfinite(values)] == 1.0)


def test_residual_bootstrap_removes_step_and_is_deterministic():
    rng = np.random.default_rng(5)
    values = np.r_[rng.normal(0, 0.2, 24), rng.normal(-3, 0.2, 24)]
    observed = float(cusum_amplitude(values)[0])
    config = NullModelConfig("residual_mbb_l6", 100, block_length=6, seed=9)
    first = residual_moving_block_bootstrap_score(
        values, observed, 24, config=config, seed=9
    )
    second = residual_moving_block_bootstrap_score(
        values, observed, 24, config=config, seed=9
    )
    assert first == second
    assert first.null_p_value <= 0.025


def test_circular_shift_null_is_degenerate_for_cusum_range():
    values = np.array([0.2, -1.1, 0.7, 2.0, -0.4, 0.3, -0.9])
    ranges = circular_shift_ranges(values)
    np.testing.assert_allclose(ranges, ranges[0], rtol=0, atol=1e-12)


def test_candidate_seed_is_stable_and_method_specific():
    first = derive_candidate_seed("candidate-a", "iid", 20)
    second = derive_candidate_seed("candidate-a", "iid", 20)
    third = derive_candidate_seed("candidate-a", "block-l3", 20)
    assert first == second
    assert first != third


def test_trackwise_max_null_returns_one_score_per_candidate():
    values = np.r_[np.zeros(24), np.full(24, -3.0), np.zeros(24)]
    scores = trackwise_max_block_scores(
        values,
        [4.0, 20.0],
        config=NullModelConfig("max_block_l3", 50, block_length=3, seed=12),
        seed=12,
        window_starts=[0, 12, 24],
        window_ends=[48, 60, 72],
        decision_starts=[12, 24, 36],
        decision_ends=[36, 48, 60],
        minimum_observations=12,
        reporting_start_index=0,
        reporting_end_index=72,
    )
    assert len(scores) == 2
    assert scores[0].null_p_value >= scores[1].null_p_value
    assert all(score.method == "max_block_l3" for score in scores)
