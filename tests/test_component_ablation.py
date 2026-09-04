import numpy as np
import pandas as pd

from lr_cusum.component_ablation import (
    ComponentAblationConfig,
    build_component_ledgers,
)
from lr_cusum.core import lr_cusum


def _config(**updates):
    values = {
        "window_length": 48,
        "step": 12,
        "confidence_threshold": 0.90,
        "bootstrap_iterations": 50,
        "minimum_observations": 12,
        "seed": 11,
        "start_year": 2017,
        "end_year": 2025,
    }
    values.update(updates)
    return ComponentAblationConfig(**values)


def test_full_stage_matches_legacy_core_event_fields_exactly():
    rng = np.random.default_rng(4)
    values = np.r_[
        rng.normal(-7, 0.15, 42),
        rng.normal(-10, 0.15, 38),
        rng.normal(-6.5, 0.15, 52),
    ]
    values[[7, 8, 56, 101]] = np.nan
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    config = _config()

    ledgers = build_component_ledgers(
        values, dates, config=config, series_key="stand:VV", series_seed=123
    )
    actual = ledgers.stage_frame("F4")
    expected = lr_cusum(
        values,
        dates,
        window_length=config.window_length,
        step=config.step,
        confidence_threshold=config.confidence_threshold,
        bootstrap_iterations=config.bootstrap_iterations,
        minimum_observations=config.minimum_observations,
        seed=123,
        start_year=config.start_year,
        end_year=config.end_year,
    )

    columns = [
        "change_index", "change_time", "window_start", "window_end",
        "cusum_amplitude", "bootstrap_confidence",
    ]
    expected_frame = pd.DataFrame([event.as_dict() for event in expected])[columns]
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected_frame.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def test_forward_candidates_are_nested_before_fusion():
    values = np.r_[np.zeros(40), np.full(40, -3.0), np.zeros(52)]
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    ledgers.assert_forward_nesting()

    stages = {
        name: set(ledgers.stage_frame(name).candidate_id)
        for name in ("F1", "F2", "F3", "F4")
    }
    assert stages["F4"] <= stages["F3"] <= stages["F2"] <= stages["F1"]


def test_decrease_gate_removes_upward_candidates():
    # Put the upward step close to the final edge so the legacy maximum-CUSUM
    # index falls inside the last window's central decision interval.  F2 is
    # intentionally direction agnostic; F3 must reject the positive shift.
    values = np.r_[np.full(97, -10.0), np.full(35, -5.0)]
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    assert len(ledgers.stage_frame("F2")) > 0
    assert len(ledgers.stage_frame("F3")) == 0


def test_reporting_filter_is_common_and_does_not_change_legacy_stage():
    values = np.r_[np.zeros(10), np.full(122, -4.0)]
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    legacy = ledgers.stage_frame("F1", reporting_only=False)
    strict = ledgers.stage_frame("F1", reporting_only=True)
    assert set(strict.candidate_id) <= set(legacy.candidate_id)
    if len(strict):
        assert pd.to_datetime(strict.change_time).dt.year.between(2017, 2025).all()


def test_removal_r3_is_identical_to_forward_f3():
    rng = np.random.default_rng(9)
    values = rng.normal(-8, 0.4, 132)
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    assert set(ledgers.stage_frame("R3").candidate_id) == set(
        ledgers.stage_frame("F3").candidate_id
    )


def test_windowing_recovers_two_separated_decreases_while_global_is_single():
    values = np.r_[
        np.full(25, -5.0),
        np.full(22, -9.0),
        np.full(25, -5.0),
        np.full(25, -9.0),
        np.full(35, -5.0),
    ]
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    assert len(ledgers.stage_frame("F0")) == 1
    assert ledgers.stage_frame("F3").change_index.nunique() >= 2


def test_central_gate_only_filters_existing_scored_candidates():
    values = np.r_[np.full(40, -5.0), np.full(40, -9.0), np.full(52, -6.0)]
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    f1 = ledgers.stage_frame("F1").set_index("candidate_id")
    f2 = ledgers.stage_frame("F2").set_index("candidate_id")
    removed = f1.index.difference(f2.index)
    assert len(removed) > 0
    assert f1.loc[removed, "passes_central"].eq(False).all()
    assert np.isfinite(f1.loc[removed, "cusum_amplitude"]).all()


def test_missing_months_keep_original_calendar_positions():
    values = np.r_[np.full(58, -6.0), np.full(74, -10.0)].astype(float)
    values[[3, 4, 17, 42, 61, 89]] = np.nan
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    valid = ledgers.sliding_candidates.loc[
        ledgers.sliding_candidates.candidate_valid
    ]
    assert len(valid) > 0
    assert (valid.valid_fraction < 1.0).any()
    for row in valid.itertuples(index=False):
        assert pd.Timestamp(row.change_time) == dates[int(row.change_index)]


def test_too_few_finite_observations_produces_no_stage_candidate():
    values = np.full(132, np.nan)
    values[:10] = np.linspace(-5.0, -8.0, 10)
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    ledgers = build_component_ledgers(values, dates, config=_config())
    assert len(ledgers.stage_frame("F0")) == 0
    assert len(ledgers.stage_frame("F1")) == 0
    assert not ledgers.sliding_candidates.passes_minimum_observations.any()


def test_permutation_scores_are_deterministic_for_same_series_seed():
    rng = np.random.default_rng(14)
    values = rng.normal(-7.0, 0.3, 132)
    dates = pd.date_range("2016-01-01", periods=132, freq="MS")
    first = build_component_ledgers(
        values, dates, config=_config(), series_seed=987
    ).sliding_candidates
    second = build_component_ledgers(
        values, dates, config=_config(), series_seed=987
    ).sliding_candidates
    np.testing.assert_array_equal(
        first.permutation_confidence.to_numpy(),
        second.permutation_confidence.to_numpy(),
    )
