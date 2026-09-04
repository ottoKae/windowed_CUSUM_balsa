import numpy as np
import pandas as pd
import pytest

from lr_cusum.disturbance_intervals import (
    build_adjacent_intervals,
    match_polygon_polarizations,
    select_polygon_first_vv,
    spatial_block_bootstrap,
)


def _events():
    return pd.DataFrame({
        "sample_id": ["a", "a", "a", "a", "a", "b"],
        "polygon_id": ["a", "a", "a", "a", "a", "b"],
        "Band": ["VV_dB", "VV_dB", "VV_dB", "VH_dB", "VH_dB", "VV_dB"],
        "route": ["polygon_first"] * 5 + ["pixel_first"],
        "change_time": ["2020-01", "2021-01", "2022-01", "2020-01", "2022-02", "2020-01"],
        "track_support": [1, 1, 1, 1, 1, 1],
    })


def test_primary_selector_excludes_pixel_first():
    selected = select_polygon_first_vv(_events())
    assert len(selected) == 3
    assert selected.route.eq("polygon_first").all()
    assert selected.Band.eq("VV_dB").all()


def test_supported_interval_does_not_bridge_unsupported_middle_event():
    annotated, vh_only, matches = match_polygon_polarizations(_events())
    primary = select_polygon_first_vv(annotated)
    intervals = build_adjacent_intervals(primary)
    assert len(matches) == 2
    assert vh_only.empty
    assert intervals.interval_months.tolist() == [12, 12]
    assert intervals.both_endpoints_vh_supported.sum() == 0


def test_duplicate_polygon_month_is_rejected():
    duplicated = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate polygon-month"):
        select_polygon_first_vv(duplicated)


def test_polarization_matching_is_one_to_one():
    events = pd.DataFrame({
        "sample_id": ["a", "a", "a"],
        "polygon_id": ["a", "a", "a"],
        "Band": ["VV_dB", "VV_dB", "VH_dB"],
        "route": ["polygon_first"] * 3,
        "change_time": ["2020-01", "2020-02", "2020-01"],
        "track_support": [1, 1, 1],
    })
    annotated, vh_only, matches = match_polygon_polarizations(events)
    assert annotated.vh_supported.sum() == 1
    assert len(matches) == 1
    assert vh_only.empty


def test_two_month_polarization_tolerance_includes_exact_boundary():
    events = pd.DataFrame({
        "sample_id": ["a", "a"],
        "polygon_id": ["a", "a"],
        "Band": ["VV_dB", "VH_dB"],
        "route": ["polygon_first", "polygon_first"],
        "change_time": ["2020-01", "2020-03"],
        "track_support": [1, 1],
    })
    strict, strict_only, strict_matches = match_polygon_polarizations(
        events, tolerance_months=1,
    )
    relaxed, relaxed_only, relaxed_matches = match_polygon_polarizations(
        events, tolerance_months=2,
    )
    assert not strict.vh_supported.iloc[0]
    assert len(strict_only) == 1
    assert strict_matches.empty
    assert relaxed.vh_supported.iloc[0]
    assert relaxed_only.empty
    assert len(relaxed_matches) == 1
    assert relaxed_matches.absolute_delta_months.iloc[0] == 2


def test_spatial_block_bootstrap_is_deterministic_and_block_aware():
    intervals = pd.DataFrame({
        "sample_id": ["a", "a", "b", "c"],
        "spatial_block_id": ["x", "x", "y", "z"],
        "interval_months": [12, 24, 36, 48],
        "both_endpoints_vh_supported": [True, False, True, False],
    })
    first, first_delta = spatial_block_bootstrap(intervals, replicates=20, seed=7)
    second, second_delta = spatial_block_bootstrap(intervals, replicates=20, seed=7)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_delta, second_delta)
    assert np.isfinite(first.loc[first.cohort.eq("primary_vv"), "median_months"]).all()
