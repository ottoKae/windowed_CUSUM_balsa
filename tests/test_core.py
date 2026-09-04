import numpy as np
import pandas as pd

from lr_cusum.aggregation import aggregate_polygon_timeseries
from lr_cusum.core import generate_windows, lr_cusum
from lr_cusum.detection import DetectionConfig, detect_table, stable_seed
from lr_cusum.rotation import lifecycle_summary, rotation_periods


def test_windows_cover_every_decision_position():
    windows = list(generate_windows(108, 36, 12))
    covered = np.zeros(108, dtype=bool)
    for _, _, start, end in windows:
        covered[start:end] = True
    assert covered.all()


def test_stable_seed_stays_in_portable_rng_range():
    seed = stable_seed("17MNU_ASCENDING_TK18::large-hash-identifier", 20260820)
    assert 0 <= seed < 2**31 - 1
    assert seed == stable_seed("17MNU_ASCENDING_TK18::large-hash-identifier", 20260820)


def test_clear_cut_drop_is_detected():
    rng = np.random.default_rng(4)
    values = np.r_[rng.normal(-10, .12, 48), rng.normal(-15, .12, 60)]
    dates = pd.date_range("2017-01-01", periods=108, freq="MS")
    events = lr_cusum(
        values, dates, bootstrap_iterations=200, confidence_threshold=.90,
        seed=4, start_year=2017, end_year=2025,
    )
    assert events
    assert min(abs((event.change_time - dates[48]).days) for event in events) <= 62


def test_event_dates_are_strictly_inside_requested_decision_years():
    dates = pd.date_range("2016-01-01", periods=126, freq="MS")
    values = np.r_[
        np.full(6, -10.0),
        np.full(108, -14.0),
        np.full(12, -18.0),
    ]
    unrestricted = lr_cusum(
        values,
        dates,
        bootstrap_iterations=50,
        confidence_threshold=.8,
        start_year=None,
        end_year=None,
    )
    assert any(event.change_time.year in {2016, 2025, 2026} for event in unrestricted)
    restricted = lr_cusum(
        values,
        dates,
        bootstrap_iterations=50,
        confidence_threshold=.8,
        start_year=2017,
        end_year=2025,
    )
    assert all(2017 <= event.change_time.year <= 2025 for event in restricted)


def test_upward_change_is_rejected():
    values = np.r_[np.full(48, -15.0), np.full(60, -10.0)]
    dates = pd.date_range("2017-01-01", periods=108, freq="MS")
    assert not lr_cusum(values, dates, bootstrap_iterations=50, confidence_threshold=.8)


def test_grouped_detection_schema():
    dates = pd.date_range("2017-01-01", periods=108, freq="MS")
    frame = pd.DataFrame({
        "ID": np.repeat([1, 2], 108), "time": list(dates) * 2,
        "VV_dB": np.tile(np.r_[np.full(48, -10.0), np.full(60, -15.0)], 2),
        "VH_dB": np.tile(np.r_[np.full(48, -16.0), np.full(60, -21.0)], 2),
    })
    result = detect_table(frame, config=DetectionConfig(bootstrap_iterations=50, confidence_threshold=.8))
    assert {"ID", "Band", "change_time", "bootstrap_confidence"}.issubset(result.columns)


def test_polygon_aggregation():
    frame = pd.DataFrame({
        "time": ["2020-01-01", "2020-01-01", "2020-02-01"],
        "VV_dB": [-10.0, -12.0, -13.0],
    })
    result = aggregate_polygon_timeseries(frame, "VV_dB")
    assert result.VV_dB.tolist() == [-11.0, -13.0]


def test_rotation_and_lifecycle_include_undetected_polygons():
    events = pd.DataFrame({
        "polygon_id": ["a", "a", "b"], "Band": ["VV", "VV", "VV"],
        "change_time": ["2020-01-01", "2022-01-01", "2021-01-01"],
    })
    periods = rotation_periods(events)
    assert periods.rotation_months.iloc[0] == 24
    lifecycle = lifecycle_summary(events, ["a", "b", "c"])
    assert lifecycle.set_index("polygon_id").loc["c", "detected_events"] == 0
