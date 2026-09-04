import numpy as np
import pandas as pd

from lr_cusum.polygon_workflow import monthly_calendar
from lr_cusum.validation_closure import (
    balanced_spatial_folds,
    classical_lower_cusum_events,
    match_events_detailed,
    maximum_drop_events,
)


def test_explicit_context_calendar_has_missing_month_slots():
    calendar = monthly_calendar("2016-01", "2026-12")
    assert len(calendar) == 132
    assert str(calendar[0]) == "2016-01"
    assert str(calendar[-1]) == "2026-12"


def test_maximum_drop_detects_known_drop():
    dates = pd.date_range("2017-01-01", periods=24, freq="MS")
    values = np.zeros(24); values[12:] = -3
    events = maximum_drop_events(values, dates, threshold_db=2)
    assert [event["change_time"] for event in events] == [pd.Timestamp("2018-01-01")]


def test_classical_lower_cusum_detects_sustained_low_state():
    dates = pd.date_range("2017-01-01", periods=36, freq="MS")
    values = np.r_[np.zeros(18), np.full(18, -3.0)]
    events = classical_lower_cusum_events(
        values, dates, allowance=.25, decision_interval=3
    )
    assert events
    assert events[0]["change_time"] >= pd.Timestamp("2018-07-01")


def test_detailed_matching_is_one_to_one_and_signed():
    refs = pd.DataFrame({"sample_id": ["a", "a"], "event_month": ["2020-01", "2020-06"]})
    det = pd.DataFrame({"sample_id": ["a", "a"], "change_time": ["2020-02", "2020-09"]})
    metrics, pairs = match_events_detailed(refs, det, tolerance_months=2)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert pairs.signed_error_months.tolist() == [1]


def test_balanced_folds_keep_blocks_intact():
    refs = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e", "f"],
        "spatial_block_id": ["x", "x", "y", "z", "w", "v"],
        "event_month": ["2020-01"] * 6,
    })
    folds = balanced_spatial_folds(refs, folds=3)
    assert folds.spatial_block_id.nunique() == 5
    assert folds.groupby("spatial_block_id").outer_fold.nunique().max() == 1

