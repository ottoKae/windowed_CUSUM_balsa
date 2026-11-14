"""Local benchmark opportunity masks follow native observations, not detections."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tooling/diagnostics/detector-tda-benchmark-20261109"))
import local as adapter
from windowed_cusum.benchmarks.cusum_nrt_adapter import scan_nrt, admitted


def test_maximum_drop_missing_month_is_not_bridged_and_quiet_is_observed():
    values = np.array([[0., 0., np.nan, -3., -5., -5.], [-5.] * 6])
    eligible = adapter.maximum_drop_eligibility(values)
    np.testing.assert_array_equal(eligible[0], [False, True, False, False, True, True])
    np.testing.assert_array_equal(eligible[1], [False, True, True, True, True, True])
    dates = pd.date_range("2017-01", periods=6, freq="MS")
    events = adapter.maximum_drop_events(values[0], dates, threshold_db=1.25)
    assert [row["change_time"] for row in events] == [dates[4]]
    assert adapter.maximum_drop_events(values[1], dates, threshold_db=1.25) == []


def test_bocd_joint_support_and_calendar_direction_support_ignore_sign():
    vv = np.full((3, 36), -6.)
    vh = np.full((3, 36), -12.)
    vh[1, :13] = np.nan
    vv[2, 14:20] = np.nan
    eligible = adapter.bocd_eligibility(vv, vh)
    assert eligible[0, 1:].all() and not eligible[0, 0]
    assert not eligible[1].any()
    assert not eligible[2, 20] and eligible[2, 21]
    np.testing.assert_array_equal(eligible, adapter.bocd_eligibility(-vv, -vh))


def test_nrt_scan_union_preserves_missing_calendar_and_independence(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "OUT", tmp_path)
    values = np.full((1, 42), -6.)
    values[:, 20:] = -9.
    values[:, [3, 8, 14, 27]] = np.nan
    dates = adapter.DATES[:values.shape[1]]
    raw = scan_nrt(values, dates, before=12, after=6, samples=500, seed=20260905)
    assert (raw.context_end - raw.context_start).dt.days.max() > 19 * 31
    raw_events = admitted(raw, confidence=.98).assign(unit_id="test_unit", method="nrt_test_raw")
    decrease_events = admitted(raw, confidence=.98, decrease=True).assign(unit_id="test_unit", method="nrt_test_decrease")
    assert len(raw_events) and len(decrease_events)
    raw_mask, audit = adapter.validate_nrt_scans(values, ["test_unit"], raw_events, False)
    decrease_mask, _ = adapter.validate_nrt_scans(values, ["test_unit"], decrease_events, True)
    np.testing.assert_array_equal(raw_mask, np.isfinite(values))
    expected_decrease = raw_mask.copy()
    expected_decrease[:, 0] = False
    np.testing.assert_array_equal(decrease_mask, expected_decrease)
    assert audit["all_scan_windows"] == 20
    assert audit["observed_Q_first_peak_context_and_means_exact"]

    # Removing accepted nominations must not turn observed months into unknowns.
    reduced, _ = adapter.validate_nrt_scans(values, ["test_unit"], raw_events.iloc[:1], False)
    np.testing.assert_array_equal(raw_mask, reduced)
