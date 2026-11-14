"""Run the real current detector and aggregators without private records."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_current_pipeline.py"
SPEC = importlib.util.spec_from_file_location("current_pipeline_demo", PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def test_two_rotations_preserve_pixel_sources_and_parent_counts(tmp_path):
    out = tmp_path / "demo"
    result = pipeline.run(None, out)
    assert result["pixel_detections"] == 72
    assert result["tda_events"] == 2
    assert result["stp_groups"] == 4
    events = pd.read_csv(out / "pixel_detections.csv")
    assert set(events.peak_to_firstpost_gap_months) == {1}
    owned = pd.read_parquet(out / "stp/node_ownership.parquet")
    assert len(owned) == len(events) and owned.node_id.is_unique
    monthly = pd.read_csv(out / "stand_monthly_observation.csv")
    frozen = pd.read_parquet(out / "tda/monthly_activity.parquet")
    pd.testing.assert_frame_equal(monthly, frozen[monthly.columns], check_dtype=False)


def test_native_identity_and_parallel_replay(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    pipeline.make_demo(inputs)
    units = pd.read_csv(inputs / "units.csv")
    units["pixel_track_id"] = "native_" + units.unit_id
    units.to_csv(inputs / "units.csv", index=False)
    pipeline.run(inputs, tmp_path / "serial")
    subprocess.run([sys.executable, str(PATH), "--input-dir", str(inputs),
                    "--output", str(tmp_path / "parallel"), "--workers", "2"], check=True,
                   capture_output=True, text=True)
    for name in ("pixel_detections.csv", "tda_events.csv", "stp_groups.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "serial" / name),
                                      pd.read_csv(tmp_path / "parallel" / name), check_exact=True)
    events = pd.read_csv(tmp_path / "serial/pixel_detections.csv")
    assert events.pixel_track_id.eq("native_" + events.unit_id).all()
    owned = pd.read_parquet(tmp_path / "serial/stp/node_ownership.parquet")
    expected = {pipeline._identifier("node_", row.sample_id, row.unit_id,
                                    str(pd.Period("2016-01", freq="M") + row.month_index))
                for row in owned.itertuples()}
    assert set(owned.node_id) == expected


@pytest.mark.parametrize("case,status", [("stable", "observed"), ("missing", "unassessable"),
                                       ("partial", "partially_observed"), ("below_support", "observed")])
def test_empty_results_keep_schemas_and_observation_status(tmp_path, case, status):
    inputs = tmp_path / "input"
    inputs.mkdir()
    pipeline.make_demo(inputs)
    values = np.full((36, 126), -6.0)
    if case == "missing":
        values[:] = np.nan
    elif case == "partial":
        values[:, 60:] = np.nan
    elif case == "below_support":
        values[0] = np.load(inputs / "monthly_vv.npy")[0]
    np.save(inputs / "monthly_vv.npy", values)
    result = pipeline.run(inputs, tmp_path / "stable")
    assert result["tda_events"] == result["stp_groups"] == 0
    assert result["pixel_detections"] == (2 if case == "below_support" else 0)
    for filename, required in [("pixel_detections.csv", "change_time"),
                               ("tda_events.csv", "prediction_id"), ("stp_groups.csv", "round_id")]:
        assert required in pd.read_csv(tmp_path / "stable" / filename).columns
    stands = pd.read_csv(tmp_path / "stable/stand_summary.csv")
    assert stands.observation_status.tolist() == [status]
    assert pd.isna(stands.report_count.iloc[0]) if case == "missing" else stands.report_count.iloc[0] == 0
