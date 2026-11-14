"""The nationwide adapter calls the real frozen R2 preparation and assembler."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "tooling/diagnostics/harvest-inventory-20261104/tda.py"
spec = importlib.util.spec_from_file_location("test_inventory_tda", PATH)
tda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tda)


def inputs():
    units = pd.DataFrame(dict(unit_id=[f"u{i}" for i in range(5)]))
    members = units.assign(sample_id="s", source_row_original=1)
    values = np.tile(np.linspace(-4, -12, 126), (5, 1))
    execution = pd.DataFrame(dict(eligible_year_mask=[511] * 5))
    nodes = pd.DataFrame([dict(sample_id="s", unit_id="u0", node_id=f"n{month}", month_index=month,
        response_month=str(pd.Period("2016-01", "M") + month), window_start="2016-01", window_end="2026-06",
        candidate_id=f"c{month}", phase_id="") for month in (20, 22, 50, 51)])
    return [units, members, values, execution, nodes]


def test_actual_g1_assembler_preserves_repeat_nodes_and_parent_denominator(tmp_path):
    result = tda.aggregate_frames(*inputs(), tmp_path)
    assert len(result["reports"]) == 2
    assert result["reports"].source_event_count.tolist() == [2, 2]
    assert result["reports"].response_pixels.tolist() == [1, 1]
    assert result["reports"].total_pixels.eq(5).all()
    assert result["source_members"].node_id.tolist() == ["n20", "n22", "n50", "n51"]
    assert result["phase_relations"].relation.tolist() == ["temporal_join", "temporal_separate"]
    assert result["reports"].representative_month.tolist() == ["2017-10", "2020-04"]


def test_unknown_month_does_not_become_quiet_bridge(tmp_path):
    args = inputs()
    args[2][:, 21] = np.nan
    result = tda.aggregate_frames(*args, tmp_path)
    assert result["phase_relations"].relation.iloc[0] == "boundary_unknown"
    assert len(result["reports"]) == 1
    assert len(result["source_members"]) == 4
    assert result["source_members"].destination_type.tolist() == ["unresolved", "unresolved", "episode", "episode"]


def test_inventory_unavailable_count_is_na_not_zero(tmp_path):
    args = inputs()
    result = tda.aggregate_frames(*args, tmp_path)
    summary = pd.DataFrame(dict(sample_id=["s", "unavailable"], source_row_original=[1, 2], DPA_DESPRO=["a", "b"],
        inventory_area_ha=[1, 0.01], native_cells=[5, 0], status=["completed_with_reports", "unsupported_spatial_no_native_centres"]))
    actual = tda.stand_universe(summary, args[1], args[4], result).set_index("sample_id")
    assert actual.loc["s", "report_count"] == 2
    assert pd.isna(actual.loc["unavailable", "report_count"])
    assert actual.loc["unavailable", "processing_status"] == "insufficient_coverage"


def test_legacy_entry_preserves_actual_temporal_aggregation(tmp_path):
    legacy_spec = importlib.util.spec_from_file_location("legacy_inventory_tda", PATH.with_name("octa.py"))
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    expected = tda.aggregate_frames(*inputs(), tmp_path / "canonical")
    actual = legacy.aggregate_frames(*inputs(), tmp_path / "legacy")
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
