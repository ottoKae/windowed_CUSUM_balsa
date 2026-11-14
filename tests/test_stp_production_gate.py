"""P1a checks the actual production segmenter, not a duplicate segmenter."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tooling/diagnostics/balsa-stars-20261102/run_e1_production.py"
SPEC = importlib.util.spec_from_file_location("synthetic_production", PATH)
SYN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SYN
SPEC.loader.exec_module(SYN)
stars = SYN.stars


def assert_sources(result, n):
    owned = result["node_ownership"]
    assert len(owned) == n and owned.node_id.is_unique
    assert owned[["zone_id", "round_id"]].notna().all().all()
    assert set(owned.round_id) == set(result["rounds"].round_id)


def test_truth_never_changes_prediction():
    case = SYN.make_cases(1, 0)[1]
    wrong = dict(case, truth_rounds=97, labels=[0]*36)
    a, b = SYN.evaluate_case(case), SYN.evaluate_case(wrong)
    for key in ("reported_rounds", "split_accepted", "reason", "p_value"):
        assert a[key] == b[key]
    assert a["correct_count"] != b["correct_count"]


def test_half_month_rounding_and_disabled_parent():
    first = [6]*18+[7]*18
    result = SYN.predict(first, stars.Config(mode="disabled"))
    assert len(result["rounds"]) == 1
    assert result["rounds"].representative_month.iloc[0] == stars.month_label(7)
    assert result["rounds"].source_node_count.iloc[0] == 36
    assert_sources(result, 36)


def test_admissibility_is_production_not_truth():
    result = SYN.predict([6]*8+[18]*8, stars.Config())
    assert result["stands"].reason.iloc[0] == "insufficient_pixels"
    assert not result["stands"].admissible.iloc[0]
    assert len(result["rounds"]) == 1
    assert_sources(result, 16)


def test_support_gate_follows_real_significance_and_keeps_parent_denominator():
    _, c = np.indices((6, 6)).reshape(2, -1)
    first = np.where(c < 5, 6, 18)
    result = SYN.predict(first, stars.Config())
    row = result["stands"].iloc[0]
    assert row.p_value <= .05
    assert row.reason == "round_support_below_parent_gate"
    assert len(result["rounds"]) == 1
    assert result["rounds"].parent_pixels.iloc[0] == 36
    assert_sources(result, 36)


def test_seed_replay_and_source_ownership():
    case = SYN.make_cases(1, 0)[1]
    a = SYN.predict(case["first"], stars.Config())
    b = SYN.predict(case["first"], stars.Config())
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key], check_exact=True)
    assert len(a["rounds"]) == 2
    assert_sources(a, 36)


def test_small_truth_block_does_not_force_an_exit_reason():
    case = SYN.make_cases(1, 0)[4]
    result = SYN.evaluate_case(case)
    assert result["truth_rounds"] == 2
    assert result["reason"] in {"spatial_organization_not_significant", "single_zone_after_merge",
                                "single_round_no_partition", "round_support_below_parent_gate",
                                "spatially_supported_partition"}
    assert result["source_conservation"]


def test_all_eight_scenarios_use_production():
    cases = SYN.make_cases(1, 0)
    assert len(cases) == 8
    for case in cases:
        row = SYN.evaluate_case(case)
        assert row["source_conservation"]
        assert row["split_accepted"] == (row["reported_rounds"] > 1)
