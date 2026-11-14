"""Prove cached-prefix RNG replay equals an uninterrupted frozen A1 call."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "harvest_inventory_stp", ROOT / "tooling/diagnostics/harvest-inventory-20261104/stp.py")
STP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STP)


def synthetic_parents():
    reports, members, units = [], [], []
    for parent, size in [("a", 36), ("b", 18), ("c", 36)]:
        reports.append(dict(component_id=parent, prediction_id=parent, sample_id=parent, total_pixels=size))
        for i in range(size):
            unit = f"{parent}:u{i:02}"
            row, col = divmod(i, 6)
            first = 36 + 6 * (col >= 3) if parent == "a" else 70 + 4 * (row >= 3)
            units.append(dict(unit_id=unit, native_row=row, native_col=col, track_key="track"))
            for month in [first, first + 1]:
                members.append(dict(component_id=parent, sample_id=parent, unit_id=unit,
                                    node_id=f"{unit}:{month}", month_index=month))
    return pd.DataFrame(reports), pd.DataFrame(members), pd.DataFrame(units)


def test_cached_prefix_advances_rng_exactly_like_original_permutation_tests():
    reports, members, units = synthetic_parents()
    lookup = units.set_index("unit_id", drop=False)
    full_rng = np.random.default_rng(STP.CONFIG.seed)
    full_order = []
    full = STP.segment_sequence(reports, members, lookup, full_rng, {}, full_order, "uncached")
    prefix_tests = full["tests"].loc[full["tests"].component_id.isin(["a", "b"])]
    cached_rng = np.random.default_rng(STP.CONFIG.seed)
    cached_order = []
    replayed = STP.segment_sequence(reports, members, lookup, cached_rng,
        STP.calibration_cache(prefix_tests), cached_order, "cached_prefix")
    assert STP.state_digest(full_rng) == STP.state_digest(cached_rng)
    assert [r["rng_state_after"] for r in full_order] == [r["rng_state_after"] for r in cached_order]
    assert [r["permutation_draws"] for r in cached_order] == [999, 0, 999]
    for name, key in STP.TABLE_KEYS.items():
        STP.compare_table(replayed[name], full[name], key, list(full[name].columns))
    original = STP.stars.segment(reports, members, units, STP.CONFIG)
    for name, key in STP.TABLE_KEYS.items():
        STP.compare_table(replayed[name], original[name], key, list(original[name].columns))


def test_rng_permutation_of_length_and_observed_dates_have_identical_state():
    dates = np.array([12, 12, 14, 17, 17, 17, 20])
    left = np.random.default_rng(STP.CONFIG.seed)
    right = np.random.default_rng(STP.CONFIG.seed)
    for _ in range(STP.CONFIG.permutations):
        left.permutation(dates)
        right.permutation(len(dates))
    assert STP.state_digest(left) == STP.state_digest(right)


def test_source_validation_rejects_reused_nodes():
    reports, members, units = synthetic_parents()
    STP.validate_inputs(reports, members, units, set(reports.sample_id))
    duplicated = pd.concat([members, members.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exclusive"):
        STP.validate_inputs(reports, duplicated, units, set(reports.sample_id))


def test_frozen_cohort_actual_artifact_preserves_original_full_chain():
    out = STP.RUN / "cstr/cohort_parity"
    if not out.is_dir():
        pytest.skip("Author-local frozen cohort is not included in the public repository")
    result = {name: pd.read_parquet(out / f"{name}.parquet") for name in STP.TABLE_KEYS}
    result["stands"] = result["stands"].drop(columns="q_value_frozen_cohort")
    checks = STP.compare_frozen_chain(result)
    assert checks.passed.all() and len(checks) == 7
    assert len(result["stands"]) == 1216 and len(result["rounds"]) == 1235
    assert len(result["node_ownership"]) == 38345


def test_legacy_entry_uses_its_config_when_dynamically_loaded(monkeypatch):
    legacy_spec = importlib.util.spec_from_file_location(
        "legacy_inventory_stp", ROOT / "tooling/diagnostics/harvest-inventory-20261104/cstr.py")
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    monkeypatch.setattr(legacy, "CONFIG", replace(legacy.CONFIG, permutations=99))
    reports, members, units = synthetic_parents()
    order = []
    legacy.segment_sequence(reports, members, units.set_index("unit_id", drop=False),
        np.random.default_rng(legacy.CONFIG.seed), {}, order, "compatibility")
    assert [row["permutation_draws"] for row in order] == [99, 0, 99]
    assert legacy.run_cstr is legacy.run_stp
    assert STP.CONFIG.permutations == 999
