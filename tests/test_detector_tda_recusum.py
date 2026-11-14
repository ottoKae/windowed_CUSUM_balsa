"""Exercise the trace against the existing recursive implementations."""
import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.benchmarks.baselines import cusum_single, is_decrease, recusum_detect

spec = importlib.util.spec_from_file_location("benchmark_recusum_trace",
    ROOT / "tooling/diagnostics/detector-tda-benchmark-20261109/recusum.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
source = ROOT / "tooling/diagnostics/balsa-research-v3-20261031/controls/recusum_adapter.py"
function = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef) and node.name == "firstpost_recusum")
namespace = {"np": np, "cusum_single": cusum_single}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)


@pytest.mark.parametrize("policy,minimum", [("legacy_peak", 12), ("legacy_peak", 18), ("firstpost", 12)])
@pytest.mark.parametrize("missing", [False, True])
def test_trace_preserves_frozen_recursive_dates(policy, minimum, missing):
    values = np.repeat([-8., -12., -8., -12.], [30, 30, 30, 36])
    if missing:
        values[[29, 30, 59, 60, 61, 89, 90]] = np.nan
    options = dict(confidence=.9, bootstrap=800, seed=20, minimum_segment=minimum, maximum_depth=10)
    expected = (namespace["firstpost_recusum"] if policy == "firstpost" else recusum_detect)(values, **options)
    actual = adapter.trace_recusum(values, date_policy=policy, **options)
    assert actual["points"] == expected
    decreased = adapter.trace_recusum(values, date_policy=policy, decrease=True, **options)
    assert decreased["points"] == [point for point in expected if is_decrease(values, point)]
    assert all(decreased["opportunity"][point] for point in decreased["points"])
    assert not actual["opportunity"][~np.isfinite(values)].any()


def test_rejected_context_remains_observed_and_depth_is_not_manufactured():
    values = np.random.default_rng(3).normal(-10, .1, 126)
    rejected = adapter.trace_recusum(values, seed=20, confidence=1)
    assert rejected["segments"][0]["stop_reason"] == "confidence_rejected"
    assert rejected["points"] == [] and rejected["opportunity"].sum() == 125
    ordinary = adapter.trace_recusum(values, seed=20, confidence=.9)
    np.testing.assert_array_equal(rejected["opportunity"], ordinary["opportunity"])
    blocked = adapter.trace_recusum(values, seed=20, maximum_depth=0)
    assert not blocked["opportunity"].any()
    assert blocked["segments"][0]["stop_reason"] == "maximum_depth"


def test_tested_ancestor_support_survives_short_child():
    values = np.r_[np.full(12, -8.), np.full(114, -12.)]
    actual = adapter.trace_recusum(values, seed=20)
    assert actual["points"] == [12]
    assert any(row["stop_reason"] == "insufficient_segment_observations" for row in actual["segments"])
    assert actual["opportunity"][1:12].all()


def test_constant_context_and_direction_observation_contract():
    values = np.full(126, -10.)
    actual = adapter.trace_recusum(values, seed=20, date_policy="legacy_peak", decrease=True)
    assert actual["points"] == []
    assert actual["segments"][0]["stop_reason"] == "evaluated_zero_amplitude"
    assert not actual["segments"][0]["permutation_evaluated"]
    assert actual["opportunity"][1:].all() and not actual["opportunity"][0]
    assert actual["unknown_reason"][0] == "direction_context_unobserved"
    sparse = np.full(126, np.nan)
    sparse[::6] = -10
    assert not adapter.trace_recusum(sparse, seed=20)["opportunity"].any()
