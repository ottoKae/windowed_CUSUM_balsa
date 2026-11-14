"""Validate five frozen local detectors and expose their monthly TDA opportunities."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from common import ROOT, SOURCE, OUT, MONTHS, load_inputs, save_detector, write_json

sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.benchmarks.baselines import fpol_bocd_detect, is_decrease
from windowed_cusum.benchmarks.cusum_nrt_adapter import score_windows
from windowed_cusum.evaluation.validation_closure import maximum_drop_events, classical_lower_cusum_events

CACHE = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_v3"
ARCHIVE = ROOT.parent / "windowed_CUSUM_pixelwise/archives/stand_workflow_20260914/src/lr_cusum"
POLAR = ROOT / "runs/validation/balsa_results_ablation_20261101_v1/polarization"
METHODS = ("maximum_drop_t125", "classical_cusum_k075_h3", "bocd_h24_d6_decrease",
           "nrt_b12_a6_tc98_raw", "nrt_b12_a6_tc98_decrease")
DATES = MONTHS.to_timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, action="append")
    args = parser.parse_args()
    units, _, values = load_inputs()
    methods = args.method or list(METHODS)
    source_functions = {
        "validation_closure.py": ["maximum_drop_events", "classical_lower_cusum_events", "_separated_candidates", "month_number"],
        "baselines.py": ["fpol_bocd_detect", "_student_t_logpdf", "is_decrease"],
        "cusum_nrt_adapter.py": ["scan_nrt", "score_windows", "permutations", "admitted"],
    }
    sources = [Path(__file__), Path(__file__).with_name("common.py"), SOURCE / "master_values.npy",
               SOURCE / "master_units.parquet", SOURCE / "validation_membership.parquet",
               CACHE / "predictions/pixel_events.parquet", CACHE / "predictions/pixel_status.parquet",
               CACHE / "config/configuration_manifest.json"]
    for filename, names in source_functions.items():
        group = "evaluation" if filename == "validation_closure.py" else "benchmarks"
        current, archived = ROOT / "src/windowed_cusum" / group / filename, ARCHIVE / filename
        assert_function_identity(current, archived, names)
        sources.extend([current, archived])
    status = pd.read_parquet(CACHE / "predictions/pixel_status.parquet")
    finite = np.isfinite(values)
    for method in methods:
        old_status = status.loc[status.method.eq(method)].set_index("unit_id").reindex(units.unit_id)
        assert len(old_status) == len(units) and old_status.status.eq("completed").all()
        np.testing.assert_array_equal(old_status.valid_months.to_numpy(int), finite.sum(axis=1))
        events = pd.read_parquet(CACHE / "predictions/pixel_events.parquet", filters=[("method", "==", method)])
        events = events.dropna(axis=1, how="all").copy()
        events["change_time"] = pd.to_datetime(events.change_time)
        metadata = dict(status="completed", coverage_type="retrospective_2017_2025",
                        frozen_source_functions_ast_identical=True, parameters_selected_on_this_run=False)
        method_sources = sources.copy()
        if method == "maximum_drop_t125":
            replay = []
            for unit, x in zip(units.unit_id, values):
                replay.extend(dict(unit_id=unit, **e) for e in maximum_drop_events(x, DATES, threshold_db=1.25))
            compare_replay(events, pd.DataFrame(replay), ["score", "threshold"])
            eligible = maximum_drop_eligibility(values)
            metadata.update(family="maximum_monthly_decline", label="Maximum monthly decline",
                parameters=dict(threshold_db=1.25, minimum_separation=3),
                date_contract="First lower month of an adjacent natural-calendar-month pair; retained candidates separated by >3 months.",
                eligibility_contract="Both current and immediately preceding natural-calendar month finite; independent of drop size and candidate separation.",
                full_detector_replay_exact=True, replay_event_rows=len(replay))
        elif method == "classical_cusum_k075_h3":
            replay = []
            for unit, x in zip(units.unit_id, values):
                replay.extend(dict(unit_id=unit, **e) for e in classical_lower_cusum_events(
                    x, DATES, allowance=.75, decision_interval=3.0))
            compare_replay(events, pd.DataFrame(replay), ["score", "threshold"])
            eligible = finite & (finite.sum(axis=1) >= 12)[:, None]
            metadata.update(family="classical_lower_cusum", label="Classical lower CUSUM",
                parameters=dict(allowance=.75, decision_interval=3.0, reset_gap_months=2, minimum_separation=3),
                date_contract="Native threshold-crossing observation month; no First-post shift. Full-record median/MAD standardization.",
                eligibility_contract="Every finite month in a record with >=12 finite observations; the actual detector skips missing values and resets after gaps >2 months.",
                full_detector_replay_exact=True, replay_event_rows=len(replay))
        elif method == "bocd_h24_d6_decrease":
            polar_units = pd.read_parquet(POLAR / "units.parquet").set_index("unit_id")
            vh = np.load(POLAR / "vh_values.npy", mmap_mode="r")[polar_units.loc[units.unit_id, "local_series_index"].to_numpy()]
            np.testing.assert_array_equal(finite, np.isfinite(vh))
            tasks = [(values[i:i + 128], vh[i:i + 128]) for i in range(0, len(units), 128)]
            replay = []
            with ProcessPoolExecutor(max_workers=4) as pool:
                for batch, results in enumerate(pool.map(replay_bocd, tasks)):
                    for offset, points in enumerate(results):
                        unit = units.unit_id.iloc[batch * 128 + offset]
                        replay.extend(dict(unit_id=unit, change_time=DATES[k], raw_index=k, decrease=True) for k in points)
                    if (batch + 1) % 40 == 0:
                        print(f"BOCD actual-function replay: {min((batch + 1) * 128, len(units))}/{len(units)}", flush=True)
            compare_replay(events, pd.DataFrame(replay), ["raw_index", "decrease"])
            eligible = bocd_eligibility(values, vh)
            method_sources.extend([POLAR / "units.parquet", POLAR / "vh_values.npy"])
            metadata.update(family="bocd_prototype", label="Notebook f-pol BOCD adaptation",
                parameters=dict(hazard=1 / 24, delta_m=6, min_obs=12, decrease_radius_calendar_months=6),
                date_contract="Retrospective MAP run-length boundary valid_index[t-map_run], followed by native VV median-decrease screening; not alert availability.",
                eligibility_contract="Retrospective search-domain support: jointly finite months in a record with >=24 joint observations, and at least one finite VV value in each pre/post six-calendar-month side. No accepted MAP path or decrease sign defines this mask; it is not online availability.",
                full_detector_replay_exact=True, replay_event_rows=len(replay))
        else:
            decrease = method.endswith("decrease")
            eligible, validation = validate_nrt_scans(values, units.unit_id, events, decrease)
            metadata.update(family="cusum_nrt", label="CuSum NRT, decrease-filtered" if decrease else "CuSum NRT, raw",
                parameters=dict(before=12, after=6, window_unit="observations", samples=500,
                                confidence=.98, seed=20260905, decrease=decrease),
                date_contract="First CUSUM maximum within each 19-valid-observation window, mapped to the original month. Accepted window aliases are preserved; no First-post shift.",
                eligibility_contract="Union of locations in every scanned 19-valid-observation context, independently of Q, confidence and direction. Decrease-filtered support excludes the first observation of each window, where a before mean cannot be computed. Calendar gaps remain unfilled.",
                permutation_replayed=False, cache_validation=validation)
        eligible[:, :12] = False
        eligible[:, 120:] = False
        positions = units.set_index("unit_id").index.get_indexer(events.unit_id)
        month_index = events.change_time.dt.to_period("M").astype("int64").to_numpy() - MONTHS[0].ordinal
        assert eligible[positions, month_index].all()
        metadata["all_accepted_dates_supported_by_independent_eligibility"] = True
        save_detector(method, events, eligible, metadata, method_sources)


def maximum_drop_eligibility(values):
    finite = np.isfinite(values)
    eligible = np.zeros_like(finite)
    eligible[:, 1:] = finite[:, :-1] & finite[:, 1:]
    return eligible


def bocd_eligibility(vv, vh):
    finite = np.isfinite(vv)
    joint = finite & np.isfinite(vh)
    eligible = joint & (joint.sum(axis=1) >= 24)[:, None]
    for k in range(vv.shape[1]):
        eligible[:, k] &= finite[:, max(0, k - 6):k].any(axis=1)
        eligible[:, k] &= finite[:, k:min(vv.shape[1], k + 6)].any(axis=1)
    return eligible


def replay_bocd(task):
    vv, vh = task
    return [[k for k in fpol_bocd_detect(x, y, hazard=1 / 24, delta_m=6, min_obs=12)
             if is_decrease(x, k, radius=6) and 12 <= k < 120] for x, y in zip(vv, vh)]


def validate_nrt_scans(values, unit_ids, events, decrease):
    groups = events.groupby("unit_id", sort=False).indices
    fields = ["target_month", "context_start", "context_end", "available_through_month"]
    month_fields = {name: pd.to_datetime(events[name]).dt.to_period("M").astype("int64").to_numpy() - MONTHS[0].ordinal for name in fields}
    response = events.change_time.dt.to_period("M").astype("int64").to_numpy() - MONTHS[0].ordinal
    eligible = np.zeros_like(values, dtype=bool)
    scans = np.zeros(len(values), dtype=np.int32)
    max_q_difference = 0.0
    no_permutations = np.empty((0, 19), dtype=np.int64)
    assert not events.duplicated(["unit_id", "target_month"]).any()
    for position, (unit, x) in enumerate(zip(unit_ids, values)):
        finite = np.flatnonzero(np.isfinite(x))
        if len(finite) < 19:
            assert unit not in groups
            continue
        windows = np.lib.stride_tricks.sliding_window_view(x[finite], 19).copy()
        observed = score_windows(windows, no_permutations)
        scans[position] = len(windows)
        eligible[position, finite[1 if decrease else 0:]] = True
        selected = groups.get(unit)
        if selected is None:
            continue
        indices = np.searchsorted(finite[12:-6], month_fields["target_month"][selected])
        np.testing.assert_array_equal(finite[indices + 12], month_fields["target_month"][selected])
        np.testing.assert_array_equal(finite[indices], month_fields["context_start"][selected])
        np.testing.assert_array_equal(finite[indices + 18], month_fields["context_end"][selected])
        np.testing.assert_array_equal(finite[indices + 18], month_fields["available_through_month"][selected])
        peaks = observed[indices, 1].astype(int)
        np.testing.assert_array_equal(peaks, events.raw_boundary_index.to_numpy()[selected])
        np.testing.assert_array_equal(finite[indices + peaks], response[selected])
        expected_q = events.q.to_numpy()[selected]
        np.testing.assert_array_equal(observed[indices, 0], expected_q)
        max_q_difference = max(max_q_difference, float(np.max(np.abs(observed[indices, 0] - expected_q))))
        pre = np.array([np.mean(windows[i, :k]) if k else np.nan for i, k in zip(indices, peaks)])
        post = np.array([np.mean(windows[i, k:]) for i, k in zip(indices, peaks)])
        np.testing.assert_array_equal(pre, events.pre_mean.to_numpy()[selected])
        np.testing.assert_array_equal(post, events.post_mean.to_numpy()[selected])
        if decrease:
            assert ((peaks > 0) & (post < pre)).all()
    assert events.native_confidence.ge(.98).all() and events.n_obs.eq(19).all()
    assert events.before.eq(12).all() and events.after.eq(6).all()
    assert events.window_unit.eq("observations").all() and events.permutations.eq(500).all()
    assert events.seed.eq(20260905).all()
    target = OUT / "local_validation"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(unit_id=unit_ids, n_scanned_windows=scans,
                     n_supported_months=eligible.sum(axis=1))).to_parquet(target / f"{events.method.iloc[0]}_scan_counts.parquet", index=False)
    return eligible, dict(all_scan_windows=int(scans.sum()), cached_accepted_windows=len(events),
        observed_Q_first_peak_context_and_means_exact=True, observed_Q_max_abs_difference=max_q_difference,
        eligibility_uses_only_finite_observation_axis=True,
        eligibility_union_months_before_reporting_clip=int(eligible.sum()))


def compare_replay(cached, actual, fields):
    columns = ["unit_id", "change_time", *fields]
    left = cached[columns].sort_values(["unit_id", "change_time"]).reset_index(drop=True)
    right = actual[columns].sort_values(["unit_id", "change_time"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)


def assert_function_identity(current, archived, names):
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef)}
    left, right = functions(current), functions(archived)
    assert all(left[name] == right[name] for name in names), f"Frozen detector function changed: {current}"


if __name__ == "__main__":
    main()
