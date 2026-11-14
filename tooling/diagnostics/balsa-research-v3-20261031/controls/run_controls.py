"""Execute E3/E4 controls on the frozen First-post v2 validation domain."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/release")]
from build_sample_release import original_tables
from windowed_cusum.evaluation.catalogue_evaluation import catalogue_metrics, bootstrap_ratios
from windowed_cusum.stand.harvest_quarter_aggregation import build_quarter_aggregations, DETECTOR_METHOD
from windowed_cusum.stand.harvest_bounded_event_aggregation import build_bounded_events
from windowed_cusum.evaluation.validation_closure import maximum_drop_events
from windowed_cusum.detector.detection import stable_seed
from windowed_cusum.benchmarks.baselines import is_decrease

SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
ANALYSIS = ROOT / "runs/validation/balsa_firstpost_result_analysis_20261030_v1"
OLD = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_v3"
OUT = ROOT / "runs/validation/balsa_research_execution_20261031_v3/controls"
DATES = pd.date_range("2016-01-01", "2026-06-01", freq="MS")
METHOD = "recusum_notebook_firstpost_c09_m12_adapted"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inputs():
    signature = json.loads((SOURCE / "method_signature.json").read_text())
    for path, expected in signature["source_sha256"].items():
        assert sha(path) == expected, path
    members = pd.read_parquet(SOURCE / "validation_membership.parquet")
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    units = units.loc[units.unit_id.isin(members.unit_id)].copy()
    assert len(units) == 26890 and members.sample_id.nunique() == 468
    return units, members, signature


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    units, members, signature = inputs()
    historical = json.loads((OLD / "config/configuration_manifest.json").read_text())
    assert historical["configurations"]["maximum_drop_t125"] == {"family": "maximum_drop", "threshold_db": 1.25}
    expected_recusum = dict(family="recusum", confidence=.9, minimum_segment=12,
        maximum_depth=10, decrease=True, bootstrap=800)
    assert historical["configurations"]["recusum_decrease_c09_m12"] == expected_recusum
    files = [SOURCE / "method_signature.json", SOURCE / "master_units.parquet",
        SOURCE / "master_values.npy", SOURCE / "validation_membership.parquet",
        SOURCE / "validation/pixel_events.parquet", SOURCE / "validation/pixel_execution.parquet",
        SOURCE / "validation/complete.json", ANALYSIS / "pixel_window_trace.parquet",
        ANALYSIS / "firstpost_trace_receipt.json", OLD / "config/configuration_manifest.json",
        ROOT / "scripts/run_pixel_b20_benchmark_window.py", ROOT / "src/windowed_cusum/benchmarks/baselines.py",
        ROOT / "src/windowed_cusum/evaluation/validation_closure.py", ROOT / "src/windowed_cusum/evaluation/catalogue_evaluation.py",
        ROOT / "src/windowed_cusum/evaluation/event_scoring.py", ROOT.parent / "LR-CuSum_v100/notebooks/LR-CuSum.ipynb",
        Path(__file__), Path(__file__).with_name("recusum_adapter.py")]
    config = dict(method_signature=signature["signature"], input_pixels=len(units), input_stands=468,
        report_months=["2017-01", "2025-12"], calendar=["2016-01", "2026-06"], cpu_workers=4,
        q20=dict(input_accepted_nodes=44311, fixed_members=True, support_fraction=.2,
            representative_month="round_half_up_median_source_node_month", calendar_quarter=True,
            execution_receipts="frozen pixel_execution plus actual firstpost annual trace"),
        maximum_drop=dict(threshold_db=1.25, minimum_separation=3,
            separation_semantics="retained candidates differ by strictly more than 3 calendar months",
            source="historical v3 configuration plus explicit frozen source default",
            date="first_lower_month_of_adjacent_calendar_month_pair"),
        recusum={**expected_recusum, "radius": 6, "seed_base": 20260820,
            "source": "local LR-CuSum.ipynb cell 122; historical v3 operating point",
            "status": "Notebook-derived first-post recursive adaptation; not official implementation",
            "date": "next finite observation after first absolute CUSUM maximum",
            "split": "left includes maximum; right begins at first-post observation",
            "recursive_seeds": "seed+firstpost_split and seed+firstpost_split+1",
            "minimum_segment_semantics": "parent requires >=24 finite observations; edge splits remain permitted"},
        evaluation=dict(primary_tolerance_months=2, sensitivity_tolerances=[1, 3],
            unmatched_reports_are_FP=False, precision=None, F1=None, transferred_old_weights=False),
        source_sha256={str(p): sha(p) for p in files})
    path = OUT / "configuration_frozen.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == config, "Configuration changed"
    else:
        save(path, config)
    return units, members


def q20():
    units, members = freeze()
    dest = OUT / "q20"; dest.mkdir(exist_ok=True)
    receipt = json.loads((SOURCE / "validation/complete.json").read_text())
    assert receipt["status"] == "completed" and receipt["full_chain_passed"]
    for name in ("pixel_events.parquet", "pixel_execution.parquet"):
        assert sha(SOURCE / "validation" / name) == receipt["output_sha256"][name]
    events = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    execution = pd.read_parquet(SOURCE / "validation/pixel_execution.parquet")
    trace = pd.read_parquet(ANALYSIS / "pixel_window_trace.parquet")
    assert len(events) == 44311 and len(trace) == 9*len(units)
    assert set(execution.unit_id) == set(units.unit_id) and not trace.duplicated(["unit_id", "decision_year"]).any()
    actual = trace[trace.accepted][["unit_id", "candidate_month"]].sort_values(["unit_id", "candidate_month"]).reset_index(drop=True)
    expected = events.assign(candidate_month=events.change_time.dt.strftime("%Y-%m"))[
        ["unit_id", "candidate_month"]].sort_values(["unit_id", "candidate_month"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    for row in execution.itertuples():
        assert row.eligible_year_mask == 511
    assert trace.n_valid.ge(12).all()
    events["source_detector_method"] = events.method
    events["method"] = DETECTOR_METHOD
    status = execution[["unit_id", "pixel_track_id"]].copy()
    status["status"] = "completed"; status["method"] = DETECTOR_METHOD
    status["event_count"] = status.unit_id.map(events.groupby("unit_id").size()).fillna(0).astype(int)
    windows = trace[["unit_id", "pixel_track_id", "decision_year", "n_valid", "accepted"]].rename(
        columns={"n_valid": "valid_observations", "accepted": "accepted_event"})
    windows["passes_minimum_observations"] = windows.valid_observations.ge(12)
    windows["execution_status"] = np.where(windows.accepted_event, "completed_event", "completed_no_event")
    windows["method"] = DETECTOR_METHOD
    empty_m0 = pd.DataFrame(columns=["sample_id", "prediction_id", "predicted_month"])
    sample = members.sample_id.sort_values().iloc[0]
    small_m = members[members.sample_id.eq(sample)]
    small_u = units[units.unit_id.isin(small_m.unit_id)]
    small = build_quarter_aggregations(small_u, small_m, events[events.unit_id.isin(small_u.unit_id)],
        status[status.unit_id.isin(small_u.unit_id)], windows[windows.unit_id.isin(small_u.unit_id)], [sample], empty_m0)
    support, reports, diagnostics = build_quarter_aggregations(units, members, events, status, windows,
        members.sample_id.unique(), empty_m0)
    reports = reports[reports.method.eq("Q20")].copy()
    pd.testing.assert_frame_equal(reports[reports.sample_id.eq(sample)].reset_index(drop=True),
        small[1][small[1].method.eq("Q20")].reset_index(drop=True))
    assert diagnostics["technical_gap_stand_quarters"] == 0
    assert diagnostics["unique_pixel_month_events"] == 44311
    assert (5*reports.response_pixels >= reports.total_pixels).all()
    assert not reports.duplicated(["sample_id", "predicted_quarter"]).any()
    reports["method"] = "Q20_firstpost_v2"
    reports["interval_time_basis"] = "actual_pixel_response_months"
    support.to_parquet(dest / "quarter_support.parquet", index=False)
    reports.to_parquet(dest / "stand_reports.parquet", index=False)
    status.to_parquet(dest / "pixel_status_adapter.parquet", index=False)
    windows.to_parquet(dest / "annual_window_status_adapter.parquet", index=False)
    save(dest / "aggregation_receipt.json", {**diagnostics,
        "source_method": "windowed_w36_d12_s12_firstpost_v2", "adapter_method": DETECTOR_METHOD,
        "identity_alias_only_no_detector_rerun": True, "small_stand_parity": sample})
    print("Q20 completed", len(reports), flush=True)


def maximum_drop():
    units, members = freeze()
    dest = OUT / "maximum_drop"; dest.mkdir(exist_ok=True)
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")
    old_events = pd.read_parquet(OLD / "predictions/pixel_events.parquet",
        filters=[("method", "==", "maximum_drop_t125")])
    old_status = pd.read_parquet(OLD / "predictions/pixel_status.parquet",
        filters=[("method", "==", "maximum_drop_t125")])
    assert len(old_status) == len(units) and old_status.status.eq("completed").all()
    for col in ("pixel_track_id", "track_key", "cache_index"):
        assert old_status.set_index("unit_id").loc[units.unit_id, col].tolist() == units[col].tolist()
    records = []; started = time.monotonic()
    for row in units.itertuples():
        detected = maximum_drop_events(values[row.series_index], DATES, threshold_db=1.25,
            minimum_separation=3, start_year=2017, end_year=2025)
        records.extend(dict(unit_id=row.unit_id, change_time=event["change_time"], score=event["score"])
                       for event in detected)
    recomputed = pd.DataFrame(records).sort_values(["unit_id", "change_time"]).reset_index(drop=True)
    prior = old_events[["unit_id", "change_time", "score"]].copy()
    prior["change_time"] = pd.to_datetime(prior.change_time)
    prior = prior.sort_values(["unit_id", "change_time"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(recomputed, prior, check_dtype=False, check_exact=True)
    x = np.array([-8., -8., np.nan, -12., -8., -10., -8., -11.])
    test = maximum_drop_events(x, pd.date_range("2020-01-01", periods=8, freq="MS"),
        threshold_db=1.25, minimum_separation=3)
    assert [v["change_time"].strftime("%Y-%m") for v in test] == ["2020-08"]
    events = old_events[["unit_id", "pixel_track_id", "change_time", "candidate_id", "score", "threshold", "method"]].copy()
    reports, nodes, receipt = build_bounded_events(units, members, events,
        allow_multiple_months_per_year=True, source_method="maximum_drop_t125",
        validate_frozen_gates=False, require_annual_decision=False,
        output_method="maximum_drop_t125_B20_sameinput_v3", protocol_version="balsa_research_v3_controls")
    reports["interval_time_basis"] = "actual_pixel_response_months"
    events.to_parquet(dest / "pixel_events.parquet", index=False)
    old_status.to_parquet(dest / "pixel_execution.parquet", index=False)
    reports.to_parquet(dest / "stand_reports.parquet", index=False)
    nodes.to_parquet(dest / "source_members.parquet", index=False)
    save(dest / "execution_receipt.json", dict(status="completed", all_pixel_events_recomputed_exact=True,
        pixels=len(units), pixel_events=len(events), reports=len(reports), seconds=time.monotonic()-started,
        date_and_gap_selfcheck=True, source_events_sha256=sha(OLD / "predictions/pixel_events.parquet"),
        source_status_sha256=sha(OLD / "predictions/pixel_status.parquet"), aggregation=receipt))
    print("Maximum-drop completed", len(events), len(reports), flush=True)


def initialize_recusum():
    global VALUES
    VALUES = np.load(SOURCE / "master_values.npy", mmap_mode="r")


def recusum_batch(rows):
    from recusum_adapter import firstpost_recusum
    records, execution = [], []
    for unit, identity, index in rows:
        started = time.monotonic()
        vv = VALUES[index]
        points = firstpost_recusum(vv, seed=stable_seed(identity, 20260820))
        accepted = [p for p in points if 2017 <= DATES[p].year <= 2025 and is_decrease(vv, p, radius=6)]
        for p in accepted:
            records.append(dict(unit_id=unit, pixel_track_id=identity, change_time=DATES[p], raw_index=p,
                candidate_id=f"{METHOD}:{identity}:{DATES[p]:%Y-%m}", method=METHOD, decrease=True))
        execution.append(dict(unit_id=unit, pixel_track_id=identity, method=METHOD, status="completed",
            valid_months=int(np.isfinite(vv).sum()), raw_candidates=len(points), event_count=len(accepted),
            execution_seconds=time.monotonic()-started))
    return records, execution


def recusum():
    units, members = freeze()
    pilot = json.loads((OUT / "recusum/pilot.json").read_text())
    assert pilot["checks_passed"]
    dest = OUT / "recusum"; started = time.monotonic()
    rows = list(units[["unit_id", "pixel_track_id", "series_index"]].itertuples(index=False, name=None))
    batches = [rows[i:i+64] for i in range(0, len(rows), 64)]
    events, statuses = [], []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize_recusum) as pool:
        for i, (e, s) in enumerate(pool.map(recusum_batch, batches), 1):
            events.extend(e); statuses.extend(s)
            if i % 20 == 0 or i == len(batches):
                print(f"ReCuSum adapted {min(i*64, len(rows))}/{len(rows)}; {time.monotonic()-started:.1f}s", flush=True)
    events = pd.DataFrame(events); statuses = pd.DataFrame(statuses)
    assert len(statuses) == 26890 and statuses.status.eq("completed").all()
    events.to_parquet(dest / "pixel_events.parquet", index=False)
    statuses.to_parquet(dest / "pixel_execution.parquet", index=False)
    reports, nodes, diagnostics = build_bounded_events(units, members, events,
        allow_multiple_months_per_year=True, source_method=METHOD, validate_frozen_gates=False,
        require_annual_decision=False, output_method=METHOD+"_B20", protocol_version="balsa_research_v3_controls")
    reports["interval_time_basis"] = "actual_pixel_response_months"
    reports.to_parquet(dest / "stand_reports.parquet", index=False)
    nodes.to_parquet(dest / "source_members.parquet", index=False)
    save(dest / "execution_receipt.json", dict(status="completed_adapted_not_official", pixels=len(units),
        pixel_events=len(events), stand_reports=len(reports), seconds=time.monotonic()-started,
        independent_pilot=pilot, aggregation=diagnostics))
    print("ReCuSum adapted completed", len(events), len(reports), flush=True)


def score():
    freeze()
    tables, reference_config = original_tables()
    methods = [("B20_firstpost_v2", SOURCE / "validation/stand_reports.parquet"),
        ("Q20_firstpost_v2", OUT / "q20/stand_reports.parquet"),
        ("maximum_drop_t125_B20", OUT / "maximum_drop/stand_reports.parquet"),
        (METHOD+"_B20", OUT / "recusum/stand_reports.parquet")]
    summaries, histories, pairs, strata_all, paired = [], [], [], [], []
    main_found = None
    for name, path in methods:
        if not path.exists():
            continue
        reports = pd.read_parquet(path)
        detail, summary, history, annual, strata, _ = catalogue_metrics(name, reports,
            tables["references"], tables["eligibility"], tables["stands"])
        detail.to_csv(OUT / f"{name}_matches.csv", index=False, encoding="utf-8-sig")
        annual.to_csv(OUT / f"{name}_annual.csv", index=False, encoding="utf-8-sig")
        summaries.append(summary); histories.append(history); strata_all.append(strata)
        selected = detail[detail.timing_type.eq("month") & detail.tolerance_months.eq(2)]
        assert len(selected) == 1049
        found = selected.set_index("event_id")
        if main_found is None:
            main_found = found
            assert found.found.sum() == 930
        else:
            delta = main_found.found.astype(int)-found.found.reindex(main_found.index).astype(int)
            counts = pd.DataFrame(dict(sample_id=main_found.sample_id, delta=delta)).groupby("sample_id").delta.agg(["sum", "size"])
            ci = np.nanquantile(bootstrap_ratios(counts["sum"], counts["size"], tables["stands"]), [.025, .975])
            paired.append(dict(comparator=name, main_only=int(delta.eq(1).sum()), control_only=int(delta.eq(-1).sum()),
                main_minus_control=float(delta.mean()), ci_lower=float(ci[0]), ci_upper=float(ci[1]),
                scope="paired_conditional_R0_spatial_block_bootstrap"))
        ranks = reports.sort_values(["sample_id", "representative_month", "prediction_id"]).copy()
        ranks["report_order"] = ranks.groupby("sample_id").cumcount()
        order = ranks.set_index("prediction_id").report_order
        count_before = len(pairs)
        for sample, group in selected.groupby("sample_id", sort=True):
            rows = group.sort_values(["reference_month", "event_id"]).to_dict("records")
            for left, right in zip(rows, rows[1:]):
                both = bool(left["found"] and right["found"])
                rgap = pd.Period(right["reference_month"], freq="M").ordinal-pd.Period(left["reference_month"], freq="M").ordinal
                pgap = (pd.Period(right["representative_month"], freq="M").ordinal-
                    pd.Period(left["representative_month"], freq="M").ordinal) if both else None
                adjacent = bool(both and order[right["prediction_id"]]-order[left["prediction_id"]] == 1)
                pairs.append(dict(method=name, sample_id=sample, left_event_id=left["event_id"],
                    right_event_id=right["event_id"], both_found=both, reference_gap_months=rgap,
                    predicted_gap_months=pgap, registered_pair_reports_adjacent=adjacent,
                    interval_error_months=pgap-rgap if both and pgap > 0 else None))
        assert len(pairs)-count_before == 590
    pd.concat(summaries).to_csv(OUT / "catalogue_recall.csv", index=False, encoding="utf-8-sig")
    pd.concat(histories).to_csv(OUT / "registered_history_recovery.csv", index=False, encoding="utf-8-sig")
    pd.concat(strata_all).to_csv(OUT / "catalogue_strata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired).to_csv(OUT / "paired_primary_comparison.csv", index=False, encoding="utf-8-sig")
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(OUT / "registered_pair_details.csv", index=False, encoding="utf-8-sig")
    pair_summary = pair_frame.groupby("method").agg(registered_pairs=("both_found", "size"),
        both_found=("both_found", "sum"), registered_pair_reports_adjacent=("registered_pair_reports_adjacent", "sum"))
    pair_summary.to_csv(OUT / "registered_pair_summary.csv", encoding="utf-8-sig")
    common = pair_frame.pivot(index=["sample_id", "left_event_id", "right_event_id"], columns="method", values="interval_error_months")
    rows = []
    for name in common.columns.drop("B20_firstpost_v2"):
        shared = common[["B20_firstpost_v2", name]].dropna()
        rows.append(dict(comparator=name, common_order_valid_pairs=len(shared),
            main_interval_MAE=float(shared.B20_firstpost_v2.abs().mean()),
            control_interval_MAE=float(shared[name].abs().mean()),
            scope="conditional_common_recovered_registered_pairs_not_true_rotation_accuracy"))
    pd.DataFrame(rows).to_csv(OUT / "common_pair_interval_error.csv", index=False, encoding="utf-8-sig")
    inputs()
    save(OUT / "scoring_receipt.json", dict(completed_methods=[n for n,p in methods if p.exists()],
        source_reference_inputs={k:v for k,v in reference_config["inputs"].items()
            if k in ("references", "event_comparison", "units", "membership", "old_blocks")},
        denominators=dict(month_references=1049, multi_reference_stands=403, registered_adjacent_pairs=590),
        primary_tolerance_months=2, quarter_tolerance_used=False, old_weights_transferred=False,
        unmatched_reports_are_FP=False, precision=None, F1=None, scientific_signature_unchanged=True,
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "scoring_receipt.json"}))
    print(pd.concat(summaries).query("tolerance_months == 2")[["method", "n_reports", "frozen_month_found", "frozen_recall"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "q20", "maximum_drop", "recusum", "score"])
    args = parser.parse_args()
    {"freeze":freeze, "q20":q20, "maximum_drop":maximum_drop, "recusum":recusum, "score":score}[args.phase]()
