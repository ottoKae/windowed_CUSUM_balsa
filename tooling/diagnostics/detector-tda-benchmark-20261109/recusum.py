"""Trace the frozen ReCuSum recursion and its own monthly observation support."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from windowed_cusum.benchmarks.baselines import cusum_single, is_decrease
from windowed_cusum.detector.detection import stable_seed

METHODS = {
    "recusum_decrease_c09_m12": dict(date_policy="legacy_peak", minimum_segment=12, decrease=True),
    "recusum_raw_c09_m18": dict(date_policy="legacy_peak", minimum_segment=18, decrease=False),
    "recusum_notebook_firstpost_c09_m12_adapted": dict(date_policy="firstpost", minimum_segment=12, decrease=True),
}
LEGACY = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_v3/predictions"
FIRSTPOST = ROOT / "runs/validation/balsa_research_execution_20261031_v3/controls/recusum"
UNKNOWN_REASONS = {"": 0, "missing_observation": 1, "no_evaluated_recursive_context": 2,
    "not_reportable_under_date_policy": 3, "direction_context_unobserved": 4, "outside_report_period": 5}


def trace_recusum(values, *, seed, confidence=.9, bootstrap=800,
                  minimum_segment=12, maximum_depth=10,
                  date_policy="firstpost", decrease=False, radius=6):
    """Return unchanged detections, monthly opportunities and a recursive ledger.

    An opportunity is a finite, potentially reportable month in a segment that
    reached the original CUSUM evaluation gate. It is not a per-month hypothesis
    test. Segment support is recorded before the confidence or direction decision;
    descendants are visited only when the original recursion would visit them.
    A tested ancestor continues to support months in a subsequently blocked child.
    Reporting-year restriction belongs to the caller and does not alter recursion.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or date_policy not in {"legacy_peak", "firstpost"}:
        raise ValueError("Use a one-dimensional series and legacy_peak or firstpost dates")
    if bootstrap < 1 or minimum_segment < 6 or maximum_depth < 0 or not 0 <= confidence <= 1:
        raise ValueError("Invalid ReCuSum confidence, permutation or recursion configuration")

    finite = np.isfinite(values)
    tested_context_count = np.zeros(len(values), dtype=np.uint16)
    candidate_context_count = np.zeros(len(values), dtype=np.uint16)
    segments = []

    def visit(start, stop, local_seed, remaining_depth, parent_id):
        local = values[start:stop]
        valid = np.flatnonzero(np.isfinite(local))
        segment_id = len(segments)
        record = dict(segment_id=segment_id, parent_segment_id=parent_id,
            start_index=start, stop_index_exclusive=stop,
            depth=maximum_depth-remaining_depth, remaining_depth=remaining_depth,
            seed=int(local_seed), finite_observations=int(len(valid)),
            context_evaluated=False, permutation_evaluated=False,
            permutations_run=0, confidence_level=None, confidence_passed=False,
            peak_index=None, proposed_index=None, accepted_index=None,
            opportunity_months=0, first_opportunity_index=None,
            last_opportunity_index=None, stop_reason="")
        segments.append(record)
        if len(valid) < 2 * minimum_segment:
            record["stop_reason"] = "insufficient_segment_observations"
            return []
        if remaining_depth <= 0:
            record["stop_reason"] = "maximum_depth"
            return []

        record["context_evaluated"] = True
        tested_context_count[start + valid] += 1
        possible = valid if date_policy == "legacy_peak" else valid[1:]
        candidate_context_count[start + possible] += 1
        record.update(opportunity_months=int(len(possible)),
            first_opportunity_index=int(start + possible[0]),
            last_opportunity_index=int(start + possible[-1]))

        # confidence=0 exposes the frozen score even on rejected segments, without
        # changing its permutation stream; the original threshold is applied below.
        peak, level = cusum_single(local, confidence=0, bootstrap=bootstrap, seed=local_seed)
        if peak is None:
            record["stop_reason"] = "evaluated_zero_amplitude"
            return []
        record.update(permutation_evaluated=True, permutations_run=bootstrap,
            confidence_level=float(level), confidence_passed=bool(level >= confidence),
            peak_index=int(start + peak))
        after = valid[valid > peak]
        point = peak if date_policy == "legacy_peak" else (int(after[0]) if len(after) else None)
        record["proposed_index"] = None if point is None else int(start + point)
        if level < confidence:
            record["stop_reason"] = "confidence_rejected"
            return []
        if point is None:
            record["stop_reason"] = "no_post_peak_observation"
            return []

        point = int(point)
        record.update(accepted_index=start + point, stop_reason="accepted_split")
        if date_policy == "firstpost":
            left = visit(start, start + point, local_seed + point, remaining_depth - 1, segment_id)
            right = visit(start + point, stop, local_seed + point + 1, remaining_depth - 1, segment_id)
        else:
            left = visit(start, start + point, local_seed + point, remaining_depth - 1, segment_id) if point > 0 else []
            right = visit(start + point, stop, local_seed + point + 1, remaining_depth - 1, segment_id) if point < len(local) - 1 else []
        return sorted(set(left + [start + point] + right))

    raw_points = visit(0, len(values), seed, maximum_depth, None)
    direction_observed = np.asarray([
        finite[max(0, index-radius):index].any()
        and finite[index:min(len(values), index+radius)].any()
        for index in range(len(values))
    ], dtype=bool)
    opportunity = candidate_context_count > 0
    if decrease:
        opportunity &= direction_observed
    points = [point for point in raw_points if not decrease or is_decrease(values, point, radius=radius)]
    unknown_reason = np.full(len(values), "", dtype=object)
    unknown_reason[~finite] = "missing_observation"
    unknown_reason[finite & (tested_context_count == 0)] = "no_evaluated_recursive_context"
    unknown_reason[finite & (tested_context_count > 0) & (candidate_context_count == 0)] = "not_reportable_under_date_policy"
    if decrease:
        unknown_reason[(candidate_context_count > 0) & ~direction_observed] = "direction_context_unobserved"
    return dict(points=points, raw_points=raw_points, opportunity=opportunity,
        tested_context_count=tested_context_count, candidate_context_count=candidate_context_count,
        direction_observed=direction_observed, unknown_reason=unknown_reason,
        segments=segments, date_policy=date_policy, decrease=decrease)


def main():
    from common import OUT, SOURCE, MONTHS, digest, load_inputs, save_detector, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "full", "reexport"))
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    chosen = {args.method: METHODS[args.method]} if args.method else METHODS
    if args.phase == "reexport":
        reexport_trace(chosen)
        return
    units, _, values = load_inputs()
    rows = list(units[["unit_id", "pixel_track_id", "series_index"]].itertuples(index=False, name=None))
    pilot_results = []
    for method, config in chosen.items():
        source_dir = FIRSTPOST if config["date_policy"] == "firstpost" else LEGACY
        source_file = source_dir / "pixel_events.parquet"
        events = pd.read_parquet(source_file, filters=[("method", "=", method)])
        if not events.unit_id.isin(units.unit_id).all():
            raise ValueError(f"Frozen {method} contains units outside the current membership")
        expected = events.assign(index=pd.to_datetime(events.change_time).dt.to_period("M").astype("int64")-MONTHS[0].ordinal)
        expected = expected.groupby("unit_id")["index"].apply(lambda x: sorted(set(x))).to_dict()
        sources = [source_file, Path(__file__), ROOT / "src/windowed_cusum/benchmarks/baselines.py",
            SOURCE / "master_values.npy", SOURCE / "master_units.parquet", SOURCE / "validation_membership.parquet"]
        if config["date_policy"] == "firstpost":
            sources.append(ROOT / "tooling/diagnostics/balsa-research-v3-20261031/controls/recusum_adapter.py")
        else:
            sources.append(ROOT / "scripts/run_pixel_b20_benchmark_window.py")
        before = {str(p): digest(p) for p in sources}
        selected = [rows[i] for i in np.linspace(0, len(rows)-1, 32, dtype=int)] if args.phase == "pilot" else rows
        batches = [selected[i:i+64] for i in range(0, len(selected), 64)]
        started = perf_counter()
        statuses, ledger = [], []
        eligible_rows, context_rows, candidate_rows, direction_rows, reason_rows = [], [], [], [], []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(SOURCE / "master_values.npy"), config)) as pool:
            for batch_number, records in enumerate(pool.map(trace_batch, batches), 1):
                for unit, identity, trace, seconds in records:
                    accepted = [point for point in trace["points"] if 12 <= point <= 119]
                    if accepted != expected.get(unit, []):
                        failure = dict(status="failed_frozen_parity", method_id=method, unit_id=unit,
                            expected=expected.get(unit, []), observed=accepted, date_policy=config["date_policy"])
                        write_json(OUT / "recusum_parity_failure.json", failure)
                        raise ValueError(f"Frozen ReCuSum parity failed for {method}: {unit}")
                    opportunity = trace["opportunity"].copy()
                    opportunity[:12] = False
                    opportunity[120:] = False
                    assert all(opportunity[point] for point in accepted)
                    reason = trace["unknown_reason"].copy()
                    reason[:12] = "outside_report_period"
                    reason[120:] = "outside_report_period"
                    eligible_rows.append(opportunity)
                    context_rows.append(trace["tested_context_count"])
                    candidate_rows.append(trace["candidate_context_count"])
                    direction_rows.append(trace["direction_observed"])
                    reason_rows.append(np.asarray([UNKNOWN_REASONS[value] for value in reason], dtype=np.uint8))
                    ledger.extend(dict(unit_id=unit, pixel_track_id=identity, method_id=method, **record)
                                  for record in trace["segments"])
                    statuses.append(dict(unit_id=unit, pixel_track_id=identity, method_id=method,
                        status="completed", frozen_event_parity=True, raw_candidates=len(trace["raw_points"]),
                        accepted_report_months=len(accepted), evaluated_contexts=sum(r["context_evaluated"] for r in trace["segments"]),
                        tested_contexts=len(trace["segments"]), eligible_report_months=int(opportunity.sum()),
                        execution_seconds=seconds))
                if batch_number % 20 == 0 or batch_number == len(batches):
                    print(f"{method}: {len(statuses)}/{len(selected)} pixels; {perf_counter()-started:.1f}s", flush=True)
        elapsed = perf_counter()-started
        assert all(digest(p) == h for p, h in before.items()), "Frozen sources changed during ReCuSum replay"
        eligible = np.stack(eligible_rows)
        segment_table = pd.DataFrame(ledger)
        summary = dict(method_id=method, status="passed", pixels=len(selected),
            accepted_event_nodes=sum(r["accepted_report_months"] for r in statuses),
            segments=len(ledger), evaluated_contexts=int(segment_table.context_evaluated.sum()),
            permutation_tests=int(segment_table.permutation_evaluated.sum()),
            cpu_seconds=sum(r["execution_seconds"] for r in statuses), elapsed_seconds=elapsed,
            estimated_full_four_worker_seconds=sum(r["execution_seconds"] for r in statuses)/len(selected)*len(units)/4,
            frozen_accepted_nodes_exact=True, parameters=dict(confidence=.9, bootstrap=800,
                maximum_depth=10, radius=6, seed_base=20260820, **config), input_sha256=before)
        if args.phase == "pilot":
            pilot_results.append(summary)
            target = OUT / "recusum_pilot" / method
            target.mkdir(parents=True, exist_ok=True)
            segment_table.to_parquet(target / "segments.parquet", index=False)
            pd.DataFrame(statuses).to_parquet(target / "pixel_status.parquet", index=False)
            continue

        assert [r["unit_id"] for r in statuses] == units.unit_id.tolist()
        assert not eligible[~np.isfinite(values)].any()
        assert summary["accepted_event_nodes"] == len(events.drop_duplicates(["unit_id", "change_time"]))
        target = OUT / "detectors" / method
        target.mkdir(parents=True, exist_ok=True)
        segment_table.to_parquet(target / "segments.parquet", index=False)
        pd.DataFrame(statuses).to_parquet(target / "pixel_status.parquet", index=False)
        np.savez_compressed(target / "monthly_trace.npz", unit_id=units.unit_id.to_numpy(dtype=str),
            months=MONTHS.astype(str).to_numpy(dtype=str), tested_context_count=np.stack(context_rows),
            candidate_context_count=np.stack(candidate_rows), direction_observed=np.stack(direction_rows),
            unknown_reason_code=np.stack(reason_rows))
        labels = {"recusum_decrease_c09_m12": "ReCuSum legacy peak, decrease-filtered (m12)",
            "recusum_raw_c09_m18": "ReCuSum legacy peak, raw (m18)",
            "recusum_notebook_firstpost_c09_m12_adapted": "ReCuSum first-post recursive adaptation, decrease-filtered (m12)"}
        metadata = dict(**summary, family="recusum", label=labels[method],
            date_contract="legacy CUSUM peak at every split; no date shift" if config["date_policy"] == "legacy_peak"
                else "first finite observation after CUSUM peak at every split; corrected split and seed offsets",
            eligibility_contract="finite reportable month in an actually evaluated recursive context, registered before CL acceptance; tested ancestors remain valid support; decrease arms additionally require observable radius-6 pre/post direction windows",
            coverage_type="recursive_context_search_support_not_individual_month_tests",
            no_windowed_cusum_eligibility_reused=True, non_significant_contexts_remain_observed=True,
            zero_amplitude_contexts="observed deterministic non-detection; permutations not needed",
            descendant_contexts="only original accepted-parent recursion is visited; no hypothetical rejected-parent descendants",
            unknown_reason_codes=UNKNOWN_REASONS,
            source_events_preserved=True, trace_output_sha256={name: digest(target/name) for name in
                ("segments.parquet", "pixel_status.parquet", "monthly_trace.npz")})
        save_detector(method, events, eligible, metadata, sources)
        write_json(target / "trace_receipt.json", metadata)
    if args.phase == "pilot":
        write_json(OUT / "recusum_pilot.json", dict(status="passed", configurations=pilot_results))


def reexport_trace(methods):
    """Convert the locally generated cached month axis without rerunning detection."""
    from common import OUT, digest, write_json

    source = str(Path(__file__).resolve())
    source_hash = digest(source)
    records = []
    for method in methods:
        target = OUT / "detectors" / method
        scientific_hashes = {name: digest(target / name) for name in
            ("pixel_events.parquet", "monthly_eligibility.npz")}
        path = target / "monthly_trace.npz"
        trace_before = digest(path)
        # The old archive was generated here with an object-valued month axis.
        with np.load(path, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        old_dtype = str(arrays["months"].dtype)
        arrays["months"] = np.asarray(arrays["months"], dtype=str)
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as archive:
            assert all(np.array_equal(archive[name], value) and archive[name].dtype == value.dtype
                       for name, value in arrays.items())
        assert scientific_hashes == {name: digest(target / name) for name in scientific_hashes}
        trace_after = digest(path)
        receipt_before, receipt_after, prior_source_hashes = {}, {}, {}
        for name in ("trace_receipt.json", "detector_receipt.json"):
            receipt_path = target / name
            receipt_before[name] = digest(receipt_path)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            prior_source_hashes[name] = receipt["input_sha256"][source]
            receipt["input_sha256"][source] = source_hash
            receipt["trace_output_sha256"]["monthly_trace.npz"] = trace_after
            write_json(receipt_path, receipt)
            receipt_after[name] = digest(receipt_path)
        records.append(dict(method_id=method, old_months_dtype=old_dtype,
            new_months_dtype=str(arrays["months"].dtype), all_array_values_unchanged=True,
            scientific_output_sha256_unchanged=scientific_hashes,
            trace_sha256_before=trace_before, trace_sha256_after=trace_after,
            receipt_sha256_before=receipt_before, receipt_sha256_after=receipt_after,
            source_sha256_before=prior_source_hashes, source_sha256_after=source_hash))
        print(f"{method}: cached trace month axis re-exported; scientific outputs unchanged", flush=True)
    write_json(OUT / "recusum_trace_serialization_receipt.json", dict(status="passed",
        serialization_only=True, detection_or_permutations_rerun=False, configurations=records))


def initialize(path, config):
    global VALUES, CONFIG
    VALUES = np.load(path, mmap_mode="r")
    CONFIG = config


def trace_batch(rows):
    records = []
    for unit, identity, index in rows:
        started = perf_counter()
        trace = trace_recusum(VALUES[index], seed=stable_seed(identity, 20260820), **CONFIG)
        records.append((unit, identity, trace, perf_counter()-started))
    return records


if __name__ == "__main__":
    main()
