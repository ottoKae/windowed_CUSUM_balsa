"""Run the frozen BFASTmonitor configuration per FC001 master cell, then B20."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "runs/validation/existing_labels_w36_20260927_v1/inputs"
DEFAULT_RUN = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_bfast_v1"
DEFAULT_RSCRIPT = Path(shutil.which("Rscript") or "Rscript")
DEFAULT_R_LIBRARY = ROOT / "runs/benchmark_extension_dependencies_20260912/R-library"
sys.path.insert(0, str(ROOT / "scripts"))
import fc001_validation_detection as fc  # noqa: E402
from windowed_cusum.stand.harvest_bounded_event_aggregation import build_bounded_events  # noqa: E402

METHOD = "bfast_hist2_h025_a005"
YEARS = tuple(range(2019, 2026))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_chunk(payload):
    number, jobs_path, output_dir, rscript, library, runner = payload
    result = output_dir / "results.json"
    if not result.exists():
        subprocess.run([str(rscript), str(runner), str(library), str(jobs_path), str(output_dir), "compact"],
                       check=True, stdout=subprocess.DEVNULL)
    return number, result


def freeze(args):
    files = [Path(__file__), ROOT / "scripts/run_bfast_monitor.R", INPUTS / "units.parquet",
             INPUTS / "membership.parquet"]
    payload = {
        "schema_version": 1, "method": METHOD, "monitor_years": list(YEARS),
        "history_years": 2, "order": 1, "h": .25, "level": .05,
        "minimum_history_fraction": .75, "minimum_monitor_observations": 6,
        "compact_r_output": True, "chunk_size": args.chunk_size, "limit": args.limit,
        "rscript": str(args.rscript.resolve()), "r_library": str(args.r_library.resolve()),
        "files": {str(p.resolve()): digest(p) for p in files},
    }
    destination = args.run_dir / "config/configuration_manifest.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise ValueError("BFAST frozen inputs/configuration changed; use a new run directory")
    if not destination.exists():
        atomic_json(destination, payload)


def prepare(args, units):
    jobs_dir = args.run_dir / "raw/jobs"; jobs_dir.mkdir(parents=True, exist_ok=True)
    support_rows = []; job_rows = []; chunk_number = 0
    for trajectory_index, row in enumerate(units.to_dict("records")):
        vv, dates, _ = fc.series_for(row)
        for year in YEARS:
            history = (dates.year >= year - 2) & (dates.year < year)
            monitor = dates.year == year
            history_count = int(np.isfinite(vv[history]).sum())
            monitor_count = int(np.isfinite(vv[monitor]).sum())
            eligible = history_count >= 18 and monitor_count >= 6
            support_rows.append({"trajectory_index": trajectory_index, "unit_id": row["unit_id"],
                                 "pixel_track_id": row["pixel_track_id"], "monitor_year": year,
                                 "history_valid_months": history_count, "monitor_valid_months": monitor_count,
                                 "eligible": eligible})
            if not eligible:
                continue
            mask = (dates.year >= year - 2) & (dates.year <= year)
            job_rows.append({"trajectory_index": trajectory_index, "monitor_year": year,
                             "start_year": year - 2, "order": 1, "h": .25, "level": .05,
                             "parameter_id": METHOD,
                             "values": [float(v) if np.isfinite(v) else None for v in vv[mask]]})
            if len(job_rows) >= args.chunk_size:
                atomic_json(jobs_dir / f"{chunk_number:05d}.json", job_rows)
                job_rows = []; chunk_number += 1
    if job_rows:
        atomic_json(jobs_dir / f"{chunk_number:05d}.json", job_rows)
    support = pd.DataFrame(support_rows)
    support.to_parquet(args.run_dir / "monitoring_support.parquet", index=False)
    return support, sorted(jobs_dir.glob("*.json"))


def main(args):
    args.run_dir.mkdir(parents=True, exist_ok=True); freeze(args)
    units = pd.read_parquet(INPUTS / "units.parquet").sort_values(["track_key", "cache_index"]).reset_index(drop=True)
    if args.limit is not None:
        units = units.iloc[:args.limit].copy()
    support_path = args.run_dir / "monitoring_support.parquet"
    if support_path.exists():
        support = pd.read_parquet(support_path); job_paths = sorted((args.run_dir / "raw/jobs").glob("*.json"))
    else:
        support, job_paths = prepare(args, units)
    tasks = []
    for number, jobs_path in enumerate(job_paths):
        output = args.run_dir / "raw/results" / f"{number:05d}"; output.mkdir(parents=True, exist_ok=True)
        tasks.append((number, jobs_path, output, args.rscript, args.r_library, ROOT / "scripts/run_bfast_monitor.R"))
    receipts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_chunk, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            number, result = future.result(); receipts[number] = result
            print(f"bfast_chunks={completed}/{len(tasks)}", flush=True)
    events = []; completed_keys = set()
    for number in sorted(receipts):
        rows = json.loads(receipts[number].read_text(encoding="utf-8"))
        expected = json.loads(job_paths[number].read_text(encoding="utf-8"))
        if len(rows) != len(expected) or any(row.get("status") != "success" for row in rows):
            raise ValueError(f"BFAST chunk {number} is incomplete or contains failed fits")
        for result in rows:
            key = (int(result["trajectory_index"]), int(result["monitor_year"])); completed_keys.add(key)
            boundary = result.get("breakpoint"); magnitude = result.get("alarm_residual_mean")
            if boundary is None or magnitude is None or float(magnitude) >= 0:
                continue
            year = int(np.floor(boundary)); month = int(round((float(boundary) - year) * 12)) + 1
            if month == 13:
                year += 1; month = 1
            date = pd.Timestamp(year=year, month=month, day=1)
            if date.year != int(result["monitor_year"]):
                raise ValueError("BFAST alarm lies outside its scheduled monitor year")
            unit = units.iloc[int(result["trajectory_index"])]
            raw = json.dumps([METHOD, unit.pixel_track_id, str(date.date())], separators=(",", ":"))
            events.append({"unit_id": unit.unit_id, "block_id": unit.block_id,
                           "pixel_track_id": unit.pixel_track_id, "track_key": unit.track_key,
                           "cache_index": int(unit.cache_index), "method": METHOD, "change_time": date,
                           "magnitude": float(magnitude),
                           "candidate_id": METHOD + ":" + hashlib.sha256(raw.encode()).hexdigest()[:24]})
    eligible = support.loc[support.eligible]
    if len(completed_keys) != len(eligible) or completed_keys != set(zip(eligible.trajectory_index, eligible.monitor_year)):
        raise ValueError("BFAST completed-job identity does not equal eligible support")
    events = pd.DataFrame(events)
    if events.empty:
        events = pd.DataFrame(columns=["unit_id", "block_id", "pixel_track_id", "method", "change_time", "candidate_id"])
    events.to_parquet(args.run_dir / "pixel_events.parquet", index=False)
    membership = pd.read_parquet(INPUTS / "membership.parquet")
    if args.limit is not None:
        membership = membership.loc[membership.unit_id.isin(units.unit_id)].copy()
    reports, members, diagnostic = build_bounded_events(
        units, membership, events, allow_multiple_months_per_year=True, source_method=None,
        validate_frozen_gates=False, require_annual_decision=False, output_method="B20:" + METHOD,
        protocol_version="common_pixel_b20_monitoring_v1")
    method_dir = args.run_dir / "methods" / METHOD; method_dir.mkdir(parents=True, exist_ok=True)
    reports["interval_time_basis"] = "actual_pixel_alarm_months"
    reports.to_parquet(method_dir / "stand_reports.parquet", index=False)
    members.to_parquet(method_dir / "source_members.parquet", index=False)
    atomic_json(method_dir / "aggregation_receipt.json", diagnostic)
    atomic_json(args.run_dir / "completed.json", {"eligible_pixel_year_jobs": len(eligible),
                "negative_pixel_alarms": len(events), "stand_reports": len(reports),
                "reporting_stands": int(reports.sample_id.nunique()) if len(reports) else 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--r-library", type=Path, default=DEFAULT_R_LIBRARY)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
