"""Run the frozen FC001 native pixels through common local benchmarks and B20.

This driver is intentionally thin: numerical detectors are imported from the
frozen/current lr_cusum sources, native monthly caches remain read-only, and the
existing B20 allocator is called after method-specific acceptance.  SQLite
checkpoints make every method/configuration resumable and distinguish successful
zero-event executions from failures and unexecuted pixels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
PIXELWISE = ROOT.parent / "windowed_CUSUM_pixelwise"
ARCHIVE = ROOT / "src"
CURRENT_SRC = ROOT / "src"
INPUTS = ROOT / "runs/validation/existing_labels_w36_20260927_v1/inputs"
OLD_REMOVALS = ROOT / "runs/validation/jag_event_accuracy_recalculation_20260928_v1/pixel_removals"
DEFAULT_RUN = ROOT / "runs/validation/pixel_b20_benchmark_window_public_replay"
DETECTOR_CONFIG = ROOT / "configs/standard_w36.json"

# Use the frozen detector package for every implementation it contains.  This is
# also the package used by the accepted W36 FC001 run.

sys.path.insert(0, str(ROOT / "scripts"))
import fc001_validation_detection as fc  # noqa: E402
from harvest_bounded_event_aggregation import build_bounded_events  # noqa: E402
from lr_cusum import paper  # noqa: E402
from lr_cusum.baselines import fpol_bocd_detect, is_decrease, recusum_detect  # noqa: E402
from lr_cusum.cusum_nrt_adapter import admitted, scan_nrt  # noqa: E402
from lr_cusum.detection import stable_seed  # noqa: E402
from lr_cusum.validation_closure import classical_lower_cusum_events, maximum_drop_events  # noqa: E402


from lr_cusum import window_geometry as GEOMETRY


CONFIGS = {
    "recusum_decrease_c09_m12": {"family": "recusum", "confidence": .9, "minimum_segment": 12,
                                  "maximum_depth": 10, "decrease": True, "bootstrap": 800},
    "recusum_raw_c09_m18": {"family": "recusum", "confidence": .9, "minimum_segment": 18,
                             "maximum_depth": 10, "decrease": False, "bootstrap": 800},
    "maximum_drop_t125": {"family": "maximum_drop", "threshold_db": 1.25},
    "classical_cusum_k075_h3": {"family": "classical_cusum", "allowance": .75,
                                  "decision_interval": 3.0},
    "bocd_h24_d6_decrease": {"family": "bocd", "hazard": 1 / 24, "delta_m": 6,
                               "min_obs": 12, "decrease": True},
    "nrt_b12_a6_tc98_raw": {"family": "nrt", "before": 12, "after": 6,
                              "confidence": .98, "samples": 500, "decrease": False},
    "nrt_b12_a6_tc98_decrease": {"family": "nrt", "before": 12, "after": 6,
                                   "confidence": .98, "samples": 500, "decrease": True},
    "windowed_w24_d12_s12": {"family": "windowed", "window_months": 24,
                               "decision_months": 12, "stride_months": 12},
    "windowed_w48_d12_s12": {"family": "windowed", "window_months": 48,
                               "decision_months": 12, "stride_months": 12},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def method_source_hashes():
    paths = [Path(fc.__file__), Path(paper.__file__), ROOT / "scripts/harvest_bounded_event_aggregation.py",
             ROOT / "scripts/harvest_quarter_aggregation.py", Path(GEOMETRY.__file__), Path(__file__)]
    return {str(path.resolve()): sha256(path) for path in paths}


def freeze(run):
    units = INPUTS / "units.parquet"; membership = INPUTS / "membership.parquet"
    payload = {
        "schema_version": 1,
        "scope": {"stands": 468, "master_pixels": 26890, "calendar": ["2016-01", "2026-06"],
                  "reporting": ["2017-01", "2025-12"]},
        "configurations": CONFIGS,
        "inputs": {str(path.resolve()): sha256(path) for path in (units, membership, DETECTOR_CONFIG)},
        "sources": method_source_hashes(),
        "actual_imports": {"paper": str(Path(paper.__file__).resolve()),
                           "fc001": str(Path(fc.__file__).resolve())},
        "common_b20": {"months": 3, "minimum_support": .2, "fixed_denominator": True,
                       "allow_multiple_months_per_year": True,
                       "ranking": ["unique_pixels_desc", "pixel_month_nodes_desc", "window_start_asc"]},
    }
    path = run / "config/configuration_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError("Frozen configuration/source/input changed; use a new run directory")
    if not path.exists():
        write_json(path, payload)
    return payload


def read_series(row):
    index, values, valid, n_obs, calendar, present = fc.read_track(row["track_key"])
    pos = int(row["cache_index"]); member = index.loc[pos]
    if member.pixel_track_id != row["pixel_track_id"]:
        raise ValueError("cache/native pixel identity mismatch")
    mask = valid[pos]
    if values.shape[1] < 2:
        raise ValueError("frozen cache does not contain VV and VH")
    vv = np.where(mask, values[pos, 0], np.nan).astype(np.float64)
    vh = np.where(mask, values[pos, 1], np.nan).astype(np.float64)
    if np.any(mask & ((n_obs[pos] <= 0) | ~np.isfinite(values[pos]).all(axis=0))):
        raise ValueError("positive valid mask contradicts cache values/n_obs")
    return vv, vh, pd.DatetimeIndex(calendar), present


def _event_id(method, pixel_track_id, number, date):
    key = json.dumps([method, pixel_track_id, int(number), str(pd.Timestamp(date).date())], separators=(",", ":"))
    return method + ":" + hashlib.sha256(key.encode()).hexdigest()[:24]


def detect(row, method):
    start = time.perf_counter()
    base = {key: row[key] for key in ("unit_id", "block_id", "pixel_track_id", "track_key", "cache_index")}
    result = {**base, "method": method, "status": "failed", "events": [], "error": None,
              "valid_months": None, "source_months": None, "execution_seconds": 0.0}
    try:
        vv, vh, dates, present = read_series(row); spec = CONFIGS[method]
        result["valid_months"] = int(np.isfinite(vv).sum()); result["source_months"] = int(present.sum())
        family = spec["family"]; events = []
        if family == "recusum":
            points = recusum_detect(vv, confidence=spec["confidence"], bootstrap=spec["bootstrap"],
                                    minimum_segment=spec["minimum_segment"], maximum_depth=spec["maximum_depth"],
                                    seed=stable_seed(row["pixel_track_id"], 20260820))
            for point in points:
                if (not spec["decrease"] or is_decrease(vv, point, radius=6)) and 2017 <= dates[point].year <= 2025:
                    events.append({"change_time": dates[point], "raw_index": int(point),
                                   "decrease": bool(is_decrease(vv, point, radius=6))})
        elif family == "maximum_drop":
            events = maximum_drop_events(vv, dates, threshold_db=spec["threshold_db"])
        elif family == "classical_cusum":
            events = classical_lower_cusum_events(vv, dates, allowance=spec["allowance"],
                                                   decision_interval=spec["decision_interval"])
        elif family == "bocd":
            points = fpol_bocd_detect(vv, vh, hazard=spec["hazard"], delta_m=spec["delta_m"],
                                      min_obs=spec["min_obs"])
            for point in points:
                if is_decrease(vv, point, radius=6) and 2017 <= dates[point].year <= 2025:
                    events.append({"change_time": dates[point], "raw_index": int(point), "decrease": True})
        elif family == "nrt":
            raw = scan_nrt(vv[None, :], dates, before=spec["before"], after=spec["after"],
                           samples=spec["samples"], seed=20260905)
            chosen = admitted(raw, confidence=spec["confidence"], decrease=spec["decrease"])
            chosen = chosen.loc[chosen.change_time.dt.year.between(2017, 2025)]
            events = chosen.drop(columns="trajectory_index", errors="ignore").to_dict("records")
        elif family == "windowed":
            cfg = paper.load_paper_config(DETECTOR_CONFIG)
            geometry = GEOMETRY.Geometry(spec["window_months"], spec["decision_months"], spec["stride_months"])
            events = GEOMETRY.detect_series(vv, dates, pixel_track_id=row["pixel_track_id"],
                                             geometry=geometry, config=cfg).to_dict("records")
        else:
            raise ValueError(family)
        normalized = []
        for number, event in enumerate(events):
            event = dict(event); date = pd.Timestamp(event["change_time"])
            event["change_time"] = date.isoformat()
            event["candidate_id"] = _event_id(method, row["pixel_track_id"], number, date)
            normalized.append(event)
        result.update(status="completed", events=normalized)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["execution_seconds"] = time.perf_counter() - start
    return result


def detect_batch(batch):
    return [detect(row, method) for row in batch for method in row.pop("_methods")]


def connect(run):
    path = run / "checkpoints/local_methods.sqlite"; path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path); db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS detections (pixel_track_id TEXT, method TEXT, status TEXT, payload TEXT, PRIMARY KEY(pixel_track_id,method))")
    return db


def export(run, units, db, methods):
    payload = {(pixel, method): json.loads(value) for pixel, method, value in
               db.execute("SELECT pixel_track_id,method,payload FROM detections")}
    statuses, events = [], []
    for row in units.to_dict("records"):
        identity = {key: row[key] for key in ("unit_id", "block_id", "pixel_track_id", "track_key", "cache_index")}
        for method in methods:
            item = payload.get((row["pixel_track_id"], method))
            if item is None:
                statuses.append({**identity, "method": method, "status": "not_executed", "event_count": None})
                continue
            statuses.append({**identity, "method": method, "status": item["status"], "error": item["error"],
                             "valid_months": item["valid_months"], "source_months": item["source_months"],
                             "execution_seconds": item["execution_seconds"],
                             "event_count": len(item["events"]) if item["status"] == "completed" else None})
            events.extend({**identity, "method": method, **event} for event in item["events"])
    status = pd.DataFrame(statuses)
    event_frame = (pd.DataFrame(events) if events else
                   pd.DataFrame(columns=["unit_id", "block_id", "pixel_track_id", "method", "change_time", "candidate_id"]))
    out = run / "predictions"; out.mkdir(parents=True, exist_ok=True)
    status.to_parquet(out / "pixel_status.parquet", index=False)
    event_frame.to_parquet(out / "pixel_events.parquet", index=False)
    status.groupby(["method", "status"], dropna=False).size().rename("pixel_count").reset_index().to_csv(
        out / "execution_summary.csv", index=False, encoding="utf-8-sig")
    return status, event_frame


def aggregate(run, units, methods):
    membership = pd.read_parquet(INPUTS / "membership.parquet")
    events = pd.read_parquet(run / "predictions/pixel_events.parquet")
    rows = []
    for method in methods:
        selected = events.loc[events.method.eq(method)].copy()
        reports, members, diagnostics = build_bounded_events(
            units, membership, selected,
            allow_multiple_months_per_year=True,
            source_method=None, validate_frozen_gates=False, require_annual_decision=False,
            output_method="B20:" + method, protocol_version="common_pixel_b20_v1",
        )
        destination = run / "methods" / method; destination.mkdir(parents=True, exist_ok=True)
        reports["interval_time_basis"] = "actual_pixel_response_months"
        reports.to_parquet(destination / "stand_reports.parquet", index=False)
        members.to_parquet(destination / "source_members.parquet", index=False)
        write_json(destination / "aggregation_receipt.json", diagnostics)
        rows.append({"method": method, "n_pixel_events": len(selected), "n_reports": len(reports),
                     "n_reporting_stands": reports.sample_id.nunique() if len(reports) else 0,
                     **{key: diagnostics[key] for key in ("deduplicated_pixel_month_events", "duplicate_pixel_month_rows",
                                                          "assigned_nodes", "residual_nodes")}})
    pd.DataFrame(rows).to_csv(run / "evaluation/aggregation_summary.csv", index=False, encoding="utf-8-sig")


def baseline(run, units):
    source_events = pd.read_parquet(OLD_REMOVALS / "F4_pixel_events.parquet")
    membership = pd.read_parquet(INPUTS / "membership.parquet")
    reports, members, diagnostics = build_bounded_events(units, membership, source_events)
    old = pd.read_csv(OLD_REMOVALS / "F4_stand_reports.csv")
    compare = ["sample_id", "prediction_id", "representative_month", "response_pixels", "total_pixels"]
    left = reports[compare].sort_values(compare[:2]).reset_index(drop=True)
    right = old[compare].sort_values(compare[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    destination = run / "methods/windowed_w36_d12_s12"; destination.mkdir(parents=True, exist_ok=True)
    reports["interval_time_basis"] = "actual_pixel_response_months"
    reports.to_parquet(destination / "stand_reports.parquet", index=False)
    members.to_parquet(destination / "source_members.parquet", index=False)
    write_json(destination / "aggregation_receipt.json", diagnostics)
    write_json(run / "verification/w36_baseline_reproduction.json", {
        "exact": True, "pixel_events": len(source_events), "stand_reports": len(reports),
        "source": str((OLD_REMOVALS / "F4_pixel_events.parquet").resolve()),
        "source_sha256": sha256(OLD_REMOVALS / "F4_pixel_events.parquet")})


def run(args):
    args.run_dir.mkdir(parents=True, exist_ok=True); freeze(args.run_dir)
    units = pd.read_parquet(INPUTS / "units.parquet").sort_values(["track_key", "cache_index"])
    if len(units) != 26890 or units.unit_id.duplicated().any() or units.pixel_track_id.duplicated().any():
        raise ValueError("frozen master-pixel inventory changed")
    methods = args.method or list(CONFIGS)
    unknown = set(methods) - set(CONFIGS)
    if unknown: raise ValueError(f"unknown methods: {sorted(unknown)}")
    if args.stage == "freeze": return
    if args.stage == "baseline": baseline(args.run_dir, units); return
    if args.stage in ("run", "all"):
        selected = units.head(args.limit) if args.limit else units
        with connect(args.run_dir) as db:
            done = {(pixel, method): status for pixel, method, status in
                    db.execute("SELECT pixel_track_id,method,status FROM detections")}
            tasks = []
            for row in selected.to_dict("records"):
                missing = [m for m in methods if (row["pixel_track_id"], m) not in done or
                           (args.retry_failed and done[(row["pixel_track_id"], m)] == "failed")]
                if missing: tasks.append({**row, "_methods": missing})
            batches = [tasks[i:i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
            print(f"pixels={len(selected)} pixel_method_tasks={sum(len(x['_methods']) for x in tasks)} batches={len(batches)}", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(detect_batch, batch) for batch in batches]
                for count, future in enumerate(as_completed(futures), 1):
                    results = future.result()
                    db.executemany("INSERT OR REPLACE INTO detections VALUES (?,?,?,?)",
                                   [(x["pixel_track_id"], x["method"], x["status"], json.dumps(x, default=str))
                                    for x in results])
                    db.commit()
                    if count % 10 == 0 or count == len(futures):
                        print(f"completed_batches={count}/{len(futures)}", flush=True)
            export(args.run_dir, units, db, methods)
    if args.stage in ("aggregate", "all"):
        aggregate(args.run_dir, units, methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "baseline", "run", "aggregate", "all"], default="all")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--method", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.batch_size < 1: parser.error("workers and batch size must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(arguments)
