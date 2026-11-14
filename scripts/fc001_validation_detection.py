"""Frozen FC001 probability-sample detection, using existing native pixel caches.

The numerical methods are imported unchanged. SQLite checkpoints distinguish
completed zero-event series, unavailable inputs, failures and unexecuted keys.
Only the parent process writes checkpoints; workers evaluate bounded batches.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import importlib.metadata
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
SOURCE = ROOT / "src"
CACHE = ROOT / "data/local/paper_v1/polygon_pixels/cache"
CONFIG = ROOT / "configs/benchmarks/legacy_w36.json"
DEFAULT_RUN = ROOT / "runs/sampling/fc001_probability_validation_20260925_v1"
METHODS = ("windowed_w36_d12_s12", "recusum_decrease_c09_m12")

from windowed_cusum.detector import paper  # noqa: E402
from windowed_cusum.benchmarks.baselines import recusum_detect, is_decrease  # noqa: E402
from windowed_cusum.detector.component_ablation import ComponentAblationConfig, build_component_ledgers, derive_variant_events
from windowed_cusum.detector.detection import stable_seed  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def freeze_methods(run):
    config = paper.load_paper_config(CONFIG)
    assert (config.window_length, config.step, config.bands) == (36, 12, ("VV_dB",))
    assert Path(paper.__file__).resolve() == (SOURCE / "windowed_cusum/detector/paper.py").resolve()
    selected = ROOT / "runs/protocol_v2_1061_20260911/selection/folds/recusum_decrease.csv"
    folds = pd.read_csv(selected)
    assert len(folds) == 5 and set(folds.parameter_id) == {"recusum_c0.9_m12"}
    payload = {
        "schema_version": 1,
        "scope": "same native pixel support, frozen before first-stage block draw",
        "methods": {
            METHODS[0]: {"entry": "windowed_cusum.detector.paper.scan_series_n1 + derive_variant_events(F4)",
                         "config": config.as_dict(), "config_path": str(CONFIG),
                         "config_sha256": sha256(CONFIG)},
            METHODS[1]: {"entry": "windowed_cusum.benchmarks.baselines.recusum_detect + is_decrease",
                         "config": {"confidence": .9, "minimum_segment": 12,
                                    "maximum_depth": 10, "bootstrap": 800,
                                    "seed_base": 20260820, "direction_radius_months": 6},
                         "selection_basis": "all five existing development folds selected c0.9_m12; no new labels used",
                         "selection_source": str(selected), "selection_sha256": sha256(selected),
                         "seed_rule": "stable_seed(original pixel_track_id, 20260820)",
                         "date_rule": "unaltered baseline breakpoint index mapped back to original calendar"},
        },
        "input": {"cache_root": str(CACHE), "band": "VV_dB",
                  "calendar": ["2016-01", "2026-06"],
                  "mask": "existing joint VV/VH finite and n_obs>0; preserve explicit monthly NaN",
                  "spatial_support": "frozen assigned native 30 m pixel; never aggregate to plot median"},
        "reporting_calendar": ["2017-01", "2025-12"],
        "source_root": str(SOURCE),
        "source_sha256": {p.relative_to(SOURCE).as_posix(): sha256(p)
                          for p in sorted((SOURCE / "windowed_cusum").rglob("*.py"))},
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("numpy", "pandas", "scipy", "numba", "pyarrow")},
        "excluded_from_sampling_union": {
            "CCDC": "official Earth Engine execution not activated for new pixel population",
            "maximum_drop/classical_cusum/other": "no single deployment configuration fixed for this sample"},
        "later_methods": "may be evaluated with the same frozen probability sample and weights; do not redraw it",
    }
    # Normalize tuples so comparison to persisted JSON has identical types.
    payload = json.loads(json.dumps(payload))
    path = run / "config/freeze_methods.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Frozen implementation/config changed; use a new run, never overwrite")
    else:
        json_write(path, payload)
    return payload


@lru_cache(maxsize=3)
def read_track(track_key):
    if Path(track_key).name != track_key or ":" in track_key:
        raise ValueError("invalid track key")
    index = pd.read_parquet(CACHE / f"{track_key}.pixels.parquet").set_index("cache_index")
    with np.load(CACHE / f"{track_key}.npz", allow_pickle=False) as archive:
        values, valid = archive["values"], archive["valid"].astype(bool)
        n_obs, calendar = archive["n_obs"], pd.DatetimeIndex(archive["calendar"])
        present = archive["source_month_present"].astype(bool)
    if not calendar.equals(pd.date_range("2016-01-01", "2026-06-01", freq="MS")):
        raise ValueError("calendar differs from frozen 126 monthly positions")
    return index, values, valid, n_obs, calendar, present


def series_for(row):
    index, values, valid, n_obs, calendar, present = read_track(row["track_key"])
    position = int(row["cache_index"])
    member = index.loc[position]
    if member.pixel_track_id != row["pixel_track_id"]:
        raise ValueError("cache_index/pixel_track_id mismatch")
    for field in ("native_row", "native_col"):
        if field in row and int(row[field]) != int(member[field]):
            raise ValueError(f"native geometry mismatch: {field}")
    mask = valid[position]
    if np.any(mask & ((n_obs[position] <= 0) | ~np.isfinite(values[position]).all(axis=0))):
        raise ValueError("invalid positive cache mask")
    vv = np.where(mask, values[position, 0], np.nan).astype(np.float64)
    return vv, calendar, present


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def execute_pixel(row, methods):
    payloads = []
    try:
        vv, dates, present = series_for(row)
        read_error = None
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"
    for method in methods:
        started = time.perf_counter()
        result = {key: row[key] for key in ("unit_id", "block_id", "pixel_track_id", "track_key", "cache_index")}
        result.update(method=method, status="failed", events=[], windows=[], error=read_error,
                      valid_months=None, execution_seconds=0.0)
        try:
            if read_error:
                raise ValueError(read_error)
            result["valid_months"] = int(np.isfinite(vv).sum())
            if method == METHODS[0]:
                config = paper.load_paper_config(CONFIG)
                ledger = paper.scan_series_n1(vv, dates, identifier=row["pixel_track_id"],
                                             band="VV_dB", config=config).sliding_candidates.copy()
                events = derive_variant_events(pd.DataFrame(), ledger, "F4", reporting_only=True)
                accepted = set(events.candidate_id)
                ledger["accepted_event"] = ledger.candidate_id.isin(accepted)
                # Old f4_rejection_reason precedes N1 replacement; recompute labels
                # from returned N1 gate columns rather than repeat the detector.
                def status(window):
                    if not window.passes_window_calendar:
                        return "outside_output_calendar"
                    if not window.passes_minimum_observations:
                        return "input_insufficient"
                    if window.accepted_event:
                        return "completed_event"
                    return "completed_no_event"
                ledger["execution_status"] = [status(x) for x in ledger.itertuples(index=False)]
                ledger = ledger.drop(columns=["f4_rejection_reason"])
                result["windows"], result["events"] = records(ledger), records(events)
                eligible = ledger.passes_window_calendar & ledger.passes_minimum_observations
                result["status"] = "completed" if eligible.any() else "input_insufficient"
            elif method == METHODS[1]:
                enough = int(np.isfinite(vv).sum()) >= 24
                points = recusum_detect(vv, confidence=.9, minimum_segment=12, maximum_depth=10,
                                        bootstrap=800, seed=stable_seed(row["pixel_track_id"], 20260820))
                accepted = [int(i) for i in points if is_decrease(vv, int(i), radius=6)
                            and pd.Timestamp("2017-01-01") <= dates[int(i)] <= pd.Timestamp("2025-12-01")]
                result["events"] = [{"change_time": dates[i].isoformat(), "change_index": i,
                                     "Band": "VV_dB"} for i in accepted]
                # ReCuSum has one full-series execution, not W36 annual windows.
                result["windows"] = [{"candidate_scope": "full_series_recursive",
                                      "window_start": dates[0].isoformat(), "window_end": dates[-1].isoformat(),
                                      "valid_observations": int(np.isfinite(vv).sum()),
                                      "execution_status": "completed" if enough else "input_insufficient",
                                      "raw_breakpoints": points, "accepted_event_count": len(accepted)}]
                result["status"] = "completed" if enough else "input_insufficient"
            else:
                raise ValueError(method)
            result["source_months"] = int(present.sum())
            result["error"] = None
        except Exception as exc:
            result.update(status="failed", events=[], windows=[], error=f"{type(exc).__name__}: {exc}")
        result["execution_seconds"] = time.perf_counter() - started
        payloads.append(result)
    return payloads


def execute_batch(batch):
    return [result for row, methods in batch for result in execute_pixel(row, methods)]


def connect(run):
    path = run / "predictions/checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS detections (pixel_track_id TEXT, method TEXT, status TEXT, payload TEXT, PRIMARY KEY(pixel_track_id,method))")
    return db


def legacy_reuse(rows):
    """Reuse the old 456-plot transfer only when execution and inputs are proven.

    A found event alone is insufficient. This path requires the frozen per-key
    execution table, verified receipts, unchanged native cache and source code.
    Older buffer output has no equivalent per-key status table and is not adopted.
    """
    old = PIXELWISE / "runs/pixelwise_20260914_v1"
    receipt_path = old / "cusum_transfer/COMPLETE.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    input_path = old / "inputs/COMPLETE.json"
    input_receipt = json.loads(input_path.read_text(encoding="utf-8"))
    archive_path = SOURCE.parent / "archive_manifest.json"
    for path in (input_path, archive_path):
        if receipt["inputs"].get(str(path)) != sha256(path):
            raise ValueError(f"Legacy execution receipt no longer matches {path}")
    for relative in ("cusum_transfer/events.parquet", "cusum_transfer/pixel_status.parquet"):
        path = old / relative
        if receipt["outputs"].get(str(path)) != sha256(path):
            raise ValueError(f"Legacy execution output changed: {path}")
    for path in (CONFIG, old / "inputs/pixel_index.parquet"):
        if input_receipt["outputs"].get(str(path)) != sha256(path):
            raise ValueError(f"Legacy input changed: {path}")
    archived = json.loads(archive_path.read_text(encoding="utf-8"))["files"]
    hashes = {str(Path(item["archived"]).resolve()): item["sha256"] for item in archived}
    for path in (SOURCE / "windowed_cusum").rglob("*.py"):
        if hashes.get(str(path.resolve())) != sha256(path):
            raise ValueError(f"Frozen archive implementation changed: {path}")
    status = pd.read_parquet(old / "cusum_transfer/pixel_status.parquet").set_index("pixel_track_id")
    event_frame = pd.read_parquet(old / "cusum_transfer/events.parquet")
    grouped = {key: frame for key, frame in event_frame.groupby("pixel_track_id")}
    verified_tracks = set()
    payloads = []
    for row in rows:
        key, track = row["pixel_track_id"], row["track_key"]
        if key not in status.index or status.loc[key, "status"] not in {"evaluated_no_event", "evaluated_with_events"}:
            continue
        if track not in verified_tracks:
            paths = [CACHE / f"{track}.npz", CACHE / f"{track}.pixels.parquet"]
            if not all(input_receipt["inputs"].get(str(path)) == sha256(path) for path in paths):
                continue
            verified_tracks.add(track)
        vv, dates, present = series_for(row)
        events = grouped.get(key, event_frame.iloc[:0])
        if len(events) != int(status.loc[key, "events"]):
            raise ValueError("Legacy executed status/event counts disagree")
        config = paper.load_paper_config(CONFIG)
        seed = stable_seed(f"{key}:VV_dB", config.candidate_seed)
        # One-bootstrap helper rebuilds geometry only. Its statistical values
        # are deliberately discarded and never claimed to be original N1 scores.
        geometry = build_component_ledgers(
            vv, dates, config=ComponentAblationConfig(window_length=36, step=12,
                confidence_threshold=1.0, bootstrap_iterations=1, minimum_observations=12,
                seed=config.candidate_seed, start_year=2017, end_year=2025),
            series_key=f"{key}:VV_dB", series_seed=seed, metadata={}).sliding_candidates
        fields = ["candidate_id", "window_start", "window_end", "decision_start", "decision_end",
                  "valid_observations", "window_observations", "passes_window_calendar", "passes_minimum_observations"]
        windows = geometry[fields].copy()
        windows["accepted_event"] = windows.candidate_id.isin(set(events.candidate_id))
        windows["execution_status"] = np.select(
            [~windows.passes_window_calendar, ~windows.passes_minimum_observations, windows.accepted_event],
            ["outside_output_calendar", "input_insufficient", "reused_completed_event"],
            default="reused_completed_no_event")
        windows["diagnostics_origin"] = "geometry rebuilt; original rejected-candidate scores unavailable"
        payloads.append({**{field: row[field] for field in ("unit_id", "block_id", "pixel_track_id", "track_key", "cache_index")},
                         "method": METHODS[0], "status": "completed", "events": records(events),
                         "windows": records(windows), "error": None, "valid_months": int(np.isfinite(vv).sum()),
                         "source_months": int(present.sum()), "execution_seconds": 0.0,
                         "execution_origin": str(receipt_path), "execution_receipt_sha256": sha256(receipt_path)})
    return payloads


def export_results(run, units, db):
    statuses, windows, events = [], [], []
    by_key = {(key, method): json.loads(payload)
              for key, method, payload in db.execute("SELECT pixel_track_id,method,payload FROM detections")}
    for row in units.to_dict("records"):
        for method in METHODS:
            result = by_key.get((row["pixel_track_id"], method))
            base = {key: row[key] for key in ("unit_id", "block_id", "pixel_track_id", "track_key", "cache_index")}
            base["method"] = method
            if result is None:
                statuses.append({**base, "status": "not_executed", "event_count": None})
                continue
            statuses.append({**base, **{key: result.get(key) for key in
                                      ("status", "error", "valid_months", "execution_seconds", "source_months", "execution_origin")},
                             "event_count": len(result["events"]) if result["status"] == "completed" else None})
            windows.extend({**base, **window} for window in result["windows"])
            events.extend({**base, **event} for event in result["events"])
    out = run / "predictions"
    frames = {"pixel_status": pd.DataFrame(statuses), "window_status": pd.DataFrame(windows),
              "events": pd.DataFrame(events) if events else pd.DataFrame(columns=["unit_id", "block_id", "pixel_track_id", "method", "change_time"])}
    for name, frame in frames.items():
        temporary = out / f"{name}.partial.parquet"
        frame.to_parquet(temporary, index=False)
        temporary.replace(out / f"{name}.parquet")
    status = frames["pixel_status"]
    summary = status.groupby(["block_id", "method", "status"]).size().rename("pixel_count").reset_index()
    summary.to_csv(out / "block_status.csv", index=False, encoding="utf-8-sig")
    print(status.groupby(["method", "status"]).size().to_string(), flush=True)


def run(args):
    frozen = freeze_methods(args.run_dir)
    if args.stage == "freeze":
        print(args.run_dir / "config/freeze_methods.json", flush=True)
        return
    if args.units is None:
        raise ValueError("--units is required for run/export")
    units = pd.read_parquet(args.units)
    required = {"unit_id", "block_id", "pixel_track_id", "track_key", "cache_index"}
    if not required.issubset(units) or units.pixel_track_id.duplicated().any() or units.unit_id.duplicated().any():
        raise ValueError("units require unique native pixel/unit identities and cache fields")
    signature = {"units_sha256": sha256(args.units), "units_path": str(args.units.resolve()),
                 "methods_sha256": sha256(args.run_dir / "config/freeze_methods.json"),
                 "adapter_sha256": sha256(__file__),
                 "cache_sha256": {str(CACHE / f"{track}{suffix}"): sha256(CACHE / f"{track}{suffix}")
                                  for track in sorted(units.track_key.unique())
                                  for suffix in (".npz", ".pixels.parquet")}}
    manifest = args.run_dir / "predictions/input_signature.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != signature:
        raise ValueError("Input/adapter identity differs from checkpoint; use new run")
    if not manifest.exists():
        json_write(manifest, signature)
    with connect(args.run_dir) as db:
        if args.stage == "run":
            selected = units
            if args.block:
                selected = selected[selected.block_id.isin(args.block)]
            if args.pilot_only:
                if "pilot8" not in selected:
                    raise ValueError("pilot8 field required for --pilot-only")
                selected = selected[selected.pilot8.eq(True)]
            selected = selected.sort_values(["track_key", "cache_index"])
            if args.limit:
                selected = selected.head(args.limit)
            done = {(key, method): status for key, method, status in
                    db.execute("SELECT pixel_track_id,method,status FROM detections")}
            # Compatible historical zero-event records are reusable too.
            reuse_rows = [row for row in selected.to_dict("records")
                          if (row["pixel_track_id"], METHODS[0]) not in done]
            if reuse_rows:
                reused = legacy_reuse(reuse_rows)
                db.executemany("INSERT OR REPLACE INTO detections VALUES (?,?,?,?)",
                               [(r["pixel_track_id"], r["method"], r["status"], json.dumps(r)) for r in reused])
                db.commit()
                done.update({(r["pixel_track_id"], r["method"]): r["status"] for r in reused})
                print(f"legacy_executed_pixels_reused={len(reused)}", flush=True)
            tasks = []
            for row in selected.to_dict("records"):
                missing = [method for method in METHODS if (row["pixel_track_id"], method) not in done
                           or (args.retry_failed and done[(row["pixel_track_id"], method)] == "failed")]
                if missing:
                    tasks.append((row, missing))
            batches = [tasks[i:i + args.batch_size] for i in range(0, len(tasks), args.batch_size)]
            print(f"pixels_to_execute={len(tasks)} workers={args.workers} batches={len(batches)}", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending = [pool.submit(execute_batch, batch) for batch in batches]
                for count, future in enumerate(as_completed(pending), 1):
                    results = future.result()
                    db.executemany("INSERT OR REPLACE INTO detections VALUES (?,?,?,?)",
                                   [(r["pixel_track_id"], r["method"], r["status"], json.dumps(r)) for r in results])
                    db.commit()
                    if count % 10 == 0 or count == len(batches):
                        print(f"completed_batches={count}/{len(batches)}", flush=True)
        export_results(args.run_dir, units, db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["freeze", "run", "export"], default="run")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--units", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block", action="append")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.batch_size < 1:
        parser.error("workers and batch size must be positive")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(arguments)
