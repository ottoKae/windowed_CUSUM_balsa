"""Run the frozen official EWMACD monitor per FC001 master cell, then B20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "runs/validation/existing_labels_w36_20260927_v1/inputs"
DEFAULT_RUN = ROOT / "runs/validation/pixel_b20_benchmark_window_20261004_ewmacd_v1"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import fc001_validation_detection as fc  # noqa: E402
from harvest_bounded_event_aggregation import build_bounded_events  # noqa: E402
from lr_cusum.ewmacd_adapter import run_ewmacd, ewma_trace  # noqa: E402

METHOD = "ewmacd_hist2_l03_p1"
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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(args):
    files = [Path(__file__), Path(run_ewmacd.__code__.co_filename), INPUTS / "units.parquet",
             INPUTS / "membership.parquet"]
    payload = {"schema_version": 1, "method": METHOD, "monitor_years": list(YEARS),
               "history_years": 2, "lambda": .3, "persistence": 1,
               "minimum_history_fraction": .75, "minimum_monitor_observations": 6,
               "batch_size": args.batch_size, "limit": args.limit, "gee_project": args.project,
               "files": {str(p.resolve()): digest(p) for p in files}}
    destination = args.run_dir / "config/configuration_manifest.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != payload:
        raise ValueError("EWMACD frozen inputs/configuration changed; use a new run directory")
    if not destination.exists():
        atomic_json(destination, payload)


def prepare(units):
    series = []; calendar = None; support = []
    for trajectory_index, row in enumerate(units.to_dict("records")):
        vv, dates, _ = fc.series_for(row); series.append(vv); calendar = dates
        for year in YEARS:
            history = (dates.year >= year - 2) & (dates.year < year)
            monitor = dates.year == year
            h_count = int(np.isfinite(vv[history]).sum()); m_count = int(np.isfinite(vv[monitor]).sum())
            support.append({"trajectory_index": trajectory_index, "unit_id": row["unit_id"],
                            "pixel_track_id": row["pixel_track_id"], "monitor_year": year,
                            "history_valid_months": h_count, "monitor_valid_months": m_count,
                            "eligible": h_count >= 18 and m_count >= 6})
    return np.asarray(series, dtype=np.float64), pd.DatetimeIndex(calendar), pd.DataFrame(support)


def request(values, dates, project):
    error = None
    for attempt in range(3):
        try:
            return run_ewmacd(values, dates, history_end=int(dates[-1].year), weight=.3, persistence=1)
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise error


def main(args):
    args.run_dir.mkdir(parents=True, exist_ok=True); freeze(args)
    import ee
    ee.Initialize(project=args.project)
    units = pd.read_parquet(INPUTS / "units.parquet").sort_values(["track_key", "cache_index"]).reset_index(drop=True)
    if args.limit is not None:
        units = units.iloc[:args.limit].copy()
    values, dates, support = prepare(units)
    support.to_parquet(args.run_dir / "monitoring_support.parquet", index=False)
    event_rows = []; completed = set(); raw_root = args.run_dir / "raw"
    for year in YEARS:
        identities = support.loc[(support.monitor_year == year) & support.eligible, "trajectory_index"].to_numpy(int)
        mask = (dates.year >= year - 2) & (dates.year <= year); calendar = dates[mask]
        for start in range(0, len(identities), args.batch_size):
            ids = identities[start:start + args.batch_size]
            destination = raw_root / str(year) / f"{start:06d}.json"
            if destination.exists():
                records = json.loads(destination.read_text(encoding="utf-8"))
            else:
                records = request(values[ids, None, :][:, :, mask], calendar, args.project)
                atomic_json(destination, records)
            trace = ewma_trace(records, values[ids, None, :][:, :, mask], calendar)
            trace["trajectory_index"] = ids[trace.trajectory_index.to_numpy(int)]
            for trajectory_index in ids:
                completed.add((int(trajectory_index), year))
            alarm = trace.loc[(trace.month.dt.year == year) & trace.score.lt(0)].sort_values("month")
            alarm = alarm.groupby("trajectory_index", as_index=False).head(1)
            for event in alarm.itertuples(index=False):
                unit = units.iloc[int(event.trajectory_index)]; date = pd.Timestamp(event.month)
                raw = json.dumps([METHOD, unit.pixel_track_id, str(date.date())], separators=(",", ":"))
                event_rows.append({"unit_id": unit.unit_id, "block_id": unit.block_id,
                                   "pixel_track_id": unit.pixel_track_id, "track_key": unit.track_key,
                                   "cache_index": int(unit.cache_index), "method": METHOD,
                                   "change_time": date, "score": float(event.score),
                                   "candidate_id": METHOD + ":" + hashlib.sha256(raw.encode()).hexdigest()[:24]})
            print(f"ewmacd year={year} batch={start // args.batch_size + 1}/{int(np.ceil(len(identities)/args.batch_size))}",
                  flush=True)
    eligible = support.loc[support.eligible]
    if completed != set(zip(eligible.trajectory_index, eligible.monitor_year)):
        raise ValueError("EWMACD completed support differs from eligible pixel-years")
    events = pd.DataFrame(event_rows)
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
    parser.add_argument("--project", default="gee-r1536803768")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
