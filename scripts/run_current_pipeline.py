"""Run the paper's first-post detector, TDA and STP on supplied or synthetic pixels."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lr_cusum import star_segmentation as stp
from lr_cusum.harvest_quarter_aggregation import _identifier


def run(input_dir: Path | None, output: Path, workers=1, stp_replay=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=False)
    if input_dir is None:
        input_dir = output / "synthetic_input"
        input_dir.mkdir()
        make_demo(input_dir)
    units = pd.read_csv(input_dir / "units.csv", dtype={"unit_id": str, "track_key": str})
    membership = pd.read_csv(input_dir / "membership.csv", dtype={"unit_id": str, "sample_id": str})
    values = np.load(input_dir / "monthly_vv.npy", allow_pickle=False)
    if units.empty or values.shape != (len(units), 126):
        raise ValueError("Expected one row per unit and 126 columns (2016-01 through 2026-06)")
    if not units.unit_id.is_unique or not membership.unit_id.is_unique:
        raise ValueError("Use unique native pixels, with one stand membership per pixel")
    if set(units.unit_id) != set(membership.unit_id):
        raise ValueError("Units and membership must contain the same pixels")
    identities = units.pixel_track_id if "pixel_track_id" in units else units.unit_id
    if identities.isna().any() or not identities.is_unique:
        raise ValueError("Pixel randomization identities must be nonmissing and unique")
    detector = load("public_firstpost", "balsa-national-firstpost-20261030/detector_engine.py")
    tda = load("public_tda", "harvest-inventory-20261104/tda.py")
    assembly = load("public_episode_schema", "balsa-harvest-episode-20261101/episode_assembly.py")
    detector.initialize(input_dir / "monthly_vv.npy", ROOT / "configs/current_pipeline.json")
    rows = [(unit, identity, i) for i, (unit, identity) in enumerate(zip(units.unit_id, identities))]
    if workers == 1:
        payloads = [json.loads(payload) for _, payload in detector.batch(rows)]
    else:
        payloads = []
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                 initargs=(input_dir / "monthly_vv.npy",)) as pool:
            batches = [rows[i:i+64] for i in range(0, len(rows), 64)]
            for i, batch in enumerate(pool.map(detect_batch, batches), 1):
                payloads.extend(json.loads(payload) for _, payload in batch)
                if i % 25 == 0 or i == len(batches):
                    print(f"Detected {min(i*64, len(rows))}/{len(rows)} pixels", flush=True)
    execution = pd.DataFrame([{k: v for k, v in p.items() if k != "events"} for p in payloads])
    event_columns = list(dict.fromkeys(detector.geometry.COLUMNS + [
        "unit_id", "pixel_track_id", "method", "peak_marker_month", "peak_to_firstpost_gap_months"]))
    events = pd.DataFrame([event for p in payloads for event in p["events"]], columns=event_columns)
    reports = pd.DataFrame(columns=assembly.COMPONENT_COLUMNS + [
        "span_over_12", "legacy_component_id", "legacy_prediction_id"])
    groups = pd.DataFrame(columns=stp.ROUND_COLUMNS)
    execution.to_csv(output / "pixel_execution.csv", index=False)
    events.to_csv(output / "pixel_detections.csv", index=False)
    if events.empty:
        summary = dict(pixels=len(units), pixel_detections=0, tda_events=0, stp_groups=0)
    else:
        nodes = events[["unit_id", "change_time", "change_index", "window_start", "window_end", "candidate_id"]].rename(columns={"change_index": "month_index"}).copy()
        nodes["response_month"] = nodes.pop("change_time").str[:7]
        nodes["sample_id"] = nodes.unit_id.map(membership.set_index("unit_id").sample_id)
        nodes["node_id"] = [_identifier("node_", r.sample_id, r.unit_id, r.response_month)
                            for r in nodes.itertuples()]
        nodes["window_start"] = pd.to_datetime(nodes.window_start)
        nodes["window_end"] = pd.to_datetime(nodes.window_end)
        nodes["phase_id"] = ""
        temporal = tda.aggregate_frames(units, membership, values, execution, nodes, output / "tda")
        reports = temporal["reports"]
        formal = temporal["source_members"].loc[
            temporal["source_members"].component_id.isin(reports.component_id)]
        summary = dict(pixels=len(units), pixel_detections=len(events), tda_events=len(reports), stp_groups=0)
        if len(reports):
            if stp_replay is None:
                spatial = stp.segment(reports, formal, units, stp.Config())
            else:
                adapter = load("public_stp_replay", "harvest-inventory-20261104/stp.py")
                adapter.OLD = stp_replay / "legacy"
                adapter.A1 = stp_replay / "calibration"
                adapter.CHAIN = stp_replay / "chain"
                adapter.MASTER = stp_replay / "master_units.parquet"
                spatial = adapter.run_stp(reports, formal, units[["unit_id", "native_row", "native_col", "track_key"]], output / "stp_replay",
                                          set(membership.sample_id))["tables"]
            (output / "stp").mkdir()
            for name, frame in spatial.items():
                frame.to_parquet(output / "stp" / f"{name}.parquet", index=False)
            groups = spatial["rounds"]
            assert set(spatial["node_ownership"].node_id) == set(formal.node_id)
            assert spatial["node_ownership"].node_id.is_unique
            summary["stp_groups"] = len(spatial["rounds"])
    reports.to_csv(output / "tda_events.csv", index=False)
    groups.to_csv(output / "stp_groups.csv", index=False)
    monthly, stands = observation_summary(units, membership, values, execution, events, reports)
    monthly.to_csv(output / "stand_monthly_observation.csv", index=False)
    stands.to_csv(output / "stand_summary.csv", index=False)
    summary.update(stands=len(stands),
                   unassessable_stands=int(stands.observation_status.eq("unassessable").sum()),
                   partially_observed_stands=int(stands.observation_status.eq("partially_observed").sum()))
    summary.update(synthetic=input_dir == output / "synthetic_input",
                   stp_groups_are_additional_harvests=False, historical_cached_random_stream_replayed=stp_replay is not None)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def initialize_worker(values_path):
    global worker_detector
    worker_detector = load("public_worker_detector", "balsa-national-firstpost-20261030/detector_engine.py")
    worker_detector.initialize(values_path, ROOT / "configs/current_pipeline.json")


def detect_batch(rows):
    return worker_detector.batch(rows)


def observation_summary(units, membership, values, execution, events, reports):
    """Expose the frozen monthly observation rule, including stands with no events."""
    months = np.arange(12, 120)
    dates = pd.period_range("2017-01", "2025-12", freq="M").astype(str)
    row_index = pd.Series(np.arange(len(units)), index=units.unit_id)
    records, summaries = [], []
    for stand, members in membership.groupby("sample_id", sort=True):
        indices = row_index.loc[members.unit_id].to_numpy()
        mask = execution.eligible_year_mask.to_numpy()[indices]
        eligible = ((mask[:, None] >> ((months - 12) // 12)) & 1).astype(bool)
        opportunity = (np.isfinite(values[indices][:, months]) & eligible).sum(axis=0)
        selected = events.loc[events.unit_id.isin(members.unit_id)]
        active = selected.groupby("change_index").unit_id.nunique().reindex(months, fill_value=0).to_numpy()
        states = np.where(active > 0, "active", np.where(opportunity >= .5 * len(members),
                          "observed_no_candidate", "unobserved_or_ineligible"))
        known = int(np.count_nonzero(states != "unobserved_or_ineligible"))
        records.extend(dict(sample_id=stand, month=month, active_pixels=int(a),
                            opportunity_pixels=int(v), total_pixels=len(members), state=state)
                       for month, a, v, state in zip(dates, active, opportunity, states))
        summaries.append(dict(sample_id=stand, total_pixels=len(members),
            observed_or_active_months=known, unknown_months=108-known,
            observation_status="unassessable" if known == 0 else "observed" if known == 108 else "partially_observed",
            observation_reason="no_observed_or_active_report_month" if known == 0 else
                               "some_months_unknown" if known < 108 else "all_report_months_observed_or_active",
            report_count=int(reports.sample_id.eq(stand).sum()) if known else pd.NA))
    table = pd.DataFrame(summaries)
    table["report_count"] = table.report_count.astype("Int64")
    return pd.DataFrame(records), table


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tooling/diagnostics" / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_demo(directory):
    """Invented 6x6 stand with two removals, staged across its two halves."""
    rows, cols = np.indices((6, 6)).reshape(2, -1)
    units = pd.DataFrame(dict(unit_id=[f"demo_pixel_{i:02d}" for i in range(36)],
                              native_row=rows, native_col=cols, track_key="demo_track"))
    membership = units[["unit_id"]].assign(sample_id="demo_stand", source_row_original=1)
    rng = np.random.default_rng(20261114)
    t = np.arange(126)
    values = -6 + rng.normal(0, .12, (36, 126)) + .12 * np.sin(2 * np.pi * t / 12)
    onset = np.where(cols < 3, 30, 34)
    onset[(rows == 5) & (cols == 0)] = 32
    for i, first in enumerate(onset):
        for start in (first, first + 48):
            age = t - start
            values[i] -= np.where((age >= 0) & (age < 18), 6 * (1 - age / 18), 0)
    units.to_csv(directory / "units.csv", index=False)
    membership.to_csv(directory / "membership.csv", index=False)
    np.save(directory / "monthly_vv.npy", values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Omit to generate invented data")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is not overwritten")
    parser.add_argument("--workers", type=int, default=1, help="Independent detector worker processes")
    parser.add_argument("--stp-replay", type=Path, help="Frozen dataset reproduction/stp_replay directory")
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output, args.workers, args.stp_replay), indent=2))
