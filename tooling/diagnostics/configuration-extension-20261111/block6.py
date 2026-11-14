"""Six-month calendar-block null with the frozen first-post detector and TDA."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OLD = ROOT / "runs/validation/balsa_research_execution_20261031_v3/simulation/validation_block3"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain/variants/block3"
OUT = ROOT / "runs/validation/configuration_extension_20261111_v1/block6"
CONFIG = ROOT / "configs/firstpost_w36_d12_s12_b20_20261030_v2.json"
REFERENCE = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030")]
import detector_engine as engine
from lr_cusum.temporal_nulls import calendar_block_permutation_score

BLOCK_LENGTH = 6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    reference = pd.read_csv(REFERENCE)
    cohort = set(reference.sample_id)
    assert len(reference) == 1049 and len(cohort) == 459
    tda_path = ROOT / "tooling/diagnostics/harvest-inventory-20261104/tda.py"
    spec = importlib.util.spec_from_file_location("extension_block6_tda", tda_path)
    tda = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tda
    spec.loader.exec_module(tda)
    units, members, values, frozen_execution, full_nodes = tda.load_inputs("validation")
    selected = units.unit_id.isin(members.loc[members.sample_id.isin(cohort), "unit_id"])
    values = values[selected.to_numpy()]
    units = units.loc[selected].reset_index(drop=True)
    members = members.loc[members.sample_id.isin(cohort)].copy()
    frozen_execution = frozen_execution.loc[units.unit_id]
    full_nodes = full_nodes.loc[full_nodes.sample_id.isin(cohort)].copy()
    assert members.sample_id.nunique() == 459 and set(members.unit_id) == set(units.unit_id)
    paths = [REFERENCE, CONFIG, SOURCE / "master_values.npy", SOURCE / "master_units.parquet",
             SOURCE / "validation_membership.parquet", SOURCE / "validation/pixel_events.parquet",
             SOURCE / "validation/pixel_execution.parquet", SOURCE / "validation/source_members.parquet",
             OLD / "pixel_events.parquet", OLD / "pixel_execution.parquet", FROZEN / "octa/reports.parquet",
             FROZEN / "octa/monthly_activity.parquet", tda_path, Path(engine.__file__),
             ROOT / "src/lr_cusum/temporal_nulls.py", ROOT / "src/lr_cusum/window_geometry.py", Path(__file__)]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    archived_events = pd.read_parquet(OLD / "pixel_events.parquet")
    archived_events = archived_events.loc[archived_events.unit_id.isin(units.unit_id)]
    archived_execution = pd.read_parquet(OLD / "pixel_execution.parquet").set_index("unit_id")

    initialize(3)
    pilot_ids = set(units.groupby("track_key", sort=True).head(2).unit_id)
    pilot_ids.update(archived_events.loc[archived_events.near_threshold_refined, "unit_id"].head(20))
    pilot = units.loc[units.unit_id.isin(pilot_ids)]
    actual_events, actual_execution = unpack(engine.batch(pixel_rows(pilot)))
    columns = [c for c in archived_events if c != "method"]
    normalize_dates(actual_events)
    normalize_dates(archived_events)
    pd.testing.assert_frame_equal(
        actual_events[columns].sort_values("candidate_id").reset_index(drop=True),
        archived_events.loc[archived_events.unit_id.isin(pilot_ids), columns].sort_values("candidate_id").reset_index(drop=True),
        check_dtype=False, check_exact=True)
    for col in ("valid_months", "eligible_year_mask", "pre_null_candidates", "refined_candidates"):
        pd.testing.assert_series_equal(actual_execution.set_index("unit_id")[col].sort_index(),
            archived_execution.loc[pilot.unit_id, col].sort_index(), check_dtype=False, check_exact=True)

    replay_nodes = make_nodes(archived_events, members, full_nodes)
    replay = tda.aggregate_frames(units, members, values, frozen_execution, replay_nodes, OUT / "parity_block3")
    for name, key in (("reports", "prediction_id"), ("monthly_activity", "month_index")):
        expected = pd.read_parquet(FROZEN / "octa" / f"{name}.parquet")
        expected = expected.loc[expected.sample_id.isin(cohort)]
        keys = [key] if name == "reports" else ["sample_id", key]
        pd.testing.assert_frame_equal(replay[name][expected.columns].sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=True)
    save(OUT / "parity_block3.json", dict(status="passed", pilot_pixels=len(pilot),
        pilot_tracks=pilot.track_key.nunique(), accepted_events_exact=True, refined_and_base_scores_exact=True,
        tda_reports_exact=True, tda_reference_stands=459, tda_reports=len(replay["reports"])))
    print(f"Block3 parity passed: {len(pilot)} pixels; {len(replay['reports'])} TDA reports", flush=True)

    rows = pixel_rows(units)
    chunks = [rows[i:i+32] for i in range(0, len(rows), 32)]
    payloads = []
    with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(6,)) as pool:
        for i, results in enumerate(pool.map(engine.batch, chunks), 1):
            payloads.extend(results)
            if i % 64 == 0 or i == len(chunks):
                print(f"Block6 {min(i*32, len(rows))}/{len(rows)} pixels", flush=True)
    events, execution = unpack(payloads)
    normalize_dates(events)
    execution["method"] = "windowed_w36_d12_s12_firstpost_block6_extension_v1"
    execution = execution.set_index("unit_id").loc[units.unit_id]
    for col in ("valid_months", "eligible_year_mask", "pre_null_candidates"):
        pd.testing.assert_series_equal(execution[col], frozen_execution[col], check_dtype=False, check_exact=True)
    events.to_parquet(OUT / "pixel_events.parquet", index=False)
    execution.reset_index().to_parquet(OUT / "pixel_execution.parquet", index=False)
    events[["unit_id", "change_time", "candidate_id", "window_start", "window_end"]].to_parquet(
        OUT / "node_aliases.parquet", index=False)
    nodes = make_nodes(events, members, full_nodes)
    nodes.to_parquet(OUT / "pixel_nodes.parquet", index=False)
    result = tda.aggregate_frames(units, members, values, execution, nodes, OUT / "octa")
    monthly = result["monthly_activity"]
    pd.testing.assert_frame_equal(
        monthly[["sample_id", "month_index", "opportunity_pixels", "total_pixels"]].sort_values(["sample_id", "month_index"]).reset_index(drop=True),
        replay["monthly_activity"][["sample_id", "month_index", "opportunity_pixels", "total_pixels"]].sort_values(["sample_id", "month_index"]).reset_index(drop=True),
        check_dtype=False, check_exact=True)
    assert set(result["source_members"].node_id) == set(nodes.node_id)
    assert result["source_members"].node_id.is_unique
    assert (5*result["reports"].response_pixels >= result["reports"].total_pixels).all()
    eligibility = np.zeros(values.shape, dtype=bool)
    for bit in range(9):
        eligibility[:, 12+12*bit:24+12*bit] = ((execution.eligible_year_mask.to_numpy() >> bit) & 1).astype(bool)[:, None]
    np.savez_compressed(OUT / "monthly_eligibility.npz", eligibility=eligibility,
        unit_id=units.unit_id.to_numpy(dtype=str), months=tda.preparation.MONTHS.astype(str))
    assert inputs == {p: sha(ROOT / p) for p in inputs}
    save(OUT / "receipt.json", dict(status="passed", variant_id="block6", block_length=6,
        alpha=.025, permutations=3999, refined_permutations=19999, refine_interval=[.015,.035],
        candidate_seed=20260820, geometry=dict(W=36,D=12,S=12), firstpost_date_policy=True,
        reference_stands=459, reference_records=1049, pixels=len(units),
        accepted_pixel_events=len(events), distinct_pixel_month_nodes=len(nodes), tda_reports=len(result["reports"]),
        pre_null_candidates=int(execution.pre_null_candidates.sum()),
        refined_candidates=int(execution.refined_candidates.sum()),
        opportunity_equal_baseline=True, block3_exact_replay=True, stp_run=False,
        seconds=perf_counter()-start, input_sha256=inputs))
    print(f"Block6 finished: {len(events)} detections, {len(nodes)} nodes, {len(result['reports'])} reports", flush=True)


def initialize(length):
    global BLOCK_LENGTH
    BLOCK_LENGTH = length
    engine.initialize(SOURCE / "master_values.npy", CONFIG)
    engine.geometry.iid_permutation_score = block_score
    engine.METHOD = f"windowed_w36_d12_s12_firstpost_block{length}_extension_v1"


def block_score(window, amplitude, *, config, seed):
    engine.COUNTS[int(config.replicates != engine.STATE["config"].permutations)] += 1
    return calendar_block_permutation_score(window, amplitude,
        config=replace(config, method=f"calendar_block{BLOCK_LENGTH}", block_length=BLOCK_LENGTH), seed=seed)


def pixel_rows(units):
    return list(units[["unit_id", "pixel_track_id", "series_index"]].itertuples(index=False, name=None))


def unpack(results):
    payloads = [json.loads(payload) for _, payload in results]
    events = pd.DataFrame([event for payload in payloads for event in payload.pop("events")])
    return events, pd.DataFrame(payloads)


def normalize_dates(frame):
    for col in ("change_time", "window_start", "window_end", "decision_start", "decision_end"):
        frame[col] = pd.to_datetime(frame[col])


def make_nodes(events, members, full_nodes):
    events = events.assign(response_month=events.change_time.dt.strftime("%Y-%m"))
    chosen = events.sort_values(["unit_id", "response_month", "candidate_id"], kind="stable").drop_duplicates(["unit_id", "response_month"])
    nodes = chosen[["unit_id", "response_month", "change_index", "window_start", "window_end", "candidate_id"]].rename(columns={"change_index": "month_index"}).copy()
    nodes["sample_id"] = nodes.unit_id.map(members.set_index("unit_id").sample_id)
    frozen = full_nodes.set_index(["unit_id", "response_month"]).node_id
    nodes["node_id"] = [frozen.get((u,m), "node_"+hashlib.sha256(f"{u}|{m}".encode()).hexdigest()[:24]) for u,m in zip(nodes.unit_id,nodes.response_month)]
    nodes["phase_id"] = ""
    assert nodes.node_id.is_unique and nodes.month_index.between(12,119).all()
    return nodes.reset_index(drop=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
