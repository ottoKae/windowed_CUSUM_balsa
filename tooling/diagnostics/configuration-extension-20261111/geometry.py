"""Extend frozen first-post window geometry and pass detections through TDA."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OUT = ROOT / "runs/validation/configuration_extension_20261111_v1/geometry"
CONFIG = ROOT / "configs/firstpost_w36_d12_s12_b20_20261030_v2.json"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain/variants/full"
REFERENCE = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030")]
import detector_engine as engine

spec = importlib.util.spec_from_file_location("extension_tda", ROOT / "tooling/diagnostics/harvest-inventory-20261104/tda.py")
TDA = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = TDA
spec.loader.exec_module(TDA)

ARMS = {"W24_D12_S12": (24, 12, 12), "W36_D12_S6": (36, 12, 6), "W36_D6_S6": (36, 6, 6)}
GEOMETRY = None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = pd.read_csv(REFERENCE)
    cohort = set(refs.sample_id)
    units, membership, values, execution, frozen_nodes = TDA.load_inputs("validation")
    selected = membership.loc[membership.sample_id.isin(cohort)].unit_id
    take = units.unit_id.isin(selected).to_numpy()
    units, values = units.loc[take].reset_index(drop=True), values[take]
    membership = membership.loc[membership.sample_id.isin(cohort)].copy()
    execution = execution.loc[units.unit_id]
    frozen_nodes = frozen_nodes.loc[frozen_nodes.sample_id.isin(cohort)].copy()
    assert len(refs) == 1049 and len(cohort) == membership.sample_id.nunique() == 459
    assert units.unit_id.is_unique and set(units.unit_id) == set(membership.unit_id)
    paths = [SOURCE / name for name in ("master_values.npy", "master_units.parquet", "validation_membership.parquet",
             "validation/pixel_events.parquet", "validation/pixel_execution.parquet", "validation/source_members.parquet")]
    paths += [CONFIG, REFERENCE, Path(__file__), Path(engine.__file__), Path(engine.geometry.__file__),
              ROOT / "src/windowed_cusum/detector/core.py", ROOT / "src/windowed_cusum/detector/temporal_nulls.py", Path(TDA.__file__),
              Path(TDA.preparation.__file__), Path(TDA.baseline.__file__)]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(OUT / "input_manifest.json", dict(input_sha256=protected, stands=459, references=1049,
        pixels=len(units), arms=ARMS, actual_geometry_independent_of_legacy_config=True))
    selfcheck(units, membership, values, execution, frozen_nodes)
    if "--selfcheck-only" in sys.argv:
        return
    for name, dimensions in ARMS.items():
        run_arm(name, dimensions, units, membership, values, execution, frozen_nodes)
    assert protected == {p: sha(ROOT / p) for p in protected}
    save(OUT / "receipt.json", dict(status="passed", variants=list(ARMS), reference_stands=459,
        reference_records=1049, fixed_pixels=len(units), protected_inputs_unchanged=True,
        scoring_config=json.loads(CONFIG.read_text())["detector"], actual_geometries=ARMS,
        date_policy="first_valid_observation_after_first_CUSUM_maximum", output_level="TDA parents",
        detector_and_tda_code_unchanged=True, input_sha256=protected))


def initialize(dimensions):
    global GEOMETRY
    engine.initialize(SOURCE / "master_values.npy", CONFIG)
    # The legacy configuration supplies scoring settings; Geometry owns W, D and S.
    GEOMETRY = engine.geometry.Geometry(*dimensions)


def batch(rows):
    result = []
    for unit_id, identity, index in rows:
        values = engine.STATE["values"][index]
        engine.COUNTS[:] = [0, 0]
        events = engine.geometry.detect_series(values, engine.STATE["dates"], pixel_track_id=identity,
                                               geometry=GEOMETRY, config=engine.STATE["config"])
        base_calls, refined_calls = engine.COUNTS
        records = events.to_dict("records")
        for event in records:
            event.update(unit_id=unit_id, pixel_track_id=identity)
            for key in ("change_time", "window_start", "window_end", "decision_start", "decision_end"):
                event[key] = pd.Timestamp(event[key]).strftime("%Y-%m-%d")
        result.append((identity, json.dumps(dict(unit_id=unit_id, pixel_track_id=identity, series_index=index,
            valid_months=int(np.isfinite(values).sum()), pre_null_candidates=base_calls,
            refined_candidates=refined_calls, events=records), allow_nan=False)))
    return result


def eligibility(values, dimensions):
    allowed = np.zeros(values.shape, dtype=bool)
    rows = []
    geometry = engine.geometry.Geometry(*dimensions)
    for order, (ws, we, ds, de) in enumerate(engine.geometry.windows(values.shape[1], geometry)):
        valid = np.isfinite(values[:, ws:we]).sum(axis=1) >= 12
        start, end = max(12, ds), min(120, de)
        allowed[:, start:end] |= valid[:, None]
        rows.append(dict(window_id=order, window_start=ws, window_end_exclusive=we,
            decision_start=ds, decision_end_exclusive=de, eligible_pixels=int(valid.sum()),
            nominal_W=dimensions[0], nominal_D=dimensions[1], stride_S=dimensions[2]))
    assert allowed[:, :12].sum() == allowed[:, 120:].sum() == 0
    return allowed, pd.DataFrame(rows)


def aggregate(directory, units, membership, values, execution, nodes, allowed):
    directory.mkdir(parents=True, exist_ok=True)
    TDA.preparation.OUT = directory
    frames, templates = TDA.preparation.prepare_phases(units, membership, values, execution, nodes.copy())
    monthly = frames["monthly_activity"]
    opportunities = pd.DataFrame(np.isfinite(values[:, 12:120]) & allowed[:, 12:120], index=units.unit_id)
    opportunities["sample_id"] = membership.set_index("unit_id").sample_id.reindex(units.unit_id).to_numpy()
    counts = opportunities.groupby("sample_id").sum()
    monthly["opportunity_pixels"] = [int(counts.loc[r.sample_id, r.month_index-12]) for r in monthly.itertuples()]
    monthly["opportunity_fraction"] = monthly.opportunity_pixels / monthly.total_pixels
    monthly["state"] = np.where(monthly.active_pixels > 0, "active", np.where(
        2 * monthly.opportunity_pixels >= monthly.total_pixels, "observed_no_candidate", "unobserved_or_ineligible"))
    monthly.to_parquet(directory / "monthly_activity.parquet", index=False)
    edges = TDA.preparation.temporal_edges(frames["phases"], monthly, templates, 1)
    edges.to_parquet(directory / "phase_relations.parquet", index=False)
    result = TDA.baseline.assemble_hflag(frames["phases"], edges, frames["nodes"], monthly, method="R2_TIME")
    for name, frame in result.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    assert set(result["source_members"].node_id) == set(nodes.node_id)
    assert result["source_members"].node_id.is_unique and result["reports"].prediction_id.is_unique
    assert (5 * result["reports"].response_pixels >= result["reports"].total_pixels).all()
    expected_active = nodes.groupby(["sample_id", "month_index"]).unit_id.nunique()
    assert all(row.active_pixels == expected_active.get((row.sample_id, row.month_index), 0) for row in monthly.itertuples())
    return result


def selfcheck(units, membership, values, execution, nodes):
    initialize((36, 12, 12))
    frozen = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    refined = set(frozen.loc[frozen.near_threshold_refined, "unit_id"].head(3))
    selected = pd.concat([units.groupby("track_key").head(1), units.loc[units.unit_id.isin(refined)]]).drop_duplicates("unit_id")
    columns = ["change_index", "cusum_amplitude", "pre_mean", "post_mean", "post_minus_pre", "null_p_value",
               "exceedance_count", "permutations", "near_threshold_refined", "candidate_id"]
    for row in selected.itertuples():
        payload = json.loads(batch([(row.unit_id, row.pixel_track_id, row.series_index)])[0][1])
        actual = pd.DataFrame(payload["events"], columns=columns).sort_values("candidate_id").reset_index(drop=True)
        expected = frozen.loc[frozen.unit_id.eq(row.unit_id), columns].sort_values("candidate_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    for dimensions in ARMS.values():
        initialize(dimensions)
        dates = engine.STATE["dates"]
        series = np.where(dates < pd.Timestamp("2020-07-01"), -8., -12.)
        for missing, expected in ((False, "2020-07-01"), (True, "2020-09-01")):
            current = series.copy()
            if missing:
                current[(dates >= "2020-07-01") & (dates < "2020-09-01")] = np.nan
            accepted = engine.geometry.detect_series(current, dates, pixel_track_id="geometry_known_step",
                                                      geometry=GEOMETRY, config=engine.STATE["config"])
            assert set(accepted.change_time) == {pd.Timestamp(expected)}
        coverage, _ = eligibility(np.ones((1, 126)), dimensions)
        assert coverage[0, 12:120].all()
    allowed, schedule = eligibility(values, (36, 12, 12))
    saved = np.load(FROZEN / "monthly_eligibility.npz")
    order = pd.Index(saved["unit_id"]).get_indexer(units.unit_id)
    np.testing.assert_array_equal(allowed, saved["eligibility"][order])
    result = aggregate(OUT / "baseline_parity/octa", units, membership, values, execution, nodes, allowed)
    expected = pd.read_parquet(FROZEN / "octa/reports.parquet")
    expected = expected.loc[expected.sample_id.isin(membership.sample_id)]
    pd.testing.assert_frame_equal(result["reports"].sort_values("prediction_id").reset_index(drop=True),
        expected.sort_values("prediction_id").reset_index(drop=True), check_dtype=False, check_exact=True)
    assert len(expected) == 1208
    schedule.to_csv(OUT / "baseline_parity/window_schedule.csv", index=False)
    save(OUT / "selfcheck.json", dict(status="passed", baseline_detector_bit_exact_pixels=len(selected),
        baseline_pixel_ids=selected.unit_id.tolist(), baseline_eligibility_bit_exact=True,
        baseline_reports_bit_exact=1208, known_firstpost_and_missing_step_passed=True,
        all_geometries_cover_all_reporting_months=True))
    print(f"Baseline parity passed: {len(selected)} real pixels, eligibility and 1208 TDA reports", flush=True)


def run_arm(name, dimensions, units, membership, values, execution, frozen_nodes):
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    if (directory / "receipt.json").exists():
        return
    start = perf_counter()
    with sqlite3.connect(directory / "pixel_checkpoints.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS pixels (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        done = {row[0] for row in db.execute("SELECT identity FROM pixels")}
        rows = list(units.loc[~units.pixel_track_id.isin(done), ["unit_id", "pixel_track_id", "series_index"]].itertuples(index=False, name=None))
        chunks = [rows[i:i+32] for i in range(0, len(rows), 32)]
        with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(dimensions,)) as pool:
            for i, part in enumerate(pool.map(batch, chunks), 1):
                db.executemany("INSERT INTO pixels VALUES (?, ?)", part)
                db.commit()
                if i % 64 == 0 or i == len(chunks):
                    print(f"{name}: {len(done)+min(32*i,len(rows))}/{len(units)} pixels; {perf_counter()-start:.1f}s", flush=True)
        payloads = [json.loads(row[0]) for row in db.execute("SELECT payload FROM pixels ORDER BY identity")]
    assert len(payloads) == len(units)
    events = pd.DataFrame([event for payload in payloads for event in payload.pop("events")])
    for key in ("change_time", "window_start", "window_end", "decision_start", "decision_end"):
        events[key] = pd.to_datetime(events[key])
    events["response_month"] = events.change_time.dt.strftime("%Y-%m")
    events["method"] = name
    events.to_parquet(directory / "pixel_events.parquet", index=False)
    pd.DataFrame(payloads).to_parquet(directory / "pixel_execution.parquet", index=False)
    events[["unit_id", "response_month", "candidate_id", "window_start", "window_end"]].to_parquet(directory / "node_aliases.parquet", index=False)
    chosen = events.sort_values(["unit_id", "response_month", "candidate_id"], kind="stable").drop_duplicates(["unit_id", "response_month"])
    nodes = chosen[["unit_id", "response_month", "change_index", "window_start", "window_end", "candidate_id"]].rename(columns={"change_index":"month_index"}).copy()
    nodes["sample_id"] = nodes.unit_id.map(membership.set_index("unit_id").sample_id)
    frozen_ids = frozen_nodes.set_index(["unit_id", "response_month"]).node_id
    nodes["node_id"] = [frozen_ids.get((u,m), "node_"+hashlib.sha256(f"{u}|{m}".encode()).hexdigest()[:24])
                        for u,m in zip(nodes.unit_id,nodes.response_month)]
    nodes["phase_id"] = ""
    assert nodes.node_id.is_unique and nodes.month_index.between(12,119).all() and nodes.sample_id.notna().all()
    nodes.to_parquet(directory / "pixel_nodes.parquet", index=False)
    allowed, schedule = eligibility(values, dimensions)
    np.savez_compressed(directory / "monthly_eligibility.npz", eligibility=allowed,
                        unit_id=units.unit_id.to_numpy(dtype=str), months=TDA.preparation.MONTHS.astype(str))
    schedule.to_csv(directory / "window_schedule.csv", index=False)
    result = aggregate(directory / "octa", units, membership, values, execution, nodes, allowed)
    baseline_allowed, _ = eligibility(values, (36,12,12))
    save(directory / "receipt.json", dict(status="passed", actual_geometry=dict(zip(("W","D","S"),dimensions)),
        scoring_config=json.loads(CONFIG.read_text())["detector"], pixels=len(units), reference_stands=459,
        accepted_window_candidates=len(events), unique_pixel_month_nodes=len(nodes), reports=len(result["reports"]),
        pre_null_candidates=sum(p["pre_null_candidates"] for p in payloads),
        refined_candidates=sum(p["refined_candidates"] for p in payloads),
        eligibility_entries_changed=int((allowed!=baseline_allowed).sum()),
        eligibility="finite>=12 in any admitting context; OR overlapping decision intervals; report months 12:120",
        elapsed_seconds=perf_counter()-start, workers=4, unchanged_TDA_rules=True,
        aliases_preserved_and_pixel_months_deduplicated=True,
        output_sha256={str(p.relative_to(directory)):sha(p) for p in directory.rglob("*") if p.is_file() and p.suffix in (".parquet", ".npz", ".csv")}))
    print(f"{name} completed: {len(nodes)} unique nodes, {len(result['reports'])} TDA parents", flush=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
