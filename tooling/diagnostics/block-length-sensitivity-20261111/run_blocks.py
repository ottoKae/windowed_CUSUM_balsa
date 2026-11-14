"""Add L4/L5 by reusing the frozen block6 adapter, detector and TDA kernels."""
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tooling/diagnostics/configuration-extension-20261111"))
import block6 as frozen
import numpy as np
import pandas as pd

OUT = ROOT / "runs/validation/block_length_sensitivity_20261111_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("Loading frozen cohort and TDA inputs")
    reference = pd.read_csv(frozen.REFERENCE)
    cohort = set(reference.sample_id)
    assert len(reference) == 1049 and len(cohort) == 459
    old_receipt = json.loads((frozen.OUT / "receipt.json").read_text(encoding="utf-8"))
    assert old_receipt["status"] == "passed"
    for path, expected in old_receipt["input_sha256"].items():
        assert frozen.sha(ROOT / path) == expected, path
    tda_path = ROOT / "tooling/diagnostics/harvest-inventory-20261104/tda.py"
    spec = importlib.util.spec_from_file_location("block_length_tda", tda_path)
    tda = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tda
    spec.loader.exec_module(tda)
    units, members, values, execution, full_nodes = tda.load_inputs("validation")
    selected = units.unit_id.isin(members.loc[members.sample_id.isin(cohort), "unit_id"])
    values = values[selected.to_numpy()]
    units = units.loc[selected].reset_index(drop=True)
    members = members.loc[members.sample_id.isin(cohort)].copy()
    execution = execution.loc[units.unit_id]
    full_nodes = full_nodes.loc[full_nodes.sample_id.isin(cohort)].copy()
    assert len(units) == 25555 and members.sample_id.nunique() == 459
    assert set(units.unit_id) == set(members.unit_id)
    paths = [frozen.REFERENCE, frozen.CONFIG, frozen.SOURCE / "master_values.npy",
        frozen.SOURCE / "master_units.parquet", frozen.SOURCE / "validation_membership.parquet",
        frozen.SOURCE / "validation/pixel_events.parquet", frozen.SOURCE / "validation/pixel_execution.parquet",
        frozen.SOURCE / "validation/source_members.parquet", Path(frozen.__file__),
        Path(frozen.engine.__file__), tda_path, ROOT / "src/windowed_cusum/detector/core.py",
        ROOT / "src/windowed_cusum/detector/temporal_nulls.py", ROOT / "src/windowed_cusum/detector/window_geometry.py", Path(__file__)]
    inputs = {str(path.relative_to(ROOT)): frozen.sha(path) for path in paths}
    frozen.save(OUT / "input_manifest.json", dict(input_sha256=inputs, reference_stands=459,
        reference_records=1049, fixed_pixels=25555, added_lengths=[4,5], main_configuration_unchanged=True))
    log("Frozen inputs loaded; replaying small L3/L6 samples")
    checks = []
    for length, directory in ((3, frozen.OLD), (6, frozen.OUT)):
        archived_events = pd.read_parquet(directory / "pixel_events.parquet")
        archived_events = archived_events.loc[archived_events.unit_id.isin(units.unit_id)]
        archived_execution = pd.read_parquet(directory / "pixel_execution.parquet").set_index("unit_id")
        pilot_ids = set(units.groupby("track_key", sort=True).head(1).unit_id)
        pilot_ids.update(archived_events.loc[archived_events.near_threshold_refined,"unit_id"].head(8))
        pilot = units.loc[units.unit_id.isin(pilot_ids)]
        frozen.initialize(length)
        actual_events, actual_execution = frozen.unpack(frozen.engine.batch(frozen.pixel_rows(pilot)))
        frozen.normalize_dates(actual_events)
        frozen.normalize_dates(archived_events)
        cols = [col for col in archived_events if col != "method"]
        pd.testing.assert_frame_equal(actual_events[cols].sort_values("candidate_id").reset_index(drop=True),
            archived_events.loc[archived_events.unit_id.isin(pilot_ids),cols].sort_values("candidate_id").reset_index(drop=True),
            check_dtype=False, check_exact=True)
        for col in ("valid_months","eligible_year_mask","pre_null_candidates","refined_candidates"):
            pd.testing.assert_series_equal(actual_execution.set_index("unit_id")[col].sort_index(),
                archived_execution.loc[pilot.unit_id,col].sort_index(),check_dtype=False,check_exact=True)
        checks.append(dict(length=length,pixels=len(pilot),tracks=pilot.track_key.nunique(),
            scores_dates_and_execution_exact=True))
    frozen.save(OUT / "pilot_parity.json", dict(status="passed",checks=checks))
    log("L3/L6 score, date and execution replay passed")
    baseline_monthly = pd.read_parquet(frozen.OUT / "octa/monthly_activity.parquet")
    opportunity_cols = ["sample_id","month_index","opportunity_pixels","total_pixels"]
    expected_opportunity = baseline_monthly[opportunity_cols].sort_values(["sample_id","month_index"]).reset_index(drop=True)
    eligibility = np.zeros(values.shape,dtype=bool)
    for bit in range(9):
        eligibility[:,12+12*bit:24+12*bit] = ((execution.eligible_year_mask.to_numpy() >> bit) & 1).astype(bool)[:,None]
    rows = frozen.pixel_rows(units)
    chunks = [rows[i:i+32] for i in range(0,len(rows),32)]
    for length in (4,5):
        started = perf_counter()
        destination = OUT / f"block{length}"
        destination.mkdir(exist_ok=True)
        log(f"L{length}: starting {len(rows)} pixels with four workers")
        payloads = []
        with ProcessPoolExecutor(max_workers=4,initializer=frozen.initialize,initargs=(length,)) as pool:
            for i,results in enumerate(pool.map(frozen.engine.batch,chunks),1):
                payloads.extend(results)
                if i % 64 == 0 or i == len(chunks):
                    log(f"L{length}: {min(i*32,len(rows))}/{len(rows)} pixels")
        events, observed_execution = frozen.unpack(payloads)
        frozen.normalize_dates(events)
        observed_execution["method"] = f"windowed_w36_d12_s12_firstpost_block{length}_extension_v1"
        observed_execution = observed_execution.set_index("unit_id").loc[units.unit_id]
        for col in ("valid_months","eligible_year_mask","pre_null_candidates"):
            pd.testing.assert_series_equal(observed_execution[col],execution[col],check_dtype=False,check_exact=True)
        events.to_parquet(destination / "pixel_events.parquet",index=False)
        observed_execution.reset_index().to_parquet(destination / "pixel_execution.parquet",index=False)
        events[["unit_id","change_time","candidate_id","window_start","window_end"]].to_parquet(
            destination / "node_aliases.parquet",index=False)
        nodes = frozen.make_nodes(events,members,full_nodes)
        nodes.to_parquet(destination / "pixel_nodes.parquet",index=False)
        result = tda.aggregate_frames(units,members,values,observed_execution,nodes,destination / "octa")
        pd.testing.assert_frame_equal(result["monthly_activity"][opportunity_cols].sort_values(
            ["sample_id","month_index"]).reset_index(drop=True),expected_opportunity,check_dtype=False,check_exact=True)
        assert set(result["source_members"].node_id) == set(nodes.node_id)
        assert result["source_members"].node_id.is_unique
        assert (5*result["reports"].response_pixels >= result["reports"].total_pixels).all()
        np.savez_compressed(destination / "monthly_eligibility.npz",eligibility=eligibility,
            unit_id=units.unit_id.to_numpy(dtype=str),months=tda.preparation.MONTHS.astype(str))
        assert inputs == {p:frozen.sha(ROOT / p) for p in inputs}
        receipt = dict(status="passed",variant_id=f"block{length}",block_length=length,
            alpha=.025,permutations=3999,refined_permutations=19999,refine_interval=[.015,.035],
            candidate_seed=20260820,geometry=dict(W=36,D=12,S=12),firstpost_date_policy=True,
            reference_stands=459,reference_records=1049,pixels=len(units),
            accepted_pixel_events=len(events),distinct_pixel_month_nodes=len(nodes),tda_reports=len(result["reports"]),
            pre_null_candidates=int(observed_execution.pre_null_candidates.sum()),
            refined_candidates=int(observed_execution.refined_candidates.sum()),
            opportunity_equal_baseline=True,prior_block3_and_block6_exact_replay=True,stp_run=False,
            source_members_all_assigned_once=True,parent_support_fraction=.2,seconds=perf_counter()-started,input_sha256=inputs)
        frozen.save(destination / "receipt.json",receipt)
        log(f"L{length} completed: {len(events)} detections, {len(result['reports'])} TDA reports")


def log(message):
    print(message,flush=True)
    with (OUT / "run.log").open("a",encoding="utf-8") as stream:
        stream.write(message+"\n")


if __name__ == "__main__":
    main()
