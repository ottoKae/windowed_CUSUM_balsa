"""Apply Temporal Detection Aggregation (TDA) to the frozen national pixel ledger."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OLD = ROOT / "runs/validation/balsa_harvest_episode_20261101_v3"
BASE = ROOT / "runs/validation/balsa_sts_episode_v4_1_20261102_v1/baselines/methods/R2_TIME"
DEFAULT_OUTPUT = ROOT / "runs/validation/harvest_inventory_20261104_v1/octa"
EPISODE = ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101"
BASE_CODE = ROOT / "tooling/diagnostics/balsa-sts-v4_1-20261102/baselines.py"
sys.path[:0] = [str(EPISODE), str(BASE_CODE.parent)]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preparation = import_file("inventory_frozen_episode_preparation", EPISODE / "run_episode_experiment.py")
baseline = import_file("inventory_frozen_episode_baseline", BASE_CODE)


def aggregate_frames(units, membership, values, execution, nodes, output):
    """Reuse unchanged preparation, G1 edges and R2 assembler; redirect output only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The historical preparation function owns file writes through this path.
    preparation.OUT = output
    frames, templates = preparation.prepare_phases(units, membership, values, execution, nodes.copy())
    edges = preparation.temporal_edges(frames["phases"], frames["monthly_activity"], templates, 1)
    result = baseline.assemble_hflag(frames["phases"], edges, frames["nodes"],
                                   frames["monthly_activity"], method="R2_TIME")
    edges.to_parquet(output / "phase_relations.parquet", index=False)
    for name, frame in result.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    return {**frames, "phase_relations": edges, **result}


def load_inputs(scope):
    membership = pd.read_parquet(SOURCE / f"{scope}_membership.parquet")
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    units = units.loc[units.unit_id.isin(membership.unit_id)].sort_values("unit_id").reset_index(drop=True)
    execution = pd.read_parquet(SOURCE / scope / "pixel_execution.parquet").set_index("unit_id").loc[units.unit_id]
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")[units.series_index.to_numpy()]
    events = pd.read_parquet(SOURCE / scope / "pixel_events.parquet")
    sources = pd.read_parquet(SOURCE / scope / "source_members.parquet")
    assert membership.unit_id.is_unique and units.unit_id.is_unique and sources.node_id.is_unique
    assert set(units.unit_id) == set(membership.unit_id) and len(events) == len(sources)
    assert events.method.eq(preparation.engine.METHOD).all() and execution.method.eq(preparation.engine.METHOD).all()
    np.testing.assert_array_equal(execution.series_index, units.series_index)
    np.testing.assert_array_equal(execution.valid_months, np.isfinite(values).sum(axis=1))
    for bit in range(9):
        finite = np.isfinite(values[:, bit * 12:bit * 12 + 36]).sum(axis=1)
        np.testing.assert_array_equal(((execution.eligible_year_mask.to_numpy() >> bit) & 1).astype(bool), finite >= 12)
    events = events.assign(response_month=events.change_time.dt.strftime("%Y-%m"))
    nodes = sources[["sample_id", "unit_id", "node_id", "response_month"]].merge(
        events[["unit_id", "response_month", "change_index", "window_start", "window_end", "candidate_id"]],
        on=["unit_id", "response_month"], validate="one_to_one").rename(columns={"change_index": "month_index"})
    assert len(nodes) == len(sources) and nodes.node_id.is_unique
    assert nodes.month_index.between(12, 119).all()
    assert nodes.sample_id.eq(nodes.unit_id.map(membership.set_index("unit_id").sample_id)).all()
    np.testing.assert_array_equal(preparation.MONTHS[nodes.month_index.to_numpy()], nodes.response_month)
    nodes["phase_id"] = ""
    return units, membership, values, execution, nodes


def assert_parity(actual):
    """Legacy method-dependent IDs are metadata; compare every scientific column."""
    checks = []
    key_map = dict(nodes=["node_id"], monthly_activity=["sample_id", "month_index"], phases=["phase_id"],
                   phase_template_points=["phase_id", "offset"], phase_template_members=["phase_id", "unit_id"],
                   phase_relations=["edge_id"], components=["component_id"], reports=["prediction_id"],
                   source_members=["node_id"], edge_decisions=["edge_id"], progress=["component_id", "response_month"])
    for name, keys in key_map.items():
        if name == "phase_relations":
            expected = pd.read_parquet(OLD / "phase_relations_B1.parquet").query("G == 1")
        else:
            parent = BASE if name in {"components", "reports", "source_members", "edge_decisions", "progress"} else OLD
            expected = pd.read_parquet(parent / f"{name}.parquet")
        observed = actual[name].loc[actual[name].sample_id.isin(expected.sample_id)].copy()
        columns = [c for c in expected if not c.startswith("legacy_")]
        pd.testing.assert_frame_equal(observed.sort_values(keys).reset_index(drop=True)[columns],
            expected.sort_values(keys).reset_index(drop=True)[columns], check_dtype=False, check_exact=True)
        checks.append(dict(table=name, rows=len(expected), columns=columns,
                           excluded_metadata=[c for c in expected if c.startswith("legacy_")], passed=True))
    return checks


def stand_universe(summary, membership, nodes, result):
    table = summary[["sample_id", "source_row_original", "DPA_DESPRO", "inventory_area_ha", "native_cells", "status"]].copy()
    total = membership.groupby("sample_id").unit_id.nunique()
    np.testing.assert_array_equal(table.sample_id.map(total).fillna(0).astype(int), table.native_cells)
    table = table.rename(columns={"status": "source_processing_status"})
    table["processing_status"] = np.where(table.native_cells.gt(0), "completed", "insufficient_coverage")
    table["total_pixels"] = table.sample_id.map(total).fillna(0).astype(int)
    table["accepted_nodes"] = table.sample_id.map(nodes.groupby("sample_id").size()).fillna(0).astype(int)
    table["n_components"] = table.sample_id.map(result["components"].groupby("sample_id").size()).fillna(0).astype("Int64")
    table["report_count"] = table.sample_id.map(result["reports"].groupby("sample_id").size()).fillna(0).astype("Int64")
    table.loc[table.processing_status.ne("completed"), ["n_components", "report_count"]] = pd.NA
    table["method"] = "R2_TIME"
    table["report_start_month"], table["record_cutoff_month"] = "2017-01", "2025-12"
    table["input_cutoff_month"] = "2026-06"
    return table


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    start = time.perf_counter()
    inputs = [SOURCE / name for name in ("master_units.parquet", "master_values.npy", "national_membership.parquet",
        "validation_membership.parquet", "products/stand_summary.csv", "protocol.json", "method_signature.json",
        "national/complete.json", "validation/complete.json")]
    inputs += [SOURCE / scope / name for scope in ("validation", "national") for name in
               ("pixel_events.parquet", "pixel_execution.parquet", "source_members.parquet")]
    inputs += list(BASE.glob("*.parquet")) + [OLD / f"{name}.parquet" for name in
        ("nodes", "monthly_activity", "phases", "phase_template_points", "phase_template_members", "phase_relations_B1")]
    algorithm_sources = [BASE_CODE] + [EPISODE / name for name in
        ("run_episode_experiment.py", "episode_relations.py", "episode_assembly.py")]
    protected = {str(path): digest(path) for path in inputs + algorithm_sources}
    source_hash = digest(__file__)
    write_json(output / "input_manifest.json", dict(protected_sha256=protected, adapter_sha256=source_hash))
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), *algorithm_sources]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    receipt = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        method="R2_TIME", display_name="TDA", cusum_rerun=False, candidate_changes=False,
        G=1, H=108, H12="flag_only", support="5 * distinct_pixels >= original_total_pixels",
        representative_month="floor(median(all_source_node_month_indices)+0.5)",
        unknown_gap_policy="frozen classify_gap", report_period=["2017-01", "2025-12"],
        adapter_sha256=source_hash)
    write_json(output / "execution_receipt.json", receipt)
    try:
        print("TDA: frozen 468-stand preparation and assembly parity", flush=True)
        validation = aggregate_frames(*load_inputs("validation"), output / "validation_parity")
        checks = assert_parity(validation)
        write_json(output / "validation_parity/checks.json", checks)
        print("TDA: national frozen candidate aggregation", flush=True)
        units, membership, values, execution, nodes = load_inputs("national")
        assert len(units) == 151549 and membership.sample_id.nunique() == 3364 and len(nodes) == 181619
        result = aggregate_frames(units, membership, values, execution, nodes, output)
        checks_national = assert_parity(result)
        write_json(output / "national_contains_validation_checks.json", checks_national)
        reports, sources = result["reports"], result["source_members"]
        assert sources.node_id.is_unique and set(sources.node_id) == set(nodes.node_id)
        assert reports.prediction_id.is_unique and (5 * reports.response_pixels >= reports.total_pixels).all()
        formal = sources.loc[sources.destination_type.eq("episode")]
        np.testing.assert_array_equal(reports.prediction_id.map(formal.groupby("prediction_id").unit_id.nunique()), reports.response_pixels)
        universe = stand_universe(pd.read_csv(SOURCE / "products/stand_summary.csv"), membership, nodes, result)
        assert len(universe) == 3498 and universe.sample_id.is_unique
        assert universe.processing_status.eq("completed").sum() == 3364 and universe.report_count.isna().sum() == 134
        universe.to_parquet(output / "stand_universe.parquet", index=False)
        universe.to_csv(output / "stand_summary.csv", index=False)
        changed = [path for path, expected in protected.items() if digest(path) != expected]
        assert not changed and digest(__file__) == source_hash, f"Input or source changed: {changed}"
        receipt.update(status="passed", inventory_stands=3498, evaluable_stands=3364,
            unavailable_stands=134, members=len(units), accepted_nodes=len(nodes),
            components=len(result["components"]), reports=len(reports),
            stands_with_reports=reports.sample_id.nunique(), zero_report_stands=int(universe.report_count.eq(0).sum()),
            source_destinations=sources.destination_type.value_counts().to_dict(),
            validation_parity_passed=True, national_contains_validation_exact_scientific_fields=True,
            protected_inputs_changed=[], full_history_precision=None,
            scope_note="algorithmic reports; no new independent harvest truth or national accuracy estimate")
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start,
            output_sha256={str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                           if path.is_file() and path.name != "execution_receipt.json"})
        write_json(output / "execution_receipt.json", receipt)
    return receipt


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x:
        x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False, default=str))
