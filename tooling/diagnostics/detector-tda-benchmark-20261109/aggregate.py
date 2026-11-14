"""Apply unchanged production TDA to each detector's nodes and observation domain."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from common import ROOT, SOURCE, OUT, MONTHS, load_inputs, load_module, save_detector, write_json, digest

TDA = load_module("benchmark_production_tda", ROOT / "tooling/diagnostics/harvest-inventory-20261104/tda.py")
BASELINE = "windowed_cusum_w36_firstpost"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain/variants/full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.baseline:
        baseline_inputs()
    methods = [args.method] if args.method else [p.parent.name for p in sorted((OUT / "detectors").glob("*/detector_receipt.json"))]
    for method in methods:
        target = OUT / "methods" / method
        if (target / "aggregation_receipt.json").exists():
            print(f"{method}: aggregation already completed", flush=True)
            continue
        aggregate(method, target)


def baseline_inputs():
    units, _, _ = load_inputs()
    path = FROZEN / "monthly_eligibility.npz"
    with np.load(path) as saved:
        order = pd.Index(saved["unit_id"]).get_indexer(units.unit_id)
        if np.any(order < 0):
            raise ValueError("Frozen windowed CuSum membership mismatch")
        eligible = saved["eligibility"][order]
    events = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    save_detector(BASELINE, events, eligible, {
        "family": "windowed_cusum", "label": "windowed CuSum",
        "parameters": {"W": 36, "D": 12, "S": 12, "alpha_ts": 0.025},
        "date_contract": "next valid observation after cumulative maximum; frozen First-post v2",
        "eligibility_contract": "admissible decision month within a context with at least 12 finite observations",
        "coverage_type": "frozen_context_eligibility",
        "status": "frozen_candidates_reused; no detection rerun",
    }, [SOURCE / "validation/pixel_events.parquet", path, SOURCE / "method_signature.json"])


def temporal_inputs(units, membership, values, eligible, nodes):
    """Construct calendar/phase tables; all joining and admission use frozen code."""
    positions = pd.Series(np.arange(len(units)), index=units.unit_id)
    node_groups = nodes.groupby("sample_id").groups
    monthly_rows, phase_rows = [], []
    for sample_id, members in membership.groupby("sample_id", sort=True):
        selected = nodes.loc[node_groups.get(sample_id, [])]
        indices = positions.loc[members.unit_id].to_numpy()
        total = len(indices)
        counts = selected.groupby("month_index").unit_id.nunique().reindex(range(12, 120), fill_value=0).to_numpy()
        observed = (eligible[indices, 12:120] & np.isfinite(values[indices, 12:120])).sum(axis=0)
        states = np.where(counts > 0, "active", np.where(2 * observed >= total,
                          "observed_no_candidate", "unobserved_or_ineligible"))
        for month, count, known, state in zip(range(12, 120), counts, observed, states):
            monthly_rows.append({"sample_id": sample_id, "month_index": month, "month": str(MONTHS[month]),
                "active_pixels": int(count), "opportunity_pixels": int(known), "opportunity_fraction": float(known / total),
                "state": state, "total_pixels": total})
        for order, (a, b) in enumerate(TDA.preparation.active_phases(counts)):
            start, end = a + 12, b + 12
            phase_id = "phase_" + hashlib.sha256(f"{sample_id}|{start}|{end}".encode()).hexdigest()[:24]
            local = selected.loc[selected.month_index.between(start, end)]
            nodes.loc[local.index, "phase_id"] = phase_id
            phase_rows.append({"sample_id": sample_id, "phase_id": phase_id, "phase_order": order,
                "start_index": start, "end_index": end, "start_month": str(MONTHS[start]),
                "end_month": str(MONTHS[end]), "total_pixels": total,
                "n_nodes": len(local), "n_response_pixels": local.unit_id.nunique()})
    monthly = pd.DataFrame(monthly_rows)
    phases = pd.DataFrame(phase_rows, columns=["sample_id", "phase_id", "phase_order", "start_index", "end_index",
        "start_month", "end_month", "total_pixels", "n_nodes", "n_response_pixels"])
    # TDA disables shape-based vetoes. Empty templates avoid introducing a morphology module.
    templates = {p: np.full(13, np.nan) for p in phases.phase_id}
    edges = TDA.preparation.temporal_edges(phases, monthly, templates, 1)
    if edges.empty:
        edges = pd.DataFrame(columns=["sample_id", "left_phase_id", "right_phase_id", "edge_id", "relation", "distance"])
    return phases, monthly, edges


def aggregate(method, target):
    start = perf_counter()
    units, membership, values = load_inputs()
    source = OUT / "detectors" / method
    receipt = json.loads((source / "detector_receipt.json").read_text(encoding="utf-8"))
    for name, expected in receipt["output_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Detector output changed: {method}/{name}")
    events = pd.read_parquet(source / "pixel_events.parquet")
    with np.load(source / "monthly_eligibility.npz") as saved:
        np.testing.assert_array_equal(saved["unit_id"], units.unit_id.to_numpy())
        np.testing.assert_array_equal(saved["months"], MONTHS.astype(str).to_numpy())
        eligible = saved["eligibility"]
    nodes = events[["unit_id", "response_month", "month_index"]].drop_duplicates().sort_values(
        ["unit_id", "month_index"], kind="stable").reset_index(drop=True)
    nodes["sample_id"] = nodes.unit_id.map(membership.set_index("unit_id").sample_id)
    old = pd.read_parquet(SOURCE / "validation/source_members.parquet")
    old_ids = old.set_index(["unit_id", "response_month"]).node_id
    nodes["node_id"] = [old_ids.get((u, m), "node_" + hashlib.sha256(f"{u}|{m}".encode()).hexdigest()[:24])
                        for u, m in zip(nodes.unit_id, nodes.response_month)]
    nodes["phase_id"] = ""
    phases, monthly, edges = temporal_inputs(units, membership, values, eligible, nodes)
    result = TDA.baseline.assemble_hflag(phases, edges, nodes, monthly, method="R2_TIME")
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in {"pixel_nodes": nodes, "phases": phases, "monthly_activity": monthly,
                        "phase_relations": edges, **result}.items():
        frame.to_parquet(target / f"{name}.parquet", index=False)
    aliases = events[["unit_id", "response_month"]].reset_index(names="source_event_row")
    aliases = aliases.merge(nodes[["unit_id", "response_month", "node_id"]], on=["unit_id", "response_month"], validate="many_to_one")
    aliases.to_parquet(target / "node_aliases.parquet", index=False)
    baseline_parity = None
    if method == BASELINE:
        for name, keys in [("reports", ["prediction_id"]), ("components", ["component_id"]),
                           ("progress", ["component_id", "response_month"])]:
            expected = pd.read_parquet(FROZEN / "octa" / f"{name}.parquet")
            pd.testing.assert_frame_equal(result[name].sort_values(keys).reset_index(drop=True),
                expected.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=True)
        expected_monthly = pd.read_parquet(FROZEN / "octa/monthly_activity.parquet")
        pd.testing.assert_frame_equal(monthly, expected_monthly[monthly.columns], check_dtype=False, check_exact=True)
        baseline_parity = True
    record = {"status": "completed", "method_id": method, "detector_receipt_sha256": digest(source / "detector_receipt.json"),
        "n_raw_nominations": len(events), "n_unique_nodes": len(nodes), "n_parents_execution": len(result["reports"]),
        "fixed_members": 26890, "execution_stands": 468, "node_conservation": len(nodes) == len(result["source_members"]),
        "source_destinations": result["source_members"].destination_type.value_counts().to_dict(),
        "monthly_states": monthly.state.value_counts().to_dict(),
        "baseline_exact_parity": baseline_parity, "seconds": perf_counter() - start,
        "tda": {"G": 1, "support": 0.2, "observed_fraction": 0.5, "H12": "flag only", "STS_veto": False},
        "source_sha256": {str(p): digest(p) for p in [Path(__file__), Path(TDA.baseline.__file__),
            Path(TDA.preparation.__file__), ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101/episode_assembly.py"]},
    }
    write_json(target / "aggregation_receipt.json", record)
    print(f"{method}: {len(nodes)} nodes -> {len(result['reports'])} TDA parents ({record['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
