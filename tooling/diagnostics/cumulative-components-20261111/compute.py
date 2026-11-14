"""Frozen first-post candidates -> cumulative controls -> unchanged TDA."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"),
               str(ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"),
               str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030")]
import detector_engine as engine
import fullchain_ablation as chain

TDA = chain.TDA
SOURCE = TDA.SOURCE
OUT = ROOT / "runs/validation/cumulative_components_20261111_v1"
TRACE = ROOT / "runs/validation/balsa_firstpost_result_analysis_20261030_v1/pixel_window_trace.parquet"
CONFIG = ROOT / "configs/firstpost_w36_d12_s12_b20_20261030_v2.json"
DATES = pd.date_range("2016-01-01", "2026-06-01", freq="MS")
VARIANTS = {"F0_full_record": "no_local_window", "F1_local_windows": "no_central_gate",
            "F2_decision_interval": "full"}


def main():
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    units, membership, values, execution, frozen_nodes = TDA.load_inputs("validation")
    assert len(units) == 26890 and membership.sample_id.nunique() == 468
    protected = [SOURCE / name for name in ("master_values.npy", "master_units.parquet",
        "validation_membership.parquet", "validation/pixel_events.parquet",
        "validation/pixel_execution.parquet", "validation/source_members.parquet")]
    protected += [TRACE, CONFIG, ROOT / "manuscript_JAG/sections/03_methods.tex"]
    dependencies = [Path(__file__), Path(engine.__file__), Path(chain.__file__), Path(TDA.__file__),
        Path(TDA.preparation.__file__), Path(TDA.baseline.__file__),
        *sorted((ROOT / "src/lr_cusum").glob("*.py"))]
    hashes = {str(p): chain.digest(p) for p in protected + dependencies}
    protocol = dict(status="running", stages=list(VARIANTS) + ["F3_direction", "F4_permutation"],
        calendar=["2016-01", "2026-06"], report_period=["2017-01", "2025-12"],
        execution_stands=468, evaluation_stands=459, references=1049, native_pixels=26890,
        date_policy="first valid observation after the first global cumulative maximum",
        F0="one full-record context; no direction or permutation screening",
        F1="W36/S12; no central, direction or permutation screening",
        F2="W36/D12/S12; no direction or permutation screening",
        F3="reuse frozen no_permutation_gate", F4="reuse frozen full",
        TDA="unchanged G1, distinct-member support>=20%, unknown boundaries unresolved, H12 flag only",
        eligibility="context finite>=12 and permitted report month, independent of acceptance",
        input_sha256=hashes)
    chain.write_json(OUT / "protocol.json", protocol)
    trace = pd.read_parquet(TRACE)
    trace["candidate_id"] = trace.pixel_track_id + ":VV_dB|sliding|w" + trace.logical_window.astype(str)
    assert len(trace) == 9 * len(units) and not trace.duplicated(["unit_id", "logical_window"]).any()
    full = full_record_candidates(units, values)
    full.to_parquet(OUT / "full_record_candidate_ledger.parquet", index=False)
    choices = {"F0_full_record": select_candidates(full),
               "F1_local_windows": select_candidates(trace),
               "F2_decision_interval": select_candidates(trace, central=True)}
    global_accepted = pd.read_parquet(ROOT / "runs/validation/balsa_firstpost_result_analysis_20261030_v1/contributions/global_pixel_events.parquet")
    raw_global = choices["F0_full_record"].set_index("unit_id")
    for a, b in (("candidate_index", "change_index"), ("cusum_amplitude", "cusum_amplitude"),
                 ("pre_mean", "pre_mean"), ("post_mean", "post_mean")):
        np.testing.assert_array_equal(raw_global.loc[global_accepted.unit_id, a], global_accepted[b])
    receipts = []
    for variant, selected in choices.items():
        destination = OUT / "variants" / variant
        destination.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(destination / "selected_window_candidates.parquet", index=False)
        nodes, aliases = make_nodes(selected, membership, frozen_nodes)
        nodes.to_parquet(destination / "pixel_nodes.parquet", index=False)
        aliases.to_parquet(destination / "node_aliases.parquet", index=False)
        eligible = chain.eligibility_matrix(VARIANTS[variant], values)
        np.savez_compressed(destination / "monthly_eligibility.npz", eligibility=eligible,
            unit_id=units.unit_id.to_numpy(dtype=str), months=DATES.strftime("%Y-%m").to_numpy(dtype=str))
        print(f"{variant}: {len(selected)} nominations, {len(nodes)} unique pixel-months", flush=True)
        result = aggregate(variant, units, membership, values, execution, nodes, eligible)
        receipts.append(dict(variant=variant, window_candidates=len(selected), pixel_months=len(nodes),
            non_downward_candidates=int(selected.post_minus_pre.ge(0).sum()),
            parents_all_468=len(result["reports"]), mode="computed"))
        if variant == "F2_decision_interval":
            compare_direction_stage(destination, result)
        chain.write_json(destination / "receipt.json", dict(status="passed", **receipts[-1]))
    pd.DataFrame(receipts).to_csv(OUT / "execution_summary.csv", index=False)
    assert all(chain.digest(p) == h for p, h in hashes.items()), "Protected input changed"
    protocol.update(status="passed", seconds=perf_counter()-started,
        global_frozen_candidates_exactly_reproduced=len(global_accepted),
        F2_F3_scientific_identity=True, new_permutation_scores=0, source_inputs_unchanged=True)
    chain.write_json(OUT / "protocol.json", protocol)
    print(f"Candidate and TDA stages passed in {protocol['seconds']:.1f}s", flush=True)


def full_record_candidates(units, values):
    """Use the production statistic and date mapping; do not run its screening."""
    records = []
    for row, vector in zip(units.itertuples(index=False), values):
        valid = np.flatnonzero(np.isfinite(vector))
        result = dict(unit_id=row.unit_id, pixel_track_id=row.pixel_track_id, series_index=row.series_index,
            candidate_id=row.pixel_track_id + ":VV_dB|global|w0", logical_window=0,
            window_start_index=0, window_end_exclusive=len(vector), decision_start_index=12,
            decision_end_exclusive=120, n_valid=len(valid), candidate_index=-1,
            n_pre=0, n_post=0, pre_mean=np.nan, post_mean=np.nan, post_minus_pre=np.nan,
            candidate_in_decision=False, cusum_amplitude=np.nan)
        if len(valid) >= 12:
            amplitude, first_post = engine.first_post_statistic(vector[valid])
            result["cusum_amplitude"] = amplitude
            if first_post >= 0:
                candidate = int(valid[first_post])
                before, after = vector[:candidate], vector[candidate:]
                pre_mean, post_mean = float(np.nanmean(before)), float(np.nanmean(after))
                result.update(candidate_index=candidate, n_pre=int(np.isfinite(before).sum()),
                    n_post=int(np.isfinite(after).sum()), pre_mean=pre_mean, post_mean=post_mean,
                    post_minus_pre=post_mean-pre_mean, candidate_in_decision=12 <= candidate < 120)
        records.append(result)
    return pd.DataFrame(records)


def select_candidates(trace, central=False):
    eligible = (trace.n_valid.ge(12) & trace.candidate_index.between(12, 119)
        & trace.n_pre.gt(0) & trace.n_post.gt(0)
        & np.isfinite(trace.pre_mean) & np.isfinite(trace.post_mean))
    if central:
        eligible &= trace.candidate_in_decision.eq(True)
    return trace.loc[eligible].copy()


def make_nodes(trace, membership, frozen_nodes):
    selected = trace.copy()
    selected["month_index"] = selected.candidate_index.astype(int)
    selected["response_month"] = DATES[selected.month_index].strftime("%Y-%m")
    selected["window_start"] = DATES[selected.window_start_index.astype(int)]
    selected["window_end"] = DATES[selected.window_end_exclusive.astype(int)-1]
    aliases = selected[["unit_id", "response_month", "candidate_id", "window_start", "window_end"]].copy()
    chosen = selected.sort_values(["unit_id", "response_month", "candidate_id"], kind="stable").drop_duplicates(["unit_id", "response_month"])
    nodes = chosen[["unit_id", "response_month", "month_index", "window_start", "window_end", "candidate_id"]].reset_index(drop=True)
    nodes["sample_id"] = nodes.unit_id.map(membership.set_index("unit_id").sample_id)
    frozen = frozen_nodes.set_index(["unit_id", "response_month"]).node_id
    nodes["node_id"] = [frozen.get((u,m), "node_"+hashlib.sha256(f"{u}|{m}".encode()).hexdigest()[:24])
                        for u,m in zip(nodes.unit_id, nodes.response_month)]
    nodes["phase_id"] = ""
    assert nodes.node_id.is_unique and nodes.sample_id.notna().all()
    return nodes, aliases


def aggregate(variant, units, membership, values, execution, nodes, eligible):
    destination = OUT / "variants" / variant / "octa"
    destination.mkdir(parents=True, exist_ok=True)
    TDA.preparation.OUT = destination
    frames, templates = TDA.preparation.prepare_phases(units, membership, values, execution, nodes.copy())
    monthly = frames["monthly_activity"]
    opportunity = pd.DataFrame(np.isfinite(values[:,12:120]) & eligible[:,12:120], index=units.unit_id)
    opportunity["sample_id"] = membership.set_index("unit_id").sample_id.reindex(units.unit_id).to_numpy()
    counts = opportunity.groupby("sample_id").sum()
    monthly["opportunity_pixels"] = [int(counts.loc[r.sample_id, r.month_index-12]) for r in monthly.itertuples()]
    monthly["opportunity_fraction"] = monthly.opportunity_pixels / monthly.total_pixels
    monthly["state"] = np.where(monthly.active_pixels.gt(0), "active", np.where(
        2*monthly.opportunity_pixels >= monthly.total_pixels, "observed_no_candidate", "unobserved_or_ineligible"))
    chain.check_monthly(monthly, membership, units, values, eligible, nodes)
    monthly.to_parquet(destination / "monthly_activity.parquet", index=False)
    edges = TDA.preparation.temporal_edges(frames["phases"], monthly, templates, 1)
    result = TDA.baseline.assemble_hflag(frames["phases"], edges, frames["nodes"], monthly, method="R2_TIME")
    edges.to_parquet(destination / "phase_relations.parquet", index=False)
    for name, frame in result.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)
    assert set(result["source_members"].node_id) == set(nodes.node_id)
    assert result["source_members"].node_id.is_unique
    assert (5*result["reports"].response_pixels >= result["reports"].total_pixels).all()
    return result


def compare_direction_stage(destination, result):
    frozen = chain.OUT / "variants/no_permutation_gate"
    actual_nodes = pd.read_parquet(destination / "pixel_nodes.parquet")
    frozen_nodes = pd.read_parquet(frozen / "pixel_nodes.parquet")
    columns = [c for c in actual_nodes if c != "phase_id"]
    pd.testing.assert_frame_equal(actual_nodes[columns].sort_values("node_id").reset_index(drop=True),
        frozen_nodes[columns].sort_values("node_id").reset_index(drop=True), check_dtype=False, check_exact=True)
    with np.load(destination / "monthly_eligibility.npz") as actual, np.load(frozen / "monthly_eligibility.npz") as expected:
        np.testing.assert_array_equal(actual["unit_id"], expected["unit_id"])
        np.testing.assert_array_equal(actual["eligibility"], expected["eligibility"])
    for name in ("reports", "source_members"):
        expected = pd.read_parquet(frozen / "octa" / f"{name}.parquet")
        # Historical alias IDs document the B20 source, not TDA membership or timing.
        expected = expected.drop(columns=["source_alias_node_id"], errors="ignore")
        pd.testing.assert_frame_equal(result[name][expected.columns], expected,
                                      check_dtype=False, check_exact=True)
    chain.write_json(OUT / "direction_stage_parity.json", dict(status="passed", distinct_nodes=len(actual_nodes),
        reports_all_468=len(result["reports"]), nodes=True, eligibility=True, reports=True, source_members=True,
        excluded_metadata=["source_alias_node_id"]))


if __name__ == "__main__":
    main()
