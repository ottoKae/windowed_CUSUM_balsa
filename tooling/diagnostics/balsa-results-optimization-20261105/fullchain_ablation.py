"""Frozen A-group candidates -> configuration-specific opportunity -> TDA/STP."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from run import ROOT, RUN, SOURCE, OLD, A1, digest, load_module, write_json

TDA = load_module("fullchain_inventory_tda", ROOT / "tooling/diagnostics/harvest-inventory-20261104/tda.py")
STP = load_module("fullchain_inventory_stp", ROOT / "tooling/diagnostics/harvest-inventory-20261104/stp.py")
stars = STP.stars
OUT = RUN / "fullchain"
# Frozen storage IDs and directories remain unchanged; current labels are TDA/STP.
VARIANTS = ("full", "no_local_window", "no_central_gate", "no_permutation_gate", "no_direction_gate", "cstr_off")
BATCHES = {"Q1": VARIANTS[1:-1], "Q2": ("W48", "W60"),
    "Q3": ("VH", "VV_union_VH", "VV_intersection_VH"), "Q4": ("block3",),
    "Q5": ("alpha_0.005", "alpha_0.01", "alpha_0.05", "alpha_0.10")}
ALL_VARIANTS = VARIANTS + sum((BATCHES[q] for q in ("Q2", "Q3", "Q4", "Q5")), ())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "temporal", "spatial", "seeds"])
    parser.add_argument("--variant", type=lambda value: "stp_off" if value=="cstr_off" else value,
                        choices=tuple("stp_off" if value=="cstr_off" else value for value in ALL_VARIANTS))
    parser.add_argument("--batch", choices=BATCHES, default="Q1")
    args = parser.parse_args()
    if args.variant == "stp_off":
        args.variant = "cstr_off"
    assert json.loads((RUN/"audit/receipt.json").read_text())["status"] == "passed"
    if args.stage != "temporal":
        gate = json.loads((RUN/"synthetic/implementation_gate.json").read_text())
        assert gate["status"] == "passed" and gate["production_sha256"] == digest(ROOT/"src/lr_cusum/star_segmentation.py")
    if args.stage == "baseline":
        baseline()
    elif args.stage == "temporal":
        assert args.variant and args.variant not in ("full", "cstr_off")
        temporal(args.variant)
    elif args.stage == "spatial":
        spatial(args.batch)
    else:
        seeds()


def baseline():
    directory = OUT/"variants/full"
    if (directory/"receipt.json").exists():
        raise FileExistsError("Baseline already completed")
    units, membership, values, execution, nodes = TDA.load_inputs("validation")
    start = perf_counter()
    result = TDA.aggregate_frames(units, membership, values, execution, nodes, directory/"octa")
    checks = TDA.assert_parity(result)
    eligibility = eligibility_matrix("full", values)
    check_monthly(result["monthly_activity"], membership, units, values, eligibility, nodes)
    save_inputs(directory, nodes, eligibility, units, [SOURCE/"validation/pixel_events.parquet"])
    formal = result["source_members"].loc[result["source_members"].component_id.isin(result["reports"].component_id)]
    replay = STP.run_stp(result["reports"], formal, units, directory/"cstr", set(membership.sample_id), True)
    write_json(directory/"cstr/execution_receipt.json", dict(status="passed",**replay["metadata"]))
    write_json(OUT/"baseline_parity.json", dict(status="passed", octa=checks,
        cstr=replay["parity"].to_dict("records"), eligibility_equal=True))
    write_json(directory/"receipt.json",dict(status="completed", variant="full", n_nodes=len(nodes),
        n_parents=len(result["reports"]), n_children=len(replay["tables"]["rounds"]),
        seconds=perf_counter()-start, production_sha256=digest(ROOT/"src/lr_cusum/star_segmentation.py")))


def eligibility_matrix(variant, values):
    allowed = np.zeros(values.shape, dtype=bool)
    if variant == "no_local_window":
        allowed[:,12:120] = (np.isfinite(values).sum(axis=1)>=12)[:,None]
    else:
        for w in range(9):
            width = int(variant[1:]) if variant in ("W48", "W60") else 36
            margin = (width-12)//2
            ws, we = max(0,12+12*w-margin), min(values.shape[1],24+12*w+margin)
            eligible = np.isfinite(values[:,ws:we]).sum(axis=1)>=12
            start,end = (max(12,12*w),min(120,12*w+36)) if variant == "no_central_gate" else (12+12*w,24+12*w)
            allowed[:,start:end] |= eligible[:,None]
    return allowed


def check_monthly(monthly, membership, units, values, eligible, nodes):
    opportunities = pd.DataFrame(np.isfinite(values[:,12:120]) & eligible[:,12:120],index=units.unit_id)
    opportunities["sample_id"] = membership.set_index("unit_id").sample_id.reindex(units.unit_id).to_numpy()
    counts = opportunities.groupby("sample_id").sum()
    expected = np.array([counts.loc[r.sample_id,r.month_index-12] for r in monthly.itertuples()])
    np.testing.assert_array_equal(monthly.opportunity_pixels,expected)
    active = nodes.groupby(["sample_id","month_index"]).unit_id.nunique()
    expected_active = np.array([active.get((r.sample_id,r.month_index),0) for r in monthly.itertuples()])
    np.testing.assert_array_equal(monthly.active_pixels,expected_active)


def temporal(variant):
    directory = OUT/"variants"/variant
    if (directory/"temporal_receipt.json").exists():
        raise FileExistsError(f"Temporal result already exists: {variant}")
    directory.mkdir(parents=True,exist_ok=True)
    start = perf_counter()
    units, membership, values, execution, full_nodes = TDA.load_inputs("validation")
    events_path, ledger_path = candidate_paths(variant)
    events,ledger = pd.read_parquet(events_path),pd.read_parquet(ledger_path)
    before = {str(p.relative_to(ROOT)):digest(p) for p in (events_path,ledger_path)}
    events["response_month"] = pd.to_datetime(events.change_time).dt.strftime("%Y-%m")
    # Multiple windows can accept the same pixel-month. Keep all aliases but one scientific node.
    aliases = events[["unit_id","response_month","candidate_id","window_start","window_end"]].copy()
    aliases.to_parquet(directory/"node_aliases.parquet",index=False)
    chosen = events.sort_values(["unit_id","response_month","candidate_id"],kind="stable").drop_duplicates(["unit_id","response_month"])
    nodes = ledger[["sample_id","unit_id","node_id","response_month"]].merge(
        chosen[["unit_id","response_month","change_index","window_start","window_end","candidate_id"]],
        on=["unit_id","response_month"],validate="one_to_one").rename(columns={"change_index":"month_index"})
    assert len(nodes)==len(ledger)==len(chosen)
    frozen = full_nodes.set_index(["unit_id","response_month"]).node_id
    keys = list(zip(nodes.unit_id,nodes.response_month))
    nodes["source_alias_node_id"] = nodes.node_id
    nodes["node_id"] = [frozen.get(key,"node_"+hashlib.sha256(f"{u}|{m}".encode()).hexdigest()[:24]) for key,(u,m) in zip(keys,keys)]
    assert nodes.node_id.is_unique and nodes.month_index.between(12,119).all()
    assert nodes.sample_id.eq(nodes.unit_id.map(membership.set_index("unit_id").sample_id)).all()
    full_keys=set(zip(full_nodes.unit_id,full_nodes.month_index)); here=set(zip(nodes.unit_id,nodes.month_index))
    if variant in ("no_central_gate","no_permutation_gate"):
        assert full_keys <= here
    if variant == "no_direction_gate":
        assert full_keys == here
    if variant in ("VV_union_VH", "alpha_0.05", "alpha_0.10"):
        assert full_keys <= here
    if variant in ("VV_intersection_VH", "alpha_0.005", "alpha_0.01"):
        assert here <= full_keys
    nodes["phase_id"]=""
    if variant in BATCHES["Q3"]:
        mask = json.loads((OUT/"polarization_mask_identity.json").read_text())
        assert mask["status"]=="passed" and mask["reuse_eligibility_permitted"]
        for p,h in mask["input_sha256"].items():assert digest(Path(p))==h
    if variant == "VH":
        polar = ROOT/"runs/validation/balsa_results_ablation_20261101_v1/polarization"
        axes = pd.read_parquet(polar/"units.parquet").set_index("unit_id")
        values = np.load(polar/"vh_values.npy",mmap_mode="r")[axes.loc[units.unit_id,"local_series_index"].to_numpy()]
    eligible=eligibility_matrix(variant,values)
    if variant in ("W48", "W60"):
        original = pd.read_parquet(events_path.with_name("pixel_execution.parquet")).set_index("unit_id")
        packed = sum(eligible[:,12+12*w].astype(int)*(1<<w) for w in range(9))
        np.testing.assert_array_equal(packed,original.loc[units.unit_id,"eligible_year_mask"].to_numpy())
        width=int(variant[1:]); margin=(width-12)//2
        decision=pd.to_datetime(events.decision_start).dt.to_period("M").astype("int64")-pd.Period("2016-01").ordinal
        actual_start=pd.to_datetime(events.window_start).dt.to_period("M").astype("int64")-pd.Period("2016-01").ordinal
        actual_end=pd.to_datetime(events.window_end).dt.to_period("M").astype("int64")-pd.Period("2016-01").ordinal+1
        np.testing.assert_array_equal(actual_start,np.maximum(0,decision-margin))
        np.testing.assert_array_equal(actual_end,np.minimum(126,decision+12+margin))
    save_inputs(directory,nodes,eligible,units,[events_path,ledger_path])
    TDA.preparation.OUT=directory/"octa"
    TDA.preparation.OUT.mkdir(exist_ok=True)
    frames,templates=TDA.preparation.prepare_phases(units,membership,values,execution,nodes.copy())
    monthly=frames["monthly_activity"]
    opportunity=pd.DataFrame(np.isfinite(values[:,12:120]) & eligible[:,12:120],index=units.unit_id)
    opportunity["sample_id"]=membership.set_index("unit_id").sample_id.reindex(units.unit_id).to_numpy()
    bystand=opportunity.groupby("sample_id").sum()
    monthly["opportunity_pixels"]=[int(bystand.loc[r.sample_id,r.month_index-12]) for r in monthly.itertuples()]
    monthly["opportunity_fraction"]=monthly.opportunity_pixels/monthly.total_pixels
    monthly["state"]=np.where(monthly.active_pixels>0,"active",np.where(
        2*monthly.opportunity_pixels>=monthly.total_pixels,"observed_no_candidate","unobserved_or_ineligible"))
    check_monthly(monthly,membership,units,values,eligible,nodes)
    monthly.to_parquet(directory/"octa/monthly_activity.parquet",index=False)
    monthly.to_parquet(directory/"monthly_opportunity.parquet",index=False)
    edges=TDA.preparation.temporal_edges(frames["phases"],monthly,templates,1)
    result=TDA.baseline.assemble_hflag(frames["phases"],edges,frames["nodes"],monthly,method="R2_TIME")
    edges.to_parquet(directory/"octa/phase_relations.parquet",index=False)
    for name,frame in result.items():frame.to_parquet(directory/"octa"/f"{name}.parquet",index=False)
    assert set(result["source_members"].node_id)==set(nodes.node_id) and result["source_members"].node_id.is_unique
    assert (5*result["reports"].response_pixels>=result["reports"].total_pixels).all()
    assert before=={p:digest(ROOT/p) for p in before}
    write_json(directory/"temporal_receipt.json",dict(status="completed",variant=variant,n_nodes=len(nodes),
        n_parents=len(result["reports"]),added_nodes=len(here-full_keys),lost_nodes=len(full_keys-here),
        input_sha256=before,seconds=perf_counter()-start,detector_rerun=False))
    print(f"{variant}: {len(nodes)} nodes, {len(result['reports'])} TDA parents",flush=True)


def candidate_paths(variant):
    root=ROOT/"runs/validation"
    filename="pixel_events.parquet"
    if variant=="no_local_window":
        p=root/"balsa_firstpost_result_analysis_20261030_v1/contributions/global_pixel_events.parquet"
        return p,p.with_name("global_source_members.parquet")
    if variant in ("W48","W60"):
        parent=root/"balsa_research_execution_20261031_v3/windows"/variant.lower()
    elif variant in BATCHES["Q3"]:
        names=dict(VH="vh_firstpost",VV_union_VH="vv_union_vh_firstpost",VV_intersection_VH="vv_intersection_vh_firstpost")
        parent=root/"balsa_results_ablation_20261101_v1/polarization"/names[variant]
        filename="aggregation_input_events.parquet"
    elif variant=="block3":
        parent=root/"balsa_research_execution_20261031_v3/simulation/validation_block3"
    elif variant.startswith("alpha_"):
        parent=root/"balsa_alpha_sensitivity_20261104_v1"/f"alpha_{float(variant[6:]):g}"
    else:parent=root/"balsa_results_ablation_20261101_v1/components"/variant
    return parent/filename,parent/"source_members.parquet"


def save_inputs(directory,nodes,eligible,units,paths):
    directory.mkdir(parents=True,exist_ok=True)
    nodes.to_parquet(directory/"pixel_nodes.parquet",index=False)
    np.savez_compressed(directory/"monthly_eligibility.npz",eligibility=eligible,
                        unit_id=units.unit_id.to_numpy(dtype=str),months=TDA.preparation.MONTHS.astype(str))
    write_json(directory/"eligibility_axes.json",dict(shape=list(eligible.shape),
        eligibility="context finite>=12 AND permitted report month; no p/peak/acceptance condition",
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def signature(report,local,units,config):
    indexed = units if units.index.name == "unit_id" else units.set_index("unit_id")
    positions=indexed.loc[sorted(local.unit_id.unique()),["native_row","native_col","track_key"]]
    data=dict(nodes=local[["unit_id","month_index"]].sort_values(["unit_id","month_index"]).values.tolist(),
        positions=positions.reset_index().values.tolist(),parent_pixels=int(report.total_pixels),
        config=asdict(config),protocol="frozen_prefix_then_Q1_sorted_signatures_v1")
    data["config"].pop("method")
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def spatial(batch_name="Q1"):
    assert json.loads((OUT/"baseline_parity.json").read_text())["status"]=="passed"
    units=pd.read_parquet(SOURCE/"master_units.parquet")
    indexed_units=units.set_index("unit_id")
    config=stars.Config()
    rng=np.random.default_rng(config.seed)
    base_reports=pd.read_parquet(OLD/"reports.parquet")
    base_members=pd.read_parquet(OLD/"source_members.parquet")
    base_members=base_members.loc[base_members.component_id.isin(base_reports.component_id)]
    frozen_cache=STP.calibration_cache(pd.read_parquet(A1/"calibration_A1.parquet"))
    prefix_order=[]
    replay=STP.segment_sequence(base_reports,base_members,units.set_index("unit_id",drop=False),rng,
                                 frozen_cache,prefix_order,"frozen_prefix")
    STP.compare_frozen_chain(replay)
    cache={}
    bg=dict(tuple(base_members.groupby("component_id",sort=False)))
    for report in base_reports.itertuples():cache[signature(report,bg[report.component_id],indexed_units,config)]=frozen_cache[report.component_id]
    for earlier in list(BATCHES)[:list(BATCHES).index(batch_name)]:
        previous=OUT/"calibration/20261102"/earlier
        assert json.loads((previous/"receipt.json").read_text())["status"]=="completed"
        assert json.loads((previous/"rng_audit.json").read_text())["status"]=="passed"
        saved=json.loads((previous/"checkpoint.json").read_text())
        pool=json.loads((previous/"pool.json").read_text())
        assert pool["prefix_state"]==rng.bit_generator.state
        assert saved["signatures"]==pool["signatures"] and set(saved["tests"])==set(pool["signatures"])
        restore_cache_keys(saved["tests"])
        assert not set(saved["tests"]) & set(cache)
        cache.update(saved["tests"])
        rng.bit_generator.state=saved["rng_state"]
    pending,variants={},{}
    usage=[]
    for variant in BATCHES[batch_name]:
        directory=OUT/"variants"/variant
        assert json.loads((directory/"temporal_receipt.json").read_text())["status"]=="completed"
        reports=pd.read_parquet(directory/"octa/reports.parquet")
        members=pd.read_parquet(directory/"octa/source_members.parquet")
        members=members.loc[members.component_id.isin(reports.component_id)]
        grouped=dict(tuple(members.groupby("component_id",sort=False)))
        mapping={}
        for report in reports.itertuples():
            key=signature(report,grouped[report.component_id],indexed_units,config)
            mapping[report.component_id]=key
            if key not in cache:pending.setdefault(key,(report,grouped[report.component_id]))
        usage.append(dict(variant=variant,parents=len(reports),previous_calibration_reused=sum(k in cache for k in mapping.values()),
                          new_signatures=len(set(mapping.values())-set(cache))))
        variants[variant]=(reports,members,mapping)
    batch=OUT/"calibration/20261102"/batch_name
    batch.mkdir(parents=True,exist_ok=True)
    keys=sorted(pending)
    pool=dict(signatures=keys,config=asdict(config),prefix_state=rng.bit_generator.state)
    if (batch/"pool.json").exists():assert json.loads((batch/"pool.json").read_text())==pool
    else:write_json(batch/"pool.json",pool)
    checkpoint=batch/"checkpoint.json"
    done={}
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text());assert saved["signatures"]==keys
        done=saved["tests"];rng.bit_generator.state=saved["rng_state"]
        restore_cache_keys(done)
    start=perf_counter()
    for i,key in enumerate(keys):
        if key in done:continue
        report,local=pending[key]
        row,zones,rounds,owned,flat=stars.segment_component(report.component_id,report.sample_id,
            report.prediction_id,local,units.loc[units.unit_id.isin(local.unit_id)],int(report.total_pixels),config,rng)
        frame=pd.DataFrame([dict(component_id=report.component_id,**flat)])
        done[key]=STP.calibration_cache(frame)[report.component_id]
        if (i+1)%25==0 or i+1==len(keys):
            temp=checkpoint.with_suffix(".new")
            write_json(temp,dict(signatures=keys,tests=done,rng_state=rng.bit_generator.state))
            temp.replace(checkpoint)
            print(f"{batch_name} calibration {i+1}/{len(keys)}; {perf_counter()-start:.1f}s",flush=True)
    cache.update(done)
    pd.DataFrame(usage).to_csv(batch/"variant_usage.csv",index=False)
    for variant,(reports,members,mapping) in variants.items():
        directory=OUT/"variants"/variant/"cstr";directory.mkdir(exist_ok=True)
        local_units=units.loc[units.unit_id.isin(members.unit_id)]
        result=stars.segment(reports,members,local_units,config,lambda component:cache[mapping[component]])
        STP.check_source_conservation(result,reports,members)
        for name,frame in result.items():frame.to_parquet(directory/f"{name}.parquet",index=False)
        write_json(directory/"execution_receipt.json",dict(status="passed",parents=len(reports),
            children=len(result["rounds"]),split_parents=int(result["stands"].destination_type.eq("zone_rounds").sum()),
            source_conservation=True,calibration_batch=batch_name,seed=config.seed))
        print(f"{variant}: {len(result['rounds'])} STP children",flush=True)
    write_json(batch/"receipt.json",dict(status="completed",new_unique_signatures=len(keys),
        rng_final=rng.bit_generator.state,seconds=perf_counter()-start))


def restore_cache_keys(tests):
    for value in tests.values():
        for field in ("observed","means","sds"):
            value[field]={None if k=="null" else int(k):v for k,v in value[field].items()}


def seeds():
    reports=pd.read_parquet(OLD/"reports.parquet")
    members=pd.read_parquet(OLD/"source_members.parquet")
    members=members.loc[members.component_id.isin(reports.component_id)]
    units=pd.read_parquet(SOURCE/"master_units.parquet")
    for seed in range(20261102,20261107):
        directory=OUT/"random_stability"/str(seed)
        if (directory/"receipt.json").exists():continue
        directory.mkdir(parents=True,exist_ok=True)
        config=stars.Config(seed=seed)
        if seed in (20261102,20261103):
            tag="A1" if seed==20261102 else "A7_seed2"
            cache=STP.calibration_cache(pd.read_parquet(A1/f"calibration_{tag}.parquet"))
            result=stars.segment(reports,members,units,config,lambda component:cache[component])
            for name,key in (("stands","component_id"),("rounds","round_id")):
                old=pd.read_parquet(A1/f"{name}_{tag}.parquet")
                fields=[c for c in old if c!="method"]
                STP.compare_table(result[name],old,key,fields)
            origin="verified_frozen_calibration_reuse"
        else:
            print(f"Calibrating baseline seed {seed}",flush=True)
            result=STP.segment_sequence(reports,members,units.set_index("unit_id",drop=False),
                np.random.default_rng(seed),{},[],f"seed{seed}")
            origin="new_permutation_calibration"
        STP.check_source_conservation(result,reports,members)
        for name,frame in result.items():frame.to_parquet(directory/f"{name}.parquet",index=False)
        write_json(directory/"receipt.json",dict(status="completed",seed=seed,origin=origin,config=asdict(config),
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (OLD/"reports.parquet",OLD/"source_members.parquet",SOURCE/"master_units.parquet")}))
        print(f"Seed {seed}: {len(result['rounds'])} children",flush=True)


if __name__ == "__main__":
    main()
