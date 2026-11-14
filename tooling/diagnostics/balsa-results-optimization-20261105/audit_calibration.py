"""Reconstruct the Q1 RNG ledger without repeating any permutation statistics."""
import json
import argparse

import numpy as np
import pandas as pd

from fullchain_ablation import OUT,SOURCE,BATCHES,signature,stars,STP,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",choices=BATCHES,default="Q1")
    args=parser.parse_args()
    batch=OUT/"calibration/20261102"/args.batch
    pool=json.loads((batch/"pool.json").read_text())
    saved=json.loads((batch/"checkpoint.json").read_text())
    assert pool["signatures"]==saved["signatures"] and len(saved["tests"])==len(pool["signatures"])
    units=pd.read_parquet(SOURCE/"master_units.parquet").set_index("unit_id",drop=False)
    config=stars.Config()
    pending={};memberships=[]
    keys=set(pool["signatures"])
    origins={}
    for previous in list(BATCHES)[:list(BATCHES).index(args.batch)+1]:
        prior=json.loads((OUT/"calibration/20261102"/previous/"pool.json").read_text())
        origins.update({key:previous for key in prior["signatures"]})
    for variant in BATCHES[args.batch]:
        directory=OUT/"variants"/variant/"octa"
        reports=pd.read_parquet(directory/"reports.parquet")
        members=pd.read_parquet(directory/"source_members.parquet")
        grouped=dict(tuple(members.groupby("component_id",sort=False)))
        for report in reports.itertuples():
            local=grouped[report.component_id]
            key=signature(report,local,units,config)
            memberships.append(dict(variant_id=variant,component_id=report.component_id,signature=key,
                new_to_batch=key in keys,first_calibration_batch=origins.get(key,"Q0")))
            if key in keys:pending.setdefault(key,(report,local))
    rng=np.random.default_rng()
    rng.bit_generator.state=pool["prefix_state"]
    rows=[]
    for order,key in enumerate(pool["signatures"]):
        report,local=pending[key]
        cached=saved["tests"][key]
        for field in ("observed","means","sds"):
            cached[field]={None if k=="null" else int(k):v for k,v in cached[field].items()}
        local_units=units.loc[local.unit_id.unique()].reset_index(drop=True)
        before=STP.state_digest(rng)
        row,_,_,_,_=stars.segment_component(report.component_id,report.sample_id,report.prediction_id,
            local,local_units,int(report.total_pixels),config,rng,cached=cached)
        n=0
        if row["admissible"]:
            n=int(local_units.groupby("track_key").size().max())
            for _ in range(config.permutations):rng.permutation(n)
        rows.append(dict(order=order,signature=key,sample_id=report.sample_id,admissible=row["admissible"],
            tested_track_pixels=n,draws=config.permutations if n else 0,
            rng_state_before=before,rng_state_after=STP.state_digest(rng)))
    assert rng.bit_generator.state==saved["rng_state"],"Reconstructed draw sequence differs from executed checkpoint"
    order_table=pd.DataFrame(rows)
    order_table.to_csv(batch/"calibration_order.csv",index=False)
    mapping=pd.DataFrame(memberships)
    mapping.to_parquet(batch/"variant_parent_signatures.parquet",index=False)
    joined=mapping.merge(order_table[["signature","admissible","draws"]],on="signature",how="left")
    usage=joined.groupby("variant_id").agg(parent_reports=("component_id","size"),
        new_parent_signatures=("new_to_batch","sum"),new_admissible_tests=("admissible","sum"))
    usage["previous_calibration_reused"]=usage.parent_reports-usage.new_parent_signatures
    usage["frozen_baseline_reused"]=joined.assign(frozen=joined.first_calibration_batch.eq("Q0")).groupby("variant_id").frozen.sum()
    usage.to_csv(batch/"variant_usage.csv")
    write_json(batch/"rng_audit.json",dict(status="passed",replayed_prefix_state=True,
        exact_final_state_equal=True,new_parent_signatures=len(rows),new_admissible_tests=int(order_table.admissible.sum()),
        permutations_replayed=int(order_table.draws.sum()),permutation_statistics_recomputed=False))
    print(usage.to_string(),flush=True)


if __name__=="__main__":main()
