"""Verify frozen alpha acceptance, polarization fusion and per-variant opportunity axes."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

from fullchain_ablation import ROOT,OUT,SOURCE,BATCHES,candidate_paths,eligibility_matrix
from run import digest,write_json,load_module


def main():
    root=ROOT/"runs/validation/balsa_alpha_sensitivity_20261104_v1"
    scored=pd.read_parquet(root/"scored_candidates.parquet")
    assert len(scored)==79301 and scored.permutations_uniform.eq(19999).all()
    np.testing.assert_array_equal(scored.p_uniform,(scored.exceedance_uniform+1)/20000)
    previous=set();alpha_rows=[];hashes={str(root/"scored_candidates.parquet"):digest(root/"scored_candidates.parquet")}
    for a in (.005,.01,.025,.05,.1):
        path=root/f"alpha_{a:g}/pixel_events.parquet"
        events=pd.read_parquet(path)
        keys=set(zip(events.unit_id,pd.to_datetime(events.change_time).dt.strftime("%Y-%m")))
        accepted=scored.loc[scored.p_uniform.le(a)]
        expected=set(zip(accepted.unit_id,accepted.candidate_month))
        assert keys==expected and previous<=keys
        if a==.025:
            frozen=pd.read_parquet(SOURCE/"validation/pixel_events.parquet")
            assert keys==set(zip(frozen.unit_id,frozen.change_time.dt.strftime("%Y-%m")))
        alpha_rows.append(dict(alpha=a,n_nodes=len(keys),uniform_score_acceptance_exact=True,nested=True))
        previous=keys;hashes[str(path)]=digest(path)
    polar=ROOT/"runs/validation/balsa_results_ablation_20261101_v1/polarization"
    sys.path.insert(0,str(ROOT/"tooling/diagnostics/balsa-results-ablation-20261101/polarization"))
    module=load_module("verify_frozen_fusion",ROOT/"tooling/diagnostics/balsa-results-ablation-20261101/polarization/run_polarization.py")
    vv=pd.read_parquet(SOURCE/"validation/pixel_events.parquet")
    vh=pd.read_parquet(polar/"vh_firstpost/pixel_events.parquet")
    fused,_=module.fuse(vv,vh)
    for name,expected in fused.items():
        path=polar/name/"aggregation_input_events.parquet"
        actual=pd.read_parquet(path)
        fields=["candidate_id","unit_id","change_index","Band","null_p_value"]
        pd.testing.assert_frame_equal(actual[fields].sort_values("candidate_id").reset_index(drop=True),
            expected[fields].sort_values("candidate_id").reset_index(drop=True),check_dtype=False,check_exact=True)
        hashes[str(path)]=digest(path)
    rows=[]
    for batch in ("Q2","Q3","Q4","Q5"):
        for variant in BATCHES[batch]:
            directory=OUT/"variants"/variant
            axes=json.loads((directory/"eligibility_axes.json").read_text())
            paths=list(candidate_paths(variant))
            if batch=="Q2":paths.append(paths[0].with_name("pixel_execution.parquet"))
            paths += [SOURCE/"master_values.npy",SOURCE/"master_units.parquet",SOURCE/"validation_membership.parquet"]
            if batch=="Q3":paths += [polar/"units.parquet",polar/"vh_values.npy",OUT/"polarization_mask_identity.json"]
            if batch=="Q5":paths.append(root/"scored_candidates.parquet")
            axes["input_sha256"].update({str(p.relative_to(ROOT)):digest(p) for p in paths})
            axes["window_months"]=int(variant[1:]) if batch=="Q2" else 36
            axes["decision_months"]=12;axes["stride_months"]=12
            axes["polarization_mask_receipt"]="fullchain/polarization_mask_identity.json" if batch=="Q3" else None
            write_json(directory/"eligibility_axes.json",axes)
            with np.load(directory/"monthly_eligibility.npz") as eligible:
                matrix=eligible["eligibility"]
            base=np.load(OUT/"variants/full/monthly_eligibility.npz")["eligibility"]
            difference=int((matrix!=base).sum())
            if batch!="Q2":assert difference==0
            rows.append(dict(variant=variant,batch=batch,n_eligible_pixel_months=int(matrix.sum()),
                eligibility_different_from_full=difference,source_hashes=len(axes["input_sha256"])))
    pd.DataFrame(rows).to_csv(OUT/"sensitivity_input_audit.csv",index=False)
    write_json(OUT/"sensitivity_input_audit.json",dict(status="passed",alpha=alpha_rows,
        alpha_0025_bridge_exact=True,frozen_fusion_exact=True,input_sha256=hashes,detector_rerun=False))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":main()
