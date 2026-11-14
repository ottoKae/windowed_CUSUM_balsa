"""Package structure diagnostics and existing evidence pointers without new truth labels."""
import json
import shutil
import numpy as np
import pandas as pd
from run import ROOT,RUN,SOURCE,A1,digest,write_json
from fullchain_ablation import ALL_VARIANTS


def main():
    out=RUN/"aggregation";out.mkdir(exist_ok=True)
    full=RUN/"fullchain"
    refs=pd.read_csv(ROOT/"runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv")
    cohort=set(refs.sample_id)
    structure=[];fallback=[];parents=[]
    for variant in ALL_VARIANTS:
        if variant=="cstr_off":continue
        d=full/"variants"/variant
        reports=pd.read_parquet(d/"octa/reports.parquet")
        members=pd.read_parquet(d/"octa/source_members.parquet")
        states=pd.read_parquet(d/"cstr/stands.parquet")
        parents.append(states.assign(variant_id=variant))
        for (destination,reason),g in states.groupby(["destination_type","reason"],dropna=False):
            fallback.append(dict(variant_id=variant,destination=destination,reason=reason,n_parents=len(g)))
        for scope,part in [("execution_468",reports),("reference_459",reports.loc[reports.sample_id.isin(cohort)])]:
            for stratum,g in [("all",part),("neither_boundary_censored",part.loc[~part.left_censored & ~part.right_censored])]:
                response=g.response_months.map(json.loads).map(len)
                structure.append(dict(variant_id=variant,scope=scope,censor_stratum=stratum,n_reports=len(g),
                    span_median=g.span_months.median(),span_p90=g.span_months.quantile(.9),span_max=g.span_months.max(),
                    response_density_mean=(response/g.span_months).mean(),empty_month_slots=int((g.span_months-response).sum()),
                    reports_with_gaps=int(response.lt(g.span_months).sum()),n_over_12=int(g.span_months.gt(12).sum()),
                    n_left_censored=int(g.left_censored.sum()),n_right_censored=int(g.right_censored.sum()),
                    n_unknown_history=int(g.history_has_unknown.sum())))
        for destination,g in members.groupby("destination_type"):
            structure.append(dict(variant_id=variant,scope="execution_468_sources",censor_stratum="not_applicable",
                destination_type=destination,n_nodes=len(g),n_distinct_pixels=g.unit_id.nunique()))
    pd.DataFrame(structure).to_csv(out/"report_structure.csv",index=False)
    pd.DataFrame(fallback).to_csv(out/"spatial_fallbacks.csv",index=False)
    pd.concat(parents,ignore_index=True).to_parquet(out/"spatial_parent_explanations.parquet",index=False)
    states=parents[0]
    assert states.variant_id.eq("full").all()
    example_ids=[f":{i:06d}:" for i in (10387,24880,6928,22909,7462)]
    selected=states.loc[states.destination_type.eq("zone_rounds") | states.sample_id.map(lambda s:any(x in s for x in example_ids))].copy()
    links_path=ROOT/"runs/validation/balsa_r2_b20_paired_evidence_20261102_v1/links/report_source_links.parquet"
    links=pd.read_parquet(links_path);links=links.loc[links.method.eq("R2_TIME")]
    evidence=[]
    for p in selected.itertuples():
        local=links.loc[links.current_prediction_id.eq(p.parent_prediction_id)]
        evidence.append(dict(sample_id=p.sample_id,parent_prediction_id=p.parent_prediction_id,
            accepted_subdivision=p.destination_type=="zone_rounds",reason=p.reason,
            existing_old_report_links=len(local),pointer_links=int(local.evidence_pointer_eligible.sum()),
            strictly_inheritable_occurrence_links=int(local.occurrence_fact_inheritable.sum()),
            independent_spatial_boundary="unavailable",spatial_subdivision_verdict="unknown",
            evidence_file=str(links_path.relative_to(ROOT)),
            interpretation="source pointers are not adjudicated child-boundary truth"))
    pd.DataFrame(evidence).to_csv(out/"spatial_evidence_cases.csv",index=False)
    known=[("22909","supports consolidation of 61 same-change core nodes and preservation of later change; four extension nodes unknown",
        "balsa_r2_b20_paired_evidence_20261102_v1/PAIRED_VALIDATION_REPORT_ZH.md"),
        ("7462","2021 canopy removal supported; early and late tails remain unresolved; 35-month envelope is not operation duration",
        "balsa_r2_longchain_diagnosis_20261102_v1")]
    pd.DataFrame(known,columns=["stand","existing_conclusion","source_relative_to_runs_validation"]).to_csv(out/"existing_case_conclusions.csv",index=False)
    for name in ("ablation_mechanism.csv","ablation_label_metrics.csv","ablation_paired_vs_A1.csv"):
        shutil.copyfile(A1/name,out/f"frozen_cstr_{name}")
    null_root=ROOT/"runs/validation/balsa_research_execution_20261031_v3/simulation"
    null_path=null_root/"simulation_summary.csv"
    assert json.loads((null_root/"verification_receipt.json").read_text())["status"] == "passed"
    shutil.copyfile(null_path,out/"frozen_temporal_null_simulation.csv")
    for source,target in [("summary_by_stage.csv","catalogue_metrics.csv"),("paired_effects.csv","paired_effects.csv")]:
        shutil.copyfile(full/source,out/target)
    summary=pd.read_csv(full/"summary_by_stage.csv")
    summary[[c for c in summary if c in ["variant_id","output_level","matching_rule","n_stands","n_references"] or "pair" in c or "recovered" in c or "interval" in c]].to_csv(out/"history_metrics.csv",index=False)
    pd.read_parquet(full/"reference_transitions.parquet").to_csv(out/"reference_transitions.csv",index=False)
    write_json(out/"receipt.json",dict(status="completed",accepted_full_parents=int(states.destination_type.eq("zone_rounds").sum()),
        evidence_case_parents=len(selected),independent_boundary_accuracy_computed=False,new_labels_created=False,
        existing_evidence_sha256={str(links_path):digest(links_path),str(null_path):digest(null_path)},case_inference="purposeful_local_evidence_not_population_accuracy"))
    print(f"Structure and evidence packaged; {len(selected)} case parents; no new boundary truth",flush=True)


if __name__=="__main__":main()
