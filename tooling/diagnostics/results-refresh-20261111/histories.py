"""Recompute descriptive national histories from frozen TDA/STP products."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
PRODUCT = ROOT / "runs/validation/harvest_inventory_20261104_v1/products"
OUT = ROOT / "runs/validation/results_refresh_20261111_v1/histories"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths = [PRODUCT/name for name in ("stand_history.csv","event_reports.csv","adjacent_report_intervals.csv")]
    stands, events, archived_intervals = [pd.read_csv(path) for path in paths]
    parents = events.loc[events.method.eq("OCTA")].copy()
    archived = archived_intervals.loc[archived_intervals.method.eq("OCTA") & archived_intervals.report_level.eq("report")].copy()
    assert stands.sample_id.is_unique and parents.prediction_id.is_unique
    parents["month_index"] = pd.PeriodIndex(pd.to_datetime(parents.representative_month),freq="M").asi8
    parents = parents.sort_values(["sample_id","month_index","prediction_id"])
    parents["interval_months"] = parents.groupby("sample_id").month_index.diff()
    intervals = parents.loc[parents.interval_months.notna(),["sample_id","prediction_id","interval_months"]].copy()
    assert sorted(intervals.interval_months)==sorted(archived.interval_months)
    assert intervals.groupby("sample_id").interval_months.apply(list).apply(sorted).to_dict()==archived.groupby("sample_id").interval_months.apply(list).apply(sorted).to_dict()
    recomputed_counts = parents.groupby("sample_id").size().reindex(stands.sample_id,fill_value=0).to_numpy()
    evaluable = stands.model_evaluable.astype(bool)
    assert stands.loc[~evaluable,"octa_count"].isna().all()
    assert np.array_equal(recomputed_counts[evaluable],stands.loc[evaluable,"octa_count"].to_numpy())
    assert (recomputed_counts[~evaluable]==0).all()
    individual_medians = intervals.groupby("sample_id").interval_months.median()
    paired = stands.set_index("sample_id").octa_interval_median_months.to_frame("stored").join(individual_medians.rename("recomputed"))
    assert np.allclose(paired.stored,paired.recomputed,equal_nan=True)
    q1,median,q3 = np.quantile(intervals.interval_months,[.25,.5,.75])
    metrics = dict(inventory_stands=len(stands),evaluable_stands=int(evaluable.sum()),
        unassessable_stands=int((~evaluable).sum()),zero_report_stands=int(((recomputed_counts==0)&evaluable).sum()),
        one_report_stands=int(((recomputed_counts==1)&evaluable).sum()),
        at_least_one_report_stands=int((recomputed_counts>=1).sum()),
        recurrent_stands=int((recomputed_counts>=2).sum()),parent_reports=len(parents),
        adjacent_intervals=len(intervals),interval_q1_months=q1,interval_median_months=median,
        interval_q3_months=q3,interval_min_months=float(intervals.interval_months.min()),
        interval_max_months=float(intervals.interval_months.max()))
    validation_ids=set(stands.loc[stands.in_reference_459,"sample_id"])
    assert len(validation_ids)==459
    metrics["parents_in_reference_459"]=int(parents.sample_id.isin(validation_ids).sum())
    assert metrics["parents_in_reference_459"]==1208
    child_path=PRODUCT/"cstr_spatial_children.csv"
    projection_path=PRODUCT/"cstr_parent_projection.csv"
    stp_stands_path=PRODUCT.parent/"cstr/stands.parquet"
    children=pd.read_csv(child_path)
    projection=pd.read_csv(projection_path)
    stp=pd.read_parquet(stp_stands_path)
    paths.extend([child_path,projection_path,stp_stands_path])
    assert children.prediction_id.is_unique
    assert set(children.parent_prediction_id)==set(parents.prediction_id)==set(projection.parent_prediction_id)
    child_counts=children.groupby("parent_prediction_id").size()
    assert (children.n_siblings==children.parent_prediction_id.map(child_counts)).all()
    assert (children.is_spatial_split==children.n_siblings.gt(1)).all()
    parent_lookup=parents.set_index("prediction_id")
    for parent_id,group in children.groupby("parent_prediction_id"):
        unit_sets=[set(value.split(";")) for value in group.source_unit_ids]
        assert set.union(*unit_sets)==set(json.loads(parent_lookup.loc[parent_id,"source_unit_ids"]))
        assert sum(map(len,unit_sets))==len(set.union(*unit_sets))
    assert stp.component_id.is_unique and set(stp.component_id)==set(parents.prediction_id)
    assert (stp.set_index("component_id").round_count.sort_index()==child_counts.sort_index()).all()
    metrics.update(stp_eligible_parents=int(stp.admissible.sum()),stp_split_parents=int(child_counts.gt(1).sum()),
        stp_split_stands=int(children.loc[children.is_spatial_split,"sample_id"].nunique()),
        stp_unsplit_parents=int(child_counts.eq(1).sum()),stp_two_group_parents=int(child_counts.eq(2).sum()),
        stp_three_group_parents=int(child_counts.eq(3).sum()),stp_final_groups_including_unsplit=len(children),
        stp_children_of_split_parents=int(children.is_spatial_split.sum()))
    child_counts.value_counts().sort_index().rename("parent_reports").rename_axis("spatial_timing_groups").to_csv(OUT/"stp_partition_counts.csv")
    stp.groupby(["admissible","reason"],dropna=False).size().rename("parent_reports").to_csv(OUT/"stp_partition_reasons.csv")
    annual=parents.groupby(pd.to_datetime(parents.representative_month).dt.year).size()
    annual.rename("parent_reports").rename_axis("year").to_csv(OUT/"annual_reports.csv")
    assert annual.index.min()>=2017 and annual.index.max()<=2025
    pd.Series(recomputed_counts[evaluable]).value_counts().sort_index().rename("stands").rename_axis("parent_count").to_csv(OUT/"stand_count_distribution.csv")
    intervals.to_csv(OUT/"recomputed_adjacent_intervals.csv",index=False)
    group_columns=[name for name in ("method","report_level") if name in events]
    groups=events.groupby(group_columns,dropna=False).agg(rows=("prediction_id","size"),stands=("sample_id","nunique"))
    groups.to_csv(OUT/"all_report_level_counts.csv")
    structure={name:{str(k):int(v) for k,v in events[name].value_counts(dropna=False).items()} for name in events if any(token in name.lower() for token in ("method","level","split","partition","parent")) and events[name].nunique(dropna=False)<20}
    region_cols=["DPA_DESPRO"]
    for name in region_cols:
        data=stands.assign(recomputed_parent_count=recomputed_counts,recurrent=recomputed_counts>=2)
        region=data.groupby(name,dropna=False).agg(inventory_stands=("sample_id","size"),evaluable_stands=("model_evaluable","sum"),parent_reports=("recomputed_parent_count","sum"),recurrent_stands=("recurrent","sum"),median_stand_count=("octa_count","median"),q1_within_stand_interval=("octa_interval_median_months",lambda x:x.quantile(.25)),median_within_stand_interval=("octa_interval_median_months","median"),q3_within_stand_interval=("octa_interval_median_months",lambda x:x.quantile(.75)))
        region["recurrent_percent_evaluable"]=100*region.recurrent_stands/region.evaluable_stands.replace(0,np.nan)
        region.to_csv(OUT/f"region_{name}.csv")
    figures=[]
    for folder in sorted((ROOT/"runs/figures").glob("national_recurrence_tda_*")):
        qa_path=folder/"qa_summary.json"
        if not qa_path.exists():
            continue
        qa=json.loads(qa_path.read_text(encoding="utf-8"))
        comparisons={}
        for key in ("source_sha256","input_sha256","sources"):
            for path,digest in qa.get(key,{}).items():
                path=Path(path)
                resolved=path if path.is_absolute() else ROOT/path
                if not resolved.exists() and str(path).startswith("methods"):
                    resolved=ROOT.parents[1]/path
                comparisons[str(path)]=resolved.exists() and hashlib.sha256(resolved.read_bytes()).hexdigest()==digest
        figures.append(dict(folder=str(folder.relative_to(ROOT)),status=qa.get("status"),
            declared_counts={k:v for k,v in qa.items() if k in ("n_stands","n_evaluable","n_reports","n_intervals","interval_q1_median_q3","zero_report_stands","stand_count","evaluable_count","parent_report_count","interval_count")},source_hash_parity=comparisons))
    assert all(all(figure["source_hash_parity"].values()) for figure in figures)
    installed_path=ROOT/"manuscript_JAG/figures/fig07_national_harvest_patterns.pdf"
    installed_text="\n".join(page.extract_text() for page in PdfReader(installed_path).pages)
    (OUT/"installed_national_figure_text.txt").write_text(installed_text,encoding="utf-8")
    product_receipt=json.loads((PRODUCT/"product_receipt.json").read_text(encoding="utf-8"))
    input_parity={}
    for name,item in product_receipt["input_sha256"].items():
        path=Path(item["path"])
        if not path.is_absolute():
            path=ROOT/path
        input_parity[name]=hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
    assert all(input_parity.values())
    results_path=ROOT/"manuscript_JAG/sections/04_results.tex"
    text=results_path.read_text(encoding="utf-8")
    active="\n".join(line.split("%",1)[0] for line in text.splitlines())
    receipt=dict(status="passed",source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        interval_recomputed_from_sorted_parent_months=True,counts_recomputed_from_parents=True,
        fixed_inventory_evaluable_zero_distinction_checked=True,
        stp_member_conservation_and_nonduplication=True,product_receipt_input_hash_parity=input_parity,
        installed_national_figure=dict(path=str(installed_path.relative_to(ROOT)),sha256=hashlib.sha256(installed_path.read_bytes()).hexdigest(),
            has_old_median_36="Median = 36 months" in installed_text,current_interval_median_months=median,
            conclusion="Installed national figure is the older result; current frozen TDA drafts are not installed."),
        methods=groups.reset_index().to_dict("records"),structure=structure,
        columns={"stands":stands.columns.tolist(),"events":events.columns.tolist(),"intervals":archived_intervals.columns.tolist()},
        available_region_columns=region_cols,figure_reviews=figures,
        active_result_subsections=re.findall(r"\\subsection\{([^}]+)\}",active),
        active_result_figures=re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",active),
        raw_detection_rerun=False,tex_or_figure_modified=False,matching_tolerance_not_used=True)
    protocol=json.loads((ROOT/"configs/firstpost_w36_d12_s12_b20_20261030_v2.json").read_text(encoding="utf-8"))
    signature=json.loads((ROOT/"runs/validation/balsa_firstpost_national_20261030_v2/method_signature.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT/"configs/firstpost_w36_d12_s12_b20_20261030_v2.json").read_bytes()).hexdigest()==signature["protocol_sha256"]
    octa_receipt=json.loads((PRODUCT.parent/"octa/execution_receipt.json").read_text(encoding="utf-8"))
    receipt["method_provenance"]={"detector":protocol["detector"],"geometry":protocol["geometry"],
        "candidate_date":protocol["date_policy"]["candidate"],"temporal_null":"IID (detector_engine imports iid_permutation_score)",
        "aggregation":"OCTA/TDA from harvest_inventory; B20 fields in the original detector protocol are not used for these counts",
        "TDA_G":octa_receipt["G"],"TDA_support":octa_receipt["support"],"TDA_representative_month":octa_receipt["representative_month"],
        "national_contains_validation_exact_scientific_fields":octa_receipt["national_contains_validation_exact_scientific_fields"]}
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    pd.Series(metrics,name="value").rename_axis("metric").to_csv(OUT/"metrics.csv")
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(metrics=metrics,methods=receipt["methods"],structure=structure,columns=receipt["columns"],regions=region_cols,figures=figures),indent=2))


if __name__=="__main__":
    main()
