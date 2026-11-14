"""Paired catalogue comparisons, sibling collapse and common-reference errors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from run import ROOT, RUN, SOURCE, write_json, display_name

sys.path.insert(0,str(ROOT/"src"))
from windowed_cusum.evaluation.event_scoring import _canonical_matching

OUT=RUN/"fullchain"
MONTH0=pd.Period("2016-01",freq="M").ordinal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",action="store_true")
    args=parser.parse_args()
    refs=pd.read_csv(ROOT/"runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv",keep_default_na=False)
    refs["reference_index"]=pd.PeriodIndex(refs.reference_month,freq="M").asi8-MONTH0
    cohort=sorted(refs.sample_id.unique())
    pairs=original_pairs(refs)
    assert len(refs)==1049 and len(cohort)==459 and len(pairs)==590
    summaries,matching,counts,histories={},{},{},{}
    variants=["full","no_local_window","no_central_gate","no_permutation_gate","no_direction_gate","cstr_off","BWA"]
    from fullchain_ablation import BATCHES
    variants += [v for q in ("Q2","Q3","Q4","Q5") for v in BATCHES[q]
                 if (OUT/"variants"/v/"cstr/execution_receipt.json").exists()]
    sets=[]
    for variant in variants:
        levels=load_reports(variant)
        for level,frame in levels.items():
            frame=frame.loc[frame.sample_id.isin(cohort)].copy()
            rules=("pm1","pm2","pm3") if level=="sibling_collapsed" else ("pm1","pm2","pm3","envelope","actual_months")
            for rule in rules:
                key=(variant,level,rule)
                matches=match_reports(refs,frame,rule,level=="sibling_collapsed")
                c=stand_counts(refs,frame,matches,cohort,level=="sibling_collapsed")
                history=pair_matches(pairs,matches)
                counts[key]=c
                matching[key]=matches
                histories[key]=history
                summaries[key]=summary_row(key,c,history,matches)
                target=OUT/"variants"/variant/"metrics";target.mkdir(parents=True,exist_ok=True)
                matches.to_parquet(target/f"{level}_{rule}_matches.parquet",index=False)
                history.to_parquet(target/f"{level}_{rule}_pairs.parquet",index=False)
    effect_rows,transitions=[],[]
    index=np.random.default_rng(20261105).integers(0,len(cohort),size=(5000,len(cohort)))
    np.save(OUT/"paired_stand_indices.npy",index)
    write_json(OUT/"paired_stand_axes.json",dict(sample_ids=cohort,seed=20261105))
    for key,summary in summaries.items():
        variant,level,rule=key
        baseline=("full",level,rule)
        if baseline not in matching:continue
        a=matching[key].set_index("event_id")
        b=matching[baseline].set_index("event_id").reindex(a.index)
        new=a.found & ~b.found;lost=~a.found & b.found;common=a.found & b.found
        shifted=common & a.report_index.ne(b.report_index)
        summary.update(n_matched_new=int(new.sum()),n_matched_lost=int(lost.sum()),
                       n_matched_month_shifted=int(shifted.sum()),n_common_references=int(common.sum()))
        assert int(a.found.sum()-b.found.sum())==int(new.sum()-lost.sum())
        local=a[["sample_id","found","prediction_id","report_index"]].rename(columns=lambda c:"variant_"+c if c!="sample_id" else c)
        for col in ("found","prediction_id","report_index"):local["full_"+col]=b[col]
        local=local.assign(variant=variant,output_level=level,matching_rule=rule,new=new,lost=lost,month_shifted=shifted)
        transitions.append(local.reset_index())
        ref_set=sorted(a.index[common]);ref_id=set_id(ref_set)
        pa=histories[key].set_index("pair_id");pb=histories[baseline].set_index("pair_id").reindex(pa.index)
        shared=pa.found & pb.found;pair_set=sorted(pa.index[shared]);pair_id=set_id(pair_set)
        sets.extend([dict(variant=variant,output_level=level,matching_rule=rule,kind="reference",set_id=ref_id,member_id=x) for x in ref_set])
        sets.extend([dict(variant=variant,output_level=level,matching_rule=rule,kind="original_pair",set_id=pair_id,member_id=x) for x in pair_set])
        common_fields=dict(common_reference_set_id=ref_id,n_common_references=len(ref_set),
            n_common_stands=int(a.loc[common,"sample_id"].nunique()),common_pair_set_id=pair_id,
            n_common_original_pairs=len(pair_set),
            full_common_date_MAE=float(b.loc[common,"absolute_error"].mean()),
            variant_common_date_MAE=float(a.loc[common,"absolute_error"].mean()),
            full_common_date_MedAE=float(b.loc[common,"absolute_error"].median()),
            variant_common_date_MedAE=float(a.loc[common,"absolute_error"].median()),
            full_common_interval_MAE=float(pb.loc[shared,"absolute_error"].mean()),
            variant_common_interval_MAE=float(pa.loc[shared,"absolute_error"].mean()),
            full_common_interval_MedAE=float(pb.loc[shared,"absolute_error"].median()),
            variant_common_interval_MedAE=float(pa.loc[shared,"absolute_error"].median()))
        common_fields["common_date_MAE_difference"]=common_fields["variant_common_date_MAE"]-common_fields["full_common_date_MAE"]
        common_fields["common_interval_MAE_difference"]=common_fields["variant_common_interval_MAE"]-common_fields["full_common_interval_MAE"]
        effect=bootstrap_effect(counts[key],counts[baseline],index,"catalogue_f1",.95)
        for error_name,left_errors,right_errors in [
            ("date",a.loc[common,["sample_id","absolute_error"]],b.loc[common,["sample_id","absolute_error"]]),
            ("interval",pa.loc[shared,["sample_id","absolute_error"]],pb.loc[shared,["sample_id","absolute_error"]])]:
            lo,hi,nvalid=common_error_ci(left_errors,right_errors,cohort,index)
            common_fields[f"common_{error_name}_MAE_difference_ci_low"]=lo
            common_fields[f"common_{error_name}_MAE_difference_ci_high"]=hi
            common_fields[f"common_{error_name}_bootstrap_valid"]=nvalid
        effect_rows.append(dict(variant=variant,output_level=level,matching_rule=rule,
            comparison_role="descriptive_not_component_selection",effect_direction="variant_minus_full",**effect,**common_fields))
    primary=[]
    family=json.loads((OUT/"analysis_family.json").read_text())
    for spec in family["contrasts"]:
        if spec["id"]=="H_D":left,right=("full","octa","pm2"),("no_local_window","octa","pm2")
        elif spec["id"]=="H_T":left,right=("full","octa","envelope"),("BWA","octa","envelope")
        else:left,right=("full","sibling_collapsed","pm2"),("cstr_off","sibling_collapsed","pm2")
        if spec["id"]=="H_T":
            a,b=[histories[k].groupby("sample_id").agg(n=("pair_id","size"),hit=("found","sum")).reindex(cohort,fill_value=0) for k in (left,right)]
            effect=bootstrap_effect(a,b,index,"original_pair_retention",1-.05/3)
        else:effect=bootstrap_effect(counts[left],counts[right],index,"catalogue_f1",1-.05/3)
        primary.append({**spec,**effect,"family_size":3,"multiplicity_adjustment":"Bonferroni","prior_results_seen":True})
    summary_table = pd.DataFrame(summaries.values())
    summary_table["output_label"] = summary_table.output_level.map(display_name)
    summary_table["variant_label"] = summary_table.variant_id.map(display_name)
    summary_table.to_csv(OUT/"summary_by_stage.csv",index=False)
    effects = pd.DataFrame(effect_rows)
    effects["output_label"] = effects.output_level.map(display_name)
    effects["variant_label"] = effects.variant.map(display_name)
    effects.to_csv(OUT/"paired_effects.csv",index=False)
    primary_table=pd.DataFrame(primary)
    primary_table["left_label"]=primary_table.left.map(display_name)
    primary_table["right_label"]=primary_table.right.map(display_name)
    primary_table.to_csv(OUT/"primary_contrasts.csv",index=False)
    pd.concat(transitions,ignore_index=True).to_parquet(OUT/"reference_transitions.parquet",index=False)
    pd.DataFrame(sets).to_parquet(OUT/"common_reference_sets.parquet",index=False)
    precision=[]
    for row in primary:
        precision.append(dict(contrast_id=row["id"],cluster_SE=row["bootstrap_se"],ci_width=row["ci_high"]-row["ci_low"],
            approximate_planning_MDE=(norm.ppf(1-.05/6)+norm.ppf(.8))*row["bootstrap_se"] if row["bootstrap_se"]>0 and abs(row["bootstrap_skewness"])<=1 else np.nan,
            MDE_status="normal_approximation_not_decision_cutoff" if row["bootstrap_se"]>0 and abs(row["bootstrap_skewness"])<=1 else "NA_degenerate_or_skewed",
            assumption="normal approximation; abs empirical skew <=1 screening, not proof of normality",
            target_power=.8,n_stands=459))
    pd.DataFrame(precision).to_csv(OUT/"detectability.csv",index=False)
    if args.seeds:seed_summary(refs,cohort)
    write_json(OUT/"evaluation_receipt.json",dict(status="completed",scope="explicit_completed_variants",variants=variants,references=1049,
        reference_stands=459,original_pairs=590,bootstrap=5000,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(primary_table.drop(columns=["left","right"]).to_string(index=False),flush=True)


def load_reports(variant):
    if variant == "stp_off":variant = "cstr_off"
    if variant=="BWA":return {"octa":pd.read_parquet(SOURCE/"validation/stand_reports.parquet")}
    directory=OUT/"variants"/("full" if variant=="cstr_off" else variant)
    parents=pd.read_parquet(directory/"octa/reports.parquet")
    if variant=="cstr_off":
        children=parents.assign(parent_prediction_id=parents.prediction_id)
    else:children=pd.read_parquet(directory/"cstr/rounds.parquet").rename(columns={"round_id":"prediction_id"})
    return dict(octa=parents,cstr=children,sibling_collapsed=children)


def match_reports(refs,reports,rule,sibling=False):
    """Use the frozen lexicographic one-to-one solver on explicit admissible edges."""
    report_groups={s:g for s,g in reports.groupby("sample_id",sort=False)}
    rows=[]
    for sample,group in refs.groupby("sample_id",sort=True):
        current=report_groups.get(sample,reports.iloc[:0])
        predictions={}
        for p in current.itertuples():
            pid=p.parent_prediction_id if sibling else p.prediction_id
            predictions.setdefault(pid,[]).append(p)
        edges,witnesses={},{}
        for r in group.itertuples():
            for pid,children in predictions.items():
                choices=[]
                for p in children:
                    month=pd.Period(p.representative_month,freq="M").ordinal-MONTH0
                    error=abs(month-r.reference_index)
                    if rule.startswith("pm"):allowed=error<=int(rule[2:])
                    elif rule=="envelope":allowed=p.first_response_month<=r.reference_month<=p.last_response_month
                    else:allowed=r.reference_month in json.loads(p.response_months)
                    if allowed:choices.append((error,month,p.prediction_id))
                if choices:
                    witness=min(choices)
                    # Frozen range scoring uses equal edge costs, not representative-date proximity.
                    edges[r.event_id,pid]=(1,witness[0] if rule.startswith("pm") else 0)
                    witnesses[r.event_id,pid]=witness
        assigned={e:p for p,e in _canonical_matching(edges).items()}
        for r in group.itertuples():
            pid=assigned.get(r.event_id,"")
            error,month,child=witnesses[r.event_id,pid] if pid else (np.nan,np.nan,"")
            rows.append(dict(event_id=r.event_id,sample_id=sample,reference_index=r.reference_index,
                found=bool(pid),prediction_id=pid,witness_child_id=child,report_index=month,absolute_error=error))
    return pd.DataFrame(rows)


def original_pairs(refs):
    rows=[]
    for sample,group in refs.groupby("sample_id",sort=True):
        ordered=list(group.sort_values(["reference_index","event_id"]).itertuples())
        for a,b in zip(ordered[:-1],ordered[1:]):
            rows.append(dict(pair_id=a.event_id+"__"+b.event_id,sample_id=sample,left=a.event_id,right=b.event_id,
                             reference_interval=b.reference_index-a.reference_index))
    return pd.DataFrame(rows)


def pair_matches(pairs,matches):
    by=matches.set_index("event_id")
    out=pairs.copy()
    out["found"]=out.left.map(by.found)&out.right.map(by.found)
    out["model_interval"]=out.right.map(by.report_index)-out.left.map(by.report_index)
    out.loc[~out.found,"model_interval"]=np.nan
    out["absolute_error"]=(out.model_interval-out.reference_interval).abs()
    return out


def stand_counts(refs,frame,matches,cohort,sibling=False):
    counts=matches.groupby("sample_id").agg(n=("event_id","size"),hit=("found","sum")).reindex(cohort)
    key="parent_prediction_id" if sibling else "prediction_id"
    counts["pred"]=frame.groupby("sample_id")[key].nunique().reindex(cohort,fill_value=0)
    return counts


def summary_row(key,c,history,matches):
    variant,level,rule=key
    n,p,h=int(c.n.sum()),int(c.pred.sum()),int(c.hit.sum())
    return dict(variant_id=variant,output_level=level,matching_rule=rule,n_stands=len(c),n_references=n,
        n_reports=p,n_matches=h,n_unique_references_matched=h,catalogue_precision=h/p if p else np.nan,
        recall=h/n,catalogue_f1=2*h/(n+p),n_original_pairs_retained=int(history.found.sum()),
        n_multiref_stands_fully_recovered=int(((c.n>=2)&c.n.eq(c.hit)).sum()),
        n_stands_fully_recovered=int(c.n.eq(c.hit).sum()),own_matched_interval_MAE=float(history.absolute_error.mean()),
        own_matched_interval_MedAE=float(history.absolute_error.median()))


def bootstrap_effect(a,b,index,metric,ci_level):
    if metric=="catalogue_f1":
        an,ad=2*a.hit.to_numpy(),(a.n+a.pred).to_numpy()
        bn,bd=2*b.hit.to_numpy(),(b.n+b.pred).to_numpy()
    else:an,ad,bn,bd=a.hit.to_numpy(),a.n.to_numpy(),b.hit.to_numpy(),b.n.to_numpy()
    draws=an[index].sum(axis=1)/ad[index].sum(axis=1)-bn[index].sum(axis=1)/bd[index].sum(axis=1)
    low,high=np.quantile(draws,[(1-ci_level)/2,(1+ci_level)/2])
    sd=float(draws.std(ddof=1))
    skew=float(np.mean(((draws-draws.mean())/sd)**3)) if sd>0 else np.nan
    return dict(metric=metric,left_estimate=float(an.sum()/ad.sum()),right_estimate=float(bn.sum()/bd.sum()),
        difference=float(an.sum()/ad.sum()-bn.sum()/bd.sum()),ci_level=ci_level,ci_low=float(low),ci_high=float(high),
        bootstrap_se=sd,bootstrap_skewness=skew,n_bootstrap=len(draws))


def seed_summary(refs,cohort):
    rows,states=[],[]
    seed_matches={}
    for seed in range(20261102,20261107):
        directory=OUT/"random_stability"/str(seed)
        children=pd.read_parquet(directory/"rounds.parquet").rename(columns={"round_id":"prediction_id"})
        children=children.loc[children.sample_id.isin(cohort)]
        parent_states=pd.read_parquet(directory/"stands.parquet")
        states.append(parent_states[["component_id","destination_type","p_value"]].assign(seed=seed))
        for sibling in (False,True):
            matches=match_reports(refs,children,"pm2",sibling)
            seed_matches[seed,sibling]=matches
            p=children.parent_prediction_id.nunique() if sibling else len(children)
            h=int(matches.found.sum())
            rows.append(dict(seed=seed,level="sibling_collapsed" if sibling else "cstr",n_reports=p,
                n_matches=h,catalogue_f1=2*h/(1049+p),n_split_parents=int(parent_states.destination_type.eq("zone_rounds").sum())))
    summary=pd.DataFrame(rows)
    summary["level_label"]=summary.level.map(display_name)
    off=load_reports("cstr_off")["sibling_collapsed"]
    off=off.loc[off.sample_id.isin(cohort)]
    off_matches=match_reports(refs,off,"pm2",True)
    off_f1=2*int(off_matches.found.sum())/(1049+off.parent_prediction_id.nunique())
    summary["difference_vs_cstr_off"]=summary.catalogue_f1-off_f1
    summary.to_csv(OUT/"random_stability/summary.csv",index=False)
    state_table=pd.concat(states)
    state_table.to_parquet(OUT/"random_stability/parent_states.parquet",index=False)
    state_table.assign(accepted=state_table.destination_type.eq("zone_rounds")).groupby("component_id").agg(
        n_accepted_seeds=("accepted","sum"),n_states=("destination_type","nunique"),
        min_p=("p_value","min"),max_p=("p_value","max")).to_csv(OUT/"random_stability/parent_flips.csv")
    summary.groupby("level").catalogue_f1.agg(["mean","std","min","max"]).to_csv(OUT/"random_stability/metric_variation.csv")
    errors=[];pairs=original_pairs(refs)
    for (seed,sibling),matches in seed_matches.items():
        a=matches.set_index("event_id")
        baseline=seed_matches[20261102,sibling]
        b=baseline.set_index("event_id").reindex(a.index)
        common=a.found & b.found
        pa=pair_matches(pairs,matches).set_index("pair_id")
        pb=pair_matches(pairs,baseline).set_index("pair_id").reindex(pa.index)
        shared=pa.found & pb.found
        errors.append(dict(seed=seed,baseline_seed=20261102,level="sibling_collapsed" if sibling else "cstr",
            n_common_references=int(common.sum()),common_reference_set_id=set_id(sorted(a.index[common])),
            baseline_common_date_MAE=b.loc[common,"absolute_error"].mean(),seed_common_date_MAE=a.loc[common,"absolute_error"].mean(),
            n_common_original_pairs=int(shared.sum()),common_pair_set_id=set_id(sorted(pa.index[shared])),
            baseline_common_interval_MAE=pb.loc[shared,"absolute_error"].mean(),seed_common_interval_MAE=pa.loc[shared,"absolute_error"].mean()))
    pd.DataFrame(errors).to_csv(OUT/"random_stability/common_reference_errors.csv",index=False)


def set_id(values):
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def common_error_ci(left,right,cohort,index):
    if len(left)==0:return np.nan,np.nan,0
    assert left.index.equals(right.index)
    delta=(left.absolute_error-right.absolute_error).groupby(left.sample_id).sum().reindex(cohort,fill_value=0).to_numpy()
    n=left.groupby("sample_id").size().reindex(cohort,fill_value=0).to_numpy()
    denominator=n[index].sum(axis=1)
    valid=denominator>0
    draws=delta[index].sum(axis=1)[valid]/denominator[valid]
    low,high=np.quantile(draws,[.025,.975]) if valid.any() else (np.nan,np.nan)
    return float(low),float(high),int(valid.sum())


if __name__=="__main__":main()
