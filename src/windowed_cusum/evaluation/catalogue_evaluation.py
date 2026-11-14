"""Frozen catalogue evaluation; functions preserved from the author workflow.

Catalogue recall is conditional on the registered reference cohort. It is not
population recall and cannot form population F1 with sampled report precision.
"""
import json
import numpy as np
import pandas as pd
from windowed_cusum.evaluation.event_scoring import _canonical_matching
TOLERANCES = (1, 2, 3)


def month_number(value):
    return pd.Period(value, freq="M").ordinal

def allowed_reference_months(reference):
    """Interval endpoint semantics remain discrete monthly open/closed bounds."""
    if reference["timing_type"] == "month":
        value = month_number(reference["reference_month"])
        return value, value
    if reference["timing_type"] != "interval":
        return None
    lower, upper = month_number(reference["interval_lower"]), month_number(reference["interval_upper"])
    semantics = str(reference.get("bound_semantics", "")).strip()
    if not semantics or semantics[0] not in "[(" or semantics[-1] not in ")]":
        raise ValueError(f"Unknown interval bounds for {reference['event_id']}: {semantics}")
    lower += int(semantics[0] == "(")
    upper -= int(semantics[-1] == ")")
    if lower > upper:
        raise ValueError(f"Empty monthly interval for {reference['event_id']}")
    return lower, upper

def frozen_recall(reference, eligibility, frame):
    """Use ONLY R0 and its frozen eligibility, never new optical month claims."""
    if reference.event_id.duplicated().any() or eligibility.event_id.duplicated().any():
        raise ValueError("Frozen references and eligibility must each be unique")
    if set(reference.event_id) != set(eligibility.event_id):
        raise ValueError("R0 and eligibility identities differ")
    qualified = reference.merge(eligibility[["event_id", "eligibility"]], on="event_id", validate="one_to_one")
    qualified = qualified.loc[qualified.eligibility.eq("eligible")]
    predictions = {key: group.to_dict("records") for key, group in frame.groupby("sample_id")}
    details, summary = [], []
    for delta in TOLERANCES:
        for stand, group in qualified.groupby("sample_id", sort=True):
            candidates, edges = predictions.get(stand, []), {}
            for ref in group.to_dict("records"):
                lo, hi = allowed_reference_months(ref)
                for candidate in candidates:
                    if ref["timing_type"] == "month":
                        distance = abs(month_number(candidate["representative_month"]) - lo)
                        if distance <= delta:
                            edges[ref["event_id"], candidate["prediction_id"]] = (1, distance)
                    elif any(lo <= month_number(m) <= hi for m in json.loads(candidate["response_months"])):
                        edges[ref["event_id"], candidate["prediction_id"]] = (0, 0)
            assignment = _canonical_matching(edges)
            winner = {event: pred for pred, event in assignment.items()}
            by_id = {p["prediction_id"]: p for p in candidates}
            for ref in group.to_dict("records"):
                pred_id = winner.get(ref["event_id"], "")
                pred_month = by_id[pred_id]["representative_month"] if pred_id else ""
                details.append(dict(tolerance_months=delta, event_id=ref["event_id"], sample_id=stand,
                                    timing_type=ref["timing_type"], date_basis=ref["date_basis"],
                                    reference_month=ref["reference_month"], prediction_id=pred_id,
                                    representative_month=pred_month, found=bool(pred_id), reference_version="R0_frozen"))
        part = pd.DataFrame(details)
        part = part.loc[part.tolerance_months.eq(delta)]
        mm, ii = part.loc[part.timing_type.eq("month")], part.loc[part.timing_type.eq("interval")]
        summary.append(dict(tolerance_months=delta, frozen_month_references=len(mm),
                            frozen_month_found=int(mm.found.sum()), frozen_month_missed=int((~mm.found).sum()),
                            frozen_recall=float(mm.found.mean()), frozen_interval_references=len(ii),
                            frozen_interval_compatible=int(ii.found.sum()), reference_version="R0_frozen"))
    return pd.DataFrame(details), pd.DataFrame(summary)

def bootstrap_ratios(numerators,denominators,stand_info,replicates=5000):
    """Conditional reference-cohort spatial block bootstrap; NOT design-based recall.

    Same seed/block ordering for all methods enables paired comparisons.
    """
    df=stand_info[['sample_id','block_id','geo_stratum']].copy()
    df['num']=df.sample_id.map(numerators).fillna(0)
    df['den']=df.sample_id.map(denominators).fillna(0)
    blocks=df.groupby(['geo_stratum','block_id'])[['num','den']].sum().reset_index()
    rng=np.random.default_rng(20260929); n=np.zeros(replicates);d=np.zeros(replicates)
    for _,g in blocks.groupby('geo_stratum',sort=True):
        g=g.sort_values('block_id');draw=rng.integers(0,len(g),size=(replicates,len(g)))
        n+=g.num.to_numpy()[draw].sum(axis=1);d+=g.den.to_numpy()[draw].sum(axis=1)
    ratio=np.divide(n,d,out=np.full(replicates,np.nan),where=d>0)
    return ratio

def catalogue_metrics(method,predictions,refs,eligibility,stands):
    details,summary=frozen_recall(refs,eligibility,predictions)
    details['method']=method;summary['method']=method
    details['is_primary_result']=details.tolerance_months.eq(2)
    summary['is_primary_result']=summary.tolerance_months.eq(2)
    summary['n_reports']=len(predictions)
    summary['interval_time_basis']=predictions.interval_time_basis.iloc[0] if len(predictions) else 'no_reports'
    history=[];annual=[];strata=[];boot=[]
    month=details.loc[details.timing_type.eq('month')].copy()
    for delta,g in month.groupby('tolerance_months'):
        g['year']=g.reference_month.str[:4].astype(int)
        counts=g.groupby('sample_id').agg(n=('event_id','size'),m=('found','sum'))
        counts['all_found']=counts.n.eq(counts.m)
        br=bootstrap_ratios(counts.m,counts.n,stands)
        ci=np.nanquantile(br,[.025,.975]);sel=summary.tolerance_months.eq(delta)
        summary.loc[sel,'conditional_block_bootstrap_ci_lower']=ci[0]
        summary.loc[sel,'conditional_block_bootstrap_ci_upper']=ci[1]
        summary.loc[sel,'ci_scope']='conditional_reference_cohort_not_population_recall'
        if delta==2:boot=br
        multi=counts.loc[counts.n.ge(2)]
        bc=bootstrap_ratios(multi.all_found.astype(int),pd.Series(1,index=multi.index),stands)
        cb=np.nanquantile(bc,[.025,.975])
        history.append(dict(method=method,tolerance_months=delta,n_multi_stands=len(multi),
            n_all_found=int(multi.all_found.sum()),complete_history_recovery=float(multi.all_found.mean()),
            ci_lower=cb[0],ci_upper=cb[1],scope='all_registered_month_events_not_all_real_events'))
        for label,mask in [('1',counts.n.eq(1)),('2',counts.n.eq(2)),('3plus',counts.n.ge(3))]:
            sub=counts.loc[mask];strata.append(dict(method=method,tolerance_months=delta,stratum='event_count_'+label,
                n_references=int(sub.n.sum()),n_found=int(sub.m.sum()),recall=float(sub.m.sum()/sub.n.sum()) if sub.n.sum() else np.nan))
        ordered=g.sort_values(['sample_id','reference_month','event_id']).copy()
        ordered['order']=ordered.groupby('sample_id').cumcount()+1
        for label,sub in [('first',ordered.loc[ordered.order.eq(1)]),('subsequent',ordered.loc[ordered.order.gt(1)])]:
            strata.append(dict(method=method,tolerance_months=delta,stratum=label,n_references=len(sub),n_found=int(sub.found.sum()),recall=float(sub.found.mean())))
        for year,sub in g.groupby('year'):
            annual.append(dict(method=method,tolerance_months=delta,reference_year=year,n_references=len(sub),
                n_found=int(sub.found.sum()),recall=float(sub.found.mean()),
                period_role='endpoint' if year in (2017,2025) else 'interior',matching='full_record_before_year_grouping'))
    return details,summary,pd.DataFrame(history),pd.DataFrame(annual),pd.DataFrame(strata),boot
