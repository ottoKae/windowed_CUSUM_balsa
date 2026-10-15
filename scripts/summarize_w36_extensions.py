"""Score the approved polarization/null checks with frozen catalogue semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lr_cusum.event_scoring import _canonical_matching
from lr_cusum.harvest_bounded_event_aggregation import build_bounded_events
from lr_cusum.catalogue_evaluation import catalogue_metrics

def run(args):
    cfg=json.loads(args.config.read_text()); base=args.run
    vv=pd.read_parquet(base/'full/pixel_events.parquet')
    vh=pd.read_parquet(base/'experiments/vh/pixel_events.parquet')
    for frame in (vv,vh):frame['month']=pd.to_datetime(frame.change_time).dt.to_period('M').astype('int64')
    groups={key:g for key,g in vh.groupby('pixel_track_id')}
    keep_vv=set();used_vh=set()
    for pixel,v in vv.groupby('pixel_track_id'):
        h=groups.get(pixel)
        if h is None:continue
        edges={(a.candidate_id,b.candidate_id):(1,abs(a.month-b.month))
               for a in v.itertuples() for b in h.itertuples()
               if abs(a.month-b.month)<=cfg['polarization']['pair_tolerance_months']}
        pairs=_canonical_matching(edges)  # VH -> VV; maximal pairs, minimum month distance.
        keep_vv.update(pairs.values());used_vh.update(pairs)
    variants={'vv_intersection_vh':vv.loc[vv.candidate_id.isin(keep_vv)],
              'vv_union_vh':pd.concat([vv,vh.loc[~vh.candidate_id.isin(used_vh)]],ignore_index=True)}
    units=pd.read_parquet(args.data/'units.parquet');membership=pd.read_parquet(args.data/'membership.parquet')
    for name,nodes in variants.items():
        out=base/'experiments'/name;out.mkdir(parents=True,exist_ok=True)
        nodes=nodes.drop(columns='month');nodes.to_parquet(out/'pixel_events.parquet',index=False)
        reports,members,receipt=build_bounded_events(units,membership,nodes,allow_multiple_months_per_year=True,
            source_method=name,validate_frozen_gates=False,require_annual_decision=False)
        reports['interval_time_basis']='actual_pixel_response_months'
        reports.to_parquet(out/'stand_reports.parquet',index=False)
        members.to_parquet(out/'source_members.parquet',index=False)
        receipt.update(pair_tolerance_months=cfg['polarization']['pair_tolerance_months'],
            date_priority='VV; unmatched VH retains VH date',paired_nodes=len(keep_vv))
        (out/'aggregation.json').write_text(json.dumps(receipt,indent=2,default=str))
    capsule=args.evaluation
    read=lambda name:pd.read_csv(capsule/(name+'.csv'),keep_default_na=False)
    bridge_path=args.data/'sample_id_public_bridge.json'
    bridge=(json.loads(bridge_path.read_text()) if bridge_path.exists()
            else {value:value for value in read('stands').sample_id})
    summaries=[];histories=[]
    sources={'vv':base/'full','vh':base/'experiments/vh','block3':base/'experiments/block3',
             **{name:base/'experiments'/name for name in variants}}
    for name,out in sources.items():
        reports=pd.read_parquet(out/'stand_reports.parquet')
        reports['sample_id']=reports.sample_id.astype(str).map(bridge)
        if reports.sample_id.isna().any():raise ValueError('Missing public stand bridge')
        details,summary,history,annual,strata,_=catalogue_metrics(name,reports,read('references'),read('eligibility'),read('stands'))
        for filename,table in [('reference_matches',details),('catalogue_recall',summary),('history_recovery',history),('annual_recall',annual)]:
            table.to_csv(out/(filename+'.csv'),index=False)
        summaries.append(summary);histories.append(history)
    output=base/'extension_summary';output.mkdir(exist_ok=True)
    pd.concat(summaries).to_csv(output/'catalogue_recall.csv',index=False)
    pd.concat(histories).to_csv(output/'history_recovery.csv',index=False)
    null=pd.read_parquet(base/'experiments/block3/pre_null_candidates.parquet')
    baseline_ids=set(vv.candidate_id);null['iid_accepted']=null.candidate_id.isin(baseline_ids)
    diagnostic={'pre_permutation_testable_candidates':len(null),'block3_refined':int(null.refined.sum()),
        'iid_accepted':int(null.iid_accepted.sum()),'block3_accepted':int(null.accepted.sum()),
        'new_block3_acceptances':int((null.accepted & ~null.iid_accepted).sum()),
        'iid_acceptances_lost':int((~null.accepted & null.iid_accepted).sum())}
    (output/'null_comparison.json').write_text(json.dumps(diagnostic,indent=2))
    seed_rows=[]
    selected=units.assign(rank=(units.seed_diagnostic_rank if 'seed_diagnostic_rank' in units else units.pixel_track_id.map(lambda x:hashlib.sha256(x.encode()).hexdigest()))).sort_values('rank').head(512)
    def identities(frame):return set(zip(frame.pixel_track_id,pd.to_datetime(frame.change_time).dt.strftime('%Y-%m')))
    baseline=identities(vv.loc[vv.pixel_track_id.isin(selected.pixel_track_id)])
    for seed in cfg['extensions']['seed_diagnostic_seeds']:
        nodes=vv.loc[vv.pixel_track_id.isin(selected.pixel_track_id)] if seed==cfg['detector']['candidate_seed'] else pd.read_parquet(base/f'experiments/seed_{seed}/pixel_events.parquet')
        found=identities(nodes)
        seed_rows.append(dict(seed=seed,pixels=512,pixel_month_nodes=len(found),retained=len(baseline & found),
            added=len(found-baseline),lost=len(baseline-found),jaccard=len(baseline & found)/len(baseline | found),
            stand_scoring='not_applicable_incomplete_stand_subset'))
    pd.DataFrame(seed_rows).to_csv(output/'seed_diagnostic.csv',index=False)
    print(pd.concat(summaries).query('tolerance_months==2')[['method','n_reports','frozen_month_found','frozen_recall']].to_string(index=False))
    print(json.dumps(diagnostic))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/standard_w36.json'))
    parser.add_argument('--data',type=Path,default=Path('data/local/standard_w36_v1'))
    parser.add_argument('--evaluation',type=Path,default=Path('data/capsules/w36_evaluation'))
    parser.add_argument('--run',type=Path,default=Path('runs/release_checks/w36_public_20261004'))
    run(parser.parse_args())
