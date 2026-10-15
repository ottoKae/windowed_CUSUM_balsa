"""Recompute frozen W36 report precision and catalogue recall from derived tables."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from lr_cusum.event_scoring import score_review_tables
from lr_cusum.b20_probability_estimation import summarize_probability_results
from lr_cusum.catalogue_evaluation import catalogue_metrics

def run(data, output):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    for filename, digest in manifest['files'].items():
        if hashlib.sha256((data/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen input changed: '+filename)
    read=lambda name:pd.read_csv(data/(name+'.csv'),keep_default_na=False)
    frame=read('sampling_frame')
    tables={name:read(name) for name in ['candidate_reviews','verified_events','candidate_event_links']}
    scores=score_review_tables(frame,**tables,tolerance_unit='month')
    precision=summarize_probability_results(frame,scores,tolerance_unit='month')
    precision['is_primary_result']=precision.tolerance_months.eq(2)
    reports=frame.copy(); reports['interval_time_basis']='actual_pixel_response_months'
    details,recall,history,annual,strata,_=catalogue_metrics('W36_B20',reports,
        read('references'),read('eligibility'),read('stands'))
    output.mkdir(parents=True,exist_ok=True)
    for name, table in [('scores',scores),('precision',precision),('reference_matches',details),
                        ('catalogue_recall',recall),('history_recovery',history),('annual_recall',annual),('strata_recall',strata)]:
        table.to_csv(output/(name+'.csv'),index=False)
    print(precision[['tolerance_months','n_T','n_F','n_U','lower_hajek','upper_hajek','ci95_lower','ci95_upper']].to_string(index=False))
    print(recall[['tolerance_months','frozen_month_found','frozen_month_references','frozen_recall']].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/capsules/w36_evaluation'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.output)
