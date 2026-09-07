#!/usr/bin/env python3
"""Prespecified subgroup and one-factor diagnostics; never select on test scores."""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from run_cusum_nrt_ccdc_comparison import ROOT, load_inputs, evaluate_bundle, parquet_write, match_events_detailed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/cusum_nrt_ccdc_20260905_v1')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/cusum_nrt_ccdc_comparison_v1.json')
    args = parser.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    out = args.run_dir / 'diagnostics'
    methods = {
        'Windowed CUSUM': pd.read_parquet(ROOT / 'data/local/paper_v1/frozen_results/geometry/nested_oof_events.parquet'),
        'CuSum NRT monthly': pd.read_parquet(args.run_dir / 'nrt_oof_events.parquet'),
        'CuSum NRT decrease': pd.read_parquet(args.run_dir / 'nrt_ablation_events/NRT_decrease_filter.parquet'),
        'CCDC VV': pd.read_parquet(args.run_dir / 'ccdc_oof_events.parquet'),
        'CCDC VV VH': pd.read_parquet(args.run_dir / 'band_ablation/VV_VH_negative.events.parquet'),
    }
    evaluation = evaluate_bundle(out, refs, methods, cfg, {
        'W minus direction-controlled NRT': ('Windowed CUSUM', 'CuSum NRT decrease'),
        'W minus dual-band CCDC': ('Windowed CUSUM', 'CCDC VV VH'),
    })
    failures = []
    for method, frame in methods.items():
        matches = evaluation.matches.loc[evaluation.matches.stage.eq(method)]
        ref_keys = set(zip(matches.sample_id, matches.reference_time))
        pred_keys = set(zip(matches.sample_id, matches.detected_time))
        for r in refs.itertuples(index=False):
            if (r.sample_id, r.event_month) not in ref_keys:
                failures.append({'method': method, 'sample_id': r.sample_id, 'event_time': r.event_month,
                    'case_type': 'unmatched_reference', 'event_id': r.event_id})
        for r in frame.itertuples(index=False):
            if (r.sample_id, r.change_time) not in pred_keys:
                failures.append({'method': method, 'sample_id': r.sample_id, 'event_time': r.change_time,
                    'case_type': 'catalogue_unmatched_prediction_not_verified_false_alarm', 'event_id': None})
    identity = refs[['sample_id', 'spatial_block_id', 'outer_fold', 'primary_tile_id', 'DPA_DESPRO']].drop_duplicates('sample_id')
    parquet_write(out / 'failure_cases.parquet', pd.DataFrame(failures).merge(identity, on='sample_id', validate='many_to_one'))
    support = pd.read_parquet(args.run_dir / 'input_support.parquet')
    refs = refs.sort_values(['sample_id', 'event_month']).copy()
    calendar_number = refs.event_month.dt.year*12 + refs.event_month.dt.month
    refs['previous_reference_gap_months'] = calendar_number.groupby(refs.sample_id).diff()
    parquet_write(out / 'close_reference_records_review_only.parquet', refs.loc[refs.previous_reference_gap_months.le(4)])
    refs['sequence_position'] = np.where(refs.groupby('sample_id').cumcount().eq(0), 'First reference', 'Subsequent reference')
    refs['reference_multiplicity'] = refs.groupby('sample_id').event_id.transform('size').clip(upper=3).astype(str).replace({'3': '3+'})
    refs['year'] = refs.event_month.dt.year.astype(str)
    refs['size_class'] = pd.cut(refs.reference_polygon_area_m2 / 10000, [0, 5, 20, np.inf], labels=['<5 ha', '5-20 ha', '>20 ha'], include_lowest=True).astype(str)
    # Explicit stand-level worst-track calendar support, not a missing-value fill.
    missing = support.assign(missing_fraction=1-support.vv_n_obs/len(dates)).groupby('sample_id').missing_fraction.max()
    refs['missing_class'] = pd.cut(refs.sample_id.map(missing), [-.001, .05, .20, 1], labels=['<=5%', '5-20%', '>20%']).astype(str)
    dimensions = ['sequence_position', 'reference_multiplicity', 'year', 'primary_tile_id', 'DPA_DESPRO', 'size_class', 'missing_class']
    blocks = sorted(refs.spatial_block_id.unique())
    draw = np.random.default_rng(cfg['bootstrap_seed']).integers(0, len(blocks), (cfg['bootstrap_replicates'], len(blocks)))
    records = []
    for method in methods:
        matched = evaluation.matches.loc[evaluation.matches.stage.eq(method)]
        keys = set(zip(matched.sample_id, matched.reference_time))
        local = refs.assign(hit=[(s, t) in keys for s, t in zip(refs.sample_id, refs.event_month)])
        for dimension in dimensions:
            for group, frame in local.groupby(dimension, observed=True):
                counts = frame.groupby('spatial_block_id').agg(n=('event_id', 'size'), hits=('hit', 'sum')).reindex(blocks, fill_value=0)
                sums = counts.to_numpy()[draw].sum(axis=1)
                good = sums[:, 0] > 0
                boot = sums[good, 1] / sums[good, 0]
                low, high = np.quantile(boot, [.025, .975])
                records.append({'method': method, 'dimension': dimension, 'group': group,
                    'references': len(frame), 'polygons': frame.sample_id.nunique(), 'blocks': frame.spatial_block_id.nunique(),
                    'recall': frame.hit.mean(), 'recall_low': low, 'recall_high': high,
                    'usable_bootstrap_replicates': int(good.sum()),
                    'interpretation': 'Exploratory; sparse blocks imply fragile intervals'})
    pd.DataFrame(records).to_csv(out / 'stratified_reference_recall.csv', index=False)
    parquet_write(out / 'annotated_references.parquet', refs)
    # Equal calendar warm-up: restrict reporting, but retain the original input history.
    start = pd.Timestamp('2019-01-01')
    period_rows = []
    for name, frame in methods.items():
        # Keep all 462 polygons, including those without a post-2018 reference.
        # Their unmatched detections remain in the catalogue-conditional denominator.
        metric, _ = match_events_detailed(refs.loc[refs.event_month.ge(start)],
            frame.loc[frame.change_time.ge(start)], tolerance_months=2, matcher_version="optimal_v2")
        period_rows.append({'method': name, 'evaluated_polygons': 462, **metric})
    pd.DataFrame(period_rows).to_csv(out / 'common_reporting_2019_2025.csv', index=False)

    selections = pd.read_csv(args.run_dir / 'ccdc_fold_selections.csv')
    variants, unavailable = {}, []
    for field, options in [('n', (3, 4, 6)), ('l', (0, .1, 1, 20))]:
        for value in options:
            label = f'CCDC {field}={value:g}'
            parts = []
            for row in selections.itertuples(index=False):
                key = re.sub(r'_n\d+', f'_n{value:g}', row.parameter_id) if field == 'n' else re.sub(r'_l[\d.]+', f'_l{value:g}', row.parameter_id)
                path = args.run_dir / 'ground' / (key+'.parquet')
                if not path.exists():
                    unavailable.append({'variant': label, 'fold': row.outer_fold, 'configuration': key, 'status': 'TECHNICALLY_UNAVAILABLE_NOT_SCORED'})
                    break
                frame = pd.read_parquet(path)
                ids = refs.loc[refs.outer_fold.eq(row.outer_fold), 'sample_id']
                parts.append(frame.loc[frame.sample_id.isin(ids)])
            else:
                variants[label] = pd.concat(parts, ignore_index=True)
    pd.DataFrame(unavailable).to_csv(out / 'ccdc_control_unavailable.csv', index=False)
    evaluate_bundle(out / 'ccdc_single_factor', refs, variants, cfg)
    print('Diagnostics complete', out, flush=True)


if __name__ == '__main__':
    main()
