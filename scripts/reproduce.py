"""Portable W36 replay. Uses the installed lr_cusum package, never sibling sources."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
from lr_cusum import paper
from lr_cusum.component_ablation import ComponentAblationConfig, build_component_ledgers
from lr_cusum.detection import stable_seed
from lr_cusum.harvest_bounded_event_aggregation import build_bounded_events
from lr_cusum.temporal_nulls import NullModelConfig, calendar_block_permutation_score

STATE = {}


def initialize(data, config, mode, seed):
    arrays = np.load(Path(data) / 'series.npz', allow_pickle=False)
    STATE.update(values=arrays['values'], dates=pd.DatetimeIndex(arrays['calendar']),
                 config=paper.load_paper_config(config), mode=mode, seed=seed)


def pixel(row):
    cfg = STATE['config']
    if STATE['seed'] is not None:
        cfg = replace(cfg, candidate_seed=STATE['seed'])
    mode = STATE['mode']; band = 'VH_dB' if mode == 'vh' else 'VV_dB'
    values = STATE['values'][row['series_index'], int(mode == 'vh')].astype(np.float64)
    dates = STATE['dates']; identity = row['pixel_track_id']
    seed_column = 'rng_offset_vh' if mode == 'vh' else 'rng_offset_vv'
    series_seed = ((int(row[seed_column]) + cfg.candidate_seed) % (2**31 - 1)
                   if seed_column in row else stable_seed(f'{identity}:{band}', cfg.candidate_seed))
    if mode != 'block3':
        events = paper.detect_series_n1(values, dates, identifier=identity, band=band, config=cfg, series_seed=series_seed)
        ledger = []
    else:
        scan = ComponentAblationConfig(window_length=cfg.window_length, step=cfg.step,
            confidence_threshold=1.0, bootstrap_iterations=1,
            minimum_observations=cfg.minimum_observations, seed=cfg.candidate_seed,
            start_year=cfg.start_year, end_year=cfg.end_year)
        candidates = build_component_ledgers(values, dates, config=scan,
            series_key=f'{identity}:{band}', series_seed=series_seed,
            metadata={'Band': band}).sliding_candidates
        # This is the complete PRE-permutation population, including IID rejections.
        eligible = (candidates.passes_window_calendar & candidates.passes_reporting_calendar
                    & candidates.candidate_valid & candidates.passes_central & candidates.passes_decrease)
        rows = []; ledger = []
        for candidate in candidates.loc[eligible].to_dict('records'):
            window = values[int(candidate['window_start_index']):int(candidate['window_end_index'])]
            base = NullModelConfig('calendar_block3', cfg.permutations, cfg.alpha, 3)
            score = calendar_block_permutation_score(window, candidate['cusum_amplitude'],
                config=base, seed=int(candidate['permutation_seed']))
            initial_p = score.null_p_value
            refined = cfg.refine_low <= initial_p <= cfg.refine_high
            if refined:
                score = calendar_block_permutation_score(window, candidate['cusum_amplitude'],
                    config=replace(base, replicates=cfg.refined_permutations),
                    seed=int(candidate['permutation_seed']))
            ledger.append(dict(candidate_id=candidate['candidate_id'], initial_p=initial_p,
                final_p=score.null_p_value, refined=refined, accepted=score.admitted,
                change_time=pd.Timestamp(candidate['change_time']).isoformat()))
            if score.admitted:
                rows.append(candidate)
        events = pd.DataFrame(rows)
    records = []
    for event in events.to_dict('records'):
        records.append(dict(unit_id=row['unit_id'], pixel_track_id=identity,
            candidate_id=event['candidate_id'], change_time=pd.Timestamp(event['change_time']).isoformat()))
    return identity, json.dumps({'events': records, 'ledger': ledger})


def batch(rows):
    return [pixel(row) for row in rows]


def run(args):
    data = args.data.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name, expected in json.loads((data/'manifest.json').read_text()).items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Input hash mismatch: '+name)
    protocol = json.loads(args.config.read_text())
    if protocol['aggregation']['months'] != 3 or protocol['aggregation']['support_numerator'] != 1 or protocol['aggregation']['support_denominator'] != 5:
        raise ValueError('This entry implements frozen B20 only; changed aggregation is a separate experiment')
    units = pd.read_parquet(data / 'units.parquet')
    membership = pd.read_parquet(data / 'membership.parquet')
    if args.stands:
        chosen = set(json.loads(args.stands.read_text(encoding='utf8')))
        membership = membership.loc[membership.sample_id.isin(chosen)]
        units = units.loc[units.unit_id.isin(membership.unit_id)]
    if args.seed_subset:
        units = units.assign(_rank=(units.seed_diagnostic_rank if 'seed_diagnostic_rank' in units
                                   else units.pixel_track_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest())))
        units = units.sort_values('_rank').head(512).drop(columns='_rank')
    receipt = dict(mode=args.mode, seed=args.seed, seed_subset=args.seed_subset,
        pixels=len(units), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        units_sha256=hashlib.sha256(units[['unit_id','pixel_track_id','series_index']].to_json(orient='records').encode()).hexdigest(),
        input_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(paper.__file__).read_bytes()).hexdigest())
    freeze = output / 'configuration.json'
    if freeze.exists() and json.loads(freeze.read_text()) != receipt:
        raise ValueError('Configuration/input changed; choose a new output directory')
    freeze.write_text(json.dumps(receipt, indent=2), encoding='utf8')
    db = sqlite3.connect(output / 'checkpoints.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS pixels (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    done = {x[0] for x in db.execute('SELECT id FROM pixels')}
    rows = units.loc[~units.pixel_track_id.isin(done)].to_dict('records')
    chunks = [rows[i:i+16] for i in range(0, len(rows), 16)]
    start = time.monotonic(); count = len(done)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(data),str(args.config.resolve()),args.mode,args.seed)) as pool:
        for results in pool.map(batch, chunks):
            db.executemany('INSERT OR REPLACE INTO pixels VALUES (?,?)', results); db.commit()
            count += len(results)
            if count % 256 < 16 or count == len(units):
                print(json.dumps(dict(completed=count,total=len(units),seconds=round(time.monotonic()-start))), flush=True)
    events=[]; null_rows=[]
    for identity, payload in db.execute('SELECT id,payload FROM pixels ORDER BY id'):
        payload=json.loads(payload); events.extend(payload['events'])
        null_rows.extend(dict(pixel_track_id=identity,**x) for x in payload['ledger'])
    db.close()
    events=pd.DataFrame(events,columns=['unit_id','pixel_track_id','candidate_id','change_time'])
    events.to_parquet(output/'pixel_events.parquet',index=False)
    if null_rows: pd.DataFrame(null_rows).to_parquet(output/'pre_null_candidates.parquet',index=False)
    if not args.seed_subset:
        reports,members,diagnostics=build_bounded_events(units,membership,events)
        reports['interval_time_basis']='actual_pixel_response_months'
        reports.to_parquet(output/'stand_reports.parquet',index=False)
        members.to_parquet(output/'source_members.parquet',index=False)
        (output/'aggregation.json').write_text(json.dumps(diagnostics,indent=2,default=str),encoding='utf8')
        receipt['reports']=len(reports)
    receipt.update(status='completed',pixel_events=len(events),actual_kernel_path=str(Path(paper.__file__).resolve()))
    (output/'completed.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['detect'])
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/standard_w36.json')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['vv','vh','block3'],default='vv')
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--stands',type=Path)
    parser.add_argument('--seed-subset',action='store_true')
    parser.add_argument('--seed',type=int)
    run(parser.parse_args())
