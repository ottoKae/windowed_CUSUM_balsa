"""Check downloaded dataset hashes and compare a real pipeline replay with frozen products."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def verify(dataset, replay, scope, output):
    dataset = dataset.resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text(encoding='utf-8'))
    for relative, info in manifest['files'].items():
        path = (dataset / relative).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError('Manifest path leaves the dataset')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != info['sha256'] or path.stat().st_size != info['bytes']:
            raise ValueError('Dataset hash mismatch: ' + relative)
    checks = dict(dataset_version=manifest['dataset_version'], manifest_files=len(manifest['files']),
                  file_hashes_passed=True)
    if replay is not None:
        expected = dataset / 'products' / scope
        units = pd.read_csv(dataset / 'timeseries' / scope / 'units.csv')
        pixels = pd.read_csv(replay / 'pixel_detections.csv', float_precision='round_trip')
        frozen = pd.read_parquet(expected / 'pixel_detections.parquet')
        compare(pixels, frozen, ['unit_id', 'candidate_id'])
        execution = pd.read_csv(replay / 'pixel_execution.csv', float_precision='round_trip')
        compare(execution, pd.read_parquet(expected / 'pixel_execution.parquet'), ['unit_id'],
                exclude={'series_index', 'method'})
        assert set(execution.unit_id) == set(units.unit_id)
        reports = pd.read_csv(replay / 'tda_events.csv', float_precision='round_trip')
        compare(reports, pd.read_parquet(expected / 'tda_reports.parquet'), ['component_id'],
                exclude={'legacy_component_id', 'legacy_prediction_id'})
        sources = pd.read_parquet(replay / 'tda/source_members.parquet')
        compare(sources, pd.read_parquet(expected / 'tda_source_members.parquet'), ['node_id'],
                exclude={'legacy_component_id', 'legacy_prediction_id'})
        for name, key in [('stands','component_id'), ('zones','zone_id'), ('rounds','round_id'),
                          ('node_ownership','node_id'), ('tests','component_id')]:
            compare(pd.read_parquet(replay / 'stp' / f'{name}.parquet'),
                    pd.read_parquet(expected / f'stp_{name}.parquet'), [key],
                    exclude={'q_value'} if scope == 'reference_stands' else set())
        checks.update(scope=scope, pixels=len(units), pixel_detections=len(pixels),
                      tda_reports=len(reports), pixel_and_tda_and_stp_parity=True,
                      float_serialization_tolerance=1e-12,
                      excluded_metadata=['execution.series_index', 'execution.method', 'legacy_component_id', 'legacy_prediction_id'],
                      q_scope_difference=scope == 'reference_stands',
                      rerun_summary=json.loads((replay/'summary.json').read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))
    return checks


def compare(actual, expected, keys, exclude=()):
    columns = [c for c in expected if c not in exclude]
    actual = actual[columns].sort_values(keys).reset_index(drop=True).copy()
    expected = expected[columns].sort_values(keys).reset_index(drop=True).copy()
    for name in columns:
        if pd.api.types.is_datetime64_any_dtype(expected[name]):
            actual[name] = pd.to_datetime(actual[name])
        elif expected[name].dtype == object:
            actual[name] = actual[name].fillna('')
            expected[name] = expected[name].fillna('')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,
                                  rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--scope', choices=['reference_stands','national_stands'], default='national_stands')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset, args.replay, args.scope, args.output)
