#!/usr/bin/env python3
"""Local component controls for the approved NRT/CCDC comparison.

Uses existing N1/component engines. Original figures and manuscript untouched.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

from run_cusum_nrt_ccdc_comparison import (
    ROOT, load_inputs, parquet_write, json_write, ground, evaluate_bundle,
)
from lr_cusum.paper import load_paper_config, scan_series_n1, detect_series_n1


def component_worker(task):
    i, values, dates, identifier, length, destination = task
    destination = Path(destination)
    path = destination / f"series_{i:04d}.parquet"
    if path.exists():
        return i
    config = replace(load_paper_config(ROOT / "configs/paper_v1.json"), window_length=length)
    ledger = scan_series_n1(values, dates, identifier=identifier, band="VV_dB",
                            config=config, all_component_candidates=True)
    rows = []
    for variant in ("F0", "F1", "F2", "F3", "F4", "R1"):
        frame = ledger.stage_frame(variant, reporting_only=True).copy()
        frame["variant"] = variant
        frame["trajectory_index"] = i
        rows.append(frame)
    parquet_write(path, pd.concat(rows, ignore_index=True))
    return i


def components(args, cfg, refs, meta, values, dates):
    for length in (48, 36):
        destination = args.run_dir / f"components_w{length}"
        tasks = [(i, values[i, 0], dates, row.polygon_track_id, length, str(destination))
                 for i, row in enumerate(meta.itertuples(index=False))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for completed, future in enumerate(as_completed([pool.submit(component_worker, t) for t in tasks]), 1):
                future.result()
                if completed % 50 == 0:
                    print(f"W{length} components {completed}/{len(meta)}", flush=True)
        raw = pd.concat([pd.read_parquet(destination / f"series_{i:04d}.parquet") for i in range(len(meta))], ignore_index=True)
        parquet_write(destination / "raw_candidates.parquet", raw)
        variants = {}
        for variant in ("F0", "F1", "F2", "F3", "F4", "R1"):
            event = ground(raw.loc[raw.variant.eq(variant)], meta, cfg)
            variants[variant] = event
            parquet_write(destination / (variant+".ground.parquet"), event)
        frozen = ROOT / ("data/local/paper_v1/frozen_results/geometry/nested_oof_events.parquet" if length == 36 else
                        "runs/numeric_refresh_20260905/representations_n1_48_12/ground_events/polygon_first_median.parquet")
        expected = pd.read_parquet(frozen)
        def keys(f):
            return set(zip(f.sample_id.astype(str), pd.to_datetime(f.change_time).dt.strftime("%Y-%m")))
        identical = keys(expected) == keys(variants["F4"])
        json_write(destination / "F4_PARITY.json", {
            "bit_exact_event_identity": identical, "expected_events": len(keys(expected)),
            "actual_events": len(keys(variants["F4"])),
            "missing_events": sorted(keys(expected)-keys(variants["F4"])),
            "extra_events": sorted(keys(variants["F4"])-keys(expected)),
        })
        if not identical:
            raise AssertionError(f"W{length}: frozen event parity failed; do not use new endpoint")
        evaluation = evaluate_bundle(destination, refs, variants, cfg,
            {"central gate": ("F4", "R1"), "windowing": ("F1", "F0"),
             "central forward": ("F2", "F1"), "direction": ("F3", "F2"), "permutation": ("F4", "F3")})
        print(evaluation.stage_metrics.to_string(index=False), flush=True)


def prefix_worker(task):
    i, values, dates, identifier, length, destination = task
    path = Path(destination) / f"series_{i:04d}.parquet"
    if path.exists():
        return i
    config = replace(load_paper_config(ROOT / "configs/paper_v1.json"), window_length=length)
    rows = []
    for end in range(13, len(dates)+1):
        events = detect_series_n1(values[:end], dates[:end], identifier=identifier,
                                  band="VV_dB", config=config)
        for month in events.change_time:
            rows.append({"trajectory_index": i, "change_time": month,
                         "available_through_month": dates[end-1]})
    parquet_write(path, pd.DataFrame(rows, columns=["trajectory_index", "change_time", "available_through_month"]))
    return i


def replay_local(args, cfg, refs, meta, values, dates):
    destination = args.run_dir / "replay_windowed"
    # This is the frozen OOF selection, not a new full-reference optimization.
    selection = pd.read_csv(ROOT / "data/local/paper_v1/frozen_results/geometry/nested_selections.csv")
    assert set(selection.selected_name) == {"w36_s12"}
    tasks = [(i, values[i, 0], dates, row.polygon_track_id, 36, str(destination))
             for i, row in enumerate(meta.itertuples(index=False))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for completed, future in enumerate(as_completed([pool.submit(prefix_worker, t) for t in tasks]), 1):
            future.result()
            if completed % 25 == 0:
                print(f"Windowed monthly prefixes {completed}/{len(meta)}", flush=True)
    raw = pd.concat([pd.read_parquet(destination / f"series_{i:04d}.parquet") for i in range(len(meta))], ignore_index=True)
    parquet_write(destination / "track_snapshots.parquet", raw)
    pieces = []
    for cutoff in dates[12:]:
        event = ground(raw.loc[raw.available_through_month.eq(cutoff)], meta, cfg)
        event["available_through_month"] = cutoff
        pieces.append(event)
    snapshots = pd.concat(pieces, ignore_index=True)
    parquet_write(destination / "ground_snapshots.parquet", snapshots)
    frozen = pd.read_parquet(ROOT / "data/local/paper_v1/frozen_results/geometry/nested_oof_events.parquet")
    last = snapshots.loc[snapshots.available_through_month.eq(dates[-1])]
    keys = lambda f: set(zip(f.sample_id, pd.to_datetime(f.change_time)))
    if keys(last) != keys(frozen):
        raise AssertionError("prefix replay endpoint differs from the frozen OOF result")
    json_write(destination / "COMPLETE.json", {"passed": True, "prefixes": len(dates)-12,
        "trajectories": len(meta), "endpoint_events": len(last), "future_observations_used": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["components", "replay"])
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/cusum_nrt_ccdc_20260905_v1")
    p.add_argument("--config", type=Path, default=ROOT / "configs/cusum_nrt_ccdc_comparison_v1.json")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    (components if args.stage == "components" else replay_local)(args, cfg, refs, meta, values, dates)


if __name__ == "__main__":
    main()
