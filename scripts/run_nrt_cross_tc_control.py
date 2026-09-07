#!/usr/bin/env python3
"""Polygon-domain cross-Tc control on real interior pixels (not full-scene NRT).

Source method_5: retain low-threshold regions intersecting a high-threshold
region, after area screening. Four-connected raster regions, closed-cell
intersection (including shared boundaries). Native projected cell areas used.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

from run_cusum_nrt_ccdc_comparison import ROOT, load_inputs, parquet_write, json_write, evaluate_bundle
from lr_cusum.cusum_nrt_adapter import scan_nrt
from lr_cusum.polygon_workflow import load_track_cache, fuse_pixel_events, fuse_tracks


def retained_cells(rows, cols, confidence, low, high, pixel_area, minimum_area=300):
    rows, cols = np.asarray(rows), np.asarray(cols)
    rr, cc = rows-rows.min(), cols-cols.min()
    shape = (int(rr.max()+1), int(cc.max()+1))
    low_mask, high_mask = np.zeros(shape, bool), np.zeros(shape, bool)
    low_mask[rr, cc] = np.asarray(confidence) >= low
    high_mask[rr, cc] = np.asarray(confidence) >= high
    low_labels, _ = ndimage.label(low_mask)
    high_labels, _ = ndimage.label(high_mask)
    low_sizes = np.bincount(low_labels.ravel()) * pixel_area
    high_sizes = np.bincount(high_labels.ravel()) * pixel_area
    valid_high = high_mask & (high_sizes[high_labels] > minimum_area)
    # Author vector polygons use intersects, including touching cell boundaries.
    touches = ndimage.binary_dilation(valid_high, structure=np.ones((3, 3), bool))
    labels = np.unique(low_labels[touches & low_mask])
    keep = np.isin(low_labels, labels) & low_mask & (low_sizes[low_labels] > minimum_area)
    return keep[rr, cc]


def worker(task):
    track, ids, destination = task
    destination = Path(destination)
    marker = destination / (track+".COMPLETE.json")
    if marker.exists():
        return track
    root = ROOT / "data/local/paper_v1/polygon_pixels"
    association = pd.read_parquet(root / "pixel_associations.parquet")
    association = association.loc[association.track_key.eq(track) & association.sample_id.isin(ids)].copy()
    tracks = pd.read_parquet(root / "polygon_tracks.parquet")
    tracks = tracks.loc[tracks.track_key.eq(track) & tracks.sample_id.isin(ids)]
    pixels, arrays = load_track_cache(track, root / "cache")
    pixels = pixels.loc[pixels.pixel_track_id.isin(association.pixel_track_id)].reset_index(drop=True)
    frames = []
    for offset in range(0, len(pixels), 500):
        path = destination / track / f"raw_{offset:05d}.parquet"
        if not path.exists():
            part = pixels.iloc[offset:offset+500]
            raw = scan_nrt(arrays["values"][part.cache_index.to_numpy(), 0], arrays["calendar"],
                           before=11, after=3, samples=500, seed=20260905)
            raw["trajectory_index"] += offset
            # Preserve all confidence levels for audit, not only admitted ones.
            parquet_write(path, raw)
        frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.merge(pixels.reset_index(names="trajectory_index")[["trajectory_index", "pixel_track_id"]],
                    on="trajectory_index", validate="many_to_one")
    raw["track_key"] = track
    raw["Band"] = "VV_dB"
    raw["cusum_amplitude"] = raw.q
    raw["selection_score"] = raw.native_confidence
    raw = raw.loc[raw.change_time.between("2017-01-01", "2025-12-01")]
    joined = raw.merge(association[["pixel_track_id", "polygon_track_id", "native_row", "native_col"]],
                       on="pixel_track_id", validate="many_to_many")
    for low in (.90, .92, .95):
        original = raw.loc[raw.native_confidence.ge(low)].copy()
        frame = fuse_pixel_events(original, association, tracks, root / "cache")
        parquet_write(destination / f"low{round(low*100)}" / (track+".parquet"), frame)
        for high in (.96, .98, 1.):
            selected = []
            for (polygon, target), group in joined.groupby(["polygon_track_id", "target_month"], sort=False):
                info = tracks.loc[tracks.polygon_track_id.eq(polygon)].iloc[0]
                transform = json.loads(info["transform"])
                area = abs(transform[0]*transform[4]-transform[1]*transform[3])
                mask = retained_cells(group.native_row.to_numpy(), group.native_col.to_numpy(),
                                      group.native_confidence.to_numpy(), low, high, area)
                if mask.any():
                    selected.append(group.loc[mask])
            if selected:
                keep = pd.concat(selected, ignore_index=True)
                # Membership-specific selection must not leak a retained shared
                # physical pixel into another polygon. Aggregate each separately.
                fused = []
                for polygon, group in keep.groupby("polygon_track_id", sort=False):
                    a = association.loc[association.polygon_track_id.eq(polygon)]
                    p = tracks.loc[tracks.polygon_track_id.eq(polygon)]
                    event = group.drop(columns=["polygon_track_id", "native_row", "native_col"])
                    fused.append(fuse_pixel_events(event, a, p, root / "cache"))
                frame = pd.concat(fused, ignore_index=True)
            else:
                frame = pd.DataFrame()
            parquet_write(destination / f"cross{round(low*100)}_{round(high*100)}" / (track+".parquet"), frame)
    json_write(marker, {"passed": True, "track": track, "unique_pixels": len(pixels),
        "domain": "polygon interior only", "window": "11/3 finite-observation monthly adaptation",
        "minimum_area_m2": 300, "post_disturbance_permanent_exclusion": False})
    return track


def select_support(refs, candidates):
    frames, selections = [], []
    from lr_cusum.validation_closure import match_events_detailed
    for fold in sorted(refs.outer_fold.unique()):
        train, test = refs.loc[refs.outer_fold.ne(fold)], refs.loc[refs.outer_fold.eq(fold)]
        ranked = []
        for ratio, events in candidates.items():
            m, _ = match_events_detailed(train, events.loc[events.sample_id.isin(train.sample_id)], tolerance_months=2, matcher_version="optimal_v2")
            ranked.append((m["f1"], m["precision"], -ratio, ratio))
        ratio = max(ranked)[-1]
        event = candidates[ratio].loc[candidates[ratio].sample_id.isin(test.sample_id)].copy()
        event["outer_fold"] = fold
        frames.append(event)
        selections.append({"outer_fold": fold, "connected_support_fraction": ratio})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(selections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/cusum_nrt_ccdc_20260905_v1")
    p.add_argument("--config", type=Path, default=ROOT / "configs/cusum_nrt_ccdc_comparison_v1.json")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    destination = args.run_dir / "spatial_cross_tc"
    ids = tuple(refs.sample_id.unique())
    tracks = sorted(meta.track_key.unique())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(worker, (t, ids, str(destination))) for t in tracks]
        for i, task in enumerate(as_completed(tasks), 1):
            print(f"cross-Tc {i}/{len(tracks)} {task.result()}", flush=True)
    methods = {}
    for name in ["low90", "low92", "low95"] + [f"cross{l}_{h}" for l in (90,92,95) for h in (96,98,100)]:
        frames = [pd.read_parquet(destination / name / (t+".parquet")) for t in tracks]
        track_events = pd.concat(frames, ignore_index=True)
        candidates = {}
        for ratio in (.1,.2,.3,.4,.5,.6):
            filtered = track_events.loc[track_events.connected_support_fraction.ge(ratio)] if not track_events.empty else track_events
            candidates[ratio] = fuse_tracks(filtered, tolerance_months=1)
        event, selections = select_support(refs, candidates)
        methods[name] = event
        parquet_write(destination / (name+".oof.parquet"), event)
        selections.to_csv(destination / (name+".selections.csv"), index=False)
    methods["Windowed Pixel-first"] = pd.read_parquet(ROOT / "runs/numeric_refresh_20260905/representations_n1_48_12/ground_events/connected.nested_oof.parquet")
    result = evaluate_bundle(destination, refs, methods, cfg,
        {"cross-Tc default minus single": ("cross92_100", "low92"),
         "Windowed pixel minus NRT spatial": ("Windowed Pixel-first", "cross92_100")})
    print(result.stage_metrics.to_string(index=False), flush=True)
    json_write(destination / "COMPLETE.json", {"passed": True, "tracks": len(tracks),
        "scope": "polygon-domain spatial adaptation; not full-scene author replication"})


if __name__ == "__main__":
    main()
