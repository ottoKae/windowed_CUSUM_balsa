#!/usr/bin/env python3
"""Offline B/C replay using maintained detection and evaluation implementations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import warnings
from types import SimpleNamespace
import importlib.util

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lr_cusum.capsule import verify, write_json, frame_write, digest
from lr_cusum.component_ablation import derive_variant_events
from lr_cusum.component_evaluation import evaluate_event_stages, common_index_paired_block_bootstrap
from lr_cusum.paper import load_paper_config, scan_series_n1
from lr_cusum.polygon_workflow import detect_pixels, fuse_pixel_events, fuse_tracks, annotate_vv_with_vh
from lr_cusum.ccdc_adapter import segment_candidates


def parity(b: Path, c: Path) -> dict:
    tracks = pd.read_parquet(b / "tables/polygon_tracks.parquet")
    assoc = pd.read_parquet(c / "tables/pixel_associations.parquet")
    with np.load(b / "inputs/stand_series.npz") as z:
        expected, calendar = z["values"], z["calendar"]
    checked = 0
    for key, group in assoc.groupby("track_key"):
        pixels = pd.read_parquet(c / f"cache/{key}.pixels.parquet")
        lookup = pixels.set_index("pixel_track_id").cache_index
        with np.load(c / f"cache/{key}.npz") as z:
            values = z["values"]
            np.testing.assert_array_equal(calendar, z["calendar"])
        for track, rows in group.groupby("polygon_track_id"):
            index = tracks.loc[tracks.polygon_track_id.eq(track), "series_index"].item()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                actual = np.nanmedian(values[lookup.loc[rows.pixel_track_id].to_numpy(int)], axis=0)
            np.testing.assert_array_equal(actual, expected[index])
            checked += 1
    if checked != len(tracks):
        raise ValueError("incomplete pixel-to-stand replay")
    return {"passed": True, "polygon_tracks": checked, "exact_values_and_nan_mask": True}


def same_table(actual, expected, keys, columns):
    a = actual.sort_values(keys).reset_index(drop=True)[columns]
    e = expected.sort_values(keys).reset_index(drop=True)[columns]
    pd.testing.assert_frame_equal(a, e, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)


def evaluate(b: Path, output: Path, candidate_source: Path | None = None) -> dict:
    refs = pd.read_parquet(b / "tables/references.parquet")
    nrt = b / "evidence/nrt_ccdc"
    stages = {
        "Windowed CUSUM": pd.read_parquet(b / "evidence/frozen/geometry/nested_oof_events.parquet"),
        "CuSum NRT monthly": pd.read_parquet(nrt / "nrt_oof_events.parquet"),
        "CCDC VV": pd.read_parquet(nrt / "ccdc_oof_events.parquet"),
        "ReCuSum decrease": pd.read_parquet(nrt / "recusum_frozen_oof.parquet"),
        "Windowed fixed 48/12": pd.read_parquet(b / "evidence/frozen/geometry/events/w48_s12.ground.parquet"),
        "CuSum NRT source defaults": pd.read_parquet(nrt / "nrt_source_defaults.parquet"),
    }
    expected = pd.read_parquet(nrt / "method_metrics.parquet")
    if set(stages) != set(expected.stage):
        raise ValueError("stored benchmark names/configurations changed")
    result = evaluate_event_stages(refs, stages, tolerances=2, fold_column="outer_fold")
    metrics = ["stage", "tp", "fp", "fn", "precision", "recall", "f1", "temporal_mae_months"]
    same_table(result.stage_metrics, expected, ["stage"], metrics)
    frame_write(output / "benchmark_metrics.parquet", result.stage_metrics)
    frame_write(output / "benchmark_matches.parquet", result.matches)
    contrasts = pd.read_parquet(nrt / "paired_contrasts.parquet")
    mapping = {r.contrast: (r.left, r.right) for r in contrasts.itertuples()}
    bs = common_index_paired_block_bootstrap(result.block_metrics, mapping,
                                            tolerance_months=2, replicates=5000, seed=20260829)
    stored = pd.read_parquet(nrt / "paired_block_replicates.parquet")
    same_table(bs.contrast_replicates, stored, ["replicate", "contrast"], stored.columns.tolist())
    summaries = bs.contrast_summary.copy()
    for i,row in summaries.iterrows():
        dist = bs.contrast_replicates.loc[bs.contrast_replicates.contrast.eq(row.contrast),"delta_"+row.metric]
        summaries.loc[i,"familywise_ci_low"] = dist.quantile(.0125)
        summaries.loc[i,"familywise_ci_high"] = dist.quantile(.9875)
    same_table(summaries,contrasts,["contrast","metric"],contrasts.columns.tolist())
    frame_write(output / "benchmark_paired_replicates.parquet", bs.contrast_replicates)
    frame_write(output / "benchmark_contrasts.parquet", summaries)

    base = b / "evidence/frozen/component_ablation"
    if candidate_source:
        receipt = json.loads((candidate_source / "REPLAY.json").read_text())
        if not (receipt.get("passed") and receipt.get("full_national") and receipt.get("component_ledger")):
            raise ValueError("complete freshly recomputed component ledgers required")
        glob = pd.read_parquet(candidate_source / "global_candidates.parquet")
        sliding = pd.read_parquet(candidate_source / "sliding_candidates.parquet")
    else:
        glob = pd.read_parquet(base / "candidate_audit/global_candidates.parquet")
        sliding = pd.read_parquet(base / "candidate_audit/sliding_candidates.parquet")
        for name, ledger in (("global", glob), ("sliding", sliding)):
            scores = pd.read_parquet(base / f"scores/{name}_n1.parquet")
            ledger["passes_permutation"] = ledger.candidate_id.isin(scores.loc[scores.admitted, "candidate_id"])
    components = {}
    for stage in ("F0", "F1", "F2", "F3", "F4", "R0", "R1", "R2", "R3"):
        track_events = derive_variant_events(glob, sliding, stage, reporting_only=True)
        track_events = track_events.loc[track_events.sample_id.isin(refs.sample_id)]
        events = fuse_tracks(track_events,tolerance_months=1)
        frozen = pd.read_parquet(base / f"events/{stage}.ground.parquet")
        frozen = frozen.loc[frozen.sample_id.isin(refs.sample_id)]
        same_table(events, frozen, ["sample_id", "Band", "change_time"],
                   ["sample_id", "Band", "change_time"])
        components[stage] = events
    evaluated = evaluate_event_stages(refs, components, tolerances=(1, 2, 3), fold_column="outer_fold")
    frame_write(output / "component_metrics.parquet", evaluated.stage_metrics)
    oldrep = pd.read_parquet(base / "bootstrap/contrast_replicates.parquet")
    mapping = {r.contrast:(r.left,r.right) for r in oldrep[["contrast","left","right"]].drop_duplicates().itertuples()}
    bs = common_index_paired_block_bootstrap(evaluated.block_metrics, mapping,
                                            tolerance_months=2, replicates=5000, seed=20260829)
    same_table(bs.contrast_replicates, oldrep, ["replicate","contrast"], oldrep.columns.tolist())
    frame_write(output / "component_paired_replicates.parquet", bs.contrast_replicates)

    events = pd.read_parquet(b / "evidence/frozen/national/events/VV_dB.ground.parquet")
    counts = events.groupby("sample_id").size()
    dates = pd.to_datetime(events.change_time)
    events["month_number"] = dates.dt.year * 12 + dates.dt.month
    intervals = events.sort_values(["sample_id","month_number"]).groupby("sample_id").month_number.diff().dropna()
    national = {"events":len(events), "polygons_with_two_or_more":int((counts>=2).sum()),
                "intervals":len(intervals), "interval_median_months":float(intervals.median())}
    if national != {"events":6658, "polygons_with_two_or_more":2194, "intervals":3509, "interval_median_months":36.0}:
        raise ValueError(f"national result drift: {national}")
    return {"passed":True, "mode":"OOF replay plus component evaluation",
            "component_source":"recomputed" if candidate_source else "frozen",
            "benchmark_point_estimates_match":True, "benchmark_5000_paired_replicates_match":True,
            "component_events_match":True, "component_5000_paired_replicates_match":True,
            "national":national}


def detect_one(job):
    row, values, dates, config, components = job
    metadata = {k:row[k] for k in ("sample_id","polygon_id","polygon_track_id","role","species_nco","tile_id","track_key")}
    metadata["route"] = "polygon_first"
    return scan_series_n1(values, dates, identifier=row["polygon_track_id"], band="VV_dB",
                          config=config, metadata=metadata, all_component_candidates=components,
                          series_seed=int(row["seed_VV_dB"]))


def ccdc_cache(b: Path, output: Path) -> dict:
    base = b / "evidence/nrt_ccdc"
    metadata = pd.read_parquet(base / "trajectory_metadata.parquet").reset_index(names="trajectory_index")
    refs = pd.read_parquet(b / "tables/references.parquet")
    selections = pd.read_parquet(base / "ccdc_fold_selections.parquet")
    pieces = []
    for parameter in selections.parameter_id.unique():
        paths = sorted((base / "official_responses" / parameter).glob("batch_*.json"))
        records = [{**row, "trajectory_index":int(p.stem.split("_")[1])+int(row["trajectory_index"])}
                   for p in paths for row in json.loads(p.read_text())]
        # Stored indices are batch-local; restore offsets exactly as the cloud driver.
        if sorted(r["trajectory_index"] for r in records) != list(range(len(metadata))):
            raise ValueError("incomplete/duplicate official CCDC response indices")
        raw = segment_candidates(records)
        selected = raw.loc[raw.confirmed & raw.decrease].merge(metadata,on="trajectory_index",validate="many_to_one")
        selected["Band"] = "VV_dB"
        selected = selected.loc[selected.change_time.between("2017-01-01","2025-12-01")]
        selected = selected.drop_duplicates(["sample_id","track_key","change_time"])
        ground = fuse_tracks(selected,tolerance_months=1)
        for fold in selections.loc[selections.parameter_id.eq(parameter),"outer_fold"]:
            rows = ground.loc[ground.sample_id.isin(refs.loc[refs.outer_fold.eq(fold),"sample_id"])].copy()
            rows["outer_fold"] = fold
            pieces.append(rows)
    events = pd.concat(pieces,ignore_index=True)
    frozen = pd.read_parquet(base / "ccdc_oof_events.parquet")
    same_table(events,frozen,["sample_id","change_time"],["sample_id","change_time","outer_fold"])
    frame_write(output / "ccdc_oof_events.parquet",events)
    result = evaluate_event_stages(refs,{"CCDC VV":events},tolerances=2,fold_column="outer_fold")
    frame_write(output / "metrics.parquet",result.stage_metrics)
    return {"passed":True,"mode":"official cached API responses; NO new cloud request",
            "predictions_match_frozen":True,"events":len(events)}


def nrt(b: Path, output: Path) -> dict:
    """Rerun the pinned monthly numerical core and its 27-config OOF selection."""
    path = Path(__file__).with_name("run_cusum_nrt_ccdc_comparison.py")
    spec = importlib.util.spec_from_file_location("benchmark_driver",path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    base = b / "evidence/nrt_ccdc"
    meta = pd.read_parquet(base / "trajectory_metadata.parquet")
    refs = pd.read_parquet(b / "tables/references.parquet")
    cfg = json.loads((b / "benchmark_config.json").read_text())
    with np.load(b / "inputs/stand_series.npz") as z:
        values = z["values"][meta.series_index.to_numpy()]
        dates = pd.DatetimeIndex(z["calendar"])
    driver.nrt_grid(SimpleNamespace(run_dir=output),cfg,meta,values,dates)
    candidates = {p.stem:pd.read_parquet(p) for p in sorted((output / "ground").glob("nrt_*.parquet"))}
    if len(candidates)!=27:
        raise ValueError("incomplete NRT grid")
    events, selections = driver.select_oof(refs,candidates)
    stored = pd.read_parquet(base / "nrt_oof_events.parquet")
    same_table(events,stored,["sample_id","change_time"],["sample_id","change_time","outer_fold"])
    expected = pd.read_parquet(base / "nrt_fold_selections.parquet")
    same_table(selections,expected,["outer_fold"],["outer_fold","parameter_id","training_f1","training_mae"])
    frame_write(output / "nrt_oof_events.parquet",events)
    frame_write(output / "nrt_selections.parquet",selections)
    return {"passed":True,"mode":"fresh monthly NRT core, 27 configurations, frozen spatial OOF",
            "parameters_and_events_match_frozen":True,"events":len(events)}


def representations(b: Path, output: Path) -> dict:
    base = b / "evidence/numeric_refresh/representations_n1_48_12"
    refs = pd.read_parquet(b / "tables/references.parquet")
    names = {
        "Polygon-first median":"polygon_first_median",
        "Polygon-first mean":"polygon_first_mean",
        "Polygon-first mode":"polygon_first_mode",
        "Single-cell spatial medoid":"polygon_first_center",
        "Pixel-first connected support":"connected.nested_oof",
        "Pixel-first temporal DBSCAN":"temporal_dbscan.nested_oof",
        "Pixel-first space--time DBSCAN":"space_time_dbscan.nested_oof",
    }
    events = {k:pd.read_parquet(base / f"ground_events/{v}.parquet") for k,v in names.items()}
    result = evaluate_event_stages(refs,events,tolerances=2,fold_column="outer_fold")
    expected = pd.read_parquet(base / "stand_representation_metrics.parquet")
    same_table(result.stage_metrics,expected,["stage"],
               ["stage","tp","fp","fn","precision","recall","f1","temporal_mae_months"])
    saved = pd.read_parquet(base / "stand_representation_contrast_replicates.parquet")
    pairs = {r.contrast:(r.left,r.right) for r in saved[["contrast","left","right"]].drop_duplicates().itertuples()}
    bs = common_index_paired_block_bootstrap(result.block_metrics,pairs,
                                           tolerance_months=2,replicates=5000,seed=20260829)
    same_table(bs.contrast_replicates,saved,["replicate","contrast"],saved.columns.tolist())
    frame_write(output / "metrics.parquet",result.stage_metrics)
    frame_write(output / "paired_replicates.parquet",bs.contrast_replicates)
    return {"passed":True,"mode":"seven fixed 48/12 N1 representation routes; OOF replay",
            "metrics_and_5000_paired_replicates_match":True}


def detect(b: Path, output: Path, limit: int, workers: int, components: bool) -> dict:
    tracks = pd.read_parquet(b / "tables/polygon_tracks.parquet").sort_values("series_index")
    if limit:
        tracks = tracks.head(limit)
    config = load_paper_config(b / "paper_v1.json")
    with np.load(b / "inputs/stand_series.npz") as z:
        dates, values = z["calendar"], z["values"]
    jobs = ((r, values[int(r["series_index"]), 0], dates, config, components) for r in tracks.to_dict("records"))
    glob, sliding = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, ledger in enumerate(pool.map(detect_one, jobs, chunksize=4), start=1):
            glob.append(ledger.global_candidates); sliding.append(ledger.sliding_candidates)
            if i % 100 == 0:
                print(f"stand tracks {i}/{len(tracks)}", flush=True)
    g, s = pd.concat(glob, ignore_index=True), pd.concat(sliding, ignore_index=True)
    frame_write(output / "global_candidates.parquet", g)
    frame_write(output / "sliding_candidates.parquet", s)
    stages = ("F0","F1","F2","F3","F4") if components else ("F4",)
    counts = {}
    for stage in stages:
        track_events = derive_variant_events(g,s,stage,reporting_only=True)
        ground = fuse_tracks(track_events,tolerance_months=1)
        frame_write(output / f"{stage}.track.parquet", track_events)
        frame_write(output / f"{stage}.ground.parquet", ground)
        counts[stage] = len(ground)
    frozen = pd.read_parquet(b / "evidence/frozen/national/events/VV_dB.track.parquet")
    frozen = frozen.loc[frozen.polygon_track_id.isin(tracks.polygon_track_id)]
    actual = pd.read_parquet(output / "F4.track.parquet")
    same_table(actual, frozen, ["polygon_track_id","Band","change_time"],
               ["polygon_track_id","Band","change_time"])
    return {"passed":True, "mode":"fresh N1 VV detection", "polygon_tracks":len(tracks),
            "full_national":len(tracks)==4001, "component_ledger":components,
            "F4_track_events_match_frozen":True, "ground_events":counts}


def pixel_first(c: Path, output: Path, limit: int, workers: int) -> dict:
    tracks = pd.read_parquet(c / "tables/polygon_tracks.parquet").sort_values("series_index")
    if limit:
        tracks = tracks.head(limit)
    assoc = pd.read_parquet(c / "tables/pixel_associations.parquet")
    assoc = assoc.loc[assoc.polygon_track_id.isin(tracks.polygon_track_id)].rename(
        columns={"local_row":"native_row","local_col":"native_col"})
    config = load_paper_config(c / "paper_v1.json")
    events = detect_pixels(assoc, c / "cache", config, workers=workers)
    frame_write(output / "pixel_events.parquet", events)
    track_events = fuse_pixel_events(events,assoc,tracks,c / "cache")
    if len(track_events):
        track_events = track_events.loc[track_events.connected_support_fraction.ge(0.20)]
    ground = fuse_tracks(track_events)
    frame_write(output / "track_events.parquet", track_events)
    frame_write(output / "ground_events.parquet", ground)
    primary, _ = annotate_vv_with_vh(ground,tolerance_months=2)
    frame_write(output / "vv_primary.parquet", primary)
    return {"passed":True, "mode":"fresh pixel-first N1 connected support >=0.20",
            "polygon_tracks":len(tracks), "full_national":len(tracks)==4001,
            "unique_pixels":assoc.pixel_track_id.nunique(), "vv_events":len(primary),
            "legacy_L0_equivalence_claimed":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify","parity","evaluate","detect","pixel-first","ccdc-cache","nrt","representations"))
    parser.add_argument("--capsules",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--limit",type=int,default=4)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--components",action="store_true")
    parser.add_argument("--candidate-source",type=Path)
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1:
        parser.error("limit must be >=0 and workers >=1")
    b,c = args.capsules/"science", args.capsules/"pixels"
    if args.command == "verify":
        result = {"science":verify(b),"pixels":verify(c)}
    elif args.command == "parity":
        verify(b); verify(c)
        result = parity(b,c)
    else:
        if not args.output:
            parser.error("--output is required")
        if args.output.exists():
            parser.error("choose a fresh output directory")
        verify(c if args.command=="pixel-first" else b)
        args.output.mkdir(parents=True)
        if args.command == "evaluate":
            result = evaluate(b,args.output,args.candidate_source)
        elif args.command == "ccdc-cache":
            result = ccdc_cache(b,args.output)
        elif args.command == "nrt":
            result = nrt(b,args.output)
        elif args.command == "representations":
            result = representations(b,args.output)
        elif args.command == "detect":
            result = detect(b,args.output,args.limit,args.workers,args.components)
        else:
            result = pixel_first(c,args.output,args.limit,args.workers)
    if args.output:
        result["input_manifest_sha256"] = {
            p.name:digest(p/"MANIFEST.json") for p in (b,c) if p.exists()}
        write_json(args.output/"REPLAY.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
