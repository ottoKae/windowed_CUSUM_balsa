#!/usr/bin/env python3
"""Run the approved monthly stand benchmark without replacing frozen outputs.

All cloud operations use the official Earth Engine Python API. Resume is keyed
by frozen input/config hashes. A failed request is not an empty detection table.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import os
import json
import re
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lr_cusum.ccdc_adapter import run_ccdc, segment_candidates, virtual_collection, read_pixels
from lr_cusum.cusum_nrt_adapter import scan_nrt, score_windows, permutations, admitted
from lr_cusum.component_evaluation import evaluate_event_stages, common_index_paired_block_bootstrap
from lr_cusum.polygon_workflow import fuse_tracks
from lr_cusum.validation_closure import match_events_detailed
from lr_cusum.capsule import verify


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def parquet_write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(args):
    args.science = getattr(args, "science", None)
    data = ROOT / "data/local/paper_v1"
    paths = {
        "config": args.config,
        "values": data / "frozen_results/national/polygon_series.npz",
        "metadata": data / "frozen_results/national/polygon_series_metadata.parquet",
        "references": data / "references/reference_ground_events.parquet",
        "frozen_blocks": ROOT / "runs/numeric_refresh_20260905/benchmark_shared_oof/method_block_metrics.parquet",
    }
    if args.science:
        verify(args.science)
        paths.update({
            "values": args.science / "inputs/stand_series.npz",
            "metadata": args.science / "tables/polygon_tracks.parquet",
            "references": args.science / "tables/references.parquet",
            "frozen_blocks": args.science / "tables/reference_blocks.parquet",
        })
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    fingerprints = {k: sha256(p) for k, p in paths.items()}
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["input_sha256"] != fingerprints:
            raise ValueError("stale resume: input/config signature changed; choose a new run directory")
    refs = pd.read_parquet(paths["references"])
    block = pd.read_parquet(paths["frozen_blocks"])[["spatial_block_id", "outer_fold"]].drop_duplicates()
    if block.spatial_block_id.duplicated().any():
        raise ValueError("conflicting frozen outer folds")
    refs = refs.drop(columns=["outer_fold"],errors="ignore").merge(
        block, on="spatial_block_id", validate="many_to_one", how="left")
    if len(refs) != 1093 or refs.sample_id.nunique() != 462 or refs.outer_fold.isna().any() or len(block) != 57:
        raise ValueError("reference/fold freeze mismatch")
    meta = pd.read_parquet(paths["metadata"])
    meta = meta.loc[meta.sample_id.isin(refs.sample_id)].reset_index(drop=True)
    if len(meta) != 474 or meta.polygon_track_id.duplicated().any():
        raise ValueError("expected 474 unique reference polygon-track sequences")
    with np.load(paths["values"], allow_pickle=False) as z:
        values = z["values"][meta.series_index.to_numpy()]
        dates = pd.DatetimeIndex(z["calendar"])
    if values.shape != (474, 2, 126):
        raise ValueError(values.shape)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        json_write(manifest_path, {
            "protocol": cfg["protocol"], "input_sha256": fingerprints,
            "input_paths": {k: str(p) for k, p in paths.items()},
            "reference_events": len(refs), "reference_polygons": 462,
            "reference_polygon_tracks": len(meta), "blocks": len(block),
            "cloud_authorized": bool(cfg.get("cloud_timeseries_authorized") or os.environ.get("S1GRITS_ALLOW_CLOUD")=="1"),
            "status": "RUNNING_NOT_COMPLETE",
            "note": "Catalogue-conditional evaluation; no verified nationwide negative denominator.",
        })
        parquet_write(args.run_dir / "references.parquet", refs)
        parquet_write(args.run_dir / "trajectory_metadata.parquet", meta)
        support = meta.copy()
        support["vv_n_obs"] = np.isfinite(values[:, 0]).sum(axis=1)
        support["vh_n_obs"] = np.isfinite(values[:, 1]).sum(axis=1)
        support["common_n_obs"] = np.isfinite(values).all(axis=1).sum(axis=1)
        parquet_write(args.run_dir / "input_support.parquet", support)
    return cfg, refs, meta, values, dates


def init_ee(cfg):
    if not cfg.get("cloud_timeseries_authorized", False) and os.environ.get("S1GRITS_ALLOW_CLOUD") != "1":
        raise PermissionError("Explicit authorization is required before cloud processing")
    project = os.environ.get("EE_PROJECT") or cfg.get("gee_project")
    if not project:
        raise ValueError("Set EE_PROJECT to your own authorized Earth Engine project")
    import ee
    ee.Initialize(project=project)
    ee.data.setDeadline(60000)
    ee.data.setMaxRetries(0)
    return ee


def attach(raw, meta):
    if raw.empty:
        return pd.DataFrame(columns=["sample_id", "polygon_id", "track_key", "Band", "role", "species_nco", "change_time"])
    out = raw.merge(meta.reset_index(names="trajectory_index"), on="trajectory_index", validate="many_to_one")
    out["Band"] = "VV_dB"
    return out


def ground(raw, meta, cfg):
    if raw.empty:
        return pd.DataFrame(columns=["sample_id", "change_time"])
    local = attach(raw, meta)
    local = local.loc[local.change_time.between(cfg["report_start"], cfg["report_end"])].copy()
    # Same detector reports recur in overlapping scans. Unique dates per track
    # precede the exact same month-cluster fusion used in the frozen benchmark.
    local = local.drop_duplicates(["sample_id", "track_key", "change_time"])
    return fuse_tracks(local, tolerance_months=cfg["track_fusion_months"])


def preflight(args, cfg):
    import requests
    source_dir = args.run_dir / "source"
    source_dir.mkdir(exist_ok=True)
    commit = cfg["nrt_source_commit"]
    base = "https://forge.inrae.fr/api/v4/projects/bertrand.ygorra%2Fcusum-deforestation_monitoring/repository/files/"
    for name in ("CuSum_NRT_example.py", "LICENSE"):
        path = source_dir / name
        if not path.exists():
            response = requests.get(base + name + "/raw", params={"ref": commit}, timeout=45)
            response.raise_for_status()
            path.write_bytes(response.content)
    source = (source_dir / "CuSum_NRT_example.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "single_change_detection")
    # Execute only this audited pure numerical function, never the script body.
    namespace = {"np": SimpleNamespace(**{k: getattr(np, k) for k in (
        "asarray", "ones", "zeros", "mean", "cumsum", "max", "min", "argmax")},
        random=SimpleNamespace(default_rng=lambda: np.random.default_rng(123)))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "audited_author_core", "exec"), namespace)
    x = np.array([-7., -7.1, -6.9, -7., -7.2, -9., -10., -10.2, -9.8, -10.])
    scores = score_windows(x[None], permutations(len(x), 500, 123))[0]
    for threshold in (0., .92, .95, .98, 1.):
        native = namespace["single_change_detection"](x[:, None], np.arange(1, len(x)+1), threshold, 500)
        expected = (int(scores[1])+1) if scores[2] >= threshold else 0
        np.testing.assert_array_equal(native, [expected])
    print("NRT author-core parity PASS", flush=True)
    ee = init_ee(cfg)
    dates = pd.date_range("2016-01-01", periods=126, freq="MS")
    t = np.arange(len(dates))
    rng = np.random.default_rng(100)
    x = -8 + .15*np.sin(2*np.pi*t/12)[None, :] + rng.normal(0, .025, (4, len(t)))
    x[1, 45:] -= 5
    x[2, 36:55] -= 5
    x[2, 80:] -= 5
    x[3, 10:15] = np.nan
    x[3, 30] = 0.0
    values = x[:, None, :]
    collection, points, projection = virtual_collection(values, dates, ["VV"])
    layers = []
    indices = [0, 10, 30, 44, 45, 125]
    for i in indices:
        image = ee.Image(collection.toList(len(dates)).get(i)).unmask(-9999).rename(f"t{i}")
        layers.append(image)
    actual = read_pixels(ee.Image.cat(layers), points, projection)
    for row in actual:
        k = row["trajectory_index"]
        for i in indices:
            expected = x[k, i] if np.isfinite(x[k, i]) else -9999
            if row[f"t{i}"] != expected:
                raise AssertionError((k, i, expected, row[f"t{i}"]))
    print("CCDC virtual-pixel roundtrip PASS", flush=True)
    started = time.perf_counter()
    response = run_ccdc(values, dates, chi=.95, observations=3)
    json_write(args.run_dir / "preflight_ccdc_raw.json", response)
    candidates = segment_candidates(response, cutoff=dates[-1])
    parquet_write(args.run_dir / "preflight_ccdc_segments.parquet", candidates)
    hits = candidates.loc[candidates.confirmed & candidates.decrease]
    if not {1, 2}.issubset(set(hits.trajectory_index)):
        raise AssertionError("official CCDC did not detect synthetic sustained decreases")
    json_write(args.run_dir / "PREFLIGHT_PASS.json", {
        "nrt_author_core_parity": True, "virtual_roundtrip_bit_exact": True,
        "ccdc_synthetic_sustained_decreases": True, "ccdc_seconds": time.perf_counter()-started,
        "ee_version": ee.__version__, "source_commit": commit,
        "source_sha256": sha256(source_dir / "CuSum_NRT_example.py"),
        "license_sha256": sha256(source_dir / "LICENSE"),
        "source_default": {"before": 11, "after": 3, "Tc": 92, "B": 500},
        "adaptations": ["explicit missing mask", "fixed RNG", "monthly stand input",
                        "bounded windows", "shared ground-event fusion"],
    })
    print("PREFLIGHT PASS", hits[["trajectory_index", "change_time", "magnitude"]].to_dict("records"), flush=True)


def nrt_grid(args, cfg, meta, values, dates):
    for before, after in itertools.product(cfg["nrt_before"], cfg["nrt_after"]):
        path = args.run_dir / "nrt" / f"raw_b{before}_a{after}.parquet"
        started = time.perf_counter()
        if not path.exists():
            raw = scan_nrt(values[:, 0], dates, before=before, after=after,
                           samples=cfg["nrt_permutations"], seed=cfg["nrt_seed"])
            parquet_write(path, raw)
        else:
            raw = pd.read_parquet(path)
        for confidence in cfg["nrt_confidence"]:
            name = f"nrt_b{before}_a{after}_tc{round(confidence*100)}"
            event = ground(admitted(raw, confidence), meta, cfg)
            parquet_write(args.run_dir / "ground" / (name+".parquet"), event)
        print(f"NRT before={before} after={after}: {len(raw)} scans, {time.perf_counter()-started:.1f}s", flush=True)


def ccdc_cached(args, cfg, values, dates, bands, params, name):
    rows = []
    size = cfg["ccdc_batch_size"]
    def request_chunk(part, offset):
        try:
            return run_ccdc(part, dates, bands, **params)
        except Exception as exc:
            if len(part) == 1:
                # Never turn an execution failure into "no change".
                raise RuntimeError(f"trajectory_index={offset}: {exc}") from exc
            middle = len(part)//2
            left = request_chunk(part[:middle], offset)
            right = request_chunk(part[middle:], offset+middle)
            return left + [{**r, "trajectory_index": middle+int(r["trajectory_index"])} for r in right]
    for start in range(0, len(values), size):
        path = args.run_dir / "ccdc_raw" / name / f"batch_{start:04d}.json"
        if path.exists():
            batch = json.loads(path.read_text(encoding="utf-8"))
        else:
            for attempt in range(2):
                try:
                    batch = request_chunk(values[start:start+size], start)
                    break
                except Exception as exc:
                    print(f"CCDC {name} batch={start} attempt={attempt+1}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == 1:
                        raise
                    time.sleep(2*(attempt+1))
            if len(batch) != min(size, len(values)-start):
                raise ValueError("incomplete cloud response")
            json_write(path, batch)
        rows.extend([{**r, "trajectory_index": start+int(r["trajectory_index"])} for r in batch])
        if not name.startswith("replay_"):
            print(f"CCDC {name} {min(start+size,len(values))}/{len(values)}", flush=True)
    return rows


def ccdc_grid(args, cfg, meta, values, dates):
    init_ee(cfg)
    for chi, obs, lam in itertools.product(cfg["ccdc_chi"], cfg["ccdc_observations"], cfg["ccdc_lambda"]):
        name = f"ccdc_p{round(chi*100)}_n{obs}_l{lam:g}"
        output = args.run_dir / "ground" / (name+".parquet")
        if output.exists():
            continue
        failure = args.run_dir / "ccdc_failures" / (name+".json")
        if failure.exists():
            print("CCDC configuration previously failed (not an empty result):", name, flush=True)
            continue
        started = time.perf_counter()
        params = {"chi": chi, "observations": obs, "regularization": lam}
        try:
            response = ccdc_cached(args, cfg, values[:, :1], dates, ("VV",), params, name)
        except Exception as exc:
            json_write(failure, {"configuration": name, "params": params, "error": str(exc),
                "status": "TECHNICALLY_UNAVAILABLE_NOT_SCORED", "elapsed_seconds": time.perf_counter()-started,
                "rule": "Whole configuration excluded for ALL folds; no polygon removed or imputed."})
            print("CCDC unavailable:", name, str(exc), flush=True)
            continue
        segments = segment_candidates(response, cutoff=dates[-1])
        parquet_write(args.run_dir / "ccdc_segments" / (name+".parquet"), segments)
        if segments.empty:
            selected = segments
        else:
            selected = segments.loc[segments.confirmed & segments.decrease]
        parquet_write(output, ground(selected, meta, cfg))
        print(f"CCDC {name} done {time.perf_counter()-started:.1f}s", flush=True)


def select_oof(refs, candidates):
    selections, chosen = [], []
    for fold in sorted(refs.outer_fold.unique()):
        train = refs.loc[refs.outer_fold.ne(fold)]
        test_ids = set(refs.loc[refs.outer_fold.eq(fold), "sample_id"])
        scores = []
        for rank, (name, frame) in enumerate(candidates.items()):
            local = frame.loc[frame.sample_id.isin(train.sample_id)]
            metrics, _ = match_events_detailed(train, local, tolerance_months=2, matcher_version="optimal_v2")
            mae = metrics["temporal_mae_months"]
            delay = int(re.search(r"_a(\d+)", name).group(1)) if name.startswith("nrt_") else int(re.search(r"_n(\d+)", name).group(1))
            scores.append((float(metrics["f1"]), -mae if np.isfinite(mae) else -np.inf, -delay, -rank, name))
        # Unmeasured latency is not invented: ties use the configured post-support
        # budget, then the preregistered grid order, not held-out performance.
        best = max(scores)
        name = best[-1]
        frame = candidates[name].loc[candidates[name].sample_id.isin(test_ids)].copy()
        frame["outer_fold"] = int(fold)
        frame["parameter_id"] = name
        chosen.append(frame)
        selections.append({"outer_fold": int(fold), "parameter_id": name,
                           "training_f1": best[0], "training_mae": -best[1],
                           "tie_break": "MAE, smaller configured confirmation budget, grid order"})
    return pd.concat(chosen, ignore_index=True), pd.DataFrame(selections)


def evaluate_bundle(destination, refs, methods, cfg, contrasts=None):
    destination.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate_event_stages(refs, methods, tolerances=(2,), fold_column="outer_fold")
    evaluation.stage_metrics.to_csv(destination / "method_metrics.csv", index=False)
    evaluation.fold_metrics.to_csv(destination / "method_fold_metrics.csv", index=False)
    parquet_write(destination / "oof_matches.parquet", evaluation.matches)
    parquet_write(destination / "method_block_metrics.parquet", evaluation.block_metrics)
    if contrasts:
        boot = common_index_paired_block_bootstrap(evaluation.block_metrics, contrasts,
            tolerance_months=2, replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed"])
        boot.stage_summary.to_csv(destination / "method_intervals.csv", index=False)
        boot.contrast_summary.to_csv(destination / "paired_contrasts.csv", index=False)
        parquet_write(destination / "paired_block_replicates.parquet", boot.contrast_replicates)
    return evaluation


def nrt_ablation(args, cfg, refs, meta, values, dates):
    paths = sorted((args.run_dir / "ground").glob("nrt_*.parquet"))
    if len(paths) != 27:
        raise ValueError("complete NRT grid required")
    chosen, selections = select_oof(refs, {p.stem: pd.read_parquet(p) for p in paths})
    selections.to_csv(args.run_dir / "nrt_fold_selections.csv", index=False)
    parquet_write(args.run_dir / "nrt_oof_events.parquet", chosen)
    variants = {"NRT native": chosen}
    variant_pieces = {}
    for selection in selections.itertuples(index=False):
        b, a, tc = map(int, re.fullmatch(r"nrt_b(\d+)_a(\d+)_tc(\d+)", selection.parameter_id).groups())
        raw = pd.read_parquet(args.run_dir / "nrt" / f"raw_b{b}_a{a}.parquet")
        ids = set(refs.loc[refs.outer_fold.eq(selection.outer_fold), "sample_id"])
        def keep(name, selected):
            events = ground(selected, meta, cfg)
            events = events.loc[events.sample_id.isin(ids)].copy()
            events["outer_fold"] = selection.outer_fold
            variant_pieces.setdefault(name, []).append(events)
        keep("NRT corrected permutation", admitted(raw, tc/100, corrected=True))
        keep("NRT decrease filter", admitted(raw, tc/100, decrease=True))
        for unit, after, samples, seed, label in (
            [("calendar", a, 500, cfg["nrt_seed"], "NRT calendar windows")]
            + [("observations", h, 500, cfg["nrt_seed"], f"NRT after={h}") for h in (0,1,3,6)]
            + [("observations", a, 3999, s, f"NRT B3999 seed={s}") for s in (20260905,20260906,20260907)]
        ):
            key = f"raw_b{b}_a{after}_{unit}_B{samples}_s{seed}"
            path = args.run_dir / "nrt_ablation_raw" / (key+".parquet")
            if not path.exists():
                frame = scan_nrt(values[:, 0], dates, before=b, after=after, samples=samples,
                                 seed=seed, window_unit=unit)
                parquet_write(path, frame)
                print("NRT ablation", key, flush=True)
            frame = pd.read_parquet(path)
            keep(label, admitted(frame, tc/100))
    variants.update({k: pd.concat(v, ignore_index=True) for k,v in variant_pieces.items()})
    for name, frame in variants.items():
        parquet_write(args.run_dir / "nrt_ablation_events" / (name.replace(" ", "_")+".parquet"), frame)
    frozen_oof = (args.science / "evidence/frozen/geometry/nested_oof_events.parquet" if args.science
                  else ROOT / "data/local/paper_v1/frozen_results/geometry/nested_oof_events.parquet")
    methods = {"Windowed CUSUM": pd.read_parquet(frozen_oof), **variants}
    result = evaluate_bundle(args.run_dir / "nrt_ablation", refs, methods, cfg,
        {"W minus NRT": ("Windowed CUSUM", "NRT native"),
         **{name+" minus native": (name,"NRT native") for name in variants if name != "NRT native"}})
    print(result.stage_metrics.to_string(index=False), flush=True)


def evaluate(args, cfg, refs):
    frozen_oof = (args.science / "evidence/frozen/geometry/nested_oof_events.parquet" if args.science
                  else ROOT / "data/local/paper_v1/frozen_results/geometry/nested_oof_events.parquet")
    methods = {"Windowed CUSUM": pd.read_parquet(frozen_oof)}
    legacy = (args.science / "evidence/nrt_ccdc/recusum_frozen_oof.parquet" if args.science
              else args.run_dir / "recusum_frozen_oof.parquet")
    if legacy.exists():
        frame = pd.read_parquet(legacy)
        check, _ = match_events_detailed(refs, frame, tolerance_months=2, matcher_version="optimal_v2")
        if (check["tp"], check["fp"], check["fn"]) != (646, 175, 447):
            raise ValueError("ReCuSum source is not the frozen shared-OOF version")
        parquet_write(args.run_dir / "recusum_frozen_oof.parquet", frame)
        methods["ReCuSum decrease"] = frame
        json_write(args.run_dir / "recusum_source.json", {"path": str(legacy), "sha256": sha256(legacy)})
    fixed = (args.science / "evidence/frozen/geometry/events/w48_s12.ground.parquet" if args.science
             else ROOT / "runs/numeric_refresh_20260905/representations_n1_48_12/ground_events/polygon_first_median.parquet")
    methods["Windowed fixed 48/12"] = pd.read_parquet(fixed)
    methods["CuSum NRT source defaults"] = pd.read_parquet(args.run_dir / "ground/nrt_b11_a3_tc92.parquet")
    for prefix, label, expected in (("nrt_", "CuSum NRT monthly", 27), ("ccdc_", "CCDC VV", 36)):
        paths = sorted((args.run_dir / "ground").glob(prefix+"*.parquet"))
        failures = list((args.run_dir / "ccdc_failures").glob("*.json")) if prefix == "ccdc_" else []
        if len(paths)+len(failures) != expected or not paths:
            raise ValueError(f"{label}: {len(paths)} complete, {len(failures)} failed / {expected}; grid audit incomplete")
        candidates = {p.stem: pd.read_parquet(p) for p in paths}
        event, selections = select_oof(refs, candidates)
        methods[label] = event
        selections.to_csv(args.run_dir / (prefix+"fold_selections.csv"), index=False)
        parquet_write(args.run_dir / (prefix+"oof_events.parquet"), event)
    evaluation = evaluate_event_stages(refs, methods, tolerances=(2,), fold_column="outer_fold")
    bootstrap = common_index_paired_block_bootstrap(
        evaluation.block_metrics,
        {"W minus NRT": ("Windowed CUSUM", "CuSum NRT monthly"),
         "W minus CCDC": ("Windowed CUSUM", "CCDC VV")},
        tolerance_months=2, replicates=cfg["bootstrap_replicates"], seed=cfg["bootstrap_seed"],
    )
    evaluation.stage_metrics.to_csv(args.run_dir / "method_metrics.csv", index=False)
    evaluation.fold_metrics.to_csv(args.run_dir / "method_fold_metrics.csv", index=False)
    parquet_write(args.run_dir / "oof_matches.parquet", evaluation.matches)
    parquet_write(args.run_dir / "method_block_metrics.parquet", evaluation.block_metrics)
    parquet_write(args.run_dir / "paired_block_replicates.parquet", bootstrap.contrast_replicates)
    bootstrap.stage_summary.to_csv(args.run_dir / "method_intervals.csv", index=False)
    contrasts = bootstrap.contrast_summary.copy()
    for i, row in contrasts.iterrows():
        dist = bootstrap.contrast_replicates.loc[bootstrap.contrast_replicates.contrast.eq(row.contrast), "delta_"+row.metric]
        contrasts.loc[i, "familywise_ci_low"] = dist.quantile(.0125)
        contrasts.loc[i, "familywise_ci_high"] = dist.quantile(.9875)
    contrasts.to_csv(args.run_dir / "paired_contrasts.csv", index=False)
    print(evaluation.stage_metrics.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "nrt", "nrt-ablation", "ccdc", "evaluate", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cusum_nrt_ccdc_comparison_v1.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cusum_nrt_ccdc_20260905_v1")
    parser.add_argument("--science",type=Path,help="Unpacked B capsule; avoids author-only source/run paths")
    args = parser.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    if args.stage in ("preflight", "all"):
        preflight(args, cfg)
    if args.stage != "preflight" and not (args.run_dir / "PREFLIGHT_PASS.json").exists():
        raise ValueError("preflight gate has not passed")
    if args.stage in ("nrt", "all"):
        nrt_grid(args, cfg, meta, values, dates)
    if args.stage in ("ccdc", "all"):
        ccdc_grid(args, cfg, meta, values, dates)
    if args.stage in ("nrt-ablation", "all"):
        nrt_ablation(args, cfg, refs, meta, values, dates)
    if args.stage in ("evaluate", "all"):
        evaluate(args, cfg, refs)


if __name__ == "__main__":
    main()
