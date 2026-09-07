#!/usr/bin/env python3
"""Official CCDC band ablation and causal monthly replay, entirely from Python."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import numpy as np
import pandas as pd

from run_cusum_nrt_ccdc_comparison import (
    ROOT, load_inputs, init_ee, json_write, parquet_write, ccdc_cached,
    ground, evaluate_bundle,
)
from lr_cusum.ccdc_adapter import run_ccdc, segment_candidates


def parameters(name):
    chi, obs, lam = re.fullmatch(r"ccdc_p(\d+)_n(\d+)_l([\d.]+)", name).groups()
    return {"chi": int(chi)/100, "observations": int(obs), "regularization": float(lam)}


def band_ablation(args, cfg, refs, meta, values, dates):
    selections = pd.read_csv(args.run_dir / "ccdc_fold_selections.csv")
    pieces = {}
    common = np.isfinite(values).all(axis=1)
    vv = np.where(common[:, None], values[:, :1], np.nan)
    dual = np.where(common[:, None], values, np.nan)
    # Explicit engineering definition; not an unverified claim of paper parity.
    ratio = 10. ** ((dual[:, 0]-dual[:, 1])/10.)
    triplet = np.concatenate([dual, ratio[:, None]], axis=1)
    for name in selections.parameter_id.unique():
        for label, array, bands in (("VV common", vv, ("VV",)),
            ("VV VH", dual, ("VV", "VH")),
            ("VV VH linear-power-ratio", triplet, ("VV", "VH", "ratio"))):
            key = "ablation_"+name+"_"+label.replace(" ", "_")
            try:
                response = ccdc_cached(args, cfg, array, dates, bands, parameters(name), key)
            except Exception as exc:
                json_write(args.run_dir / "band_ablation" / (key+".FAILED.json"), {"error": str(exc), "status": "NOT_SCORED"})
                raise
            segment = segment_candidates(response, cutoff=dates[-1])
            parquet_write(args.run_dir / "band_ablation" / (key+".segments.parquet"), segment)
            for direction in (True, False):
                selected = segment.loc[segment.confirmed & (segment.decrease if direction else True)] if not segment.empty else segment
                events = ground(selected, meta, cfg)
                folds = selections.loc[selections.parameter_id.eq(name), "outer_fold"]
                ids = set(refs.loc[refs.outer_fold.isin(folds), "sample_id"])
                event = events.loc[events.sample_id.isin(ids)].copy()
                pieces.setdefault(label + (" negative" if direction else " all breaks"), []).append(event)
    methods = {k: pd.concat(v, ignore_index=True) for k,v in pieces.items()}
    methods["CCDC VV all available negative"] = pd.read_parquet(args.run_dir / "ccdc_oof_events.parquet")
    for name, frame in methods.items():
        parquet_write(args.run_dir / "band_ablation" / (name.replace(" ", "_")+".events.parquet"), frame)
    evaluate_bundle(args.run_dir / "band_ablation", refs, methods, cfg,
        {"add VH": ("VV VH negative", "VV common negative"),
         "add ratio": ("VV VH linear-power-ratio negative", "VV VH negative"),
         "common months": ("VV common negative", "CCDC VV all available negative"),
         "negative admission": ("VV common negative", "VV common all breaks")})
    json_write(args.run_dir / "band_ablation/COMPLETE.json", {"passed": True,
        "ratio": "10**((VV_dB-VH_dB)/10); linear power ratio, not dB quotient",
        "parameters": "Frozen VV-selected OOF parameters; no retuning on held-out data"})


def prefix_identity_check(args, cfg, values, dates):
    path = args.run_dir / "CCDC_PREFIX_IDENTITY_PASS.json"
    if path.exists():
        return
    x = values[:2, :1].copy()
    params = {"chi": .95, "observations": 4, "regularization": .1}
    tested = []
    for end in (60, 96):
        expected = run_ccdc(x[:, :, :end], dates[:end], **params)
        masked = x.copy()
        masked[:, :, end:] = np.nan
        actual = run_ccdc(masked, dates, **params)
        if expected != actual:
            json_write(args.run_dir / "CCDC_PREFIX_IDENTITY_FAILED.json", {"expected": expected, "actual": actual, "end": end})
        for left, right in zip(expected, actual):
            if set(left) != set(right):
                raise AssertionError("prefix output keys differ")
            for key in left:
                if key in ("tBreak", "tStart", "tEnd", "changeProb", "numObs", "trajectory_index"):
                    np.testing.assert_array_equal(left[key], right[key])
                else:
                    np.testing.assert_allclose(left[key], right[key], atol=1e-10, rtol=1e-12)
        tested.append(end)
    json_write(path, {"passed": True, "prefix_lengths": tested,
        "test": "Exact segment dates/decisions; coefficients within atol 1e-10 / rtol 1e-12",
        "observed_roundoff": "Initial exact-dictionary check differed only at about 1e-14",
        "production_replay": "Strict input truncation, not future masking"})


def replay(args, cfg, refs, meta, values, dates):
    prefix_identity_check(args, cfg, values, dates)
    selections = pd.read_csv(args.run_dir / "ccdc_fold_selections.csv")
    jobs = []
    case_frames = []
    for name in selections.parameter_id.unique():
        folds = selections.loc[selections.parameter_id.eq(name), "outer_fold"]
        indices = np.flatnonzero(meta.sample_id.isin(refs.loc[refs.outer_fold.isin(folds), "sample_id"]))
        for end in range(13, len(dates)+1):
            cases = pd.DataFrame({"trajectory_index": indices, "prefix_end": end, "parameter_id": name})
            case_frames.append(cases)
            for start in range(0, len(cases), cfg["ccdc_batch_size"]):
                part = cases.iloc[start:start+cfg["ccdc_batch_size"]].reset_index(drop=True)
                key = f"replay_strict_{name}_end{end:03d}_{start:05d}"
                jobs.append((key, part, parameters(name)))
    parquet_write(args.run_dir / "replay_ccdc/cases.parquet", pd.concat(case_frames, ignore_index=True))
    def job(item):
        key, part, params = item
        dest = args.run_dir / "replay_ccdc/segments" / (key+".parquet")
        if dest.exists():
            return key
        end = int(part.prefix_end.iloc[0])
        assert part.prefix_end.nunique() == 1
        array = values[part.trajectory_index.to_numpy(), :1, :end].copy()
        response = ccdc_cached(args, cfg, array, dates[:end], ("VV",), params, key)
        raw = segment_candidates(response)
        if not raw.empty:
            local = raw.trajectory_index.to_numpy()
            raw["trajectory_index"] = part.trajectory_index.to_numpy()[local]
            raw["available_through_month"] = dates[part.prefix_end.to_numpy()[local]-1]
            raw["parameter_id"] = part.parameter_id.to_numpy()[local]
        parquet_write(dest, raw)
        json_write(dest.with_suffix(".status.json"), {"completed_cases": len(part), "key": key,
            "no_segment_cases": sum(not r.get("tBreak") for r in response)})
        return key
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 20 == 0:
                print(f"CCDC monthly replay {i}/{len(jobs)} batches", flush=True)
    raw = pd.concat([pd.read_parquet(args.run_dir / "replay_ccdc/segments" / (k+".parquet")) for k,_,_ in jobs], ignore_index=True)
    snapshots = []
    for cutoff in dates[12:]:
        selected = raw.loc[raw.available_through_month.eq(cutoff) & raw.confirmed & raw.decrease]
        event = ground(selected, meta, cfg)
        event["available_through_month"] = cutoff
        snapshots.append(event)
    snapshots = pd.concat(snapshots, ignore_index=True)
    parquet_write(args.run_dir / "replay_ccdc/ground_snapshots.parquet", snapshots)
    frozen = pd.read_parquet(args.run_dir / "ccdc_oof_events.parquet")
    last = snapshots.loc[snapshots.available_through_month.eq(dates[-1])]
    keys = lambda f: set(zip(f.sample_id, pd.to_datetime(f.change_time)))
    if keys(last) != keys(frozen):
        raise AssertionError("CCDC replay endpoint differs from full-series benchmark")
    json_write(args.run_dir / "replay_ccdc/COMPLETE.json", {"passed": True,
        "cases": sum(len(p) for _,p,_ in jobs), "batches": len(jobs),
        "cutoff_count": len(dates)-12, "endpoint_events": len(last)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["bands", "replay"])
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/cusum_nrt_ccdc_20260905_v1")
    p.add_argument("--config", type=Path, default=ROOT / "configs/cusum_nrt_ccdc_comparison_v1.json")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    init_ee(cfg)
    (band_ablation if args.stage == "bands" else replay)(args, cfg, refs, meta, values, dates)


if __name__ == "__main__":
    main()
