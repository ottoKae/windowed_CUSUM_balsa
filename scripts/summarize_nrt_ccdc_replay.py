#!/usr/bin/env python3
"""Reconcile causal snapshots, alert revisions and catalogue-conditional latency."""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from run_cusum_nrt_ccdc_comparison import ROOT, load_inputs, ground, parquet_write, json_write
from lr_cusum.cusum_nrt_adapter import admitted
from lr_cusum.validation_closure import _optimal_ordered_pairs, _event_match_metrics, month_number


def match_snapshot(references, detections, tolerance_months=2):
    """Reuse optimal_v2, grouping once rather than repeatedly filtering all rows.

    Only input preparation changes. The matching objective, tie-breaking and
    metric construction are the existing frozen functions, tested for parity.
    """
    refs = references[['sample_id', 'event_month']].drop_duplicates().copy()
    det = detections[['sample_id', 'change_time']].drop_duplicates().copy()
    refs['event_month'] = pd.to_datetime(refs.event_month)
    det['change_time'] = pd.to_datetime(det.change_time)
    def groups(frame, column):
        return {str(key): pd.DatetimeIndex(group[column].sort_values(kind='stable'))
                for key, group in frame.groupby('sample_id', sort=False)}
    left, right = groups(refs, 'event_month'), groups(det, 'change_time')
    empty = pd.DatetimeIndex([])
    rows = []
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key, empty), right.get(key, empty)
        am, bm = month_number(a), month_number(b)
        for i, j in _optimal_ordered_pairs(am, bm, tolerance_months):
            error = int(bm[j]-am[i])
            rows.append({'sample_id': key, 'reference_time': a[i], 'detected_time': b[j],
                         'signed_error_months': error, 'absolute_error_months': abs(error)})
    return _event_match_metrics(refs, det, rows)


def nrt_snapshots(args, cfg, refs, meta, dates, decrease=False):
    snapshot_name = 'CuSum_NRT_decrease' if decrease else 'CuSum_NRT_monthly'
    cache = args.run_dir / 'latency' / (snapshot_name+'.snapshots.parquet')
    if cache.exists():
        # These snapshots depend on selected detections, not the reference matcher.
        # As with other caches, algorithm/input changes require a new run directory.
        return pd.read_parquet(cache)
    selection = pd.read_csv(args.run_dir / "nrt_fold_selections.csv")
    chunks = []
    for row in selection.itertuples(index=False):
        b, a, tc = map(int, re.fullmatch(r"nrt_b(\d+)_a(\d+)_tc(\d+)", row.parameter_id).groups())
        raw = pd.read_parquet(args.run_dir / "nrt" / f"raw_b{b}_a{a}.parquet")
        ids = set(refs.loc[refs.outer_fold.eq(row.outer_fold), "sample_id"])
        indices = np.flatnonzero(meta.sample_id.isin(ids))
        raw = raw.loc[raw.trajectory_index.isin(indices)]
        chunks.append(admitted(raw, tc/100, decrease=decrease))
    raw = pd.concat(chunks, ignore_index=True)
    frames = []
    for cutoff in dates[12:]:
        frame = ground(raw.loc[raw.available_through_month.le(cutoff)], meta, cfg)
        frame["available_through_month"] = cutoff
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def alert_history(snapshots, cutoffs):
    """Link consecutive snapshots without consulting reference event labels.

    IDs only persist while hypotheses remain active. Re-issued withdrawn alerts
    receive new IDs; these are alarm histories, not inferred biological events.
    """
    previous = pd.DataFrame(columns=["sample_id", "event_month"])
    state, history, next_id = {}, [], 0
    for cutoff in cutoffs:
        current = snapshots.loc[snapshots.available_through_month.eq(cutoff), ["sample_id", "change_time"]]
        pair_map = {}
        if len(previous) and len(current):
            _, matches = match_snapshot(previous, current, tolerance_months=2)
            pair_map = {(r.sample_id, r.detected_time): (r.sample_id, r.reference_time)
                        for r in matches.itertuples(index=False)}
        new_state, matched_old = {}, set()
        for row in current.itertuples(index=False):
            key = (row.sample_id, row.change_time)
            old_key = pair_map.get(key)
            if old_key is None:
                next_id += 1
                record = {"sample_id": row.sample_id, "alert_id": next_id,
                          "first_alert_month": cutoff, "last_revision_month": cutoff}
                action = "issued"
            else:
                matched_old.add(old_key)
                record = dict(state[old_key])
                action = "retained" if key == old_key else "revised"
                if action == "revised":
                    record["last_revision_month"] = cutoff
            record["estimated_event_month"] = row.change_time
            new_state[key] = record
            history.append({**record, "available_through_month": cutoff, "action": action})
        for key in set(state)-matched_old:
            history.append({**state[key], "available_through_month": cutoff, "action": "withdrawn"})
        state = new_state
        previous = current.rename(columns={"change_time": "event_month"})
    return pd.DataFrame(history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "runs/cusum_nrt_ccdc_20260905_v1")
    p.add_argument("--config", type=Path, default=ROOT / "configs/cusum_nrt_ccdc_comparison_v1.json")
    args = p.parse_args()
    cfg, refs, meta, values, dates = load_inputs(args)
    for route in ("replay_windowed", "replay_ccdc"):
        if not (args.run_dir / route / "COMPLETE.json").exists():
            raise ValueError(route+" is not complete")
    snapshots = {
        "Windowed CUSUM": pd.read_parquet(args.run_dir / "replay_windowed/ground_snapshots.parquet"),
        "CCDC VV": pd.read_parquet(args.run_dir / "replay_ccdc/ground_snapshots.parquet"),
        "CuSum NRT monthly": nrt_snapshots(args, cfg, refs, meta, dates),
        "CuSum NRT decrease": nrt_snapshots(args, cfg, refs, meta, dates, decrease=True),
    }
    out = args.run_dir / "latency"
    cutoff_metrics, match_frames, histories, reference_rows = [], [], [], []
    for name, frame in snapshots.items():
        frame = frame.copy()
        frame["method"] = name
        parquet_write(out / (name.replace(" ", "_")+".snapshots.parquet"), frame)
        history = alert_history(frame, dates[12:])
        history["method"] = name
        histories.append(history)
        all_matches = []
        for step, cutoff in enumerate(dates[12:]):
            current = frame.loc[frame.available_through_month.eq(cutoff)]
            _, matches = match_snapshot(refs, current, tolerance_months=2)
            available_refs = refs.loc[refs.event_month.le(cutoff)]
            metrics, _ = match_snapshot(available_refs, current, tolerance_months=2)
            cutoff_metrics.append({"method": name, "cutoff": cutoff, **metrics})
            matches["cutoff"] = cutoff
            matches["method"] = name
            all_matches.append(matches)
            if step % 24 == 0:
                print('Latency', name, step+1, '/', len(dates)-12, flush=True)
        matched = pd.concat(all_matches, ignore_index=True)
        match_frames.append(matched)
        first = matched.groupby(["sample_id", "reference_time"]).cutoff.min().to_dict()
        active = set(zip(matched.sample_id, matched.reference_time, matched.cutoff))
        for reference in refs.itertuples(index=False):
            first_correct = first.get((reference.sample_id, reference.event_month), pd.NaT)
            for horizon in cfg["latency_horizons"]:
                at = reference.event_month + pd.DateOffset(months=horizon)
                evaluable = at <= dates[-1]
                reference_rows.append({
                    "method": name, "event_id": reference.event_id, "sample_id": reference.sample_id,
                    "spatial_block_id": reference.spatial_block_id, "reference_time": reference.event_month,
                    "horizon_months": horizon, "right_censored": not evaluable,
                    "first_correct_alert_month": first_correct,
                    "ever_correct_by_h": bool(evaluable and pd.notna(first_correct) and first_correct <= at),
                    "active_correct_at_h": bool(evaluable and (reference.sample_id, reference.event_month, at) in active),
                })
        print("Latency reconciled", name, flush=True)
    references = pd.DataFrame(reference_rows)
    parquet_write(out / "reference_latency.parquet", references)
    parquet_write(out / "alert_history.parquet", pd.concat(histories, ignore_index=True))
    parquet_write(out / "snapshot_reference_matches.parquet", pd.concat(match_frames, ignore_index=True))
    pd.DataFrame(cutoff_metrics).to_csv(out / "cutoff_metrics.csv", index=False)
    summary = []
    blocks = sorted(refs.spatial_block_id.unique())
    draw = np.random.default_rng(cfg["bootstrap_seed"]).integers(0, len(blocks), (5000, len(blocks)))
    for (name, h), group in references.groupby(["method", "horizon_months"]):
        valid = group.loc[~group.right_censored]
        counts = valid.groupby("spatial_block_id").agg(n=("event_id", "size"),
            ever=("ever_correct_by_h", "sum"), active=("active_correct_at_h", "sum")).reindex(blocks, fill_value=0)
        counts = counts.to_numpy(dtype=float)
        boot = counts[draw].sum(axis=1)
        row = {"method": name, "horizon_months": h, "evaluable_references": len(valid),
               "right_censored": int(group.right_censored.sum()), "blocks": valid.spatial_block_id.nunique()}
        for col, label in ((1, "ever_correct_recall"), (2, "active_correct_recall")):
            dist = boot[:, col]/boot[:, 0]
            row[label] = counts[:, col].sum()/counts[:, 0].sum()
            row[label+"_low"], row[label+"_high"] = np.quantile(dist, [.025,.975])
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "latency_metrics.csv", index=False)
    json_write(out / "COMPLETE.json", {"passed": True,
        "matcher_version": "optimal_v2",
        "evaluation": "Catalogue-conditional, monthly resolution; no nationwide false-alert rate",
        "distinction": "First issued hypothesis is not the first correctly localized alert",
        "right_censoring": "Reference+h beyond 2026-06 excluded at that horizon for EVERY method"})


if __name__ == "__main__":
    main()
