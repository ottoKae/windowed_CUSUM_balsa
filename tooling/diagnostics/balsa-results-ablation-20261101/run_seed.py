"""Re-score the fixed 512-pixel First-post candidate subset with three seeds."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from preflight import ROOT, SOURCE, TRACE, OUT as RUN, digest
from lr_cusum.detection import stable_seed
from lr_cusum.temporal_nulls import NullModelConfig, iid_permutation_score

OUT = RUN / "seed"
SEEDS = (20260820, 20260821, 20260822)


def score_window(window, amplitude, pixel_track_id, window_start, base_seed):
    seed = stable_seed(pixel_track_id + ":VV_dB", base_seed) + int(window_start)
    initial = iid_permutation_score(window, amplitude,
        config=NullModelConfig(method="N1_IID_P025", replicates=3999, alpha=.025), seed=seed)
    refined = .015 <= initial.null_p_value <= .035
    final = iid_permutation_score(window, amplitude,
        config=NullModelConfig(method="N1_IID_P025", replicates=19999, alpha=.025), seed=seed) if refined else initial
    return dict(seed=int(seed), initial_p=initial.null_p_value, final_p=final.null_p_value,
        refined=refined, permutations=int(final.replicates), exceedance_count=int(final.exceedance_count),
        accepted=bool(final.admitted))


def main():
    assert json.loads((RUN / "preflight_receipt.json").read_text())["passed"]
    assert (RUN / "truncation/verification_receipt.json").exists(), "Run P4 after P3 acceptance"
    assert json.loads((RUN / "truncation/verification_receipt.json").read_text())["passed"]
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "receipt.json").exists():
        raise FileExistsError("Seed diagnostic already completed")
    started = time.monotonic()
    old = pd.read_parquet(ROOT / "data/local/standard_w36_v1/units.parquet")
    selected = old.assign(identity_hash=old.pixel_track_id.map(lambda s: hashlib.sha256(s.encode()).hexdigest()))
    selected = selected.sort_values(["identity_hash", "pixel_track_id"]).head(512)
    current = pd.read_csv(RUN / "validation_units.csv")
    assert len(selected) == 512 and selected.pixel_track_id.is_unique
    assert set(selected.pixel_track_id).issubset(current.pixel_track_id)
    selected = selected[["pixel_track_id", "identity_hash"]].merge(current, on="pixel_track_id", validate="one_to_one")
    selected.to_csv(OUT / "selected_pixels.csv", index=False)
    config = dict(selection="same historical units SHA256(pixel_track_id), first 512",
        seeds=SEEDS, band="VV_dB", pixels=512, alpha=.025, replicates=3999,
        refined_replicates=19999, refinement=[.015, .035], incomplete_stand_subset=True,
        source_units_sha256=digest(ROOT / "data/local/standard_w36_v1/units.parquet"))
    (OUT / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    trace = pd.read_parquet(TRACE / "pixel_window_trace.parquet")
    subset = trace[trace.pixel_track_id.isin(selected.pixel_track_id)].copy()
    assert len(subset) == 512 * 9
    subset.to_parquet(OUT / "candidate_eligibility.parquet", index=False)
    candidates = subset[subset.pre_null].copy()
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")
    baseline = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    baseline = baseline[baseline.pixel_track_id.isin(selected.pixel_track_id)]
    expected = set(zip(baseline.pixel_track_id, baseline.change_time.dt.strftime("%Y-%m")))
    observed = set(zip(candidates.loc[candidates.accepted, "pixel_track_id"], candidates.loc[candidates.accepted, "candidate_month"]))
    assert expected == observed
    results, summaries = [], []
    for base_seed in SEEDS:
        rows = []
        for row in candidates.itertuples(index=False):
            score = ({key: getattr(row, key) for key in ("seed", "initial_p", "final_p", "refined", "permutations", "exceedance_count", "accepted")}
                if base_seed == SEEDS[0] else score_window(values[row.series_index, row.window_start_index:row.window_end_exclusive],
                    row.cusum_amplitude, row.pixel_track_id, row.window_start_index, base_seed))
            rows.append(dict(base_seed=base_seed, pixel_track_id=row.pixel_track_id, unit_id=row.unit_id,
                logical_window=row.logical_window, candidate_month=row.candidate_month,
                cusum_amplitude=row.cusum_amplitude, post_minus_pre=row.post_minus_pre, **score))
        frame = pd.DataFrame(rows)
        results.append(frame)
        chosen = frame[frame.accepted]
        found = set(zip(chosen.pixel_track_id, chosen.candidate_month))
        summaries.append(dict(base_seed=base_seed, pixels=512, pre_null_candidates=len(candidates),
            accepted_nodes=len(found), retained=len(found & expected), added=len(found - expected),
            lost=len(expected - found), jaccard=len(found & expected) / len(found | expected),
            acceptance_rate=len(chosen)/len(candidates), refined_candidates=int(frame.refined.sum()),
            scope="fixed_pixel_subset_monte_carlo_not_stand_accuracy"))
        print(json.dumps(summaries[-1]), flush=True)
    ledger = pd.concat(results, ignore_index=True)
    for _, group in ledger.groupby(["pixel_track_id", "logical_window"]):
        assert len(group) == 3
        for col in ("candidate_month", "cusum_amplitude", "post_minus_pre"):
            assert group[col].nunique() == 1, col
    for row in candidates.iloc[:6].iloc[::-1].itertuples(index=False):
        actual = score_window(values[row.series_index, row.window_start_index:row.window_end_exclusive],
            row.cusum_amplitude, row.pixel_track_id, row.window_start_index, SEEDS[0])
        for col in ("seed", "initial_p", "final_p", "refined", "permutations", "exceedance_count", "accepted"):
            assert actual[col] == getattr(row, col), col
    ledger.to_parquet(OUT / "candidate_scores.parquet", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    selected[["pixel_track_id", "track_key"]].merge(subset.groupby("pixel_track_id").n_valid.agg(["min", "max"]),
        on="pixel_track_id", validate="one_to_one").to_csv(OUT / "track_observation_support.csv", index=False)
    receipt = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started, same_candidate_statistics_all_seeds=True,
        baseline_frozen_events_exact=True, baseline_reordered_actual_score_checks=6,
        selected_pixels=512, pre_null_candidates=len(candidates), report_level_inference=False,
        source_trace_sha256=digest(TRACE / "pixel_window_trace.parquet"), script_sha256=digest(Path(__file__)),
        output_sha256={p.name: digest(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
