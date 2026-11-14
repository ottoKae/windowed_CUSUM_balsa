"""Small exact-randomization diagnosis; does not change block6 detection results."""
from dataclasses import replace
from itertools import permutations
import json
from math import factorial
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.detector.core import cusum_amplitude
from windowed_cusum.detector.detection import stable_seed
from windowed_cusum.detector.temporal_nulls import NullModelConfig, calendar_block_permutation_score, _cusum_range_finite


def main():
    out = ROOT / "runs/validation/configuration_extension_20261111_v1/block6"
    source = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
    old = ROOT / "runs/validation/balsa_research_execution_20261031_v3/simulation/validation_block3"
    execution = pd.read_parquet(out / "pixel_execution.parquet")
    ids = set(execution.unit_id)
    events3 = pd.read_parquet(old / "pixel_events.parquet")
    events3 = events3.loc[events3.unit_id.isin(ids)].copy()
    events6 = pd.read_parquet(out / "pixel_events.parquet")
    candidates = pd.read_parquet(ROOT / "runs/validation/balsa_results_ablation_20261101_v1/components/no_permutation_gate/pixel_events.parquet")
    candidates = candidates.loc[candidates.unit_id.isin(ids)]
    assert len(candidates) == int(execution.pre_null_candidates.sum())
    counts = pd.DataFrame(index=range(2017,2026))
    for name,frame in (("pre_null_candidates",candidates),("block3_accepted",events3),("block6_accepted",events6)):
        counts[name] = frame.groupby(pd.to_datetime(frame.decision_start).dt.year).size().reindex(counts.index,fill_value=0)
    counts.index.name = "decision_year"
    counts["block3_acceptance_fraction"] = counts.block3_accepted/counts.pre_null_candidates
    counts["block6_acceptance_fraction"] = counts.block6_accepted/counts.pre_null_candidates
    counts.to_csv(out / "accepted_by_decision_year.csv")
    components = pd.read_parquet(out / "octa/components.parquet")
    units = pd.read_parquet(source / "master_units.parquet").set_index("unit_id")
    values = np.load(source / "master_values.npy",mmap_mode="r")
    excluded = events3.loc[~events3.candidate_id.isin(events6.candidate_id)].sort_values("candidate_id")
    selected = []
    for calendar_length in (36,30):
        for row in excluded.itertuples():
            start = pd.Period(row.window_start,freq="M").ordinal-pd.Period("2016-01").ordinal
            end = pd.Period(row.window_end,freq="M").ordinal-pd.Period("2016-01").ordinal+1
            if end-start != calendar_length or row.null_p_value >= .01:
                continue
            window = values[int(units.loc[row.unit_id,"series_index"]),start:end]
            if np.isfinite(window).all():
                selected.append((row,start,np.asarray(window,dtype=np.float64)))
                break
    assert len(selected) == 2
    cases = []
    for row,start,window in selected:
        observed,peak = cusum_amplitude(window)
        assert observed == row.cusum_amplitude
        seed = stable_seed(f"{row.pixel_track_id}:VV_dB",20260820)+start
        scores = {}
        for length in (3,6):
            cfg = NullModelConfig(f"calendar_block{length}",3999,.025,length)
            score = calendar_block_permutation_score(window,observed,config=cfg,seed=seed)
            initial = score.null_p_value
            refined = .015 <= initial <= .035
            if refined:
                score = calendar_block_permutation_score(window,observed,config=replace(cfg,replicates=19999),seed=seed)
            scores[str(length)] = dict(initial_p=initial,final_p=score.null_p_value,
                replicates=score.replicates,refined=refined,accepted=score.admitted)
        assert scores["3"]["final_p"] == row.null_p_value and scores["3"]["replicates"] == row.permutations
        assert not scores["6"]["accepted"]
        offsets = []
        eps = 64*np.finfo(float).eps*max(1.,len(window)*float(np.max(np.abs(window))))
        for offset in range(6):
            blocks = ([window[:offset]] if offset else []) + [window[p:p+6] for p in range(offset,len(window),6)]
            statistics = []
            for order in permutations(range(len(blocks))):
                surrogate = np.concatenate([blocks[k] for k in order])
                statistics.append(_cusum_range_finite(surrogate,len(surrogate)))
            q = np.asarray(statistics)
            cycle_error = max(abs(_cusum_range_finite(np.concatenate(blocks[k:]+blocks[:k]),len(window))-observed) for k in range(len(blocks)))
            offsets.append(dict(offset=offset,blocks=len(blocks),permutations=len(q),
                p_raw=float(np.mean(q>=observed)),p_including_numerical_ties=float(np.mean(q>=observed-eps)),
                fraction_numerical_ties=float(np.mean(np.abs(q-observed)<=eps)),
                q_min=float(q.min()),q_median=float(np.median(q)),q_max=float(q.max()),
                max_cyclic_rotation_error=float(cycle_error),ideal_cyclic_minimum_p=1/factorial(len(blocks)-1)))
        exact_p = float(np.mean([r["p_raw"] for r in offsets]))
        corrected_p = (1+scores["6"]["replicates"]*exact_p)/(scores["6"]["replicates"]+1)
        cases.append(dict(candidate_id=row.candidate_id,unit_id=row.unit_id,
            window_start=str(row.window_start),window_end=str(row.window_end),calendar_months=len(window),
            valid_months=len(window),candidate_month=str(row.change_time),observed_Q=observed,
            selection="first lexical block3-accepted/block6-rejected candidate with block3 p<.01 and all months valid, at each target context length",
            monte_carlo=scores,enumeration_by_offset=offsets,exact_offset_mixture_p_raw=exact_p,
            exact_offset_mixture_p_including_numerical_ties=float(np.mean([r["p_including_numerical_ties"] for r in offsets])),
            ideal_offset_mixture_minimum_p=float(np.mean([r["ideal_cyclic_minimum_p"] for r in offsets])),
            monte_carlo_minus_exact_expected=float(scores["6"]["final_p"]-corrected_p),
            numerical_tie_tolerance_diagnostic_only=eps))
    diagnosis = dict(status="completed",scope="two purposive candidate diagnoses; frozen outputs unchanged",
        block6_accepted_by_decision_year=counts.reset_index().to_dict("records"),
        component_counts_by_reason=components.reason.value_counts().to_dict(),
        kernel_offset_rule="uniform random offset from 0 through 5 for every replicate; partial edge blocks retained",
        full36_block_counts=[6,7,7,7,7,7],terminal30_block_counts=[5,6,6,6,6,6],
        cyclic_property="CuSum range is invariant to cyclic shifts in exact arithmetic; exact >= comparisons may resolve floating-point ties differently",
        interpretation="Longer blocks preserve longer stretches of the observed low/high states, making large CuSum ranges more common in the permutation null for these candidates. The limited block orders also create ties. Neither the five-block offset-zero bound nor these two selected candidates proves that all terminal candidates cannot pass.",
        output_unchanged=True,additional_detector_run=False,cases=cases)
    (out / "diagnosis.json").write_text(json.dumps(diagnosis,indent=2)+"\n",encoding="utf-8")
    for case in cases:
        print(case["calendar_months"],case["observed_Q"],case["monte_carlo"],case["exact_offset_mixture_p_raw"],case["exact_offset_mixture_p_including_numerical_ties"])
    print(counts.to_string())


if __name__ == "__main__":
    main()
