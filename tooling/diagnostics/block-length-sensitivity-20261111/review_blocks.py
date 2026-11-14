"""Finite-block mathematics and two frozen candidate replays; no detector rerun."""
from dataclasses import replace
from fractions import Fraction
from math import factorial
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from lr_cusum.core import cusum_amplitude
from lr_cusum.detection import stable_seed
from lr_cusum.temporal_nulls import NullModelConfig, calendar_block_permutation_score


def main():
    out = ROOT / "runs/validation/block_length_sensitivity_20261111_v1/review"
    out.mkdir(exist_ok=True)
    source = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
    old_path = ROOT / "runs/validation/configuration_extension_20261111_v1/block6/diagnosis.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    units = pd.read_parquet(source / "master_units.parquet").set_index("unit_id")
    values = np.load(source / "master_values.npy", mmap_mode="r")
    geometry = []
    for n in (36, 30):
        for length in (3, 4, 5, 6):
            offsets = []
            for offset in range(length):
                sizes = ([offset] if offset else []) + [min(length, n-p) for p in range(offset,n,length)]
                m = len(sizes)
                offsets.append(dict(offset=offset, block_lengths=sizes, blocks=m,
                    labelled_orders=factorial(m), cyclic_classes_upper_bound=factorial(m-1)))
            floor = sum((Fraction(1, x["cyclic_classes_upper_bound"]) for x in offsets), Fraction()) / length
            geometry.append(dict(calendar_months=n, block_length=length, offsets=offsets,
                exact_arithmetic_offset_mixture_cyclic_floor=float(floor), floor_fraction=str(floor)))
    cases = []
    for case in old["cases"]:
        start = pd.Period(case["window_start"],freq="M").ordinal-pd.Period("2016-01",freq="M").ordinal
        end = pd.Period(case["window_end"],freq="M").ordinal-pd.Period("2016-01",freq="M").ordinal+1
        window = np.asarray(values[int(units.loc[case["unit_id"],"series_index"]),start:end], dtype=np.float64)
        observed, _ = cusum_amplitude(window)
        assert observed == case["observed_Q"] and np.isfinite(window).all()
        seed = stable_seed(case["candidate_id"].split("|")[0],20260820)+start
        scores = {}
        for length in (3, 4, 5, 6):
            cfg = NullModelConfig(f"calendar_block{length}",3999,.025,length)
            initial = calendar_block_permutation_score(window,observed,config=cfg,seed=seed)
            refined = .015 <= initial.null_p_value <= .035
            final = calendar_block_permutation_score(window,observed,config=replace(cfg,replicates=19999),seed=seed) if refined else initial
            scores[str(length)] = dict(initial_p=initial.null_p_value, final_p=final.null_p_value,
                exceedances=final.exceedance_count, replicates=final.replicates,
                refined=refined, accepted=final.admitted)
            if length in (3, 6):
                assert final.null_p_value == case["monte_carlo"][str(length)]["final_p"]
        cases.append(dict(candidate_id=case["candidate_id"],unit_id=case["unit_id"],
            calendar_months=len(window),candidate_month=case["candidate_month"],
            observed_Q=observed,seed=seed,scores=scores))
    for length in (4, 5):
        accepted = pd.read_parquet(out.parent/f"block{length}/pixel_events.parquet",columns=["candidate_id","null_p_value","permutations"])
        for case in cases:
            row = accepted.loc[accepted.candidate_id.eq(case["candidate_id"])]
            score = case["scores"][str(length)]
            assert bool(len(row)) == score["accepted"]
            if len(row):
                assert len(row)==1 and row.iloc[0].null_p_value==score["final_p"] and row.iloc[0].permutations==score["replicates"]
    payload = dict(status="passed",scope="Block geometry and two purposively selected real-candidate scores only",
        two_case_block4_block5_production_output_parity=True,
        source_diagnosis=str(old_path.relative_to(ROOT)),
        source_diagnosis_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256((ROOT/"src/lr_cusum/temporal_nulls.py").read_bytes()).hexdigest(),
        geometry=geometry,cases=cases,notes=[
            "Each replicate draws a uniform calendar offset, retains partial edge blocks, and uniformly permutes their order.",
            "Missing values are removed after calendar-block assembly; their original absolute locations are not preserved.",
            "For centered residuals summing exactly to zero, a cyclic shift translates and reorders the cumulative path; its range Q is unchanged.",
            "m! labelled orders yield at most (m-1)! cyclic equivalence classes for Q at each offset, not independent observations or necessarily distinct Q values.",
            "Offset distributions are mixed with equal offset weights, not weighted by their factorial numbers of orders.",
            "The displayed cyclic probability floor holds for exhaustive probabilities in exact arithmetic. Floating ties and finite Monte Carlo scores need separate treatment.",
            "Neither L6 floor exceeds .025; block count alone does not forbid acceptance, including terminal contexts.",
            "More Monte Carlo repetitions improve sampling precision but add no independent temporal blocks.",
            "Long blocks retain both dependence and potentially harvest-related persistent state changes. This is a null-sensitivity stress test, not a validated autocorrelation correction.",
            "These two candidates were selected using L3 acceptance and L6 rejection and cannot estimate population effects or false-alarm calibration.",
            "No production code, TeX, full detector outputs, or reference labels were modified."])
    (out/"finite_block_review.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    rows = ["# Finite-block review", "", "The implemented comparison is a stress test of the temporal null at fixed W=36 and alpha=0.025. It does not establish a calibrated autocorrelation correction.", "", "| Calendar months | L | Blocks by offset | Q cyclic-class upper bound by offset | Exact-arithmetic mixture floor |", "|---:|---:|---|---|---:|"]
    for row in geometry:
        rows.append(f"| {row['calendar_months']} | {row['block_length']} | "+", ".join(str(x['blocks']) for x in row['offsets'])+" | "+", ".join(str(x['cyclic_classes_upper_bound']) for x in row['offsets'])+f" | {row['exact_arithmetic_offset_mixture_cyclic_floor']:.8g} |")
    rows += ["", "These are labelled-order equivalence bounds, not effective sample sizes. Further data-dependent ties can reduce distinct Q values. The mixture floors are exact-arithmetic reference bounds, not universal bounds on a finite Monte Carlo implementation.", "", "## Same-candidate production replay", "", "| Calendar months | Q | L3 p | L4 p | L5 p | L6 p |", "|---:|---:|---:|---:|---:|---:|"]
    for case in cases:
        rows.append(f"| {case['calendar_months']} | {case['observed_Q']:.8f} | "+" | ".join(f"{case['scores'][str(length)]['final_p']:.6f}" for length in (3,4,5,6))+" |")
    rows += ["", "Both cases preserve the production candidate seed and 3,999/19,999 adaptive scoring rule. They were selected previously as L3-accepted/L6-rejected cases, so they diagnose mechanisms rather than estimate average effects.", "", "## Interpretation", "", "A larger block preserves a longer observed high/low-state sequence. With only 30-36 calendar months it simultaneously leaves fewer rearrangeable pieces; Q-invariant cyclic orders and other ties further limit the distinct statistic values. Retaining event-related structure in the null can lower detection sensitivity. This should be described as a joint finite-window and null-specification effect, not proof that dependence-aware nulls are invalid or that IID scores control false alarms.", "", "The 20% distinct-member TDA gate amplifies pixel-level attrition: the prior L6 run retained 1,216 nodes, formed 543 components, and removed 520 for insufficient support. It is therefore necessary to separate pixel screening from the final stand-event recall.", "", "Block-length selection in dependent resampling is method- and dependence-specific; [Politis and White, author-hosted paper](https://www.math.ucsd.edu/~politis/SBblock-revER.pdf) provides relevant context. Its bootstrap theory does not directly validate this raw-level calendar-block permutation test.", ""]
    (out/"FINITE_BLOCK_REVIEW.md").write_text("\n".join(rows),encoding="utf-8")
    print(json.dumps(dict(geometry=[{k:v for k,v in x.items() if k!='offsets'} | {'blocks':[y['blocks'] for y in x['offsets']]} for x in geometry],cases=cases),indent=2))


if __name__ == "__main__":
    main()
