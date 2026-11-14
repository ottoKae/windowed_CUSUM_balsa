"""Production STP synthetic validation; labels are used only after prediction."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.stand import star_segmentation as stars

OUT = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/synthetic"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    gate = json.loads((args.out / "implementation_gate.json").read_text(encoding="utf-8"))
    if gate["status"] != "passed" or gate["production_sha256"] != digest(ROOT / "src/windowed_cusum/stand/star_segmentation.py"):
        raise ValueError("P1a must pass on the current production implementation")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    inputs, truth = [], []
    for case in cases:
        reports, members, units = input_tables(case["first"], case["case_id"])
        inputs.append(members.merge(units, on="unit_id").assign(
            scenario=case["scenario"], trial=case["trial"], algorithm_seed=case["seed"]))
        truth.append(dict(case_id=case["case_id"], truth_rounds=case["truth_rounds"],
                          labels=json.dumps(case["labels"]), scenario=case["scenario"]))
    pd.concat(inputs, ignore_index=True).to_parquet(args.out / "inputs.parquet", index=False)
    pd.DataFrame(truth).to_parquet(args.out / "truth.parquet", index=False)
    code = {str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), ROOT / "src/windowed_cusum/stand/star_segmentation.py")}
    receipt_path = args.out / "receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior["code_sha256"] != code:
            raise ValueError("Do not mix different implementations in synthetic checkpoints")
    path = args.out / "trials.parquet"
    rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
    completed = {x["case_id"] for x in rows}
    pending = [x for x in cases if x["case_id"] not in completed]
    start = perf_counter()
    receipt = dict(status="running", code_sha256=code, total_trials=len(cases), completed=len(rows),
                   config=asdict(stars.Config()), data_seed=20261105,
                   seed_contract="SeedSequence([20261102, case_order]); independent of data RNG",
                   scope="conditional_STP_synthetic_not_end_to_end_harvest_accuracy")
    write_json(receipt_path, receipt)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(evaluate_case, pending, chunksize=1):
            rows.append(row)
            if len(rows) % 40 == 0 or len(rows) == len(cases):
                pd.DataFrame(rows).to_parquet(path, index=False)
                receipt.update(completed=len(rows), elapsed_this_run_seconds=perf_counter()-start)
                write_json(receipt_path, receipt)
                print(f"P1b {len(rows)}/{len(cases)}; {perf_counter()-start:.1f}s", flush=True)
    table = pd.DataFrame(rows)
    summaries = []
    for name, group in table.groupby("scenario", sort=True):
        n, accepted = len(group), int(group.split_accepted.sum())
        finite = group.p_value.notna()
        summaries.append(dict(scenario=name, trials=n, accepted=accepted,
            split_acceptance_rate=accepted/n, split_ci_low=binomial_interval(accepted, n)[0],
            split_ci_high=binomial_interval(accepted, n)[1], computable_tests=int(finite.sum()),
            test_rejections=int((group.p_value <= .05).sum()),
            test_rejection_rate=float((group.loc[finite, "p_value"] <= .05).mean()),
            correct_count_rate=float(group.correct_count.mean()),
            count_MAE=float(group.report_count_error.abs().mean()),
            spatial_IoU=float(group.spatial_iou.mean()),
            source_conservation=bool(group.source_conservation.all())))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / "summary.csv", index=False)
    summary.loc[summary.scenario.str.startswith("NULL_")].to_csv(args.out / "null_calibration.csv", index=False)
    table.groupby(["scenario", "reason"]).size().rename("trials").reset_index().to_csv(
        args.out / "fallback_reasons.csv", index=False)
    receipt.update(status="passed", completed=len(rows), elapsed_this_run_seconds=perf_counter()-start,
                   source_conservation=bool(table.source_conservation.all()),
                   gate_means_engineering_not_positive_scientific_result=True)
    write_json(receipt_path, receipt)
    print(summary.to_string(index=False), flush=True)


def input_tables(first, case_id="synthetic", total_pixels=None):
    first = np.asarray(first, dtype=np.int64)
    side = int(round(np.sqrt(len(first))))
    r, c = np.divmod(np.arange(len(first)), side)
    unit_ids = [f"{case_id}:u{i:03d}" for i in range(len(first))]
    units = pd.DataFrame(dict(unit_id=unit_ids, native_row=r, native_col=c, track_key="synthetic_track"))
    members = pd.DataFrame(dict(component_id=case_id, sample_id=case_id, unit_id=unit_ids,
                               node_id=[f"{case_id}:n{i:03d}" for i in range(len(first))], month_index=first))
    reports = pd.DataFrame([dict(component_id=case_id, sample_id=case_id, prediction_id=case_id,
                                total_pixels=len(first) if total_pixels is None else total_pixels)])
    return reports, members, units


def predict(first, config, case_id="synthetic", total_pixels=None):
    return stars.segment(*input_tables(first, case_id, total_pixels), config)


def make_cases(repeats=120, null_repeats=300):
    rng = np.random.default_rng(20261105)
    r, c = np.indices((6, 6)).reshape(2, -1)
    cases = []
    for trial in range(repeats):
        specs = [
            ("S1_single_wave", rng.integers(6, 10, 36), 1, None),
            ("S2_two_blocks", np.where(c < 3, 6, 18), 2, (c >= 3).astype(int)),
            ("S3_ramp", c.copy(), 1, None),
            ("S4_stripes", np.where(c % 2 == 0, 6, 18), 2, c % 2),
            ("S5_undersupported", np.where((c < 5) | (r >= 1), 6, 18), 2,
             ((c >= 5) & (r < 1)).astype(int)),
            ("S6_quadrants", np.where(r < 3, np.where(c < 3, 6, 8), np.where(c < 3, 18, 20)),
             2, (r >= 3).astype(int)),
            ("S7_smooth_noise", np.round(8 + 3*np.sin(c/2.) + rng.normal(0, 1.5, 36)).astype(int), 1, None),
            ("S8_iid", rng.integers(0, 30, 36), 1, None)]
        for name, first, count, labels in specs:
            cases.append(dict(case_id=f"{name}_{trial:03d}", scenario=name, trial=trial,
                              first=first.tolist(), truth_rounds=count,
                              labels=None if labels is None else labels.tolist()))
    for side in (5, 6, 8, 10):
        for trial in range(null_repeats):
            name = f"NULL_{side*side}"
            cases.append(dict(case_id=f"{name}_{trial:03d}", scenario=name, trial=trial,
                              first=rng.integers(0, 30, side*side).tolist(), truth_rounds=1, labels=None))
    for order, case in enumerate(cases):
        case["seed"] = int(np.random.SeedSequence([20261102, order]).generate_state(1)[0])
    return cases


def evaluate_case(case):
    result = predict(case["first"], stars.Config(seed=case["seed"]), case["case_id"])
    row = result["stands"].iloc[0]
    ownership = result["node_ownership"]
    ids = input_tables(case["first"], case["case_id"])[1].node_id
    conserved = ownership.node_id.is_unique and set(ownership.node_id) == set(ids)
    if not conserved or ownership[["zone_id", "round_id"]].isna().any().any():
        raise AssertionError("Production source ownership violated")
    iou, extra = np.nan, np.nan
    if case["labels"] is not None:
        truth = np.asarray(case["labels"])
        predicted = ownership.set_index("node_id").loc[ids, "round_id"].to_numpy()
        truth_groups, predicted_groups = np.unique(truth), np.unique(predicted)
        scores = np.array([[np.count_nonzero((truth == a) & (predicted == b)) /
                            np.count_nonzero((truth == a) | (predicted == b))
                            for b in predicted_groups] for a in truth_groups])
        a, b = linear_sum_assignment(-scores)
        iou = float(scores[a, b].sum() / len(truth_groups))
        extra = len(predicted_groups)-len(b)
    count = len(result["rounds"])
    return dict(case_id=case["case_id"], scenario=case["scenario"], trial=case["trial"], seed=case["seed"],
        truth_rounds=case["truth_rounds"], reported_rounds=count,
        split_accepted=bool(row.destination_type == "zone_rounds" and count > 1),
        reason=row.reason, admissible=bool(row.admissible), p_value=row.p_value,
        report_count_error=count-case["truth_rounds"], correct_count=count == case["truth_rounds"],
        spatial_iou=iou, extra_spatial_groups=extra, source_conservation=conserved)


def binomial_interval(k, n):
    return (0. if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1. if k == n else float(beta.ppf(.975, k+1, n-k)))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
