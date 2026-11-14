"""Score supplied TDA reports using the paper's range matcher and paired bootstrap."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/diagnostics/balsa-results-optimization-20261105")]
import evaluate as evaluation


def score(reference_path, report_paths, output, draws_path=None, seed=20261114, ci=.95):
    refs = pd.read_csv(reference_path, dtype={"event_id": str, "sample_id": str})
    if refs.empty or not refs.event_id.is_unique:
        raise ValueError("Reference records need nonempty, unique event_id values")
    reference_months = calendar_months(refs.reference_month, "Reference")
    refs["reference_index"] = reference_months.asi8 - evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = evaluation.original_pairs(refs)
    if pairs.empty:
        pairs = pd.DataFrame(columns=["pair_id", "sample_id", "left", "right", "reference_interval"])
    draws = (np.load(draws_path, allow_pickle=False) if draws_path else
             np.random.default_rng(seed).integers(0, len(cohort), (5000, len(cohort))))
    if draws.shape != (5000, len(cohort)) or not np.issubdtype(draws.dtype, np.integer) or draws.min() < 0 or draws.max() >= len(cohort):
        raise ValueError("Draws must be 5000 x number-of-stands integer indices into sorted stand IDs")
    output.mkdir(parents=True, exist_ok=False)
    counts, summaries, contrasts = [], [], []
    for variant, path in report_paths:
        reports = pd.read_csv(path, dtype={"prediction_id": str, "sample_id": str})
        if not reports.prediction_id.is_unique:
            raise ValueError("Report prediction_id must be unique")
        dates = {name: calendar_months(reports[name], "Report") for name in
                 ("first_response_month", "last_response_month", "representative_month")}
        if (dates["first_response_month"] > dates["last_response_month"]).any():
            raise ValueError("Report first_response_month must not exceed last_response_month")
        if ((dates["representative_month"] < dates["first_response_month"]) |
                (dates["representative_month"] > dates["last_response_month"])).any():
            raise ValueError("Report representative_month must lie within the detection range")
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        admitted = reports.copy()
        admitted["first_response_month"] = (pd.PeriodIndex(reports.first_response_month, freq="M") - 1).astype(str)
        admitted["last_response_month"] = (pd.PeriodIndex(reports.last_response_month, freq="M") + 1).astype(str)
        matches = evaluation.match_reports(refs, admitted, "envelope")
        c = evaluation.stand_counts(refs, reports, matches, cohort)
        history = evaluation.pair_matches(pairs, matches)
        summaries.append(evaluation.summary_row((variant, "TDA", "reference_pm1_interval_overlap"), c, history, matches))
        if counts:
            contrasts.append(dict(previous=counts[-1][0], current=variant,
                                  **evaluation.bootstrap_effect(c, counts[-1][1], draws, "catalogue_f1", ci)))
        counts.append((variant, c))
        matches.to_csv(output / f"matches_{len(counts):02d}.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "adjacent_effects.csv", index=False)
    (output / "evaluation.json").write_text(json.dumps(dict(
        reference_stands=len(cohort), reference_records=len(refs), confidence_level=ci,
        bootstrap_replicates=len(draws), supplied_draws=bool(draws_path), seed=seed,
        draw_axis=cohort, matching="one-to-one TDA envelope overlap with reference month +/-1 month"), indent=2) + "\n")
    return summaries


def calendar_months(values, label):
    try:
        months = pd.PeriodIndex(values, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} months must be valid calendar months") from error
    if months.isna().any():
        raise ValueError(f"{label} months must be nonmissing")
    return months


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reports", nargs=2, action="append", metavar=("VARIANT", "CSV"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=Path)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--ci", type=float, choices=[.95, .975], default=.95)
    args = parser.parse_args()
    score(args.references, args.reports, args.output, args.draws, args.seed, args.ci)
