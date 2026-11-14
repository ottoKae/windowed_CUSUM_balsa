"""Evaluate the frozen cumulative TDA reports using exact calendar-month matching."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import score


def main():
    out = score.OUT / "pm0"
    out.mkdir(parents=True, exist_ok=True)
    reference_path = score.ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    refs = pd.read_csv(reference_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - score.evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = score.evaluation.original_pairs(refs)
    assert (len(refs), len(cohort), len(pairs)) == (1049, 459, 590)
    paths = [reference_path, Path(__file__), Path(score.__file__), Path(score.evaluation.__file__),
             score.ROOT / "src/lr_cusum/event_scoring.py"]
    summaries, counts = [], []
    reference_counts = refs.groupby(["sample_id", "reference_month"]).size()
    for variant in score.VARIANTS:
        base = (score.FROZEN / "variants" / score.REUSED[variant]
                if variant in score.REUSED else score.OUT / "variants" / variant)
        path = base / "octa/reports.parquet"
        paths.append(path)
        reports = pd.read_parquet(path)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        assert reports.prediction_id.is_unique
        matches = score.evaluation.match_reports(refs, reports, "pm0")
        stand = score.evaluation.stand_counts(refs, reports, matches, cohort)
        history = score.evaluation.pair_matches(pairs, matches)
        # At zero tolerance each stand-month is a separate complete matching group.
        report_counts = reports.groupby(["sample_id", "representative_month"]).size()
        expected = int(np.minimum(reference_counts, report_counts.reindex(reference_counts.index, fill_value=0)).sum())
        assert matches.found.sum() == expected
        found = matches.loc[matches.found]
        assert found.absolute_error.eq(0).all() and found.prediction_id.is_unique
        summaries.append(score.evaluation.summary_row((variant, "octa", "pm0"), stand, history, matches))
        counts.append(stand.assign(variant_id=variant).reset_index())
        destination = out / "variants" / variant
        destination.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(destination / "matches.parquet", index=False)
        history.to_parquet(destination / "pairs.parquet", index=False)
    summary = pd.DataFrame(summaries)
    pm1_path = score.OUT / "pm1/summary.csv"
    paths.append(pm1_path)
    pm1 = pd.read_csv(pm1_path).set_index("variant_id").loc[summary.variant_id]
    np.testing.assert_array_equal(summary.n_reports, pm1.n_reports)
    assert (summary.n_matches.to_numpy() <= pm1.n_matches.to_numpy()).all()
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(counts, ignore_index=True).to_csv(out / "per_stand_counts.csv", index=False)
    receipt = dict(status="passed", tolerance_months=0, reference_stands=459, reference_records=1049,
        matching="same stand and exact representative calendar month; one-to-one",
        independent_stand_month_count_check=True, identical_report_counts=True,
        matches_no_greater_than_pm1=True, detector_rerun=False, aggregation_rerun=False,
        input_sha256={str(p.relative_to(score.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(summary[["variant_id", "n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
