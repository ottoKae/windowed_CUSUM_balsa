"""Match reference month +/-1 intervals to frozen TDA detection envelopes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import score


def main():
    out = score.OUT / "range_overlap_pm1"
    out.mkdir(parents=True, exist_ok=True)
    reference_path = score.ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    refs = pd.read_csv(reference_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - score.evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = score.evaluation.original_pairs(refs)
    draws_path = score.FROZEN / "paired_stand_indices.npy"
    axes_path = score.FROZEN / "paired_stand_axes.json"
    indices = np.load(draws_path)
    assert (len(refs), len(cohort), len(pairs)) == (1049, 459, 590)
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort and indices.shape == (5000,459)
    paths = [reference_path, draws_path, axes_path, Path(__file__), Path(score.__file__),
             Path(score.evaluation.__file__), score.ROOT / "src/windowed_cusum/evaluation/event_scoring.py"]
    summaries, counts, stand_tables = [], {}, []
    for variant in score.VARIANTS:
        base = (score.FROZEN / "variants" / score.REUSED[variant]
                if variant in score.REUSED else score.OUT / "variants" / variant)
        path = base / "octa/reports.parquet"
        paths.append(path)
        reports = pd.read_parquet(path)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        assert reports.prediction_id.is_unique
        # r in [a-1,b+1] is equivalent to [r-1,r+1] intersecting [a,b].
        # Expansion is only a matcher adapter; original stored spans stay unchanged.
        admissible = reports.copy()
        admissible["first_response_month"] = (pd.PeriodIndex(reports.first_response_month,freq="M")-1).astype(str)
        admissible["last_response_month"] = (pd.PeriodIndex(reports.last_response_month,freq="M")+1).astype(str)
        matches = score.evaluation.match_reports(refs, admissible, "envelope")
        bounds = reports.set_index("prediction_id")
        matches["reference_lower_month"] = (pd.PeriodIndex(refs.set_index("event_id").loc[matches.event_id,"reference_month"],freq="M")-1).astype(str)
        matches["reference_upper_month"] = (pd.PeriodIndex(refs.set_index("event_id").loc[matches.event_id,"reference_month"],freq="M")+1).astype(str)
        for column in ("first_response_month", "last_response_month", "span_months"):
            matches[column] = matches.prediction_id.map(bounds[column])
        found = matches.loc[matches.found]
        assert found.prediction_id.is_unique and found.event_id.is_unique
        assert (found.first_response_month <= found.reference_upper_month).all()
        assert (found.last_response_month >= found.reference_lower_month).all()
        stand = score.evaluation.stand_counts(refs, reports, matches, cohort)
        counts[variant] = stand
        history = score.evaluation.pair_matches(pairs, matches)
        summary = score.evaluation.summary_row((variant, "octa", "reference_pm1_interval_overlap"),stand,history,matches)
        summary.update(median_span_months=float(reports.span_months.median()),
                       reports_over_12_months=int(reports.span_months.gt(12).sum()),
                       max_span_months=int(reports.span_months.max()))
        summaries.append(summary)
        stand_tables.append(stand.assign(variant_id=variant).reset_index())
        target = out / "variants" / variant
        target.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(target / "matches.parquet", index=False)
        history.to_parquet(target / "pairs.parquet", index=False)
    summary = pd.DataFrame(summaries)
    pm1_path = score.OUT / "pm1/summary.csv"
    paths.append(pm1_path)
    pm1 = pd.read_csv(pm1_path).set_index("variant_id").loc[summary.variant_id]
    np.testing.assert_array_equal(summary.n_reports,pm1.n_reports)
    assert (summary.n_matches.to_numpy() >= pm1.n_matches.to_numpy()).all()
    effects = []
    for previous,current in zip(score.VARIANTS[:-1],score.VARIANTS[1:]):
        effects.append(dict(previous_variant=previous,current_variant=current,
            effect_direction="current_minus_previous",
            **score.evaluation.bootstrap_effect(counts[current],counts[previous],indices,"catalogue_f1",.95)))
    summary.to_csv(out / "summary.csv",index=False)
    pd.DataFrame(effects).to_csv(out / "adjacent_effects.csv",index=False)
    pd.concat(stand_tables,ignore_index=True).to_csv(out / "per_stand_counts.csv",index=False)
    receipt = dict(status="passed",reference_stands=459,reference_records=1049,
        matching="same stand; [reference_month-1,reference_month+1] intersects [first_response_month,last_response_month]",
        interval_endpoints_inclusive=True,one_to_one=True,tie_break="record identifiers as frozen envelope matcher",
        bootstrap_replicates=5000,reference_interval_interpretation="fixed operational tolerance, not measured optical observation bounds",
        original_report_ranges_unchanged=True,detector_rerun=False,aggregation_rerun=False,
        input_sha256={str(p.relative_to(score.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary[["variant_id","n_reports","n_matches","catalogue_precision","recall","catalogue_f1"]].to_string(index=False))
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()
