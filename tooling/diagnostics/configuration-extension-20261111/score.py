"""Evaluate the additional geometries and block null against the frozen baseline."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/validation/configuration_extension_20261111_v1"
OLD = ROOT / "runs/validation/range_primary_20261111_v1/sensitivity"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain"
sys.path.insert(0, str(ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"))
import evaluate as evaluation

SOURCES = {
    **{name: RUN / "geometry" / name / "octa/reports.parquet"
       for name in ("W24_D12_S12", "W36_D12_S6", "W36_D6_S6")},
    "block6": RUN / "block6/octa/reports.parquet",
}


def main():
    out = RUN / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    ref_path = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    refs = pd.read_csv(ref_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = evaluation.original_pairs(refs)
    draws_path = FROZEN / "paired_stand_indices.npy"
    axes_path = FROZEN / "paired_stand_axes.json"
    draws = np.load(draws_path)
    assert (len(cohort), len(refs), len(pairs)) == (459, 1049, 590)
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort and draws.shape == (5000, 459)

    baseline_reports_path = FROZEN / "variants/full/octa/reports.parquet"
    baseline_reports = pd.read_parquet(baseline_reports_path)
    baseline_reports = baseline_reports.loc[baseline_reports.sample_id.isin(cohort)]
    matches_by = {"full": pd.read_parquet(OLD / "variants/full/matches.parquet")}
    histories = {"full": pd.read_parquet(OLD / "variants/full/pairs.parquet")}
    counts = {"full": evaluation.stand_counts(refs, baseline_reports, matches_by["full"], cohort)}
    paths = [ref_path, draws_path, axes_path, baseline_reports_path,
             OLD / "summary.csv", OLD / "paired_effects.csv", Path(__file__),
             Path(evaluation.__file__), ROOT / "src/lr_cusum/event_scoring.py"]
    summaries, effects = [], []
    for name, source in SOURCES.items():
        reports = pd.read_parquet(source)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        assert reports.prediction_id.is_unique
        admissible = reports.copy()
        # This expansion is exactly reference +/-1 versus the original report range.
        for column, shift in (("first_response_month", -1), ("last_response_month", 1)):
            admissible[column] = (pd.PeriodIndex(reports[column], freq="M") + shift).astype(str)
        matches = evaluation.match_reports(refs, admissible, "envelope")
        bounds = reports.set_index("prediction_id")
        for column in ("first_response_month", "last_response_month", "span_months"):
            matches[column] = matches.prediction_id.map(bounds[column])
        found = matches.loc[matches.found]
        first = pd.PeriodIndex(found.first_response_month, freq="M").asi8 - evaluation.MONTH0
        last = pd.PeriodIndex(found.last_response_month, freq="M").asi8 - evaluation.MONTH0
        assert found.prediction_id.is_unique and found.event_id.is_unique
        assert (first <= found.reference_index.to_numpy() + 1).all()
        assert (last >= found.reference_index.to_numpy() - 1).all()
        stand = evaluation.stand_counts(refs, reports, matches, cohort)
        history = evaluation.pair_matches(pairs, matches)
        counts[name], matches_by[name], histories[name] = stand, matches, history
        summary = evaluation.summary_row((name, "octa", "reference_pm1_interval_overlap"), stand, history, matches)
        summary.update(median_span_months=float(reports.span_months.median()),
                       span_q25=float(reports.span_months.quantile(.25)),
                       span_q75=float(reports.span_months.quantile(.75)),
                       reports_over_12_months=int(reports.span_months.gt(12).sum()),
                       proportion_over_12_months=float(reports.span_months.gt(12).mean()),
                       max_span_months=int(reports.span_months.max()))
        supplementary = evaluation.match_reports(refs, reports, "pm2")
        point_counts = evaluation.stand_counts(refs, reports, supplementary, cohort)
        point_summary = evaluation.summary_row((name, "octa", "pm2"), point_counts,
                                              evaluation.pair_matches(pairs, supplementary), supplementary)
        for column in ("n_matches", "catalogue_precision", "recall", "catalogue_f1"):
            summary["secondary_representative_pm2_" + column] = point_summary[column]
        summaries.append(summary)
        directory = out / name
        directory.mkdir(exist_ok=True)
        matches.to_parquet(directory / "matches.parquet", index=False)
        history.to_parquet(directory / "pairs.parquet", index=False)
        supplementary.to_parquet(directory / "representative_pm2_matches.parquet", index=False)
        stand.to_csv(directory / "stand_counts.csv")
        paths.append(source)

    for left, right in [(name, "full") for name in SOURCES] + [("W36_D6_S6", "W36_D12_S6")]:
        a = matches_by[left].set_index("event_id")
        b = matches_by[right].set_index("event_id").reindex(a.index)
        pa = histories[left].set_index("pair_id")
        pb = histories[right].set_index("pair_id").reindex(pa.index)
        common, common_pairs = a.found & b.found, pa.found & pb.found
        effect = dict(variant_id=left, comparator=right, effect_direction="variant_minus_comparator",
                      **evaluation.bootstrap_effect(counts[left], counts[right], draws, "catalogue_f1", .95),
                      n_matched_new=int((a.found & ~b.found).sum()),
                      n_matched_lost=int((~a.found & b.found).sum()),
                      n_common_references=int(common.sum()), n_common_original_pairs=int(common_pairs.sum()))
        for kind, x, y, mask in (("date", a, b, common), ("interval", pa, pb, common_pairs)):
            lo, hi, valid = evaluation.common_error_ci(x.loc[mask, ["sample_id", "absolute_error"]],
                                                      y.loc[mask, ["sample_id", "absolute_error"]], cohort, draws)
            effect.update({f"variant_common_{kind}_MAE": float(x.loc[mask, "absolute_error"].mean()),
                           f"comparator_common_{kind}_MAE": float(y.loc[mask, "absolute_error"].mean()),
                           f"common_{kind}_MAE_difference": float((x.loc[mask, "absolute_error"] - y.loc[mask, "absolute_error"]).mean()),
                           f"common_{kind}_MAE_difference_ci_low": lo,
                           f"common_{kind}_MAE_difference_ci_high": hi,
                           f"common_{kind}_bootstrap_valid": valid})
        effects.append(effect)

    summary = pd.concat([pd.read_csv(OLD / "summary.csv"), pd.DataFrame(summaries)], ignore_index=True)
    old_effects = pd.read_csv(OLD / "paired_effects.csv").assign(comparator="full")
    contrasts = pd.concat([old_effects, pd.DataFrame(effects)], ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    contrasts.to_csv(out / "paired_effects.csv", index=False)
    baseline = summary.loc[summary.variant_id.eq("full")].iloc[0]
    assert (baseline.n_reports, baseline.n_matches) == (1208, 945)
    (out / "receipt.json").write_text(json.dumps({
        "status": "passed", "reference_stands": 459, "reference_records": 1049,
        "original_reference_pairs": 590, "bootstrap_replicates": 5000,
        "matching": "reference +/-1 intersects original TDA range, inclusive, maximum-cardinality one-to-one, identifier tie break",
        "new_configurations": list(SOURCES), "existing_configurations_reused": 11,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(summary[["variant_id", "n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1", "n_original_pairs_retained", "median_span_months"]].to_string(index=False))
    print(pd.DataFrame(effects)[["variant_id", "comparator", "difference", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
