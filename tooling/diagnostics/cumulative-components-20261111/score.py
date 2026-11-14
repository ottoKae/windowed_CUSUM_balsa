"""Score cumulative detector configurations with frozen TDA matching and draws."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/validation/cumulative_components_20261111_v1"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain"
EVALUATION_CODE = ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"
sys.path.insert(0, str(EVALUATION_CODE))
import evaluate as evaluation

VARIANTS = (
    "F0_full_record", "F1_local_windows", "F2_decision_interval",
    "F3_direction", "F4_permutation",
)
REUSED = {"F3_direction": "no_permutation_gate", "F4_permutation": "full"}


def main():
    reference_path = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    draws_path = FROZEN / "paired_stand_indices.npy"
    axes_path = FROZEN / "paired_stand_axes.json"
    refs = pd.read_csv(reference_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = evaluation.original_pairs(refs)
    axes = json.loads(axes_path.read_text(encoding="utf-8"))
    indices = np.load(draws_path)
    assert (len(refs), len(cohort), len(pairs)) == (1049, 459, 590)
    assert refs.event_id.is_unique and axes["sample_ids"] == cohort
    assert indices.shape == (5000, 459) and indices.min() >= 0 and indices.max() < 459
    input_paths = [reference_path, draws_path, axes_path, EVALUATION_CODE / "evaluate.py",
                   EVALUATION_CODE / "run.py", ROOT / "src/lr_cusum/event_scoring.py", Path(__file__)]
    summaries, counts, matched, per_stand = [], {}, {}, []
    for variant in VARIANTS:
        base = FROZEN / "variants" / REUSED[variant] if variant in REUSED else OUT / "variants" / variant
        report_path = base / "octa/reports.parquet"
        input_paths.append(report_path)
        reports = pd.read_parquet(report_path)
        assert reports.prediction_id.is_unique
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        matches = evaluation.match_reports(refs, reports, "pm2")
        stand = evaluation.stand_counts(refs, reports, matches, cohort)
        history = evaluation.pair_matches(pairs, matches)
        counts[variant], matched[variant] = stand, matches
        summary = evaluation.summary_row((variant, "octa", "pm2"), stand, history, matches)
        summary.update(output_label="TDA", source="frozen_reuse" if variant in REUSED else "new_cumulative",
                       adjacent_pair_recall=summary["n_original_pairs_retained"] / len(pairs))
        summaries.append(summary)
        stand = stand.copy()
        stand["variant_id"] = variant
        stand["catalogue_precision"] = stand.hit / stand.pred.replace(0, np.nan)
        stand["recall"] = stand.hit / stand.n
        stand["catalogue_f1"] = 2 * stand.hit / (stand.n + stand.pred)
        stand["n_original_pairs_retained"] = history.groupby("sample_id").found.sum().reindex(cohort, fill_value=0)
        per_stand.append(stand.reset_index())
        target = OUT / "variants" / variant / "metrics"
        target.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(target / "octa_pm2_matches.parquet", index=False)
        history.to_parquet(target / "octa_pm2_pairs.parquet", index=False)
    summary_table = pd.DataFrame(summaries)
    frozen_summary_path = FROZEN / "summary_by_stage.csv"
    input_paths.append(frozen_summary_path)
    frozen_summary = pd.read_csv(frozen_summary_path)
    parity_columns = ["n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1",
                      "n_original_pairs_retained", "n_multiref_stands_fully_recovered"]
    for variant, original in REUSED.items():
        expected = frozen_summary.loc[(frozen_summary.variant_id == original)
            & (frozen_summary.output_level == "octa") & (frozen_summary.matching_rule == "pm2"), parity_columns]
        actual = summary_table.loc[summary_table.variant_id == variant, parity_columns]
        np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=0, atol=1e-14)
    effects, transitions = [], []
    for previous, current in zip(VARIANTS[:-1], VARIANTS[1:]):
        effect = evaluation.bootstrap_effect(counts[current], counts[previous], indices, "catalogue_f1", .95)
        a = matched[current].set_index("event_id")
        b = matched[previous].set_index("event_id").reindex(a.index)
        new, lost = a.found & ~b.found, ~a.found & b.found
        common = a.found & b.found
        effects.append(dict(previous_variant=previous, current_variant=current,
            effect_direction="current_minus_previous", **effect,
            n_report_difference=int(counts[current].pred.sum() - counts[previous].pred.sum()),
            n_match_difference=int(counts[current].hit.sum() - counts[previous].hit.sum()),
            n_matched_new=int(new.sum()), n_matched_lost=int(lost.sum()),
            n_common_references=int(common.sum()), n_common_month_shifted=int((common & a.report_index.ne(b.report_index)).sum())))
        change = a[["sample_id", "found", "prediction_id", "report_index"]].add_prefix("current_")
        change = change.join(b[["found", "prediction_id", "report_index"]].add_prefix("previous_"))
        transitions.append(change.assign(previous_variant=previous, current_variant=current,
                                        new=new, lost=lost).reset_index())
    effect_table = pd.DataFrame(effects)
    frozen_effects_path = FROZEN / "paired_effects.csv"
    input_paths.append(frozen_effects_path)
    old = pd.read_csv(frozen_effects_path)
    old = old.loc[(old.variant == "no_permutation_gate") & (old.output_level == "octa")
                  & (old.matching_rule == "pm2") & (old.metric == "catalogue_f1")].iloc[0]
    last = effect_table.iloc[-1]
    np.testing.assert_allclose([last.difference, last.ci_low, last.ci_high],
                              [-old.difference, -old.ci_high, -old.ci_low], rtol=0, atol=1e-14)
    summary_table.to_csv(OUT / "summary.csv", index=False)
    pd.concat(per_stand, ignore_index=True).to_csv(OUT / "per_stand_metrics.csv", index=False)
    effect_table.to_csv(OUT / "adjacent_effects.csv", index=False)
    pd.concat(transitions, ignore_index=True).to_parquet(OUT / "adjacent_reference_transitions.parquet", index=False)
    receipt = dict(status="passed", reference_stands=len(cohort), references=len(refs), original_pairs=len(pairs),
        tolerance_months=2, output_level="TDA_parents", bootstrap_replicates=len(indices),
        bootstrap_draws="reused_exact_frozen_array", effect_direction="current_minus_previous",
        last_two_metric_parity=True, final_increment_ci_parity=True, variants=list(VARIANTS),
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths})
    (OUT / "scoring_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary_table.to_string(index=False), flush=True)
    print(effect_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
