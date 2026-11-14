"""Re-evaluate frozen cumulative TDA reports with a one-month matching tolerance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import score


def main():
    root, evaluation = score.ROOT, score.evaluation
    out = score.OUT / "pm1"
    out.mkdir(parents=True, exist_ok=True)
    reference_path = root / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    draws_path = score.FROZEN / "paired_stand_indices.npy"
    axes_path = score.FROZEN / "paired_stand_axes.json"
    refs = pd.read_csv(reference_path, keep_default_na=False)
    refs["reference_index"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8 - evaluation.MONTH0
    cohort = sorted(refs.sample_id.unique())
    pairs = evaluation.original_pairs(refs)
    indices = np.load(draws_path)
    assert (len(refs), len(cohort), len(pairs)) == (1049, 459, 590)
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort and indices.shape == (5000, 459)
    paths = [reference_path, draws_path, axes_path, Path(__file__), Path(score.__file__),
             score.EVALUATION_CODE / "evaluate.py", root / "src/windowed_cusum/evaluation/event_scoring.py"]
    summaries, counts, per_stand = [], {}, []
    for variant in score.VARIANTS:
        base = (score.FROZEN / "variants" / score.REUSED[variant]
                if variant in score.REUSED else score.OUT / "variants" / variant)
        report_path = base / "octa/reports.parquet"
        paths.append(report_path)
        reports = pd.read_parquet(report_path)
        reports = reports.loc[reports.sample_id.isin(cohort)].copy()
        matches = evaluation.match_reports(refs, reports, "pm1")
        stand = evaluation.stand_counts(refs, reports, matches, cohort)
        history = evaluation.pair_matches(pairs, matches)
        counts[variant] = stand
        summary = evaluation.summary_row((variant, "octa", "pm1"), stand, history, matches)
        summary["adjacent_pair_recall"] = summary["n_original_pairs_retained"] / len(pairs)
        summaries.append(summary)
        per_stand.append(stand.assign(variant_id=variant).reset_index())
        target = out / "variants" / variant
        target.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(target / "matches.parquet", index=False)
        history.to_parquet(target / "pairs.parquet", index=False)
    summary = pd.DataFrame(summaries)
    pm2_path = score.OUT / "summary.csv"
    pm2 = pd.read_csv(pm2_path).set_index("variant_id").loc[summary.variant_id]
    np.testing.assert_array_equal(summary.n_reports, pm2.n_reports)
    assert (summary.n_matches.to_numpy() <= pm2.n_matches.to_numpy()).all()
    frozen_summary_path = score.FROZEN / "summary_by_stage.csv"
    frozen = pd.read_csv(frozen_summary_path)
    columns = ["n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1", "n_original_pairs_retained"]
    for variant, original in score.REUSED.items():
        expected = frozen.loc[(frozen.variant_id == original) & (frozen.output_level == "octa")
                              & (frozen.matching_rule == "pm1"), columns]
        actual = summary.loc[summary.variant_id == variant, columns]
        np.testing.assert_allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=0, atol=1e-14)
    effects = []
    for previous, current in zip(score.VARIANTS[:-1], score.VARIANTS[1:]):
        effect = evaluation.bootstrap_effect(counts[current], counts[previous], indices, "catalogue_f1", .95)
        effects.append(dict(previous_variant=previous, current_variant=current,
                            effect_direction="current_minus_previous", **effect))
    effects = pd.DataFrame(effects)
    frozen_effects_path = score.FROZEN / "paired_effects.csv"
    old = pd.read_csv(frozen_effects_path)
    old = old.loc[(old.variant == "no_permutation_gate") & (old.output_level == "octa")
                  & (old.matching_rule == "pm1") & (old.metric == "catalogue_f1")].iloc[0]
    last = effects.iloc[-1]
    np.testing.assert_allclose([last.difference, last.ci_low, last.ci_high],
                              [-old.difference, -old.ci_high, -old.ci_low], rtol=0, atol=1e-14)
    summary.to_csv(out / "summary.csv", index=False)
    effects.to_csv(out / "adjacent_effects.csv", index=False)
    pd.concat(per_stand, ignore_index=True).to_csv(out / "per_stand_counts.csv", index=False)
    paths += [pm2_path, frozen_summary_path, frozen_effects_path]
    receipt = dict(status="passed", matching_tolerance_months=1, reference_stands=459,
        reference_events=1049, original_pairs=590, bootstrap_replicates=5000,
        detector_rerun=False, aggregation_rerun=False, same_report_counts_as_pm2=True,
        total_matches_no_greater_than_pm2=True, frozen_last_two_metric_parity=True,
        final_increment_ci_parity=True, input_sha256={str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
