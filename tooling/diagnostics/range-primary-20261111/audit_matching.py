"""Independently audit saved range matching and paired bootstrap results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "runs/validation"
FROZEN = VALIDATION / "balsa_results_optimization_20261105_v1/fullchain"
OUT = VALIDATION / "range_primary_20261111_v1/audit_matching.json"
sys.path.insert(0, str(ROOT / "src"))
from windowed_cusum.evaluation.event_scoring import _canonical_matching


def main():
    reference_path = VALIDATION / "balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    refs = pd.read_csv(reference_path, keep_default_na=False).set_index("event_id", drop=False)
    refs["month"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8
    cohort = sorted(refs.sample_id.unique())
    reference_counts = refs.groupby("sample_id").size().reindex(cohort).to_numpy()
    draws_path, axes_path = FROZEN / "paired_stand_indices.npy", FROZEN / "paired_stand_axes.json"
    draws = np.load(draws_path)
    axes = json.loads(axes_path.read_text())
    assert refs.index.is_unique and (len(refs), len(cohort)) == (1049, 459)
    assert axes["sample_ids"] == cohort and draws.shape == (5000, 459)
    assert draws.min() >= 0 and draws.max() < 459
    boot_references = reference_counts[draws].sum(axis=1)
    families = {
        "cumulative": VALIDATION / "cumulative_components_20261111_v1/range_overlap_pm1",
        "sensitivity": OUT.parent / "sensitivity",
        "benchmark": OUT.parent / "benchmark",
    }
    baseline_ids = {"cumulative": "F4_permutation", "sensitivity": "full",
                    "benchmark": "windowed_cusum_w36_firstpost"}
    audited, baseline_assignments, receipts, sources = [], {}, {}, {}
    contrast_count = 0
    for family, folder in families.items():
        receipt = json.loads((folder / "receipt.json").read_text())
        hashes = receipt.get("input_sha256", receipt.get("source_sha256"))
        for relative, expected in hashes.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            assert actual == expected, (family, relative, "input hash mismatch")
            sources[relative] = actual
        for path in (reference_path, draws_path, axes_path):
            assert hashes[str(path.relative_to(ROOT))] == hashlib.sha256(path.read_bytes()).hexdigest()
        receipts[family] = {"input_hashes_checked": len(hashes), "shared_reference_and_draw_hashes": True}
        summary = pd.read_csv(folder / "summary.csv").set_index("variant_id")
        estimates, boot_f1 = {}, {}
        for variant, row in summary.iterrows():
            if family == "cumulative":
                reused = {"F3_direction": "no_permutation_gate", "F4_permutation": "full"}
                base = FROZEN / "variants" / reused[variant] if variant in reused else folder.parent / "variants" / variant
                report_path = base / "octa/reports.parquet"
                match_path = folder / "variants" / variant / "matches.parquet"
            elif family == "sensitivity":
                report_path = FROZEN / "variants" / variant / "octa/reports.parquet"
                match_path = folder / "variants" / variant / "matches.parquet"
            else:
                report_path = VALIDATION / "detector_tda_benchmark_20261109_v1/methods" / variant / "reports.parquet"
                match_path = folder / "methods" / variant / "matches.parquet"
            reports = pd.read_parquet(report_path)
            reports = reports.loc[reports.sample_id.isin(cohort)].set_index("prediction_id", drop=False)
            matches = pd.read_parquet(match_path).set_index("event_id", drop=False)
            assert reports.index.is_unique and matches.index.is_unique
            assert set(matches.index) == set(refs.index)
            assert matches.sample_id.equals(refs.sample_id.reindex(matches.index))
            found = matches.loc[matches.found].copy()
            assert found.prediction_id.is_unique and found.prediction_id.isin(reports.index).all()
            original = reports.loc[found.prediction_id].set_axis(found.index)
            assert np.array_equal(original.sample_id, found.sample_id)
            reference_month = refs.loc[found.index, "month"].to_numpy()
            first = pd.PeriodIndex(original.first_response_month, freq="M").asi8
            last = pd.PeriodIndex(original.last_response_month, freq="M").asi8
            assert ((first <= reference_month + 1) & (last >= reference_month - 1)).all()
            assert np.array_equal(original.span_months.to_numpy(), last - first + 1)
            for column in ("first_response_month", "last_response_month", "span_months"):
                if column in found:
                    assert np.array_equal(found[column], original[column])
            for column, offset in (("reference_lower_month", -1), ("reference_upper_month", 1)):
                if column in matches:
                    expected = refs.loc[matches.index, "month"].to_numpy() + offset
                    assert np.array_equal(pd.PeriodIndex(matches[column], freq="M").asi8, expected)

            # Construct admissibility directly from unchanged intervals, independently of the matcher adapter.
            canonical, maximum_count, eligible_edges = {}, 0, 0
            report_groups = dict(tuple(reports.groupby("sample_id")))
            for stand, reference_group in refs.groupby("sample_id"):
                current = report_groups.get(stand, reports.iloc[:0])
                if current.empty:
                    continue
                a = pd.PeriodIndex(current.first_response_month, freq="M").asi8
                b = pd.PeriodIndex(current.last_response_month, freq="M").asi8
                r = reference_group.month.to_numpy()[:, None]
                allowed = (a[None, :] <= r + 1) & (b[None, :] >= r - 1)
                rows, cols = linear_sum_assignment(allowed, maximize=True)
                maximum_count += int(allowed[rows, cols].sum())
                x, y = np.where(allowed)
                edges = {(reference_group.index[i], current.index[j]): (1, 0) for i, j in zip(x, y)}
                eligible_edges += len(edges)
                canonical.update({ref: report for report, ref in _canonical_matching(edges).items()})
            actual = found.prediction_id.to_dict()
            assert actual == canonical and len(found) == maximum_count
            n_report, n_match = len(reports), len(found)
            precision, recall, f1 = n_match / n_report, n_match / 1049, 2 * n_match / (n_report + 1049)
            assert (row.n_stands, row.n_references, row.n_reports, row.n_matches) == (459, 1049, n_report, n_match)
            np.testing.assert_allclose([row.catalogue_precision, row.recall, row.catalogue_f1],
                                       [precision, recall, f1], rtol=0, atol=1e-14)
            report_counts = reports.groupby("sample_id").size().reindex(cohort, fill_value=0).to_numpy()
            match_counts = found.groupby("sample_id").size().reindex(cohort, fill_value=0).to_numpy()
            boot_f1[variant] = 2 * match_counts[draws].sum(axis=1) / (report_counts[draws].sum(axis=1) + boot_references)
            estimates[variant] = f1
            if variant == baseline_ids[family]:
                assert (n_report, n_match) == (1208, 945)
                baseline_assignments[family] = actual
            audited.append(dict(family=family, variant=variant, reports=n_report, references=1049,
                                matches=n_match, eligible_edges=eligible_edges,
                                original_interval_predicate=True, one_to_one=True,
                                maximum_cardinality_independently_verified=True, identifier_tie_break_verified=True,
                                original_ranges_preserved=True, summary_metrics_verified=True,
                                matches_sha256=hashlib.sha256(match_path.read_bytes()).hexdigest()))
        effects = pd.read_csv(folder / ("adjacent_effects.csv" if family == "cumulative" else "paired_effects.csv"))
        for effect in effects.itertuples():
            if family == "cumulative":
                left, right = effect.current_variant, effect.previous_variant
            elif family == "sensitivity":
                left, right = effect.variant_id, "full"
            else:
                left, right = baseline_ids[family], effect.comparator
            delta = boot_f1[left] - boot_f1[right]
            alpha = (1 - effect.ci_level) / 2
            low, high = np.quantile(delta, [alpha, 1 - alpha])
            np.testing.assert_allclose([effect.difference, effect.ci_low, effect.ci_high],
                [estimates[left] - estimates[right], low, high], rtol=0, atol=1e-14)
            assert effect.n_bootstrap == 5000
            contrast_count += 1
    assert baseline_assignments["cumulative"] == baseline_assignments["sensitivity"]
    assert baseline_assignments["cumulative"] == baseline_assignments["benchmark"]
    result = dict(status="passed", audit_scope="saved results only; no detector or TDA rerun",
        reference_stands=459, reference_records=1049, families=3, configurations=len(audited),
        saved_reference_rows_checked=sum(x["references"] for x in audited),
        matched_rows_checked=sum(x["matches"] for x in audited),
        direct_predicate="original_first <= reference_month+1 and original_last >= reference_month-1",
        no_double_tolerance=True, original_ranges_preserved=True,
        bootstrap_shape=list(draws.shape), bootstrap_contrasts_independently_recomputed=contrast_count,
        shared_baseline_reports=1208, shared_baseline_matches=945, baseline_assignments_identical=True,
        baseline_precision=945/1208, baseline_recall=945/1049, baseline_f1=2*945/(1208+1049),
        receipt_checks=receipts, configurations_checked=audited,
        source_sha256=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("configurations_checked", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
