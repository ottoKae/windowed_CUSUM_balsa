"""Independent checks of block-length outputs, interval matches and bootstrap."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "runs/validation"
RUN = VALIDATION / "block_length_sensitivity_20261111_v1"
FROZEN = VALIDATION / "balsa_results_optimization_20261105_v1/fullchain"
SOURCE = VALIDATION / "balsa_firstpost_national_20261030_v2"
SOURCES = {"full": FROZEN / "variants/full", "block3": FROZEN / "variants/block3",
           "block4": RUN / "block4", "block5": RUN / "block5",
           "block6": VALIDATION / "configuration_extension_20261111_v1/block6"}


def main():
    refs = pd.read_csv(VALIDATION / "balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv").set_index("event_id", drop=False)
    refs["ordinal"] = pd.PeriodIndex(refs.reference_month, freq="M").asi8
    cohort = sorted(refs.sample_id.unique())
    members = pd.read_parquet(SOURCE / "validation_membership.parquet")
    members = members.loc[members.sample_id.isin(cohort)]
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    units = units.loc[units.unit_id.isin(members.unit_id)].sort_values("unit_id").reset_index(drop=True)
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")[units.series_index.to_numpy()]
    assert (len(refs), len(cohort), len(units)) == (1049, 459, 25555)
    assert members.unit_id.is_unique and set(members.unit_id) == set(units.unit_id)
    n_members = members.groupby("sample_id").unit_id.nunique()
    eligibility = np.zeros(values.shape, dtype=bool)
    for year in range(9):
        eligible = np.isfinite(values[:, 12*year:min(values.shape[1], 12*year+36)]).sum(axis=1) >= 12
        eligibility[:, 12+12*year:24+12*year] = eligible[:, None]
    opportunities = pd.DataFrame(eligibility[:, 12:120] & np.isfinite(values[:, 12:120]), index=units.unit_id)
    opportunities["sample_id"] = members.set_index("unit_id").sample_id.reindex(units.unit_id).to_numpy()
    expected_opportunity = opportunities.groupby("sample_id").sum()
    draws = np.load(FROZEN / "paired_stand_indices.npy")
    assert draws.shape == (5000, 459)
    assert json.loads((FROZEN / "paired_stand_axes.json").read_text())["sample_ids"] == cohort
    ref_counts = refs.groupby("sample_id").size().reindex(cohort).to_numpy()
    summary = pd.read_csv(RUN / "evaluation/summary.csv").set_index("variant_id")
    effects = pd.read_csv(RUN / "evaluation/paired_effects.csv")
    checks, estimates, boots, hashes = [], {}, {}, {}
    for name, source in SOURCES.items():
        reports = pd.read_parquet(source / "octa/reports.parquet")
        nodes = pd.read_parquet(source / "pixel_nodes.parquet")
        owned = pd.read_parquet(source / "octa/source_members.parquet")
        monthly = pd.read_parquet(source / "octa/monthly_activity.parquet")
        components = pd.read_parquet(source / "octa/components.parquet")
        reports, nodes, owned, monthly, components = [frame.loc[frame.sample_id.isin(cohort)].copy()
            for frame in (reports, nodes, owned, monthly, components)]
        assert reports.prediction_id.is_unique and nodes.node_id.is_unique and owned.node_id.is_unique
        assert set(nodes.node_id) == set(owned.node_id)
        assert nodes.month_index.between(12, 119).all()
        assert nodes.sample_id.eq(nodes.unit_id.map(members.set_index("unit_id").sample_id)).all()
        assert reports.total_pixels.eq(reports.sample_id.map(n_members)).all()
        assert (5*reports.response_pixels >= reports.total_pixels).all()
        contributing = owned.loc[owned.prediction_id.isin(reports.prediction_id)]
        groups = contributing.groupby("prediction_id")
        assert reports.response_pixels.eq(reports.prediction_id.map(groups.unit_id.nunique())).all()
        assert reports.first_response_month.eq(reports.prediction_id.map(groups.response_month.min())).all()
        assert reports.last_response_month.eq(reports.prediction_id.map(groups.response_month.max())).all()
        first = pd.PeriodIndex(reports.first_response_month, freq="M").asi8
        last = pd.PeriodIndex(reports.last_response_month, freq="M").asi8
        np.testing.assert_array_equal(reports.span_months, last-first+1)
        stored = np.load(source / "monthly_eligibility.npz")
        order = pd.Index(stored["unit_id"]).get_indexer(units.unit_id)
        assert (order >= 0).all()
        np.testing.assert_array_equal(stored["eligibility"][order], eligibility)
        actual_opportunity = np.array([expected_opportunity.loc[r.sample_id, r.month_index-12] for r in monthly.itertuples()])
        np.testing.assert_array_equal(monthly.opportunity_pixels, actual_opportunity)
        np.testing.assert_array_equal(monthly.total_pixels, monthly.sample_id.map(n_members))
        active = nodes.groupby(["sample_id", "month_index"]).unit_id.nunique()
        np.testing.assert_array_equal(monthly.active_pixels,
            [active.get((r.sample_id, r.month_index), 0) for r in monthly.itertuples()])
        expected_state = np.where(monthly.active_pixels > 0, "active", np.where(
            2*monthly.opportunity_pixels >= monthly.total_pixels, "observed_no_candidate", "unobserved_or_ineligible"))
        np.testing.assert_array_equal(monthly.state, expected_state)
        matches = pd.read_parquet(RUN / "evaluation/variants" / name / "matches.parquet").set_index("event_id", drop=False)
        assert matches.index.is_unique and set(matches.index) == set(refs.index)
        found = matches.loc[matches.found]
        assert found.prediction_id.is_unique
        original = reports.set_index("prediction_id").loc[found.prediction_id].set_axis(found.index)
        assert original.sample_id.equals(refs.sample_id.reindex(found.index))
        original_first = pd.PeriodIndex(original.first_response_month, freq="M").asi8
        original_last = pd.PeriodIndex(original.last_response_month, freq="M").asi8
        reference_month = refs.loc[found.index, "ordinal"].to_numpy()
        assert ((original_first <= reference_month+1) & (original_last >= reference_month-1)).all()
        for column in ("first_response_month", "last_response_month", "span_months"):
            np.testing.assert_array_equal(found[column], original[column])
        maximum = 0
        report_groups = dict(tuple(reports.groupby("sample_id")))
        for stand, reference_group in refs.groupby("sample_id"):
            if stand not in report_groups:
                continue
            predicted = report_groups[stand]
            a = pd.PeriodIndex(predicted.first_response_month, freq="M").asi8
            b = pd.PeriodIndex(predicted.last_response_month, freq="M").asi8
            r = reference_group.ordinal.to_numpy()[:, None]
            edges = (a[None, :] <= r+1) & (b[None, :] >= r-1)
            row, col = linear_sum_assignment(edges, maximize=True)
            maximum += int(edges[row, col].sum())
        assert maximum == len(found)
        n_reports, n_matches = len(reports), len(found)
        row = summary.loc[name]
        assert (row.n_reports, row.n_matches, row.accepted_pixel_month_nodes, row.temporal_components) == (n_reports, n_matches, len(nodes), len(components))
        assert row.components_below_support == int(components.reason.eq("support_below_20_percent").sum())
        metrics = [n_matches/n_reports, n_matches/1049, 2*n_matches/(n_reports+1049)]
        np.testing.assert_allclose(row[["catalogue_precision", "recall", "catalogue_f1"]].to_numpy(dtype=float), metrics, atol=1e-14, rtol=0)
        retained, pair_total = 0, 0
        flags = matches.found.to_dict()
        for _, group in refs.groupby("sample_id"):
            order = group.reset_index(drop=True).sort_values(["reference_month", "event_id"]).event_id.tolist()
            retained += sum(flags[a] and flags[b] for a,b in zip(order[:-1], order[1:]))
            pair_total += len(order)-1
        assert pair_total == 590 and row.n_original_pairs_retained == retained
        report_counts = reports.groupby("sample_id").size().reindex(cohort, fill_value=0).to_numpy()
        matched_counts = found.groupby("sample_id").size().reindex(cohort, fill_value=0).to_numpy()
        estimates[name] = metrics[-1]
        boots[name] = 2*matched_counts[draws].sum(axis=1)/(report_counts[draws].sum(axis=1)+ref_counts[draws].sum(axis=1))
        checks.append(dict(variant_id=name, reports=n_reports, matches=n_matches, nodes=len(nodes),
            original_pairs_retained=retained, whole_stand_support=True, original_bounds=True,
            eligibility_independently_recomputed=True, monthly_opportunity_and_state=True,
            one_to_one=True, maximum_cardinality_independent=True))
    for effect in effects.itertuples():
        low, high = np.quantile(boots[effect.variant_id]-boots[effect.comparator], [.025, .975])
        np.testing.assert_allclose([effect.difference, effect.ci_low, effect.ci_high],
            [estimates[effect.variant_id]-estimates[effect.comparator], low, high], atol=1e-14, rtol=0)
        assert effect.n_bootstrap == 5000
    receipts = [RUN / "input_manifest.json", RUN / "evaluation/receipt.json", RUN / "block4/receipt.json",
                RUN / "block5/receipt.json", SOURCES["block6"] / "receipt.json"]
    for path in receipts:
        receipt = json.loads(path.read_text())
        for relative, expected in receipt["input_sha256"].items():
            actual = sha(ROOT / relative)
            assert actual == expected, relative
            hashes[relative] = actual
    result = dict(status="passed", reference_stands=459, reference_records=1049, fixed_pixels=25555,
        original_reference_pairs=590, variants=checks, matching="inclusive original report range versus reference +/-1",
        independent_maximum_cardinality=True, one_to_one=True, original_ranges_unchanged=True,
        support_threshold="5 * unique responding pixels >= full fixed membership",
        eligibility_same_and_recomputed_from_raw_finiteness=True,
        bootstrap_replicates=5000, independent_f1_contrasts=len(effects),
        source_hashes_checked=len(hashes), input_sha256=hashes, audit_script_sha256=sha(Path(__file__)),
        tex_modified=False, detector_or_aggregation_rerun=False)
    (RUN / "audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "input_sha256"}, indent=2))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
