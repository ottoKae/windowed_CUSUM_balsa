"""Score calendar-block lengths 3--6 with the frozen reference-overlap rule."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/validation/block_length_sensitivity_20261111_v1"
FROZEN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1/fullchain"
EXTENSION = ROOT / "runs/validation/configuration_extension_20261111_v1"
SOURCES = {
    "full": FROZEN / "variants/full",
    "block3": FROZEN / "variants/block3",
    "block4": RUN / "block4",
    "block5": RUN / "block5",
    "block6": EXTENSION / "block6",
}
sys.path.insert(0, str(ROOT / "tooling/diagnostics/balsa-results-optimization-20261105"))
import evaluate as evaluation


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
    assert json.loads(axes_path.read_text())["sample_ids"] == cohort
    assert draws.shape == (5000, 459)
    paths = [ref_path, draws_path, axes_path, Path(__file__), Path(evaluation.__file__),
             ROOT / "src/windowed_cusum/evaluation/event_scoring.py"]
    summaries, counts, all_matches = [], {}, {}
    for name, source in SOURCES.items():
        frames = {}
        for kind, relative in (("reports", "octa/reports.parquet"),
                               ("components", "octa/components.parquet"),
                               ("nodes", "pixel_nodes.parquet")):
            path = source / relative
            frame = pd.read_parquet(path)
            frames[kind] = frame.loc[frame.sample_id.isin(cohort)].copy()
            paths.append(path)
        reports, components, nodes = (frames[k] for k in ("reports", "components", "nodes"))
        assert reports.prediction_id.is_unique and nodes.node_id.is_unique
        assert (5 * reports.response_pixels >= reports.total_pixels).all()
        admissible = reports.copy()
        # Expanding the report once is equivalent to intersecting reference +/-1.
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
        counts[name], all_matches[name] = stand, matches
        row = evaluation.summary_row((name, "tda", "reference_pm1_interval_overlap"), stand, history, matches)
        row.update(block_length_months=0 if name == "full" else int(name[-1]),
                   accepted_pixel_month_nodes=len(nodes), temporal_components=len(components),
                   components_below_support=int(components.reason.eq("support_below_20_percent").sum()),
                   median_span_months=float(reports.span_months.median()),
                   reports_over_12_months=int(reports.span_months.gt(12).sum()))
        summaries.append(row)
        target = out / "variants" / name
        target.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(target / "matches.parquet", index=False)
        history.to_parquet(target / "pairs.parquet", index=False)
        stand.to_csv(target / "stand_counts.csv")

    table = pd.DataFrame(summaries)
    previous_path = EXTENSION / "evaluation/summary.csv"
    previous = pd.read_csv(previous_path).set_index("variant_id")
    columns = ["n_reports", "n_matches", "catalogue_precision", "recall", "catalogue_f1", "n_original_pairs_retained"]
    for name in ("full", "block3", "block6"):
        np.testing.assert_allclose(table.set_index("variant_id").loc[name, columns].to_numpy(dtype=float),
                                   previous.loc[name, columns].to_numpy(dtype=float), rtol=0, atol=1e-15)
    paths.append(previous_path)
    effects = []
    contrasts = [(v, "full") for v in SOURCES if v != "full"]
    contrasts += [(f"block{length}", f"block{length-1}") for length in (4, 5, 6)]
    for left, right in contrasts:
        a = all_matches[left].set_index("event_id")
        b = all_matches[right].set_index("event_id").reindex(a.index)
        effects.append(dict(variant_id=left, comparator=right,
                            **evaluation.bootstrap_effect(counts[left], counts[right], draws, "catalogue_f1", .95),
                            references_gained=int((a.found & ~b.found).sum()),
                            references_lost=int((~a.found & b.found).sum())))
    table.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "paired_effects.csv", index=False)
    receipt = dict(status="passed", reference_stands=459, reference_records=1049,
                   original_reference_pairs=590, bootstrap_replicates=5000,
                   geometry=dict(W=36, D=12, S=12), alpha=.025,
                   initial_permutations=3999, refined_permutations=19999,
                   matching="reference +/-1 intersects original TDA range; inclusive; maximum-cardinality one-to-one; identifier tie break",
                   replayed_old_scores_exact=True, manuscript_modified=False,
                   source_directories={k: str(p.relative_to(ROOT)) for k, p in SOURCES.items()},
                   input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(table[["variant_id", "accepted_pixel_month_nodes", "n_reports", "n_matches",
                 "catalogue_precision", "recall", "catalogue_f1"]].to_string(index=False))
    print(pd.DataFrame(effects)[["variant_id", "comparator", "difference", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
