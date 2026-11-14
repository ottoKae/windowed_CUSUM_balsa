"""Reproduce frozen baselines and isolate H admission at the five old STS cuts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "runs/validation/balsa_harvest_episode_20261101_v3"
NATIONAL = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
H_AUDIT = ROOT / "runs/validation/balsa_h12_audit_20261102_v1"
OUT = ROOT / "runs/validation/balsa_sts_episode_v4_1_20261102_v1/baselines"
CODE = ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101"
sys.path[:0] = [str(CODE), str(ROOT / "src"), str(ROOT / "tooling/release"),
               str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030"),
               str(ROOT / "tooling/diagnostics/balsa-h12-audit-20261102")]
from episode_assembly import assemble_episodes
from label_evaluation import evaluate_labels
from build_sample_release import original_tables
from run_h12_audit import stabilize_ids
import detector_engine


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_receipt = json.loads((OLD / "execution_receipt.json").read_text())
    old_manifest = json.loads((OLD / "input_manifest.json").read_text())
    national_signature = json.loads((NATIONAL / "method_signature.json").read_text())
    h_receipt = json.loads((H_AUDIT / "execution_receipt.json").read_text())
    frozen_config = json.loads((OLD / "configuration_frozen.json").read_text())
    thresholds = [row for row in frozen_config["methods"] if row["method"].startswith("B2R_")
                  and not row["sts_disabled"]]
    assert [row["method"].rsplit("_", 1)[1] for row in thresholds] == ["Q10", "Q25", "Q50", "Q75", "Q90"]
    source_expected = {**national_signature["source_sha256"], **old_receipt["source_sha256"],
                       **h_receipt["code_sha256"]}
    names = ["nodes", "phases", "monthly_activity", "phase_relations_B1"]
    old_files = [f"{name}.parquet" for name in names] + ["configuration_frozen.json", "input_manifest.json",
                  "evaluation/label_metrics.csv"]
    old_methods = ["B1_G1_H12"] + [row["method"] for row in thresholds]
    old_files += [f"methods/{method}/{name}.parquet" for method in old_methods
                  for name in ["components", "reports", "source_members", "edge_decisions", "progress"]]
    expected = dict(source_expected)
    for relative in old_files:
        expected[str(OLD / relative)] = old_receipt["output_sha256"][str(Path(relative))]
    for relative in ["master_units.parquet", "master_values.npy", "validation_membership.parquet",
                     "validation/pixel_events.parquet", "validation/pixel_execution.parquet",
                     "validation/stand_reports.parquet", "validation/source_members.parquet", "protocol.json",
                     "method_signature.json"]:
        path = NATIONAL / relative
        expected[str(path)] = old_manifest["protected_sha256"][str(path)]
    for record in old_manifest["reference_inputs"].values():
        expected[record["path"]] = record["sha256"]
    for name in ["components", "reports", "source_members", "edge_decisions", "progress"]:
        relative = str(Path("methods/B1_G1_H108") / f"{name}.parquet")
        expected[str(H_AUDIT / relative)] = h_receipt["output_sha256"][relative]
    expected[str(H_AUDIT / "evaluation/label_metrics.csv")] = h_receipt["output_sha256"][str(Path("evaluation/label_metrics.csv"))]
    audit = audit_hashes(expected)
    write_json(OUT / "frozen_hash_audit.json", audit)
    if not audit["passed"]:
        raise ValueError("Frozen source/input drift; see frozen_hash_audit.json before continuing")
    tex = {str(path): digest(path) for path in (ROOT / "manuscript_JAG").rglob("*.tex")}
    protected = {**expected, **tex}
    write_json(OUT / "input_manifest.json", dict(protected_sha256=protected,
        source_sha256=source_expected, historical_threshold_source=str(OLD / "configuration_frozen.json"),
        thresholds=thresholds, historical_tex_is_not_the_current_baseline=True))

    nodes, phases, monthly, all_edges = [pd.read_parquet(OLD / f"{name}.parquet") for name in names]
    edges = all_edges.loc[all_edges.G.eq(1)].copy()
    membership = pd.read_parquet(NATIONAL / "validation_membership.parquet")
    units = pd.read_parquet(NATIONAL / "master_units.parquet")
    units = units.loc[units.unit_id.isin(membership.unit_id)].sort_values("unit_id").reset_index(drop=True)
    events = pd.read_parquet(NATIONAL / "validation/pixel_events.parquet")
    protocol = json.loads((NATIONAL / "protocol.json").read_text())
    b20_reports, b20_ledger, b20_receipt = detector_engine.aggregate(units, membership, events, protocol)
    assert_frames(b20_reports, pd.read_parquet(NATIONAL / "validation/stand_reports.parquet"), "prediction_id")
    assert_frames(b20_ledger, pd.read_parquet(NATIONAL / "validation/source_members.parquet"), "node_id")
    assert len(units) == 26890 and units.unit_id.is_unique and membership.unit_id.is_unique
    assert len(nodes) == len(b20_ledger) == 44311 and nodes.node_id.is_unique
    assert_frames(nodes[["node_id", "sample_id", "unit_id", "response_month"]],
                  b20_ledger[["node_id", "sample_id", "unit_id", "response_month"]], "node_id")
    assert monthly.sample_id.nunique() == membership.sample_id.nunique() == 468
    assert set(monthly.sample_id) == set(membership.sample_id)
    assert len(monthly) == 468 * 108 and monthly.month_index.min() == 12 and monthly.month_index.max() == 119
    assert len(phases) == 3882 and edges.relation.eq("temporal_join").sum() == 996
    assert (edges.relation.eq("temporal_join") & edges.distance.notna()).sum() == 383
    universe = membership.groupby("sample_id").unit_id.nunique().rename("total_pixels").to_frame()
    universe["accepted_nodes"] = nodes.groupby("sample_id").size().reindex(universe.index, fill_value=0)
    assert universe.accepted_nodes.eq(0).sum() == 2
    universe.reset_index().to_parquet(OUT / "stand_universe.parquet", index=False)
    tables, reference_config = original_tables()
    for item in reference_config["inputs"].values():
        assert expected[str(Path(item["path"]))] == item["sha256"]
    old_metrics = pd.read_csv(OLD / "evaluation/label_metrics.csv")
    h_metrics = pd.read_csv(H_AUDIT / "evaluation/label_metrics.csv")
    legacy, canonical, parity, effects = {}, {}, [], []

    def save(method, result, new_method, expected_metrics=None):
        scored = evaluate(method, result["reports"], tables)
        if expected_metrics is not None:
            observed = scored["summary"].sort_values("tolerance_months").reset_index(drop=True)
            frozen = expected_metrics.loc[expected_metrics.method.eq(method)].sort_values("tolerance_months").reset_index(drop=True)
            pd.testing.assert_frame_equal(observed, frozen, check_dtype=False)
        write_result(OUT / "legacy" / method, result)
        legacy[method] = scored
        for frame in result.values():
            frame["method"] = new_method
        canonicalize(result)
        if "components" in result:
            for key in ["components", "reports"]:
                result[key]["span_over_12"] = result[key].span_months.gt(12)
        write_result(OUT / "methods" / new_method, result)
        new_score = evaluate(new_method, result["reports"], tables)
        canonical[new_method] = new_score
        for tolerance in [1, 2, 3]:
            a = scored["matches"].query("tolerance_months==@tolerance").set_index("event_id")
            b = new_score["matches"].query("tolerance_months==@tolerance").set_index("event_id")
            parity.append(dict(legacy_method=method, scientific_method=new_method, tolerance_months=tolerance,
                legacy_found=int(a.found.sum()), scientific_found=int(b.found.sum()),
                changed_found_labels=int(a.found.ne(b.found).sum())))
        row = new_score["summary"].set_index("tolerance_months").loc[2]
        print(f"{new_method}: {int(row.n_reports_all)} all / {int(row.n_reports_cohort)} monthly reports; "
              f"{int(row.matched_labels)}/1049; P_label={row.precision_label:.6f}, F1_label={row.F1_label:.6f}", flush=True)
        return scored

    save("B0", dict(reports=b20_reports, source_members=b20_ledger), "R0_B20", old_metrics)
    base = assemble_episodes(phases, edges, nodes, monthly, max_gap=1, max_span=12,
                             threshold=np.inf, method="B1_G1_H12")
    frozen_base = pd.read_parquet(OLD / "methods/B1_G1_H12/components.parquet")
    for name, frame in base.items():
        assert_frames(frame, pd.read_parquet(OLD / "methods/B1_G1_H12" / f"{name}.parquet"),
                      "node_id" if name == "source_members" else "edge_id" if name == "edge_decisions" else
                      ["component_id", "response_month"] if name == "progress" else "component_id")
    save("B1_G1_H12", base, "R1_H12", old_metrics)
    relaxed = assemble_hflag(phases, edges, nodes, monthly, method="B1_G1_H108", scientific_ids=False)
    relaxed = {name: frame.drop(columns="span_over_12", errors="ignore") for name, frame in relaxed.items()}
    stabilize_ids(relaxed, frozen_base)
    for name, frame in relaxed.items():
        assert_frames(frame, pd.read_parquet(H_AUDIT / "methods/B1_G1_H108" / f"{name}.parquet"),
                      "node_id" if name == "source_members" else "edge_id" if name == "edge_decisions" else
                      ["component_id", "response_month"] if name == "progress" else "component_id")
    save("B1_G1_H108", relaxed, "R2_TIME", h_metrics)

    for config in thresholds:
        old_name, theta = config["method"], config["theta"]
        quantile = old_name.rsplit("_", 1)[1]
        result12 = assemble_episodes(phases, edges, nodes, monthly, max_gap=1, max_span=12,
                                    threshold=theta, method=old_name)
        frozen_components = pd.read_parquet(OLD / "methods" / old_name / "components.parquet")
        assert_frames(result12["components"], frozen_components, "component_id")
        before = save(old_name, result12, f"R3_OLD_{quantile}_H12", old_metrics)
        method = old_name.replace("H12", "H108")
        result108 = assemble_hflag(phases, edges, nodes, monthly, method=method,
                                  threshold=theta, scientific_ids=False)
        result108 = {name: frame.drop(columns="span_over_12", errors="ignore") for name, frame in result108.items()}
        stabilize_ids(result108, frozen_components)
        after = save(method, result108, f"R3_OLD_{quantile}_HFLAG")
        a = before["summary"].set_index("tolerance_months").loc[2]
        b = after["summary"].set_index("tolerance_months").loc[2]
        am = before["matches"].query("tolerance_months==2").set_index("event_id")
        bm = after["matches"].query("tolerance_months==2").set_index("event_id")
        effects.append(dict(quantile=quantile, exact_theta=theta, H12_reports=int(a.n_reports_cohort),
            Hflag_reports=int(b.n_reports_cohort), H12_found=int(a.matched_labels), Hflag_found=int(b.matched_labels),
            gained_references=int((~am.found & bm.found).sum()), lost_references=int((am.found & ~bm.found).sum()),
            H12_F1_label=a.F1_label, Hflag_F1_label=b.F1_label,
            identity_scope="same_old_method_ids_for_H_only_comparison_not_physical_episode_truth"))

    for scope, scores in [("legacy", legacy), ("scientific", canonical)]:
        evaluation = OUT / "evaluation" / "legacy" if scope == "legacy" else OUT / "evaluation"
        evaluation.mkdir(parents=True, exist_ok=True)
        for key, filename in [("summary", "label_metrics.csv"), ("matches", "reference_matches.parquet"),
                              ("per_stand", "per_stand_counts.parquet"), ("interval_summary", "interval_compatibility.csv")]:
            frame = pd.concat([item[key] for item in scores.values()], ignore_index=True)
            if filename.endswith("parquet"):
                frame.to_parquet(evaluation / filename, index=False)
            else:
                frame.to_csv(evaluation / filename, index=False)
    pd.DataFrame(parity).to_csv(OUT / "scientific_identity_parity.csv", index=False)
    pd.DataFrame(effects).to_csv(OUT / "historical_theta_H_effects.csv", index=False)
    primary = {method: score["summary"].set_index("tolerance_months").loc[2] for method, score in canonical.items()}
    for method, expected_counts in [("R0_B20", (1267, 1255, 930)), ("R1_H12", (1139, 1131, 866)),
                                    ("R2_TIME", (1216, 1208, 925))]:
        row = primary[method]
        assert (int(row.n_reports_all), int(row.n_reports_cohort), int(row.matched_labels)) == expected_counts
    post_audit = audit_hashes(protected)
    assert post_audit["passed"], "Source/input/TeX changed during baseline run"
    write_json(OUT / "execution_receipt.json", dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        frozen_source_and_input_hashes_match=True, protected_inputs_and_tex_unchanged=True,
        b20_exact_aggregation=b20_receipt, historical_assembly_and_metrics_reproduced=True,
        stands=468, zero_candidate_stands=2, pixels=26890, source_nodes=44311,
        phases=3882, G1_join_edges=996, G1_calculable_join_edges=383,
        source_identity_contract="sha256([sample_id,sorted source_node_ids]); excludes method",
        canonical_found_count_parity=all(row["legacy_found"] == row["scientific_found"] for row in parity),
        changed_individual_label_outcomes=sum(row["changed_found_labels"] for row in parity),
        methods_scored=len(canonical), historical_thresholds=thresholds,
        historical_thresholds_are_not_OOF_calibration=True,
        code_sha256={str(Path(__file__)): digest(__file__), **{path: digest(path) for path in source_expected}},
        output_sha256={str(path.relative_to(OUT)): digest(path) for path in OUT.rglob("*")
                       if path.is_file() and path.name != "execution_receipt.json"}))


def canonicalize(result):
    """Replace IDs in place using stand and sorted source nodes, retaining legacy IDs."""
    sources = result["components"] if "components" in result else result["reports"]
    key = "component_id" if "components" in result else "prediction_id"
    mapping = {}
    for row in sources.itertuples(index=False):
        identity = json.dumps([row.sample_id, sorted(json.loads(row.source_node_ids))], separators=(",", ":"))
        mapping[getattr(row, key)] = "episode_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    assert len(mapping) == len(sources) and len(set(mapping.values())) == len(sources)
    for name in ["components", "reports", "source_members", "progress"]:
        if name not in result:
            continue
        frame = result[name]
        for column in ["component_id", "prediction_id"]:
            if column not in frame:
                continue
            if "legacy_" + column not in frame:
                frame["legacy_" + column] = frame[column]
            frame[column] = frame[column].map(lambda value: mapping[value] if value else "")
    return result


def assemble_hflag(phases, edges, nodes, monthly, *, method, threshold=np.inf, scientific_ids=True):
    """Use the frozen assembler with G1/H108; optionally use method-independent IDs."""
    result = assemble_episodes(phases, edges, nodes, monthly, max_gap=1, max_span=108,
                               threshold=threshold, method=method)
    for name in ["components", "reports"]:
        result[name]["span_over_12"] = result[name].span_months.gt(12)
    return canonicalize(result) if scientific_ids else result


def evaluate(method, reports, tables):
    return evaluate_labels(method, reports, tables["references"], tables["eligibility"], tables["stands"])


def audit_hashes(expected):
    records = [dict(path=str(path), expected_sha256=sha, actual_sha256=digest(path))
               for path, sha in expected.items()]
    for row in records:
        row["matches"] = row["expected_sha256"] == row["actual_sha256"]
    return dict(passed=all(row["matches"] for row in records), files=records)


def assert_frames(actual, expected, key):
    pd.testing.assert_frame_equal(actual.sort_values(key).reset_index(drop=True),
        expected[actual.columns].sort_values(key).reset_index(drop=True), check_dtype=False, check_exact=True)


def write_result(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
