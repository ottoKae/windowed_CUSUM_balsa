"""Isolate the H admission gate from phase joining on frozen v3 responses."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = ROOT / "runs/validation/balsa_harvest_episode_20261101_v3"
FROZEN = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OUT = ROOT / "runs/validation/balsa_h12_audit_20261102_v1"
CODE = ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101"
sys.path[:0] = [str(CODE), str(ROOT / "tooling/release"), str(ROOT / "src")]
from episode_assembly import assemble_episodes
from label_evaluation import evaluate_labels, paired_summary
from build_sample_release import original_tables


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous_receipt = json.loads((PREVIOUS / "execution_receipt.json").read_text())
    input_paths = [PREVIOUS / f"{name}.parquet" for name in ["nodes", "monthly_activity", "phases", "phase_relations_B1"]]
    for path in input_paths:
        assert digest(path) == previous_receipt["output_sha256"][path.name], path
    for path, expected in previous_receipt["source_sha256"].items():
        if Path(path).name in {"episode_assembly.py", "episode_relations.py", "label_evaluation.py"}:
            assert digest(path) == expected, path
    nodes, monthly, phases, all_edges = [pd.read_parquet(path) for path in input_paths]
    tables, references_config = original_tables()
    baselines, base_sources = {}, {}
    for g in range(4):
        directory = PREVIOUS / "methods" / f"B1_G{g}_H12"
        baselines[g] = pd.read_parquet(directory / "components.parquet")
        base_sources[g] = pd.read_parquet(directory / "source_members.parquet")
        input_paths.extend(directory / f"{name}.parquet" for name in ["components", "source_members", "reports"])
    long = baselines[1].loc[baselines[1].reason.eq("long_chain_unresolved")]
    assert len(long) == 93 and long.source_event_count.sum() == 11317
    assert len(nodes) == 44311 and nodes.node_id.is_unique
    baseline_reports_path = FROZEN / "validation/stand_reports.parquet"
    baseline_members_path = FROZEN / "validation/source_members.parquet"
    input_paths += [baseline_reports_path, baseline_members_path, PREVIOUS / "evaluation/label_metrics.csv"]
    input_paths += [Path(record["path"]) for record in references_config["inputs"].values()]
    input_paths += list((ROOT / "manuscript_JAG").rglob("*.tex"))
    protected = {str(path): digest(path) for path in input_paths}
    write_json(OUT / "input_manifest.json", dict(protected_sha256=protected,
        source_signature=previous_receipt["source_sha256"], accepted_nodes=44311, fixed_phases=3882,
        frozen_long_components=93, frozen_long_nodes=11317))
    configs = [(1, h) for h in [18, 24, 36, 108]] + [(g, 108) for g in [0, 2, 3]]
    write_json(OUT / "configuration_frozen.json", dict(
        date="2026-11-02", diagnostic_not_new_independent_validation=True,
        original_H12_groups=list(range(4)), new_counterfactuals=[dict(G=g,H=h) for g,h in configs],
        H108_meaning="whole_report_period_nonbinding_cap_not_real_harvest_duration",
        sts_disabled=True, spatial_support=.2, q_min=.5,
        same_G_stable_component_identifiers=True, metric_scope="fixed_1049_month_labels_in_459_stands",
        tolerances_months=[1,2,3], deployment_change=False))
    results = {}
    outputs = {}
    for g, components in baselines.items():
        method = f"B1_G{g}_H12"
        outputs[method] = dict(components=components, reports=components.loc[components.destination_type.eq("episode")],
                               source_members=base_sources[g])
    outputs["B0"] = dict(reports=pd.read_parquet(baseline_reports_path))
    for g, h in configs:
        method = f"B1_G{g}_H{h}"
        edges = all_edges.loc[all_edges.G.eq(g)].copy()
        result = assemble_episodes(phases, edges, nodes, monthly, max_gap=g, max_span=h,
                                   threshold=np.inf, method=method)
        stabilize_ids(result, baselines[g])
        directory = OUT / "methods" / method
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_parquet(directory / f"{name}.parquet", index=False)
        outputs[method] = result
        print(f"{method}: reports={len(result['reports'])}, same source partition verified", flush=True)
    for method, result in outputs.items():
        evaluation = evaluate_labels(method, result["reports"], tables["references"], tables["eligibility"], tables["stands"])
        assert evaluation["summary"].n_reference_months.eq(1049).all()
        assert evaluation["summary"].n_reference_stands.eq(459).all()
        results[method] = evaluation
        primary = evaluation["summary"].set_index("tolerance_months").loc[2]
        print(f"{method}: {int(primary.matched_labels)}/1049, P={primary.precision_label:.4f}, "
              f"R={primary.recall_label:.4f}, F1={primary.F1_label:.4f}", flush=True)
    eval_dir = OUT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    combined = {key:pd.concat([result[key] for result in results.values()], ignore_index=True)
                for key in ["summary", "matches", "per_stand", "interval_summary"]}
    combined["summary"].to_csv(eval_dir / "label_metrics.csv", index=False)
    combined["matches"].to_parquet(eval_dir / "reference_matches.parquet", index=False)
    combined["per_stand"].to_parquet(eval_dir / "per_stand_counts.parquet", index=False)
    combined["interval_summary"].to_csv(eval_dir / "interval_compatibility.csv", index=False)
    primary = combined["summary"].loc[combined["summary"].tolerance_months.eq(2)].set_index("method")
    old = pd.read_csv(PREVIOUS / "evaluation/label_metrics.csv")
    for method in ["B0"] + [f"B1_G{g}_H12" for g in range(4)]:
        actual = combined["summary"].loc[combined["summary"].method.eq(method)].sort_values("tolerance_months")
        expected = old.loc[old.method.eq(method)].sort_values("tolerance_months")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    ordered_g1 = [f"B1_G1_H{h}" for h in [12,18,24,36,108]]
    for before, after in zip(ordered_g1[:-1], ordered_g1[1:]):
        assert set(outputs[before]["reports"].prediction_id).issubset(outputs[after]["reports"].prediction_id)
        assert primary.loc[before, "matched_labels"] <= primary.loc[after, "matched_labels"]
    compare = [c for c in outputs["B1_G1_H36"]["components"].columns if c != "method"]
    pd.testing.assert_frame_equal(outputs["B1_G1_H36"]["components"][compare],
                                  outputs["B1_G1_H108"]["components"][compare])
    for baseline in ["B0", "B1_G1_H12", "B1_G0_H108"]:
        paired_summary(combined["per_stand"], tables["stands"], baseline=baseline).to_csv(
            eval_dir / f"paired_vs_{baseline}.csv", index=False)
    reference_effects, releases = trace_effects(outputs, combined["matches"],
                                              pd.read_parquet(baseline_members_path), long)
    reference_effects.to_csv(eval_dir / "reference_gate_effects.csv", index=False)
    releases.to_csv(OUT / "long_chain_release.csv", index=False)
    counts = []
    for method, result in outputs.items():
        if method == "B0":
            continue
        for reason, part in result["components"].groupby("reason"):
            counts.append(dict(method=method, reason=reason, components=len(part), nodes=int(part.source_event_count.sum())))
    pd.DataFrame(counts).to_csv(eval_dir / "component_dispositions.csv", index=False)
    for path, expected in protected.items():
        assert digest(path) == expected, path
    with_h, no_h = outputs["B1_G1_H12"]["components"], outputs["B1_G1_H108"]["components"]
    released_ids = set(no_h.loc[no_h.destination_type.eq("episode"), "component_id"]) - set(
        with_h.loc[with_h.destination_type.eq("episode"), "component_id"])
    assert released_ids.issubset(set(long.component_id)) and len(released_ids) == 77
    report0 = combined["matches"].query("method=='B0' and tolerance_months==2").set_index("event_id")
    report12 = combined["matches"].query("method=='B1_G1_H12' and tolerance_months==2").set_index("event_id")
    report108 = combined["matches"].query("method=='B1_G1_H108' and tolerance_months==2").set_index("event_id")
    old_lost = report0.found & ~report12.found
    write_json(OUT / "execution_receipt.json", dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        original_inputs_and_tex_unchanged=True, original_metrics_exact=True, stable_partition_and_dates=True,
        H36_H108_G1_equal=True, monotone_admission_and_match_count=True,
        methods_scored=len(results), long_components=93, long_nodes=11317, released_components=len(released_ids),
        B0_to_H12_lost=int(old_lost.sum()), recovered_old_losses_by_H108=int((old_lost & report108.found).sum()),
        remaining_old_losses=int((old_lost & ~report108.found).sum()),
        newly_found_vs_H12=int((~report12.found & report108.found).sum()),
        individually_lost_vs_H12=int((report12.found & ~report108.found).sum()),
        independent_event_identity_confirmed=False, no_deployment_change=True,
        code_sha256={str(path):digest(path) for path in [Path(__file__), CODE/'episode_assembly.py',CODE/'episode_relations.py',CODE/'label_evaluation.py']},
        output_sha256={str(path.relative_to(OUT)):digest(path) for path in OUT.rglob('*') if path.is_file() and path.name!='execution_receipt.json'}))
    print("H and phase joining audit completed; no change to the prior run", flush=True)


def stabilize_ids(result, baseline):
    """Keep H-only comparisons free of method-name lexical matching effects."""
    current = result["components"]
    assert current.source_node_ids.is_unique and baseline.source_node_ids.is_unique
    assert set(current.source_node_ids) == set(baseline.source_node_ids)
    previous = baseline.set_index("source_node_ids")
    fixed = [c for c in current.columns if c not in {
        "method", "component_id", "prediction_id", "destination_type", "reason", "source_node_ids"}]
    pd.testing.assert_frame_equal(current.set_index("source_node_ids")[fixed].sort_index(),
                                  previous[fixed].sort_index(), check_dtype=False)
    mapping = dict(zip(current.component_id, current.source_node_ids.map(previous.component_id)))
    for name in ["components", "reports", "source_members", "progress"]:
        frame = result[name]
        frame["component_id"] = frame.component_id.map(mapping)
        frame["prediction_id"] = frame.prediction_id.map(lambda value: mapping[value] if value else "")
    assert result["source_members"].node_id.is_unique


def trace_effects(outputs, matches, b0_nodes, long):
    primary = matches.loc[matches.tolerance_months.eq(2)].copy()
    methods = ["B0", "B1_G1_H12", "B1_G1_H18", "B1_G1_H24", "B1_G1_H36", "B1_G1_H108", "B1_G0_H108"]
    base = primary.loc[primary.method.eq("B0")].set_index("event_id")
    references = base[["sample_id", "reference_month"]].copy()
    for method in methods:
        part = primary.loc[primary.method.eq(method)].set_index("event_id")
        references[method + "_found"] = part.found
        references[method + "_prediction_id"] = part.prediction_id
    old_sources = outputs["B1_G1_H12"]["source_members"]
    bridge = b0_nodes[["node_id", "prediction_id"]].rename(columns={"prediction_id":"b0_prediction_id"}).merge(
        old_sources[["node_id", "component_id", "reason"]], on="node_id", validate="one_to_one")
    groups = {key: part for key, part in bridge.loc[bridge.b0_prediction_id.ne("")].groupby("b0_prediction_id")}
    sources = []
    for ref in base.itertuples():
        part = groups.get(ref.prediction_id, bridge.iloc[:0])
        sources.append(dict(event_id=ref.Index, b0_report_source_nodes=len(part),
            b0_source_nodes_in_H12_long=int(part.reason.eq("long_chain_unresolved").sum()),
            b0_source_nodes_in_supported=int(part.reason.eq("supported_model_episode").sum()),
            b0_source_nodes_in_boundary_unknown=int(part.reason.eq("boundary_unknown").sum()),
            b0_source_nodes_below_support=int(part.reason.eq("support_below_20_percent").sum()),
            b0_source_component_ids=json.dumps(sorted(part.component_id.unique().tolist()))))
    references = references.join(pd.DataFrame(sources).set_index("event_id"))
    releases = long.copy()
    for h in [12,18,24,36,108]:
        method = f"B1_G1_H{h}"
        comp = outputs[method]["components"].set_index("component_id")
        found = primary.loc[primary.method.eq(method) & primary.found]
        ref_ids = found.groupby("prediction_id").event_id.agg(list).to_dict()
        releases[f"H{h}_destination"] = releases.component_id.map(comp.destination_type)
        releases[f"H{h}_reason"] = releases.component_id.map(comp.reason)
        releases[f"H{h}_matched_labels"] = releases.component_id.map(lambda value: json.dumps(ref_ids.get(value, [])))
    ref_months = primary.loc[primary.method.eq("B0")].copy()
    candidate_rows = []
    for component in releases.itertuples():
        group = ref_months.loc[ref_months.sample_id.eq(component.sample_id)]
        inside = group.loc[group.reference_month.between(component.first_response_month, component.last_response_month)]
        representative_index = pd.Period(component.representative_month, freq="M").ordinal
        distances = pd.PeriodIndex(group.reference_month, freq="M").asi8 - representative_index
        nearby = group.loc[np.abs(distances)<=2]
        source_rows = bridge.loc[bridge.component_id.eq(component.component_id)]
        candidate_rows.append(dict(component_id=component.component_id,
            labels_inside_response_envelope=json.dumps(inside.event_id.tolist()),
            n_labels_inside_response_envelope=len(inside),
            labels_within_2_of_fixed_representative=json.dumps(nearby.event_id.tolist()),
            n_labels_within_2_of_fixed_representative=len(nearby),
            source_overlapping_B0_reports=json.dumps(sorted(set(source_rows.b0_prediction_id)-{""})),
            interpretation="temporal_and_source_association_only_not_physical_event_identity"))
    releases = releases.merge(pd.DataFrame(candidate_rows), on="component_id", validate="one_to_one")
    return references.reset_index(), releases


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
