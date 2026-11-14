"""Run the frozen-candidate Balsa episode experiment and label-relative scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/release"),
               str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030")]
from build_sample_release import original_tables
import detector_engine as engine
from episode_relations import active_phases, classify_gap, build_phase_template, response_distance
from episode_assembly import assemble_episodes
from label_evaluation import evaluate_labels, paired_summary

SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OUT = ROOT / "runs/validation/balsa_harvest_episode_20261101_v3"
PLAN = ROOT / "docs/experiments/external_execution_20261026/BALSA_HARVEST_EPISODE_RECONSTRUCTION_PLAN_20261101_v3.md"
MONTHS = pd.period_range("2016-01", "2026-06", freq="M").astype(str).to_numpy()
REPORT_INDEX = np.arange(12, 120)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_inputs():
    signature = json.loads((SOURCE / "method_signature.json").read_text())
    for path, expected in signature["source_sha256"].items():
        assert digest(path) == expected, f"Frozen source changed: {path}"
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    members = pd.read_parquet(SOURCE / "validation_membership.parquet")
    units = units.loc[units.unit_id.isin(members.unit_id)].sort_values("unit_id").reset_index(drop=True)
    assert len(units) == 26890 and members.sample_id.nunique() == 468
    assert members.unit_id.is_unique and units.unit_id.is_unique
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")[units.series_index.to_numpy()]
    assert values.shape == (26890, 126)
    events = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    execution = pd.read_parquet(SOURCE / "validation/pixel_execution.parquet").set_index("unit_id").loc[units.unit_id]
    np.testing.assert_array_equal(execution.series_index, units.series_index)
    np.testing.assert_array_equal(execution.valid_months, np.isfinite(values).sum(axis=1))
    assert execution.method.eq(engine.METHOD).all() and events.method.eq(engine.METHOD).all()
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    reports, ledger, baseline_receipt = engine.aggregate(units, members, events, protocol)
    for actual, filename, key in [(reports, "stand_reports.parquet", "prediction_id"),
                                  (ledger, "source_members.parquet", "node_id")]:
        frozen = pd.read_parquet(SOURCE / "validation" / filename)
        pd.testing.assert_frame_equal(actual.sort_values(key).reset_index(drop=True),
            frozen[actual.columns].sort_values(key).reset_index(drop=True), check_dtype=False, check_exact=True)
    assert len(events) == len(ledger) == 44311 and len(reports) == 1267
    assert ledger.node_id.is_unique and not events.duplicated(["unit_id", "change_time"]).any()
    events = events.assign(response_month=events.change_time.dt.strftime("%Y-%m"))
    nodes = ledger[["sample_id", "unit_id", "node_id", "response_month"]].merge(
        events[["unit_id", "response_month", "change_index", "window_start", "window_end", "candidate_id"]],
        on=["unit_id", "response_month"], validate="one_to_one").rename(columns={"change_index": "month_index"})
    nodes["phase_id"] = ""
    np.testing.assert_array_equal(MONTHS[nodes.month_index.to_numpy()], nodes.response_month)
    for bit in range(9):
        finite = np.isfinite(values[:, bit * 12:min(bit * 12 + 36, len(MONTHS))]).sum(axis=1)
        np.testing.assert_array_equal(((execution.eligible_year_mask.to_numpy() >> bit) & 1).astype(bool), finite >= 12)
    tables, reference_config = original_tables()
    paths = [SOURCE / name for name in ("master_units.parquet", "master_values.npy", "validation_membership.parquet",
        "validation/pixel_events.parquet", "validation/pixel_execution.parquet", "validation/stand_reports.parquet",
        "validation/source_members.parquet", "protocol.json", "method_signature.json")]
    paths += [Path(v["path"]) for v in reference_config["inputs"].values()]
    protected = {str(path): digest(path) for path in paths}
    protected.update(signature["source_sha256"])
    protected.update({str(path): digest(path) for path in (ROOT / "manuscript_JAG").rglob("*.tex")})
    write_json(OUT / "input_manifest.json", dict(protected_sha256=protected,
        method_signature=signature["signature"], source_method=engine.METHOD,
        fixed_pixels=len(units), fixed_stands=468, accepted_nodes=len(nodes),
        calendar=[MONTHS[0], MONTHS[-1]], report_period=[MONTHS[12], MONTHS[119]],
        baseline_exact_reports_and_nodes=True, baseline_aggregation=baseline_receipt,
        reference_inputs=reference_config["inputs"]))
    print("Exact B20 parity: 1267 reports / 44311 nodes; inputs and execution masks checked", flush=True)
    return units, members, values, execution, nodes, reports, tables, protected


def prepare_phases(units, members, values, execution, nodes):
    row_index = pd.Series(np.arange(len(units)), index=units.unit_id)
    monthly_rows, phase_rows, point_rows, member_rows = [], [], [], []
    templates = {}
    node_groups = nodes.groupby("sample_id").groups
    for sample_id, membership in members.groupby("sample_id", sort=True):
        indices = row_index.loc[membership.unit_id].to_numpy()
        n_total = len(indices)
        selected = nodes.loc[node_groups.get(sample_id, [])]
        counts = selected.groupby("month_index").unit_id.nunique().reindex(REPORT_INDEX, fill_value=0).to_numpy()
        masks = execution.eligible_year_mask.to_numpy()[indices]
        eligible = ((masks[:, None] >> ((REPORT_INDEX - 12) // 12)) & 1).astype(bool)
        opportunity = (np.isfinite(values[indices][:, REPORT_INDEX]) & eligible).sum(axis=0)
        states = np.where(counts > 0, "active", np.where(opportunity / n_total >= .5,
                          "observed_no_candidate", "unobserved_or_ineligible"))
        for index, count, n_valid, state in zip(REPORT_INDEX, counts, opportunity, states):
            monthly_rows.append(dict(sample_id=sample_id, month_index=int(index), month=MONTHS[index],
                active_pixels=int(count), opportunity_pixels=int(n_valid), opportunity_fraction=float(n_valid/n_total),
                state=state, total_pixels=n_total))
        for order, (start, end) in enumerate(active_phases(counts)):
            start_index, end_index = int(REPORT_INDEX[start]), int(REPORT_INDEX[end])
            phase_id = "phase_" + hashlib.sha256(f"{sample_id}|{start_index}|{end_index}".encode()).hexdigest()[:24]
            phase_nodes = selected.loc[selected.month_index.between(start_index, end_index)]
            nodes.loc[phase_nodes.index, "phase_id"] = phase_id
            earliest = phase_nodes.sort_values(["month_index", "unit_id"]).drop_duplicates("unit_id")
            local = np.full((len(earliest), 13), np.nan)
            for i, event in enumerate(earliest.itertuples()):
                offsets = event.month_index + np.arange(-6, 7)
                first = pd.Period(event.window_start, freq="M").ordinal - pd.Period("2016-01", freq="M").ordinal
                last = pd.Period(event.window_end, freq="M").ordinal - pd.Period("2016-01", freq="M").ordinal
                allowed = (offsets >= max(0, first)) & (offsets <= min(125, last))
                local[i, allowed] = values[row_index[event.unit_id], offsets[allowed]]
            template = build_phase_template(earliest.unit_id.tolist(), local)
            templates[phase_id] = template["template"]
            phase_rows.append(dict(sample_id=sample_id, phase_id=phase_id, phase_order=order,
                start_index=start_index, end_index=end_index, start_month=MONTHS[start_index], end_month=MONTHS[end_index],
                total_pixels=n_total, n_nodes=len(phase_nodes), n_response_pixels=len(earliest),
                template_computable=bool(template["computable"]), template_members=template["eligible_count"],
                singleton_interior_months=int(np.sum(counts[start+1:end] == 1))))
            for j, offset in enumerate(range(-6, 7)):
                point_rows.append(dict(sample_id=sample_id, phase_id=phase_id, offset=offset,
                    template=template["template"][j], q25=template["q25"][j], q75=template["q75"][j],
                    contributor_count=int(template["contributor_count"][j]),
                    fraction_of_phase=float(template["contributor_fraction"][j]),
                    contributor_ids=json.dumps(sorted(template["contributor_ids"][j])),
                    contributor_changes=json.dumps(template["contributor_changes"][j])))
            for i, event in enumerate(earliest.itertuples()):
                member_rows.append(dict(sample_id=sample_id, phase_id=phase_id, unit_id=event.unit_id,
                    node_id=event.node_id, candidate_month=event.response_month,
                    original_window_start=str(event.window_start), original_window_end=str(event.window_end),
                    eligible=bool(template["eligible"][i]), baseline=template["baseline"][i], scale=template["scale"][i],
                    valid_offsets=json.dumps((np.flatnonzero(np.isfinite(local[i])) - 6).tolist())))
    assert nodes.phase_id.ne("").all() and nodes.node_id.is_unique
    frames = {"monthly_activity": pd.DataFrame(monthly_rows), "phases": pd.DataFrame(phase_rows),
        "phase_template_points": pd.DataFrame(point_rows), "phase_template_members": pd.DataFrame(member_rows),
        "nodes": nodes}
    for name, frame in frames.items():
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"Prepared {len(phase_rows)} phases and {len(member_rows)} local trajectories", flush=True)
    return frames, templates


def temporal_edges(phases, monthly, templates, max_gap):
    states = {(r.sample_id, r.month_index): r.state for r in monthly.itertuples()}
    rows = []
    for sample_id, group in phases.groupby("sample_id", sort=True):
        ordered = list(group.sort_values("phase_order").itertuples())
        for left, right in zip(ordered[:-1], ordered[1:]):
            gap_indices = range(left.end_index + 1, right.start_index)
            gap = [states[sample_id, index] for index in gap_indices]
            result = classify_gap(gap, max_gap)
            comparison = response_distance(templates[left.phase_id], templates[right.phase_id])
            rows.append(dict(sample_id=sample_id, left_phase_id=left.phase_id, right_phase_id=right.phase_id,
                edge_id=left.phase_id + "__" + right.phase_id, G=max_gap, relation=result["relation"],
                gap_months=len(gap), longest_quiet_run=result["longest_quiet_run"],
                gap_has_unknown=result["has_unknown"], gap_states=json.dumps(gap),
                quiet_runs_months=json.dumps([[MONTHS[left.end_index+1+a], MONTHS[left.end_index+1+b]]
                                              for a, b in result["quiet_runs"]]),
                distance=comparison["distance"], sts_available=comparison["available"],
                common_offsets=json.dumps(comparison["common_offsets"].tolist()),
                left_end_month=MONTHS[left.end_index], right_start_month=MONTHS[right.start_index]))
    return pd.DataFrame(rows)


def threshold_candidates(edges):
    eligible = edges.loc[edges.relation.eq("temporal_join") & edges.sts_available].copy()
    if eligible.empty:
        return []
    eligible["weight"] = 1 / eligible.groupby("sample_id").sample_id.transform("size")
    eligible = eligible.sort_values(["distance", "edge_id"])
    cdf = eligible.weight.cumsum().to_numpy() / eligible.weight.sum()
    return [(q, float(eligible.iloc[min(np.searchsorted(cdf, q/100), len(eligible)-1)].distance))
            for q in (10, 25, 50, 75, 90)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan_at_execution.md").write_bytes(PLAN.read_bytes())
    units, members, values, execution, nodes, baseline, tables, protected = load_inputs()
    frames, templates = prepare_phases(units, members, values, execution, nodes)
    monthly, phases = frames["monthly_activity"], frames["phases"]
    edge_sets = {g: temporal_edges(phases, monthly, templates, g) for g in range(4)}
    pd.concat(edge_sets.values(), ignore_index=True).to_parquet(OUT / "phase_relations_B1.parquet", index=False)
    thresholds = threshold_candidates(edge_sets[1])
    configs = [(f"B1_G{g}_H{h}", g, h, np.inf) for g in range(4) for h in (6, 9, 12)]
    configs += [(f"B2R_G1_H12_Q{q}", 1, 12, theta) for q, theta in thresholds]
    configs += [("B2R_G1_H12_OFF", 1, 12, np.inf)]
    config = dict(protocol="balsa_episode_v3_label_exploration_20261101", source_method=engine.METHOD,
        primary_temporal_pilot=dict(G=1,H=12), support_fraction=.2, q_min=.5, feature_offsets=[-6,6],
        threshold_procedure="stand-balanced inverse empirical distance CDF; no label optimisation",
        scientific_status="exploratory_existing_labels_not_independent_confirmation",
        metrics_scope="fixed_month_reference_stands_2017_2025; label_relative_precision_and_F1",
        tolerances_months=[1,2,3], methods=[dict(method=m,G=g,H=h,theta=None if np.isinf(t) else t,
             sts_disabled=bool(np.isinf(t))) for m,g,h,t in configs],
        direct_label_evaluation_authorized=True, reference_identity_claims=False)
    write_json(OUT / "configuration_frozen.json", config)
    summaries, matches, per_stands, intervals, counts, cuts = [], [], [], [], [], []
    baseline_result = evaluate_labels("B0", baseline, tables["references"], tables["eligibility"], tables["stands"])
    assert baseline_result["summary"].set_index("tolerance_months").loc[2, "matched_labels"] == 930
    outputs = {"B0": baseline_result}
    scientific_b1 = None
    for method, g, h, theta in configs:
        result = assemble_episodes(phases, edge_sets[g], nodes, monthly,
                                  max_gap=g, max_span=h, threshold=theta, method=method)
        directory = OUT / "methods" / method
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in result.items():
            frame.to_parquet(directory / f"{name}.parquet", index=False)
        assert len(result["source_members"]) == len(nodes) and result["source_members"].node_id.is_unique
        if method == "B1_G1_H12":
            scientific_b1 = result
        if method == "B2R_G1_H12_OFF":
            # IDs are version-specific; compare the source partition and scientific fields.
            for name in ("reports", "components"):
                compare = [c for c in result[name].columns if c not in
                           {"method", "prediction_id", "component_id", "candidate_id"}]
                pd.testing.assert_frame_equal(scientific_b1[name][compare].reset_index(drop=True),
                                              result[name][compare].reset_index(drop=True), check_dtype=False)
        evaluation = evaluate_labels(method, result["reports"], tables["references"], tables["eligibility"], tables["stands"])
        outputs[method] = evaluation
        destinations = result["source_members"].destination_type.value_counts().to_dict()
        counts.append(dict(method=method,G=g,H=h,theta=np.nan if np.isinf(theta) else theta,
            n_reports=len(result["reports"]), n_components=len(result["components"]),
            destinations=json.dumps(destinations)))
        decisions = result["edge_decisions"]
        selected_cuts = decisions.loc[decisions.sts_status.eq("vetoed")].copy()
        if len(selected_cuts):
            selected_cuts["method"] = method
            cuts.append(selected_cuts)
        main_result = evaluation["summary"].set_index("tolerance_months").loc[2]
        print(f"{method}: reports={len(result['reports'])}, found={main_result['matched_labels']}, "
              f"P_label={main_result['precision_label']:.4f}, R={main_result['recall_label']:.4f}, "
              f"F1_label={main_result['F1_label']:.4f}, cuts={len(selected_cuts)}", flush=True)
    for method, result in outputs.items():
        summaries.append(result["summary"])
        matches.append(result["matches"])
        per_stands.append(result["per_stand"])
        intervals.append(result["interval_summary"])
    eval_dir = OUT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(eval_dir / "label_metrics.csv", index=False)
    pd.concat(matches, ignore_index=True).to_parquet(eval_dir / "reference_matches.parquet", index=False)
    per_stand = pd.concat(per_stands, ignore_index=True)
    per_stand.to_parquet(eval_dir / "per_stand_counts.parquet", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(eval_dir / "interval_compatibility.csv", index=False)
    pd.DataFrame(counts).to_csv(eval_dir / "output_counts.csv", index=False)
    pd.concat(cuts, ignore_index=True).to_parquet(eval_dir / "sts_cut_edges.parquet", index=False)
    paired_summary(per_stand, tables["stands"], baseline="B0").to_csv(eval_dir / "paired_vs_B0.csv", index=False)
    paired_summary(per_stand, tables["stands"], baseline="B1_G1_H12").to_csv(eval_dir / "paired_vs_B1.csv", index=False)
    coverage = []
    for g, edges in edge_sets.items():
        join = edges.relation.eq("temporal_join")
        coverage.append(dict(G=g, adjacent_pairs=len(edges), joinable_pairs=int(join.sum()),
            joinable_sts_computable=int((join & edges.sts_available).sum()),
            unknown_pairs=int(edges.relation.eq("boundary_unknown").sum()),
            separated_with_missing=int((edges.relation.eq("temporal_separate") & edges.gap_has_unknown).sum())))
    pd.DataFrame(coverage).to_csv(OUT / "sts_opportunity.csv", index=False)
    for path, expected in protected.items():
        assert digest(path) == expected, f"Protected input changed during execution: {path}"
    write_json(OUT / "execution_receipt.json", dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        exact_B20_parity=True, B2_OFF_scientific_parity=True, original_inputs_and_tex_unchanged=True,
        accepted_nodes=len(nodes), phases=len(phases), singleton_interior_months=int(phases.singleton_interior_months.sum()),
        methods_scored=len(outputs), real_national_rerun=False, method_selected_for_deployment=None,
        source_sha256={str(p):digest(p) for p in HERE.glob("*.py")},
        output_sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob("*") if p.is_file() and p.name != "execution_receipt.json"}))
    print("Completed real validation-domain experiment; no production replacement", flush=True)


if __name__ == "__main__":
    main()
