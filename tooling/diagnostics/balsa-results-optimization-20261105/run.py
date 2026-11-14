"""P0 audit and P1a receipt for the frozen full-framework experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/validation/balsa_results_optimization_20261105_v1"
PLAN = ROOT / "docs/experiments/balsa_results_optimization_20261105/EXECUTION_PLAN_ZH.md"
SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
OLD = ROOT / "runs/validation/balsa_sts_episode_v4_1_20261102_v1/baselines/methods/R2_TIME"
A1 = ROOT / "runs/validation/balsa_stars_ablation_20261102_v1"


def display_name(value):
    """Translate frozen storage IDs without rewriting historical tables."""
    return {"OCTA": "TDA", "CSTR": "STP", "octa": "TDA", "cstr": "STP",
            "cstr_off": "stp_off", "sibling_collapsed": "STP (siblings collapsed)"}.get(value, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "gate", "check-protected"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    elif args.stage == "gate":
        gate_receipt()
    else:
        record = json.loads((RUN / "audit/frozen_inputs.json").read_text(encoding="utf-8"))
        changed = [p for p, expected in record["sha256"].items() if digest(ROOT/p) != expected]
        write_json(RUN / "audit/protected_check.json", dict(changed=changed, passed=not changed))
        if changed:
            raise RuntimeError(f"Protected inputs changed: {changed}")
        print("Protected input hashes unchanged")


def audit():
    out = RUN / "audit"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipt.json").exists():
        raise FileExistsError("P0 receipt exists; use check-protected before resuming")
    sys.path[:0] = [str(ROOT/"src"), str(ROOT/"tooling/release"),
                   str(ROOT/"tooling/diagnostics/balsa-harvest-episode-20261101")]
    from build_sample_release import original_tables
    from label_evaluation import evaluate_labels
    ranges = load_module("audit_ranges", ROOT / "tooling/diagnostics/balsa-range-evaluation-20261102/evaluate_ranges.py")
    tables, config = original_tables()
    refs_path = ROOT / "runs/validation/balsa_sts_range_evaluation_20261102_v1/reference_semantics.csv"
    refs = pd.read_csv(refs_path, keep_default_na=False)
    inventory = pd.read_csv(SOURCE / "products/stand_summary.csv")
    membership = pd.read_parquet(SOURCE / "validation_membership.parquet")
    cohort = set(refs.sample_id)
    assert (len(inventory), len(membership), membership.sample_id.nunique(), len(refs), len(cohort)) == (3498,26890,468,1049,459)
    assert sum(refs.groupby("sample_id").size()-1) == 590
    assert cohort <= set(inventory.sample_id)
    assert inventory.native_cells.eq(0).sum() == 134
    paths = {"BWA": SOURCE / "validation/stand_reports.parquet", "OCTA": OLD / "reports.parquet",
             "CSTR": A1 / "rounds_A1.parquet"}
    protected = [*paths.values(), refs_path, SOURCE/"validation_membership.parquet",
        SOURCE/"master_units.parquet", SOURCE/"master_values.npy", SOURCE/"products/stand_summary.csv",
        SOURCE/"validation/pixel_events.parquet", SOURCE/"validation/pixel_execution.parquet",
        SOURCE/"validation/source_members.parquet", OLD/"source_members.parquet",
        A1/"calibration_A1.parquet", SOURCE/"protocol.json", SOURCE/"method_signature.json",
        ROOT/"src/lr_cusum/star_segmentation.py", PLAN]
    protected += [Path(v["path"]) for v in config["inputs"].values()]
    protected += [ROOT/"manuscript_JAG/main.tex", *sorted((ROOT/"manuscript_JAG/sections").glob("*.tex"))]
    protected += list((ROOT/"runs/validation/harvest_inventory_20261104_v1/products").glob("*.parquet"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    write_json(out / "frozen_inputs.json", dict(sha256=hashes, reference_inputs=config["inputs"]))
    shutil.copyfile(PLAN, out / "plan_at_execution.md")
    checks = []
    expected = {"BWA":(1267,1255,930,792,780), "OCTA":(1216,1208,925,871,850), "CSTR":(1235,1227,930,871,850)}
    for name, path in paths.items():
        reports = pd.read_parquet(path)
        if name == "CSTR":
            reports = reports.rename(columns={"round_id":"prediction_id"})
        point = evaluate_labels(name, reports, tables["references"], tables["eligibility"], tables["stands"])
        point["matches"].to_parquet(out / f"{name}_point_matches.parquet", index=False)
        strict, details, _, _, _ = ranges.evaluate_method(name, refs, ranges.normalize_reports(reports), {})
        details.to_parquet(out / f"{name}_range_matches.parquet", index=False)
        strict.to_csv(out / f"{name}_range_summary.csv", index=False)
        pm2 = int(point["summary"].set_index("tolerance_months").loc[2,"matched_labels"])
        hits = details.groupby("matching_rule").found.sum().to_dict()
        actual = (len(reports),int(reports.sample_id.isin(cohort).sum()),pm2,int(hits["envelope"]),int(hits["actual_months"]))
        assert actual == expected[name], (name, actual, expected[name])
        checks.append(dict(method=name, method_label=display_name(name), reports_execution=actual[0], reports_reference=actual[1],
                           pm2=actual[2], envelope=actual[3], actual_month=actual[4], passed=True))
    pd.DataFrame(checks).to_csv(out / "baseline_anchors.csv", index=False)
    pd.DataFrame([dict(path=p,sha256=h,size_bytes=(ROOT/p).stat().st_size) for p,h in hashes.items()]).to_csv(
        out / "evidence_index.csv", index=False)
    actions = [("4.1", "replace", "1049 reference SAR description"),
               ("4.2/4.3", "extend", "variant-specific eligibility, TDA and STP"),
               ("4.4", "retain_scope", "existing detector+BWA comparisons; no new benchmark claim"),
               ("4.5", "replace", "1021-prefix detector+BWA cohort, not full framework"),
               ("4.6", "replace", "frozen 3498 inventory attributes; no area estimation"),
               ("STP synthetic", "repair", "production entry, truth only after prediction")]
    pd.DataFrame(actions,columns=["section","action","basis"]).to_csv(out/"claim_actions.csv",index=False)
    family = dict(prior_results_seen=True, family_size=3, family_alpha=.05, ci_level=1-.05/3,
        bootstrap_seed=20261105, bootstrap_replicates=5000, resampling_unit="stand", contrasts=[
            dict(id="H_D",left="full",right="no_local_window",level="octa",rule="pm2",metric="catalogue_f1"),
            dict(id="H_T",left="OCTA",right="BWA",level="parent",rule="envelope",metric="original_pair_retention"),
            dict(id="H_S",left="full",right="cstr_off",level="sibling_collapsed",rule="pm2",metric="catalogue_f1")],
        descriptive_other_comparisons=True, cstr_seeds=list(range(20261102,20261107)))
    write_json(RUN/"fullchain/analysis_family.json",family)
    write_json(out/"receipt.json",dict(status="passed", anchors=checks, inventory=3498, evaluable=3364,
        unavailable=134, execution_stands=468, reference_stands=459, references=1049, original_pairs=590,
        completed_utc=datetime.now(timezone.utc).isoformat(), protected_files=len(hashes)))
    print(pd.DataFrame(checks).drop(columns="method").to_string(index=False),flush=True)


def gate_receipt():
    assert json.loads((RUN/"audit/receipt.json").read_text())["status"] == "passed"
    path = RUN/"synthetic/gate_tests.xml"
    suites = ET.parse(path).getroot().findall("testsuite")
    cases = [case.attrib["name"] for suite in suites for case in suite.findall("testcase")]
    assert len(cases) == 7 and all(int(s.attrib[k]) == 0 for s in suites for k in ("errors","failures","skipped"))
    production = ROOT/"src/lr_cusum/star_segmentation.py"
    write_json(RUN/"synthetic/implementation_gate.json",dict(status="passed",tests=cases,
        production_sha256=digest(production), test_source_sha256=digest(ROOT/"tests/test_stp_production_gate.py"),
        adapter_sha256=digest(ROOT/"tooling/diagnostics/balsa-stars-20261102/run_e1_production.py"),
        junit_sha256=digest(path), production_modified=False, supports_scientific_accuracy_claim=False))
    print("P1a passed: seven tests; production algorithm unchanged")


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
