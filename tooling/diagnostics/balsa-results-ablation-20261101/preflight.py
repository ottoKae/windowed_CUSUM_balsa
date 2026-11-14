"""Verify the frozen baseline before the approved ablations run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tooling/release"),
               str(ROOT / "tooling/diagnostics/balsa-national-firstpost-20261030")]
from build_sample_release import original_tables
from windowed_cusum.evaluation.catalogue_evaluation import catalogue_metrics
import detector_engine as engine

SOURCE = ROOT / "runs/validation/balsa_firstpost_national_20261030_v2"
TRACE = ROOT / "runs/validation/balsa_firstpost_result_analysis_20261030_v1"
OUT = ROOT / "runs/validation/balsa_results_ablation_20261101_v1"
PLAN = ROOT / "docs/experiments/external_execution_20261026/BALSA_RESULTS_ABLATION_COMPLETION_PLAN_20261101_v1.md"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "preflight_receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        for path, expected in previous["protected_sha256"].items():
            assert digest(path) == expected, path
        print("P0 already completed; protected hashes unchanged")
        return
    signature = json.loads((SOURCE / "method_signature.json").read_text())
    protected = dict(signature["source_sha256"])
    for path, expected in protected.items():
        assert digest(path) == expected, path
    previous = json.loads((ROOT / "runs/validation/balsa_research_execution_20261031_v3/preflight_receipt.json").read_text())
    protected.update({p: digest(p) for p in previous["input_sha256"]})
    protected.update({str(p): digest(p) for p in (ROOT / "manuscript_JAG").rglob("*.tex")})
    paths = [SOURCE / n for n in ("master_units.parquet", "master_values.npy", "validation_membership.parquet",
        "validation/pixel_events.parquet", "validation/stand_reports.parquet", "validation/source_members.parquet")]
    paths += [TRACE / "pixel_window_trace.parquet", TRACE / "firstpost_trace_receipt.json"]
    tables, reference_config = original_tables()
    paths += [Path(v["path"]) for k, v in reference_config["inputs"].items()
              if k in ("references", "event_comparison", "units", "membership", "old_blocks")]
    protected.update({str(p): digest(p) for p in paths})
    (OUT / "plan_at_execution.md").write_bytes(PLAN.read_bytes())
    units = pd.read_parquet(SOURCE / "master_units.parquet")
    members = pd.read_parquet(SOURCE / "validation_membership.parquet")
    units = units[units.unit_id.isin(members.unit_id)].copy()
    assert len(units) == 26890 and members.sample_id.nunique() == 468
    values = np.load(SOURCE / "master_values.npy", mmap_mode="r")
    assert values.shape[1] == 126 and units.unit_id.is_unique
    events = pd.read_parquet(SOURCE / "validation/pixel_events.parquet")
    trace = pd.read_parquet(TRACE / "pixel_window_trace.parquet")
    assert len(trace) == 242010 and trace.pre_null.sum() == 79301
    assert trace.refined.sum() == 6521 and trace.accepted.sum() == 44311
    assert not trace.duplicated(["unit_id", "logical_window"]).any()
    actual = trace[trace.accepted].copy()
    expected = events.assign(decision_year=events.change_time.dt.year)
    paired = actual.merge(expected, on=["unit_id", "decision_year"], validate="one_to_one", suffixes=("_trace", "_event"))
    assert len(paired) == len(events) == 44311
    for left, right in (("candidate_index", "change_index"), ("final_p", "null_p_value"),
        ("cusum_amplitude_trace", "cusum_amplitude_event"), ("post_minus_pre_trace", "post_minus_pre_event"),
        ("permutations_trace", "permutations_event"), ("exceedance_count_trace", "exceedance_count_event")):
        np.testing.assert_array_equal(paired[left], paired[right])
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    reports, nodes, aggregation = engine.aggregate(units, members, events, protocol)
    frozen_reports = pd.read_parquet(SOURCE / "validation/stand_reports.parquet")
    frozen_nodes = pd.read_parquet(SOURCE / "validation/source_members.parquet")
    for a, b, key in ((reports, frozen_reports, "prediction_id"), (nodes, frozen_nodes, "node_id")):
        pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True),
            b[a.columns].sort_values(key).reset_index(drop=True), check_dtype=False, check_exact=True)
    assert len(reports) == 1267 and len(nodes) == 44311
    detail, summary, *_ = catalogue_metrics("full_firstpost", reports,
        tables["references"], tables["eligibility"], tables["stands"])
    primary = detail[detail.tolerance_months.eq(2) & detail.timing_type.eq("month")]
    assert len(primary) == 1049 and primary.found.sum() == 930
    detail.to_csv(OUT / "baseline_matches.csv", index=False)
    summary.to_csv(OUT / "baseline_catalogue_recall.csv", index=False)
    units[["unit_id", "pixel_track_id", "track_key", "series_index"]].to_csv(OUT / "validation_units.csv", index=False)
    reused = {
        "no_local_window_and_polygon": TRACE / "contributions/receipt.json",
        "windows": ROOT / "runs/validation/balsa_research_execution_20261031_v3/windows/verification_receipt.json",
        "q20_and_benchmarks": ROOT / "runs/validation/balsa_research_execution_20261031_v3/controls/controls_acceptance.json",
        "iid_block3": ROOT / "runs/validation/balsa_research_execution_20261031_v3/simulation/verification_receipt.json"}
    for path, expected_hash in protected.items():
        assert digest(path) == expected_hash, path
    result = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        method_signature=signature["signature"], protected_sha256=protected,
        plan_sha256=digest(OUT / "plan_at_execution.md"), fixed_pixels=26890, fixed_stands=468,
        trace_windows=len(trace), pre_null_candidates=79301, accepted_nodes=44311,
        baseline_reports=1267, reference_events=1049, reference_found=930,
        aggregation=aggregation, exact_trace_parity=True, exact_B20_reports_and_sources=True,
        reused_receipts={k: {"path": str(p), "sha256": digest(p)} for k, p in reused.items()},
        output_sha256={name: digest(OUT / name) for name in ("baseline_matches.csv", "baseline_catalogue_recall.csv", "validation_units.csv")},
        script_sha256=digest(Path(__file__)))
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "execution_status.json").write_text(json.dumps(dict(P0="completed", P1="ready", P2="not_started",
        P3="not_started", P4="not_started", P5="not_started"), indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("passed", "fixed_pixels", "baseline_reports", "reference_found")}))


if __name__ == "__main__":
    main()
