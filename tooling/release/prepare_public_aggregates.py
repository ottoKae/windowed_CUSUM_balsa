"""Publish aggregate-only copies of the final reference scoring; never rescore."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/validation/jag_reference_score_closeout_20261015_v1"
OUT = ROOT / "results/final_reference_score_20261015"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    origins = {}
    for name in ("main_results.csv", "weighted_precision.csv", "error_explanation.csv"):
        source = SOURCE / "scoring" / name
        table = pd.read_csv(source)
        if set(table.columns) & {"sample_id", "prediction_id", "verified_event_id", "evidence_notes", "geometry"}:
            raise ValueError("Aggregate export contains individual records: " + name)
        target = name
        if name == "error_explanation.csv":
            target = "error_composition.csv"
            labels = dict(zip(
                ["唯一正确报告", "确认无对应事件或位置不符", "定时超限", "重复报告", "参考未判定", "身份计分待完成"],
                ["unique_correct_report", "no_target_event_or_wrong_location", "timing_outside_tolerance",
                 "duplicate_report", "reference_unresolved", "identity_scoring_pending"]))
            if not set(table.category).issubset(labels):
                raise ValueError("Unknown error label")
            table["category"] = table.category.map(labels)
            assert table.groupby("tolerance_months").n_reports.sum().eq(307).all()
        else:
            assert table.n_selected.eq(307).all()
            assert table[["n_T", "n_F", "n_U"]].sum(axis=1).eq(307).all()
        if target == name:
            shutil.copy2(source, OUT / target)
        else:
            table.to_csv(OUT / target, index=False)
        origins[target] = {"source": "final_reference_score_closeout/scoring/" + name,
                           "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "export_sha256": hashlib.sha256((OUT / target).read_bytes()).hexdigest(),
                           "transformation": "English category labels only" if target != name else "Byte-identical aggregate copy"}
    raw = json.loads((SOURCE / "evaluation_manifest.json").read_text(encoding="utf-8"))
    keys = ["protocol_id", "base_configuration_sha256", "detector_and_aggregation_unchanged",
            "primary_tolerance_months", "sensitivity_tolerances_months", "all_scored_tolerances_months",
            "tolerance_selection", "polarization_pair_tolerance_months", "catalogue_reference",
            "precision_evidence_version", "population_reports", "selected_reports", "sample_and_weights",
            "estimator", "qualified_complete_history_domains", "population_F1"]
    metadata = {key: raw[key] for key in keys}
    metadata["detector_configuration"] = "configs/standard_w36.json"
    metadata["note"] = "This evaluation overlay supersedes the historical M2 display default, not the frozen detector or polarization pairing."
    (OUT / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUT / "SOURCES.json").write_text(json.dumps(origins, indent=2) + "\n", encoding="utf-8")
    print(f"Published {len(origins)} aggregate tables; no observations or scoring changed.")


if __name__ == "__main__":
    main()
