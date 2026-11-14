"""Copy explicitly selected frozen aggregate results and index the paper experiments."""
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/validation/results_refresh_20261111_v1"
CONTEXT_SOURCE = ROOT / "runs/figures/harvest_context_20261113_v17_final/source_data"
CONTEXT_TABLES = {
    "national_canton_annual_reports.csv": "annual_canton_reports.csv",
    "national_canton_quarterly_reports.csv": "quarterly_canton_reports.csv",
    "national_canton_ranking.csv": "canton_ranking.csv",
    "national_canton_classes.csv": "canton_color_classes.csv",
    "national_climate_interval_distributions.csv": "interval_distributions.csv",
    "national_climate_interval_summary.csv": "interval_group_summary.csv",
}
TABLES = {
    "table03_components.csv": "components/summary.csv",
    "table03_adjacent_effects.csv": "components/adjacent_effects.csv",
    "table04_configuration.csv": "configuration/evaluation/summary.csv",
    "table04_paired_effects.csv": "configuration/evaluation/paired_effects.csv",
    "table05_benchmarks.csv": "benchmark/summary.csv",
    "table05_primary_contrasts.csv": "benchmark/primary_contrasts.csv",
    "table05_paired_effects.csv": "benchmark/paired_effects.csv",
    "temporal_blocks.csv": "blocks/evaluation/summary.csv",
    "temporal_blocks_effects.csv": "blocks/evaluation/paired_effects.csv",
    "backscatter_signatures.csv": "signatures/summary.csv",
    "backscatter_absolute.csv": "signatures/absolute_backscatter_summary.csv",
    "backscatter_aligned_curve.csv": "signatures/figure_source_aligned.csv",
    "national_metrics.csv": "histories/metrics.csv",
    "national_annual_reports.csv": "histories/annual_reports.csv",
    "national_stand_count_distribution.csv": "histories/stand_count_distribution.csv",
    "national_stp_counts.csv": "histories/stp_partition_counts.csv",
}
GROUPS = {
    "backscatter": ["balsa-results-optimization-20261105/signatures.py", "results-refresh-20261111/signatures.py"],
    "components": ["cumulative-components-20261111/compute.py", "cumulative-components-20261111/range_overlap_pm1.py"],
    "configuration": ["range-primary-20261111/sensitivity.py", "configuration-extension-20261111/geometry.py", "configuration-extension-20261111/score.py"],
    "temporal_null": ["block-length-sensitivity-20261111/run_blocks.py", "block-length-sensitivity-20261111/score.py"],
    "benchmarks": ["detector-tda-benchmark-20261109/local.py", "detector-tda-benchmark-20261109/recusum.py", "detector-tda-benchmark-20261109/cloud.py", "detector-tda-benchmark-20261109/aggregate.py", "results-refresh-20261111/benchmark.py"],
    "national": ["results-refresh-20261111/histories.py"],
}
SOURCE_DIRS = ["balsa-results-optimization-20261105", "cumulative-components-20261111",
               "range-primary-20261111", "configuration-extension-20261111",
               "block-length-sensitivity-20261111", "detector-tda-benchmark-20261109"]


def main():
    destination = ROOT / "results/summary"
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    sources = {name: SOURCE / relative for name, relative in TABLES.items()}
    sources.update({name: CONTEXT_SOURCE / relative for name, relative in CONTEXT_TABLES.items()})
    for name, path in sources.items():
        frame = pd.read_csv(path)
        forbidden = {"sample_id", "unit_id", "event_id", "prediction_id", "geometry", "longitude", "latitude", "reviewer", "evidence_notes"}
        if forbidden.intersection(frame.columns):
            raise ValueError(f"Individual-record fields in {path.relative_to(ROOT)}")
        shutil.copyfile(path, destination / name)
        records[name] = dict(source=path.relative_to(ROOT).as_posix(), rows=len(frame),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (destination / "SOURCES.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    config_path = ROOT / "configs/public_export.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected = config["files"]
    files = list(destination.glob("*"))
    for folder in SOURCE_DIRS:
        # Retain source functions/import paths, not author-only manuscript rewriting tools.
        files += [p for p in (ROOT / "tooling/diagnostics" / folder).glob("*.py")
                  if p.name not in {"update_manuscript.py", "finish.py", "finalize.py", "inventory_figures.py"}]
    for paths in GROUPS.values():
        files += [ROOT / "tooling/diagnostics" / p for p in paths]
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            raise FileNotFoundError(name)
        selected[name] = name
    catalog = {}
    for group, paths in GROUPS.items():
        catalog[group] = dict(source_files=["tooling/diagnostics/" + p for p in paths],
                             full_study_inputs="private frozen records and detector ledgers",
                             public_evaluation_entry="scripts/experiments/evaluate_reports.py")
    catalog_path = ROOT / "configs/experiments/catalog.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    for name in ["scripts/run_current_pipeline.py", "configs/current_pipeline.json", "configs/experiments/catalog.json",
                 "scripts/experiments/evaluate_reports.py", "scripts/experiments/check_published_results.py",
                 "scripts/experiments/README.md", "docs/REPRODUCIBILITY.md", "docs/PRIVACY.md",
                 "tests/test_current_pipeline.py", "tests/test_report_evaluation_entry.py",
                 "tooling/release/prepare_reviewer_materials.py", "tooling/release/update_public_ignore.py"]:
        selected[name] = name
    config["scope"] = "Current algorithm and experimental sources, synthetic examples, aggregate results, tests, manuscript and reviewed publication PDFs. No real individual records or raw imagery."
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Prepared {len(records)} aggregate tables and indexed {len(catalog)} experiment groups")


if __name__ == "__main__":
    main()
