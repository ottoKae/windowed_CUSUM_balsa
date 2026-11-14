"""Recompute every frozen detector comparison in the current range-matching design."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tooling/diagnostics/range-primary-20261111/benchmark.py"
OLD = ROOT / "runs/validation/range_primary_20261111_v1/benchmark"
OUT = ROOT / "runs/validation/results_refresh_20261111_v1/benchmark"


def main():
    start = perf_counter()
    spec = importlib.util.spec_from_file_location("results_refresh_benchmark", SOURCE)
    scorer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scorer
    spec.loader.exec_module(scorer)
    old_receipt = json.loads((OLD / "receipt.json").read_text(encoding="utf-8"))
    for path, digest in old_receipt["source_sha256"].items():
        assert sha(ROOT / path) == digest, path
    protected = {str(path.relative_to(ROOT)): sha(path) for path in OLD.rglob("*") if path.is_file()}
    assert OUT != OLD and OUT != scorer.old.OUT
    scorer.OUT = OUT
    OUT.mkdir(parents=True,exist_ok=True)
    print("Recomputing 14 benchmarks with frozen reports and 5,000 shared stand draws",flush=True)
    with (OUT / "run.log").open("w",encoding="utf-8") as stream, redirect_stdout(stream):
        scorer.main()
    generated = OUT / "results_benchmark.tex"
    fragment = generated.read_text(encoding="utf-8").replace(
        "runs/validation/range_primary_20261111_v1/benchmark/summary.csv",
        "runs/validation/results_refresh_20261111_v1/benchmark/summary.csv")
    generated.write_text(fragment,encoding="utf-8")
    comparison = []
    for name in ("summary.csv","paired_effects.csv","primary_contrasts.csv",
                 "recall_by_reference_year.csv","recall_by_reference_multiplicity.csv"):
        previous,current = pd.read_csv(OLD / name),pd.read_csv(OUT / name)
        pd.testing.assert_frame_equal(current,previous,check_exact=True)
        comparison.append(dict(file=name,rows=len(current),equal_to_previous=True))
    for method in scorer.old.METHODS:
        for name in ("matches.parquet","original_pairs.parquet"):
            pd.testing.assert_frame_equal(pd.read_parquet(OUT / "methods" / method / name),
                pd.read_parquet(OLD / "methods" / method / name),check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT / "common_reference_and_pair_sets.parquet"),
        pd.read_parquet(OLD / "common_reference_and_pair_sets.parquet"),check_exact=True)
    manuscript = ROOT / "manuscript_JAG/sections/04_results.tex"
    text = manuscript.read_text(encoding="utf-8")
    start_at = text.index(r"\subsection{Benchmark performance of harvest detectors}")
    end_at = text.index("% \\newpage",start_at)
    active = text[start_at:end_at]
    (OUT / "manuscript_section_at_refresh.tex").write_text(active,encoding="utf-8")
    normalize = lambda s: re.sub(r"\s+"," ","\n".join(line for line in s.splitlines() if not line.lstrip().startswith("%"))).strip()
    section_equal = normalize(active) == normalize(fragment)
    assert protected == {p:sha(ROOT / p) for p in protected}
    receipt = json.loads((OUT / "receipt.json").read_text(encoding="utf-8"))
    receipt.update(refresh_status="passed",refresh_source=str(SOURCE.relative_to(ROOT)),
        refresh_wrapper_sha256=sha(Path(__file__)),refresh_seconds=perf_counter()-start,
        all_14_matches_and_original_pairs_identical=True,all_common_sets_identical=True,
        numerical_result_comparison=comparison,old_outputs_unchanged=True,
        active_manuscript_section_identical_ignoring_comments=section_equal,
        manuscript_numeric_revision_required=not section_equal,
        manuscript_snapshot_sha256=sha(OUT / "manuscript_section_at_refresh.tex"))
    (OUT / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    summary = pd.read_csv(OUT / "summary.csv")
    print(summary[["method_label","n_reports","n_matches","catalogue_precision","recall","catalogue_f1"]].to_string(index=False))
    print(f"Refresh passed; active manuscript section equal: {section_equal}",flush=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


if __name__ == "__main__":
    main()
