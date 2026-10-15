# Entry points

Current reference-score aggregates are in
`../results/final_reference_score_20261015/`. `evaluate_standard.py` remains the
historical capsule replay; it does not automatically load the October 15 reference
revisions. For the actual currently included TeX figures, use
`../provenance/figures/README.md`; the old `build_all.py` order is not their current
manuscript numbering.

| Task | Entry |
|---|---|
| W36 pixel detection and B20 | `reproduce.py` |
| Frozen probability precision and catalogue recall | `evaluate_standard.py` |
| Approved polarization/null summary | `summarize_w36_extensions.py` |
| Existing pixel+B20 benchmark runner | `run_pixel_b20_benchmark_window.py` |
| Existing cloud benchmark adapters | `run_pixel_b20_cloud_benchmarks.py` |
| Manuscript figure generation (retained historical sources) | `figures/build_all.py` |

Other retained files support these entries, existing review interfaces, or tests. Historical experiment launchers are in ignored `archive/legacy_20261004/scripts`. The current package lives in `src/lr_cusum`; do not add sibling archive directories to PYTHONPATH.
