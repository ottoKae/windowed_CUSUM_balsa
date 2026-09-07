# Windowed CUSUM: JAG manuscript revision

Revision date: 2026-09-05. This is a separate, buildable English manuscript. The previous `../manuscript/` source and all eight figure PDFs are preserved. No detector code, frozen output, Git commit or remote branch was changed by this revision.

## Read and build

- `main.pdf`: compiled manuscript for review.
- `main.tex`: entry point, title, authors, abstract and highlights.
- `sections/`: complete revised Introduction, Materials, Methods, Results, Discussion, Conclusion and declarations.
- `references.bib`: 29 cited records, with verified DOI/metadata and ASCII TeX accents for classic BibTeX.
- `figures/`: byte-identical copies of the original eight figure PDFs. Their internal layouts were not redrawn.
- `audit/`: evidence receipts, derived analyses, unresolved issues and reproducibility checks. This folder is not an appendix automatically included in the submission word count.

For VS Code, open `../JAG-manuscript.code-workspace`; see [BUILDING.md](BUILDING.md)
for the checked incremental recipe, bibliography updates and recovery instructions.

In a PowerShell terminal opened in this folder:

```powershell
.\build_latex.ps1
```

The script performs `pdflatex -> bibtex -> pdflatex -> pdflatex` and checks final undefined citations/references. It expects the installed TeX Live environment. `main.tex` loads `float` before the CAS class loads `hyperref`; this removes duplicate float-destination warnings without changing figure content. The CAS title block still emits one non-fatal overfull-box warning; the rendered title page was visually inspected. No undefined citations, unresolved references or duplicate PDF destinations remain in the checked build.

## Scientific structure

The main question is whether track-specific, near-decadal Sentinel-1 records and explicit local event-admission rules can reconstruct recurrent disturbances in the monitored balsa inventory. Record duration is not claimed as a first. Sliding-window CuSum NRT is acknowledged explicitly.

1. **Signal:** reference-defined within-stand changes precede the illustrative case.
2. **Detector:** explain why local context is renewed and why estimation is separate from admission; retain the actual post-segment indexing convention.
3. **Evidence:** distinguish fixed 48/12 N1 component/representation results from fold-selected benchmark predictions.
4. **Temporal support:** separate record span, local window geometry and post-event support.
5. **Application:** counts and intervals describe SAR-detected disturbance histories, not independently verified rotations.

Methods 3.2 now explains window rationale, calendar clipping, the empty 24/12 configuration, finite-observation indexing, direction and permutation assumptions. Repetitive results and speculative market attribution were removed. The previously mixed 2026 candidate-precision statement was replaced with the actual N1 candidate count and its review limitation.

## Completed additional analyses

`audit/derive_support.py` reuses frozen Parquet outputs and does not rerun a detector. Run with the existing pandas/numpy environment:

```powershell
python .\audit\derive_support.py
python .\audit\check_revision.py
```

Reporting-span restriction, common 3,498 polygons and December 2025 endpoint:

| Span | Events | Polygons with at least two detections | Original adjacent intervals | Median interval |
|---|---:|---:|---:|---:|
| 2021–2025 | 3,998 | 1,092 | 1,349 | 30 months |
| 2019–2025 | 5,399 | 1,752 | 2,407 | 34 months |
| 2017–2025 | 6,658 | 2,194 | 3,509 | 36 months |

This is descriptive product restriction, **not** a 5/7/9-year shortened-input accuracy experiment. Longer ranges mechanically permit more events and longer intervals, and also change the sampled years.

Exploratory recall from the same frozen OOF matches:

| Reference multiplicity | Polygons | Windowed CUSUM | Decrease-filtered ReCuSum |
|---|---:|---:|---:|
| One | 55 | 0.891 | 0.764 |
| Two | 209 | 0.813 | 0.624 |
| Three or more | 198 | 0.794 | 0.553 |

The last group's paired recall gain is 0.240 [0.179, 0.289], using 5,000 identical whole-block bootstrap draws. This does not establish a cross-stratum interaction, specifically later-event recovery, full-history recovery, or stratified precision/F1. Those extensions are not claimed as completed.

## Evidence and changes requiring attention

See `audit/EVIDENCE_AND_PENDING.md` for the claim-to-source map and the work still needed before stronger claims or release. In particular:

- CuSum NRT/credible CCDC comparisons were **not** implemented or run in this writing revision.
- The abbreviated BFAST-style row is removed from the new main ranking, not silently relabelled as full BFAST. Old outputs are untouched.
- The 2026 N1 replay has 140 candidates; old 139-candidate review precision cannot be carried over.
- No repository remote URL exists in this checkout. The availability statement does not invent an `OWNER` URL or DOI.
- Authors must confirm authorship, corresponding-author marks/email, funding, conflicts, AI disclosure, data rights and final scientific interpretation. This is a review-ready draft, not a certified submission package.

`audit/revision_checks.json` records original-file and figure SHA256 checks, bibliography checks, final compilation diagnostics and transparent PDF word accounting. The 8,000-word check includes extracted references, tables, captions and declarations; it is an estimate, not a publisher-certified count.
