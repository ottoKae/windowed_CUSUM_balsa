# Repository map and version boundaries

Updated: 2026-10-15. This guide describes the author workspace and the smaller,
allowlisted public source export. Existing scientific inputs and outputs remain
in place so frozen hashes, review links and scripts retain their original paths.

## One entry point for each responsibility

1. Start with `README.md` for the current detector and evaluation version.
2. Find a data family in `provenance/DATA_CATALOG.csv`; follow its relative index
   to the original manifest or the unified local sample collection.
3. Read `data/schemas/sample_collection.md` before joining review records. The
   collection preserves stand, event, report, observation and sampling identities.
4. Use `configs/standard_w36.json` and `scripts/reproduce.py` for detection.
5. Use `results/final_reference_score_20261015/` for current precision and
   `results/w36_standard/result_index.csv` for the detector comparisons.
6. Use `provenance/figures/` to trace the actual currently included TeX assets.
7. Edit manuscript prose only in the author's `manuscript_JAG_v2/` tree.

## Versions that must remain separate

| Material | Meaning |
|---|---|
| `configs/standard_w36.json` | Immutable detector/aggregation configuration; its original evaluation field uses +/-2 months |
| `results/final_reference_score_20261015/evaluation.json` | Final reference-score version; primary display +/-3 months, with +/-1 and +/-2 months retained |
| `results/w36_standard/precision.csv` | Historical probability-score snapshot; not the latest reference revisions |
| `results/w36_standard/benchmarks.csv` and related comparisons | Current pixel+B20 detector comparisons on the frozen catalogue |
| `results/formal_validation/`, `configuration/`, `component_ablation/`, `national/`, `polarization/` | Older manuscript workflows; not interchangeable with W36+B20 validation |
| `data/local/paper_v1/` and older B/C capsules | Historical input/release contracts |
| `data/local/standard_w36_v1/` | Current author-local W36 pixel input |
| `archive/legacy_20261004/` | Retained historical code/configurations; excluded from public selection unless a specific figure dependency is explicitly selected |

Changing the evaluation tolerance never changes the detector, B20 window length,
representative month or the separate +/-2-month polarization pairing tolerance.
No detector was rerun during this organization task.

## How visual samples are consolidated

The local collection is a searchable, versioned index and normalized copy of
existing evidence. It is not a replacement editor for the original GUI/Excel
sources. Keep the raw answer, source file, source hash and row identity available.

- A stand can have several independent events. An event can have several stages,
  observations and competing detection reports.
- Do not merge events because their dates or stand IDs coincide. Only an explicit
  identity decision can establish that two reports concern the same event.
- Observation month, confirmed event date, detector month and scoring tolerance
  are different fields. A first optical observation need not be harvest onset.
- The 307 formal samples, context reports, 31 repeat interpretations, relation
  checks, catalogue references and seven history reviews retain separate roles.
- Preserve unknowns, pending work, weights, original sample selection and all
  revision lineage. Additional evidence does not create a new probability sample.
- Unclear periods within 2018--2022 remain unclear. Absence of later imagery is
  not the only completeness limitation, and observed harvests do not imply an
  exhaustively checked history.

The collection is private because it retains original identifiers and evidence
text. Public readers receive schemas, cohort summaries, aggregate scores and
source-family descriptions. Approved deidentified capsules are a separate release.

## Tests, temporary files and historical material

Reusable test source belongs in `tests/`. Disposable runs belong in
`.test-tmp/<run-id>/`; never use a dataset or a shared parent as pytest basetemp.
One-off checks and author-side interpretation helpers belong in
`tooling/diagnostics/<run-id>/`. Scientific run receipts belong in `runs/`.

The repository already has a historical `archive/` and centralized diagnostics.
No blanket deletion based on names such as `test`, `tmp` or `review` is safe:
some such files are evidence or dependencies. This cleanup retains these local
records, excludes them from the public export and makes the maintained tests
explicit. `.gitignore` does not remove files already tracked by Git.

## Public export rather than an indiscriminate upload

`tooling/release/export_public.py` uses `configs/public_export.json` and writes a
new export directory. The resulting `PUBLIC_FILES.json` is the release inventory.
Build into a fresh destination to avoid stale files from older exports. See
`tooling/README.md` for the command and verification outputs.

The public source can run the synthetic example and its selected tests. Real
replay still requires separately available inputs. Figure sources may also need
private data or licensed imagery; the figure catalogue records these limits.
Manuscript source availability does not mean the manuscript has been updated to
the final scores, and the source-only export is not a complete camera-ready bundle.

The October 15 working-tree receipt lives locally under
`runs/repository_organization_20261015_v1/`. It protects the manuscript and frozen
reference-score files and records checks performed during organization.
