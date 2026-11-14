# Repository guide

The public snapshot exposes the current first-post W36 → TDA → STP code, selected
synthetic checks, and the approved manuscript assets. The author workspace also
contains private data and earlier experiments; those are not copied wholesale.

| Responsibility | Location |
|---|---|
| Current algorithm, callable interfaces and replay limits | [ALGORITHM.md](ALGORITHM.md) |
| Numerical library | `src/lr_cusum/` |
| Current supplied-data workflow | `scripts/run_current_pipeline.py` |
| Common report evaluation and summary-table checks | `scripts/experiments/` |
| Public experiment summaries and provenance | `results/summary/` |
| Current first-post adapter | `tooling/diagnostics/balsa-national-firstpost-20261030/detector_engine.py` |
| Current TDA/STP adapters | `tooling/diagnostics/harvest-inventory-20261104/` — selected Python files only |
| Current figure renderers and their preserved helpers | [source_spec.json](../provenance/figures/source_spec.json) |
| Synthetic inputs and schemas | [data guide](../data/README.md) |
| Selected tests and disposable output | [tests guide](../tests/README.md), `.test-tmp/<run-id>/` |
| TeX, approved figures and readable PDFs | `manuscript_JAG/` |
| Public selection, hashes and release preparation | [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md) |

The selected algorithm files retain their original diagnostic-directory paths.
Existing imports, tests and provenance depend on those paths. Their dependencies
are included explicitly; other diagnostics and execution outputs remain local.
The figure helpers likewise retain their original `scripts/figures/`,
`analysis/workflows/` and diagnostic paths.

For maintenance, separate the current callable functions from historical replay
commands: the supplied-data runner calls `detector_engine.initialize/batch`,
`tda.aggregate_frames`, and `lr_cusum.star_segmentation.segment`. TDA currently
reuses `prepare_phases`, `temporal_edges` and `assemble_hflag` from the dated
helpers listed in the algorithm guide. Those files are live dependencies, not
disposable diagnostics. Keep their frozen source signatures intact when fixing
public input/output interfaces. Relocating them requires output-parity checks
and an explicit update of their callers and provenance.

Earlier W36+B20 scripts and configurations are identified as historical
compatibility interfaces. They do not substitute for the first-post detector or
TDA/STP. Earlier scoring summaries, experimental plans and review records remain
local; their presence in an author checkout does not make them current public
results.

The public export omits raw data, real row-level tables, exact geometry, private
experiment documents/results, PPTX sources and archives. Public figure source
mappings may describe private dependencies without distributing them. The main
and supplementary PDFs are readable directly; TeX compilation additionally
requires the official CAS template described in the release guide.

`configs/public_export.json` defines the public files. The default-deny
`.gitignore` is generated from that selection; ignore rules do not untrack
previously committed files. The fresh export contains no `.git` directory and
does not inherit the author repository's history. Preserve local scientific
records and compatibility paths when maintaining the author workspace.

Current experiment entry points and aggregate tables are indexed in [scripts/experiments](../scripts/experiments/README.md). Only selected summaries are public; per-stand outputs remain private.
