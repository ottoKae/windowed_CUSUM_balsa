# Current manuscript figure sources

The main manuscript uses **seven approved figures**, numbered in reading order.

Start with [FIGURE_INDEX_ZH.md](../../docs/FIGURES.md). Each figure has direct links
to its final PDF, actual Python renderer, required helper scripts, frozen source
tables or imagery, editable layout sources, and generation records. The index
also provides the appropriate command and environment for each figure.

- [FIGURE_BUILDERS.csv](FIGURE_BUILDERS.csv): compact seven-figure mapping.
- [Source specification](../../provenance/figures/source_spec.json): maintained catalog input.
- [Dependency inventory](../../provenance/figures/figure_dependencies.csv): availability and hashes.
- [Final PDF directory](../figures): only the seven numbered manuscript figures.

Regenerate the indexes from the algorithm repository root:

```powershell
python -X utf8 tooling/catalogs/build_figure_provenance.py
```

All seven installed PDFs are bound by SHA-256 to their approved run outputs.
This verifies their identity; it does not claim every renderer was rerun during
organization. Figure 1 uses the existing `infer` environment; the other entries
use `py312_torch`. Current renderers reuse frozen data without rerunning CuSum/TDA/STP.

Figure 1 and Figure 4 read older PDFs as artwork or layout input. Those two
files are explicitly marked in the index and retained during cleanup. Earlier
version directories also contain current input tables and imagery; only the
reviewed intermediate PDFs are removed, not their directories or source data.

`scripts/figures/build_all.py` and the old PowerPoint README describe an earlier
eight-figure manuscript. Use this index for the current build routes. Licensed
imagery and restricted inventory data retain their existing access conditions.
