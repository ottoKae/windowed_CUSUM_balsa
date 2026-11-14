# Seven-figure manuscript provenance

The current `manuscript_JAG/main.tex` includes seven PDF figures, named
`fig01_...pdf` through `fig07_...pdf` in manuscript order. The supplementary
entry currently contains no graphics.

For links to final PDFs, Python renderers, editable layouts, source data and
commands, open [FIGURE_INDEX_ZH.md](../../docs/FIGURES.md).
The index preserves the actual source paths. Links to raw inputs, editable
layouts and execution receipts are usable in the author environment; those
materials are not included in the public snapshot.

The maintained source is [source_spec.json](source_spec.json). In the author
workspace, `python tooling/catalogs/build_figure_provenance.py` regenerates the
following records from the local evidence:

- [figure_bindings.csv](figure_bindings.csv): active TeX references, order, hashes and scripts.
- [figure_dependencies.csv](figure_dependencies.csv): scripts, inputs and receipts with availability and hashes.
- `figure_provenance.json`: author-local full evidence record, excluded from the public snapshot.
- `figure_code_inventory.csv`: author-local current dependencies and historical scripts, excluded from the public snapshot.
- `public_code_files.json`: author-local source-code inventory, excluded from the public snapshot; it does not authorize data redistribution.
- [FIGURE_BUILDERS.csv](../../manuscript_JAG/figure_sources/FIGURE_BUILDERS.csv): compact current mapping.

Paths are relative to the algorithm repository. Paths such as `../../BALSA/`
refer to shared-workspace dependencies. Directory entries are availability
checks, not recursive checksums.

All seven final PDF hashes match their approved run outputs. Older inputs needed
by current renderers remain in place, including two protected PDFs. Cleanup and
rename receipts are under workspace
`tooling/workspace-maintenance/receipts/balsa_figures_20261113/`.

Historical output metadata retains its original filenames. The current names
and source routes in this catalog supersede the earlier eight-figure list.
