# JAG working manuscript

This is the sole working manuscript, moved from `manuscript_JAG_v2` on
2026-11-13 before the language rewrite. The scientific TeX, bibliography,
document styles and seven final figure PDFs were preserved unchanged.

- [main.tex](main.tex): manuscript entry, title, abstract and highlights.
- [sections/](sections/): editable manuscript sections.
- [references.bib](references.bib): current bibliography.
- [main.pdf](main.pdf): compiled main manuscript.
- [supplementary.tex](supplementary.tex) and [supplementary.pdf](supplementary.pdf): supplementary manuscript.
- [figures/](figures/): the seven final PDFs, named `fig01` through `fig07`.
- [Figure source index](../docs/FIGURES.md): plotting scripts, frozen data, editable artwork and rebuild commands.

Open [JAG-manuscript.code-workspace](../JAG-manuscript.code-workspace) to work
in VS Code. See [BUILDING.md](BUILDING.md) for the existing compilation workflow.
From this directory, run `./build_latex.ps1 -Fast`; the resulting `main.pdf`
stays beside `main.tex`. The next build recreates removed compiler intermediates.
The current `main.bbl` is retained for submission preparation.

The author-local `archive/` contains the frozen pre-rewrite manuscript and
historical audit records. It is excluded from the public repository, as are
editable PowerPoint files and original imagery. The public selection contains
the TeX, bibliography, compiled manuscript, supplement and seven final figures.

The partial local CAS style bundle is not redistributed. To compile on another
machine, obtain the official CAS templates as described in
[the public release guide](../docs/PUBLIC_RELEASE.md).

Research data and figure sources remain in their existing project locations.
Generated figure indexes are rebuilt from
[source_spec.json](../provenance/figures/source_spec.json); edit that source
instead of editing the generated index. Migration and cleanup receipts are in
workspace `tooling/workspace-maintenance/receipts/manuscript_reorganization_20261113/`.
