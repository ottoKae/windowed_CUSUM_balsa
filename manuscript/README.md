# Manuscript source

This directory contains the clean, portable LaTeX source for the Windowed
CUSUM balsa manuscript. It contains only the active manuscript sections, the
eight figures referenced by `main.tex`, the cited bibliography entries, and
the Elsevier CAS files needed to reproduce the current layout.

## Build

Install TeX Live or MiKTeX with the packages required by the Elsevier CAS
single-column template. From this directory, run:

```powershell
.\build_latex.ps1 -Clean
```

The script runs pdfLaTeX, BibTeX, and two final pdfLaTeX passes. A successful
build creates `main.pdf` and fails if citations or cross-references remain
undefined.

## Source layout

- `main.tex`: document metadata, abstract, highlights, and section includes.
- `sections/`: the seven active manuscript sections.
- `figures/`: exactly the eight PDFs referenced by the manuscript.
- `references.bib`: only bibliography entries cited by the active manuscript.
- `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst`: Elsevier CAS 2.4
  template files distributed under LPPL 1.3c or later.
- `thumbnails/cas-email.jpeg`: CAS asset required by the author email command.
- `figures/FIGURE_MANIFEST.md` and `figures/SHA256SUMS.txt`: figure provenance
  and integrity records.

All eight active figures have one canonical script in `scripts/figures/` and
are orchestrated by `scripts/figures/build_all.py`. Rebuilds go to `runs/` by
default; publishing over the frozen assets requires the explicit `--publish`
flag. Figure 6 also requires the ignored licensed-source configuration described
in `configs/README.md`.

## Release checks still requiring an author decision

1. Confirm the corresponding-author marks and email assignments in `main.tex`.
2. Replace `OWNER` in the repository URL and insert an archival DOI, if
   available, in `sections/07_declarations.tex`.
3. Confirm permission and attribution for the derived figure containing
   PlanetScope imagery and for the Esri World Hillshade background. No raw
   third-party imagery is included in this manuscript package.

Generated LaTeX auxiliaries (`*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.log`,
`*.out`, `*.synctex.gz`, and `*.abs`) are build products and should not be
committed. The compiled `main.pdf` may be attached to a tagged release after
the author checks above are closed.
