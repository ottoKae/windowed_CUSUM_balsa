---
name: latex-bibtex-vscode
description: Diagnose undefined citations and stale PDFs in multi-file LaTeX manuscripts; repair BibTeX and VS Code LaTeX Workshop builds, especially Windows pdfLaTeX/natbib projects.
---

# Reliable LaTeX and BibTeX builds

Resolve the active manuscript before editing or rebuilding. Follow the chapter's
`% !TeX root` and the root's `\input`, `\include`, `\bibliography` and
`\bibliographystyle`. Check the VS Code workspace root and output directory.
Different manuscript versions commonly have different files named references.bib
and main.pdf. Never assume that an entry in one version exists in another.

## Diagnosis

- Read the current `.blg` and final `.log`. BibTeX's `I didn't find a database
  entry` means an entry is missing from the database actually loaded. Repeated
  builds cannot fix that. Locate the intended existing entry, compare title/DOI,
  and restore it only to the active bibliography. Do not invent citations or
  silently add placeholder entries.
- Check duplicate keys, exact key case, braces, and input encoding. Classic
  BibTeX can truncate UTF-8 accented initials. Use TeX accent forms when repairing
  this pdfLaTeX pipeline, e.g. `Cañadas` becomes `Ca{\~n}adas` and `Álvaro`
  becomes `{\'A}lvaro`. Do not blindly convert unsupported Unicode to `?`.
  Preserve the author's name and bibliography fields. Biber/XeLaTeX migrations
  require checking the document class and style; do not migrate just to suppress
  a warning.
- A corrupt stale `.bbl` can kill the first LaTeX pass before BibTeX executes.
  Invalidate the active root's generated bibliography or use the checked clean
  build. Never delete source `.tex`, `.bib`, figures or unrelated build trees.
- Wrapped log lines can hide citation names. Inspect `.blg` and join physical
  log lines when checking final undefined-reference warnings. A PDF existing on
  disk does not prove the latest build succeeded.

## Build and VS Code

Prefer the existing project build entry. For the repository containing this
skill, use `manuscript_JAG_v2/build_latex.ps1`:

```powershell
./build_latex.ps1 -Fast
./build_latex.ps1 -Clean
```

The fast route uses latexmk; the clean route runs pdfLaTeX, BibTeX, pdfLaTeX,
pdfLaTeX. Both check the final log. Run from the intended manuscript, or invoke
the script by absolute path. Use process exit codes for native programs;
PowerShell stderr output alone does not establish failure.

Keep recipe output and PDF viewer output identical. This project outputs next
to `main.tex`, using `%DIR%`, not a second `build/main.pdf`. Preserve SyncTeX.
Use `onFileChange` for changes to bibliography dependencies and avoid automatic
cleanup on failure, which destroys useful logs. Keep a clean rebuild recipe as
a recovery option; do not clean on every save.

Open `JAG-manuscript.code-workspace` at repository root. Folder settings in
`manuscript_JAG_v2/.vscode` apply when that directory is a workspace folder;
they do not apply merely because an ancestor folder is open. Do not replace
global settings affecting unrelated projects. If adapting another project,
inspect its engines and bibliography backend before copying this configuration.

## Verification and handoff

Run one clean build and an unchanged incremental build. For substantive build
script changes, test a newly added citation, a missing key and stale bibliography
recovery in a temporary fixture, not the manuscript. Require zero undefined
citations/references in the final log, current PDF output, and expected BibTeX
entries. Explain layout warnings separately. Do not fix them by hiding diagnostics.

Report the confirmed cause, active bibliography, exact PDF path, build commands,
and tested limits. Do not promise compilation of arbitrary malformed future input.
See the tested local instructions in
`manuscript_JAG_v2/BUILDING.md` (relative to repository root).
