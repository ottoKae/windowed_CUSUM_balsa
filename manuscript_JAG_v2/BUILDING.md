# Compile this manuscript in VS Code

Open `../JAG-manuscript.code-workspace` with **File > Open Workspace from File**.
Alternatively, open this `manuscript_JAG_v2` directory itself as the folder.
Opening an ancestor directory does not activate the nested `.vscode` settings.
LaTeX Workshop is already installed on the current workstation.

1. Add BibTeX entries to **this directory's `references.bib`** and save it.
2. Use the exact entry key in `\citep{...}` or `\citet{...}` and save the chapter.
3. With `main.tex` or a chapter active, press **Ctrl+Alt+B**. The checked
   incremental recipe runs latexmk, including BibTeX when required. Dependency
   changes also trigger automatic builds once LaTeX Workshop has found the root.
4. Press **Ctrl+Alt+V** to view the resulting `main.pdf` beside `main.tex`.

**Ctrl+Shift+B** invokes the same incremental script as a VS Code build task.
For recovery, use **LaTeX Workshop: Build with recipe > Checked clean rebuild**,
or **Tasks: Run Task > JAG: checked clean rebuild**.

In PowerShell, from this directory:

```powershell
.\build_latex.ps1 -Fast     # Incremental; unchanged builds skip TeX passes.
.\build_latex.ps1           # Explicit PDFLaTeX -> BibTeX -> PDFLaTeX twice.
.\build_latex.ps1 -Clean    # Remove this root's generated auxiliaries and rebuild.
```

For a terminal in another directory, invoke the full script path with `&`.
The script sets its own working directory. TeX Live tools must be on PATH.

## What was fixed on 2026-09-05

Six cited keys were in `../manuscript/references.bib`, but missing from the
active `manuscript_JAG_v2/references.bib`: `Doblas2020harmonic`,
`Matthew2021BFAST`, `BorlafMena2023Seasonality`, `Zhao2021SARExclusion`,
`Huang2026EucalyptusAge`, and `McMahon2019EucalyptusRotations`.
They were restored from the existing local bibliography. Re-running BibTeX
without restoring these entries could not resolve them.

The workstation's global LaTeX Workshop settings used `build/main.pdf`, whereas
the manual script wrote `main.pdf` beside the root source. This workspace now
uses a single output directory and the same checked script for both routes.
The new preflight reports missing keys and the citing file before invoking TeX.
Final log checks also handle line-wrapped citation warnings.

## When a build reports a problem

- **Missing citation keys:** add the real entry to the active `references.bib`.
  Check exact spelling and case. Entries in another manuscript are not shared.
- **Duplicate key:** retain the intended entry after comparing its metadata.
- **Non-ASCII bibliography:** this classic BibTeX workflow uses TeX accents
  such as `Ca{\~n}adas` and `{\'A}lvaro`. Preserve the real name; do not replace
  accented letters with question marks. The script reports affected line numbers.
- **Invalid UTF-8 in main.bbl:** correct the source entry, then clean rebuild.
- **Old PDF or old Problems entries:** confirm the active root and output path,
  close the old PDF tab and reopen with LaTeX Workshop. Reload Window only if
  the extension still shows stale diagnostics after a successful final log.
- **Overfull boxes:** layout warnings; they do not mean BibTeX failed.

The `.log` and `.blg` remain available after failures. Source files and figures
are not deleted by clean builds. A successful build does not certify the
scientific appropriateness of a citation.

Configuration follows the [LaTeX Workshop compile documentation](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile).

## Verified on 2026-09-05

The real manuscript passed a clean four-pass build and an incremental build:
zero undefined citations/references, zero duplicate PDF destinations, one
pre-existing title-layout overfull box. An unchanged incremental check took
approximately 0.65 seconds on this workstation; changed builds take longer.

The release audit plus a clean TeX build cover newly added BibTeX entries,
missing-key diagnostics, duplicate-key diagnostics, and recovery from a stale
`.bbl`. VS Code UI interaction still requires opening the supplied workspace.
