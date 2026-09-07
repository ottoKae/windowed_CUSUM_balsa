# Figure source index

This directory records the authoritative source route for every figure in the
final JAG manuscript. The PDF files consumed by LaTeX remain in `../figures/`.
Python renderers are kept under `../../analysis/workflows/` and thin entry
points belong under `../../scripts/figures/`.

The manuscript filenames preserve an earlier numbering scheme. Printed figure
order is defined by `FIGURE_BUILDERS.csv`, not by the numeric token embedded in
the PDF filename.

Two reproducibility boundaries are intentional:

- public source tables may be committed under `../../data/public/`;
- restricted geometries, S1-GRiTS extracts, PlanetScope imagery, and Esri tile
  caches stay under ignored `../../data/local/paper_v1/` or are supplied by an
  authorized user.

The workflow diagram is the only manually authored figure. Place its editable
source at `powerpoint/windowed-CUSUM-workflow-figures-01.pptx`. Commit that file only after
checking ownership of every embedded asset and removing document metadata.
The author has confirmed redistribution permission, and the editable PPTX is
included with the frozen PDF used by the manuscript.

Run the release audit before publication:

```text
python provenance/audit_public_package.py
```
