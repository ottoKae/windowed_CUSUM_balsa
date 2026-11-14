# Public-release policy

Current scope: 2026-11-13. The owner-authorized snapshot includes the current
first-post W36 → TDA → STP implementation, selected dependencies and tests,
synthetic examples and schemas, figure-rendering code, and the unchanged author
TeX, bibliography, main/supplementary PDFs and seven approved figure PDFs.
This file supersedes the earlier source-only manuscript selection.

The exact file selection is in `configs/public_export.json`. Approved PDF paths
and SHA-256 hashes are recorded there; no other PDF or binary is admitted by
extension alone. The exported `PUBLIC_FILES.json` records the distributed bytes.
Preserving the original Python paths supports existing imports and provenance.

The following remain local:

- raw Sentinel-1/S1-GRiTS stores, optical source imagery and basemap caches;
- real pixel series, exact inventory geometry, row-level research tables and
  private interpretation/review records;
- private experiment documents, run receipts and result directories;
- editable PPTX sources, archives and the partial local CAS template bundle.

Providing the approved rendered figures does not authorize redistribution of
their underlying imagery or data. Figure source specifications describe private
dependencies so readers can trace construction without receiving those inputs.
Synthetic `DEMO_STAND_*` identifiers refer to invented sequences. No real
row-level data package or download is part of this snapshot.

Original code uses [MIT](../LICENSE). [DATA_LICENSE](../DATA_LICENSE) applies to
the material it covers and does not grant rights over private source data,
third-party imagery, basemaps, papers or templates. Retain third-party notices.
The official CAS template must be obtained separately under its own terms; see
the [release guide](../docs/PUBLIC_RELEASE.md).

The default-deny `.gitignore` is generated from the allowlist; it does not remove
previously tracked files. Publication preparation uses a fresh export without
`.git`, so the author repository's historical objects and their metadata do not
enter the candidate. Existing history is retained locally. Exporting does not
push, change repository visibility, rewrite history or deposit research data.
