# Contributing

Use Python 3.12 and the pinned constraints in `configs/requirements-reproduce.txt`.
Install normally and run the commands in [tests/README.md](tests/README.md).
Use invented input examples when reporting a bug. Keep raw imagery, exact stand
geometry, private review notes and credentials out of issues and pull requests.

The W36 detector configuration, final score snapshot and sampling design are
versioned scientific records. A change to the algorithm or reference evidence
requires a new result version; do not overwrite the frozen evidence to make a
test pass. Manuscript prose and figures remain author-controlled.

Code belongs in `src/windowed_cusum/`, entry points in `scripts/`, reusable checks in
`tests/`, and disposable output in `.test-tmp/`. Do not run author-side release or
sample-collection assembly tools unless their original local inputs are available.
Readers can verify an existing public snapshot without those inputs:

```sh
python tooling/release/export_public.py --verify .
```

`PUBLIC_FILES.json` records exact exported bytes. `.gitattributes` preserves these
bytes across Git checkouts. When changing a public file, update its SHA-256 entry
in the manifest after reviewing the change; add or remove entries with the files.
Do not disable the check or treat a matching hash as a license clearance.

Original code is MIT; study aggregates have the separate terms in `DATA_LICENSE`.
Third-party code retains its own notices. Cite the software using `CITATION.cff`;
do not invent a paper or dataset DOI.
