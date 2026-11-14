# Publication boundary

`configs/public_export.json` is the explicit file selection. `.gitignore` is
regenerated from that selection with `python tooling/release/update_public_ignore.py`.
Unlisted files stay local. Do not force-add private files or upload the whole
research directory as a ZIP.

Excluded from GitHub: raw imagery, individual trajectories/references, source
geometry, interpretation notes, private experiment receipts, archives, PPTX,
credentials, environment files, caches and local service configuration.
The reviewed numerical dataset, references and stand boundaries are available
separately on [Zenodo](https://doi.org/10.5281/zenodo.24223182) under CC BY-NC-SA 4.0;
Git exclusion does not make those published records private.
Included: author-approved paper content and final figures, code, invented examples,
schemas and checked aggregate summaries. The paper itself includes author details
and illustrated case locations; excluding raw data does not conceal information
already visible in an approved figure.

Ignore rules do not remove tracked files or old commits. Publish a fresh export
without the research repository's `.git` directory. Retain original history locally.
Do not switch the old research repository to public as a substitute for exporting.

Before releasing a changed version, rebuild the PDFs, inspect their content and
metadata/attachments, and refresh the explicitly approved hashes only after review.
Run export verification and test a clean checkout. Verification rejects
Git-tracked files outside `PUBLIC_FILES.json`; untracked local test outputs are
not part of that publication check. The scanner reports file names and categories
without printing matched secrets. Its pattern checks are not a
proof that every possible private value has been recognized.

Remote repository visibility, past releases, Actions artifacts and previously
shared attachments require their own check. A clean new snapshot does not retract
old public copies. Revoke any credential that was previously exposed.
