# Public release 1.0.0

## Scope and licenses

The rights holder approved public release of original software under MIT and
processed B/C data under CC BY 4.0 on 2026-09-05. Full spatial/raw D inputs remain
excluded. The software distribution also omits both local manuscript trees and
rendered third-party figure assets; local manuscripts and layouts are unchanged.
There is no claim that a license file or an upload plan constitutes deposition.

The candidate is staged at `dist/public_release_1.0.0_final/`:

- `software/` and `windowed_cusum_balsa-1.0.0-source.zip`: source-only distribution;
- `science.zip` and `pixels.zip`: licensed copies of the frozen real-data packages;
- `*.sha256`, `SOURCE_MANIFEST.json` and `PREPARATION.json`: integrity records.

Only explicitly allowlisted result tables enter the software package. The
author's whole working Git tree is **not** the upload payload. A later-added
real tile/track inventory is outside that allowlist. Do not run `git add .`
in the author repository as a substitute for checking the staged distribution.

## Independent-environment check

A new Python 3.12.13 venv was created with system site packages disabled.
Dependencies were downloaded afresh using the recorded constraints; the source
ZIP and both data ZIPs were extracted into a separate recipient directory.
The software was installed non-editably from that extracted source tree.

The initial run exposed a missing test dependency (`rasterio`): 85 passed,
1 failed. The `test` extra now declares it. After the fix, 87 tests passed,
including the new license/citation integrity check. Two affine deprecation
warnings are non-fatal; no detector equation or threshold was changed.

Verified from the extracted package in the new environment:

- synthetic CLI run: four accepted window candidates;
- pixel-to-stand parity: all 4,001 series and NaN masks exactly equal;
- fresh four-track N1 detection: frozen event identities reproduced;
- six-method OOF and F0--F4 evaluation: point estimates and 5,000 paired
  bootstrap replicates reproduced from frozen event ledgers;
- official CCDC cached-response replay: 907 predictions reproduced, no cloud call;
- workflow-figure builder: runs without private data.

This check does not claim a new full 176,907-pixel detection run or fresh official
CCDC execution. Full fresh national/pixel replay was verified earlier on the same
scientific inputs; see the B/C report. Linux CI and Python 3.10 remain unverified
until their jobs actually run. Full georeferenced maps require excluded D inputs.

## Publication sequence

1. Confirm software/data author order (currently transcribed from the manuscript)
   and the intended remote repository. The proposed name is
   `ottoKae/windowed_CUSUM_balsa`; it is not an assertion that the remote exists.
2. Review the source manifest and current working-tree audit. Only publish the
   staged source subset, not local data, manuscripts or unreviewed Git history.
3. Run `/ponytail-review` and correctness/privacy checks before commit/push.
4. Create the authorized GitHub repository and push the reviewed source snapshot.
   Run CI and publish a versioned release only after the checks pass.
5. Deposit both data ZIPs and checksum files together in one Zenodo dataset
   record; archive the software release separately. Retain third-party notices.
6. Add the actual repository URL, version-specific software DOI and data DOI to
   citation metadata and cross-link the records. Never insert example DOIs.
7. Download each public archive without author credentials, verify hashes and
   repeat the smoke commands before claiming public availability in the paper.

## Remaining external dependencies

No remote is configured in the local author repository. A GitHub connection
identifies the account, but repository-creation/write access must still be
established. No Zenodo publishing credential/session has been verified. Do not
paste access tokens into tracked files or chat; authenticate through the normal
local CLI/application route, or supply the created repository and deposition
record URLs. The author/citation question is still awaiting confirmation.

`provenance/PUBLICATION_STATUS.json` deliberately retains null identifiers and
`externally_published: false` until successful external operations are verified.
