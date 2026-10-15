# Real scientific reproducibility capsules

> Historical author-local workflow (2026-09-05). The commands and package counts
> below belong to the older B/C release, not the current W36+B20 public checkout.
> Their archived builders/replay scripts and rendered figures are not shipped in
> the current source-only export. For current public commands and availability,
> use [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md) and the root README. Do not run these
> historical commands as the current installation instructions.

Version: balsa_bc_20260905_v1. These are local release candidates, not a
published dataset. The owner authorized the scientific and pixel profiles.
Original code is MIT-licensed; the owner-authorized processed B/C data are
CC BY 4.0 (see LICENSE, DATA_LICENSE and CITATION.cff in the licensed delivery).
A persistent public identifier remains pending. No DOI is invented here.

The numerical baseline delivery is dist/balsa_bc_20260905_v1_release/.
The licensed delivery is prepared separately under dist/public_release_1.0.0_final/;
all scientific payload hashes must match the numerical baseline. Use the
licensed directory as --capsules, or unpack science.zip into science/ and
pixels.zip into pixels/ under your own common parent directory. The commands
below retain the baseline path for historical verification; substitute your
extracted parent path when running a published release.

## Two profiles, one frozen input version

**B — science:** 3,498 opaque polygon identities, 4,001 track-specific monthly
stand series, 1,093 historical reference events in 462 polygons, 57 evaluation
blocks and five outer folds; frozen candidate/event/score ledgers, out-of-fold
predictions, bootstrap replicates and cached official CCDC responses.
The national polygon-role partition is distinct from the reference evaluation
partition: do not substitute its folds for the 57-block evaluation protocol.

**C — pixels:** 176,907 unique pixel-track sequences, 176,935 polygon-pixel
memberships and the same 4,001 polygon tracks. Per-polygon local integer row
and column offsets retain pixel adjacency, but have no absolute origin or
geotransform. Both profiles use the original frozen pixel cache. Array order
is [sequence, polarization, month], polarization is VV then VH, in dB;
126 calendar positions cover January 2016 through June 2026.
The reporting period is January 2017 through December 2025.

Missing values remain NaN. Validity, observation counts and source-month flags
retain their original meaning: they are not interchangeable with a zero dB
value. No imputation or new masking is performed during packaging.

IDs are opaque substitutions. The private mapping is excluded. Track, tile and
block codes preserve membership, not geographic locations. Ordered block codes
retain the original seeded bootstrap ordering. Explicit per-band seeds retain
the original permutation draws after changing trajectory IDs.
This reduces direct location disclosure; it is not a claim that rich temporal
data and local shapes are impossible to reidentify.

## Excluded D layer

Exact polygon geometry, geographic/native coordinates, original catalog/Zarr,
RTC static rasters, administrative vectors, commercial optical imagery, basemap
tiles, credentials and the private ID lookup are NOT included.
Full national map reconstruction, source extraction and spatial partition
redesign therefore require separately authorized D inputs.
The repository's already rendered figures may be inspected, but B/C alone
do not reproduce their georeferenced basemaps.

## Local commands

From the repository root, in an environment with the project installed:

    python -m pip install -e ".[test]"
    python scripts/build_data_release.py real --destination dist/new_capsule_build
    python scripts/replay_capsule.py verify --capsules dist/balsa_bc_20260905_v1_release
    python scripts/replay_capsule.py parity --capsules dist/balsa_bc_20260905_v1_release
    python scripts/replay_capsule.py evaluate --capsules dist/balsa_bc_20260905_v1_release --output runs/new_evaluation
    python scripts/replay_capsule.py detect --capsules dist/balsa_bc_20260905_v1_release --output runs/new_detection_smoke --limit 4

The builder needs the author's ignored frozen source directories. A recipient
does NOT run that builder: unpack science.zip into science/ and pixels.zip into
pixels/ in the same parent, then use the replay commands on that parent.
Both archives have SHA-256 sidecars and per-file manifests.

Detection reuses the maintained N1 implementation; --limit is explicitly a
smoke test, not national reproduction. Use --limit 0 for all tracks.
Pixel-first detection and spatial fusion reuse the same implementation through
the pixel-first replay command; this is more expensive than Polygon-first.
Evaluation and bootstrap replay from frozen predictions is distinguished from
fresh signal-to-event recomputation in the receipts.

To recompute every national VV component and evaluate the new ledgers:

    python scripts/replay_capsule.py detect --capsules dist/balsa_bc_20260905_v1_release --output runs/fresh_components --limit 0 --workers 8 --components
    python scripts/replay_capsule.py evaluate --capsules dist/balsa_bc_20260905_v1_release --output runs/fresh_evaluation --candidate-source runs/fresh_components
    python scripts/replay_capsule.py pixel-first --capsules dist/balsa_bc_20260905_v1_release --output runs/fresh_pixels --limit 0 --workers 12
    python scripts/replay_capsule.py ccdc-cache --capsules dist/balsa_bc_20260905_v1_release --output runs/official_cache_replay

The additional nrt command recomputes the 27-configuration monthly numerical
benchmark and frozen spatial selection without a new cloud call.
The representations command replays the current seven-route 48/12 N1
evaluation and all 5,000 paired bootstrap replicates. Both accept --capsules
and a fresh --output like evaluate.

The last command parses official responses, restores batch-local indices, applies
confirmed-negative screening, track fusion and frozen held-out selections.
It does not run a surrogate CCDC implementation or contact Earth Engine.

For a new official cloud/grid run, install the cloud extra and provide your own
EE_PROJECT and explicit S1GRITS_ALLOW_CLOUD=1 environment settings. The maintained
run_cusum_nrt_ccdc_comparison.py accepts --science pointing to the B directory,
--run-dir pointing to a NEW output, and preflight/nrt/ccdc/evaluate stages.
The preflight downloads the pinned public author code and requires network access.
Never reuse the author's old run directory after configuration hashes change.
New cloud execution and full re-selection of all competing-method grids are
distinct from offline replay; neither is implied by a packaging success receipt.

## Version boundaries

The original frozen Polygon-first VV result contains 6,658 events. Its 3,509
adjacent intervals are detection intervals, not verified biological rotations.
The legacy Pixel-first route and later N1 route are different experiments.
The 48/12 controlled configuration and held-out-selected benchmark configuration
must not be interchanged. A later raw-data repair affects one track and is not
silently substituted into these packages.
Read VERSION_BOUNDARIES.json before comparing counts.

SOURCE_HASHES.json records input hashes and omitted table columns.
AUTHORIZATION.json records the authorized scope, not public release approval.
COMPLETE.json certifies packaging integrity, not that every expensive
experiment has been rerun. Consult the separately produced replay receipts.
