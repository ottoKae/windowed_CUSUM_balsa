# Owner-approved B/C real-data reproducibility release

Date: 2026-09-05. Status: local release candidates; not externally published.
The owner explicitly approved B/C preparation. The full spatial and raw-data
layer (D) is excluded. No Git commit, remote creation or push was performed.

## Deliverables

The final local directory is dist/balsa_bc_20260905_v1_release/.
Earlier directories with the same stem are development candidates; use the
directory ending in _release and its SHA-256 sidecars.

| Package | Contents | ZIP size |
|---|---|---:|
| science.zip | Real Polygon-first monthly inputs, references, frozen folds, event/candidate/score evidence, current seven-route N1 evidence, method OOF outputs, bootstrap replicates, official CCDC responses | 30,943,559 bytes |
| pixels.zip | Full polygon-interior monthly sequences, opaque memberships and local adjacency, validity and observation counts | 151,517,814 bytes |

ZIP SHA-256:

    science.zip  dc02c079c40a8177b98a82ed75c725332942d535ed492d6717bb78aa8eb6f240
    pixels.zip   c8d6fc6da6e1b42b2767750ce30ba3030274812b200a863c462670d38dc13ffd

The scientific payload manifest lists 1,043 files; the pixel manifest lists 87.
The manifests and completion markers are additional control files.
A separate small public aggregate/demo ZIP is also available in dist/.

## Data contract and privacy

- 3,498 validated BALSA polygon identities and 4,001 polygon-track sequences.
- 176,907 unique physical pixel-track sequences; 176,935 polygon memberships.
- 1,093 dated references in 462 polygons, 57 evaluation blocks and five folds.
- VV/VH monthly values, January 2016–June 2026 (126 positions), with events
  reported only for January 2017–December 2025 under the frozen protocol.
- 40,846,792 finite band values and 3,733,772 missing band values in C.
  Missing values were not filled. No finite zero-dB values occur in this
  particular frozen cache, and no finite two-band month has zero n_obs.
- IDs are replaced once and reused across tables. The private lookup remains
  under ignored local data. Original per-band permutation seeds are retained.
- Local integer pixel offsets preserve adjacency; absolute row/column origin,
  geometry, CRS/geotransform, source catalog/Zarr, static layers, administrative
  vectors, basemap tiles and original optical imagery are excluded.

Opaque IDs are not a formal anonymity guarantee: time-series signatures and
local shapes may still be linkable. The original deterministic seeds may also
be linkable given knowledge of the source-ID scheme. B/C reproduce the existing spatial grouping;
without D they cannot independently rebuild or geographically audit the original
guard distances or invent new spatial folds.

## Verification completed

| Check | Observed result |
|---|---|
| Unit tests | 86 passed; one non-fatal physical-core-detection warning |
| B/C aggregation parity | All 4,001 monthly Polygon-first medians and NaN masks exactly equal |
| Fresh national VV detection | All 4,001 tracks recomputed; every F4 track-event month matches frozen evidence; 6,658 fused events |
| Fresh F0–F4 → references → bootstrap | Recomputed candidate ledgers feed evaluation directly; events and all 5,000 paired replicate results match |
| Six-row NRT/CCDC benchmark replay | Point estimates and 5,000 paired replicate results match |
| Seven fixed 48/12 N1 representations | Event-level metrics and 5,000 paired replicate results match |
| Fresh monthly NRT | All 27 configurations recomputed; five-fold selections and all 3,704 OOF events match |
| Official CCDC response replay | Batch offsets restored; all 907 OOF event predictions match; no cloud request made |
| National history reconstruction | 6,658 events; 2,194 polygons with at least two events; 3,509 intervals; median 36 months |
| Pixel-first smoke test | 443 real pixels in four polygon tracks; N1 detection, local connected support, fusion and VV/VH annotation complete |
| Full Pixel-first N1 | All 176,907 pixel-track inputs and 4,001 polygon tracks complete; 6,197 VV events; all event months, track support and VH annotations match the existing N1 result |
| Independent filesystem copy | Both ZIPs extracted, checked and used for a real detection smoke test; 86 tests pass |
| Manuscript compilation in copies | Both trees compile; original files and figure layouts untouched |
| Git-visible text and file audit | Technical checks pass; no high-confidence secret hits or prohibited source data detected |
| Reachable Git history | Two commits / 132 unique blobs scanned; no high-confidence token/key hits |

The full pixel scan produced 479,593 pixel-level events: 256,120 VV and
223,473 VH. Connected support of at least 0.20 followed by multi-track fusion
produced 6,197 VV-primary polygon events, of which 3,583 have matched VH support.
These match the existing N1 result, including each event month and support
annotation. They do not replace the legacy L0 total of 6,241.

The _v1_final/_v1_rc3 QA packages and the final delivery share 175 byte-identical
Parquet/NPZ inputs. The final package adds the current representation evidence
and source documentation; it does not replace the already tested signals.
Execution receipts retain their actual input-manifest hashes rather than being
relabeled as runs against a later package.

## Fresh nationwide component counts

These are national detected-event counts, not reference matching scores:

| Stage | Fused national VV events |
|---|---:|
| F0 | 3,105 |
| F1 | 17,078 |
| F2 | 10,488 |
| F3 | 10,363 |
| F4 | 6,658 |

## Implementation changes

The maintained detector now accepts an optional explicit original seed so
opaque IDs do not alter permutation draws. Defaults and numerical detection
rules are unchanged. Pixel event records no longer require global coordinates
when the association table already supplies the local adjacency information.

The new capsule builder uses explicit column/array contracts, private persisted
ID mappings, per-file SHA-256 manifests, immutable destinations and ZIP integrity
checks. The public aggregate exporter uses an explicit file allowlist rather
than recursively copying arbitrary results.

Git audit and manifest generation now share Git's actual inclusion rules.
Chinese text is not classified as private solely because it is Chinese.
Personally configured Earth Engine projects are removed from public defaults;
new cloud processing requires the runner's own project and explicit opt-in.
The Python 3.11-only digest helper was replaced by a compatible streaming hash.

## Reproduction entry points

See docs/REAL_CAPSULES.md and scripts/replay_capsule.py.
The available commands are verify, parity, detect, pixel-first, evaluate,
representations, nrt and ccdc-cache. Fresh detection defaults to a four-track
smoke test; pass --limit 0 explicitly for national processing.

Main local evidence directories:

    runs/capsule_full_vv_components_v1/
    runs/capsule_final_component_evaluation_v1/
    runs/capsule_representations_replay_v1/
    runs/capsule_nrt_fresh_v1/
    runs/capsule_ccdc_cache_v2/
    runs/capsule_pixel_full_v1/
    runs/capsule_release_input_equivalence/

## Remaining release decisions and limits

The code and data licenses, finalized citation metadata and actual external
deposit remain pending. No DOI or download URL was invented. Rendered
third-party image content needs its own release record; this does not revoke
the owner's B/C data-preparation authorization.

The tested runtime is Windows / Python 3.12.13. The independent-copy test uses
the same installed dependencies, not a new isolated dependency environment.
The minimum Python-version matrix and the new CI workflow have not been run
on GitHub.

Both manuscript copies compile. The earlier manuscript still reports 12
duplicate PDF destinations; the JAG v2 copy reports none. Both have one
non-fatal overfull box. This task did not change layout or prose to hide them.
An additional pass left both AUX files byte-identical, confirming stable
cross-reference outputs despite the class's repeated rerun-label warning.

No new official cloud CCDC call was performed. B/C do not reconstruct the
original georeferenced map figures, source-Zarr extraction, or a redesigned
geographic validation split. Those require separately authorized D inputs.
Thus the supported claim is real time-series-to-event/statistics reproduction,
not unrestricted raw-archive-to-every-map reproduction.
