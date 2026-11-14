# Current algorithm and code entry points

Run the whole current chain with `python scripts/run_current_pipeline.py --output
.test-tmp/current-demo`. The [input/output guide](REPRODUCIBILITY.md) covers
supplied pixels and report evaluation; the [experiment index](../scripts/experiments/README.md)
maps Tables 3–5 to the actual source files and public aggregate results.

The current analysis follows **monthly native-pixel Sentinel-1 series → W36
Windowed CUSUM with first-post dating → Temporal Detection Aggregation (TDA) →
conditional Spatial Timing Partitioning (STP)**. The earlier W36+B20 release is
retained as historical code and evidence. Sharing a 36-month window does not make
the two pipelines interchangeable.

The public source keeps the original implementation paths. Selected Python files
under `tooling/diagnostics/` are active algorithm dependencies; their location
does not mean that all diagnostics, private inputs or run outputs are public.
Install the package with the `io` and `validation` extras for the array/table
interfaces below. Reading or writing the optional spatial products also requires
the `geo` extras. Run file-based entries from the repository root.

## Pixel detection and its date contract

The current adapter is
[`detector_engine.py`](../tooling/diagnostics/balsa-national-firstpost-20261030/detector_engine.py).
It uses [`window_geometry.py`](../src/lr_cusum/window_geometry.py),
[`core.py`](../src/lr_cusum/core.py),
[`temporal_nulls.py`](../src/lr_cusum/temporal_nulls.py), and the detector settings
in [`firstpost_w36_d12_s12_b20_20261030_v2.json`](../configs/firstpost_w36_d12_s12_b20_20261030_v2.json).
The configuration retains its historical B20 identifier because it also records
the earlier aggregation stage. TDA and STP have their own implementations below.

The detector uses VV, a 36-month context, a central 12-month decision interval,
and a 12-month stride. A candidate is dated to the **first finite observation
after the first CUSUM maximum**. The maximum itself remains in the left segment;
the next finite observation starts the right segment. Missing calendar months
remain missing: this is not a literal one-month increment. A peak with no later
valid observation is rejected, and the corrected candidate date must pass the
decision and reporting gates. This statistical response date does not establish
the physical onset date of harvest.

The existing callable interface is (the public runner uses `configs/current_pipeline.json`):

- `initialize(values_path, config_path)` reads a NumPy `.npy` matrix and the
  first-post configuration. The frozen calendar is January 2016–June 2026,
  with one column per month and one row per native pixel; unavailable observations
  are `NaN`.
- `detect(values, pixel_track_id)` processes one row and returns accepted events,
  including the candidate date, index, context and decision intervals, and
  permutation-test fields. Call `initialize` first.
- `batch(rows)` accepts `(unit_id, pixel_track_id, series_index)` tuples and adds
  execution information, eligible-year masks and peak-to-response date gaps.

Initialization binds the first-post statistic into the geometry module within
the worker process. This is the existing process-local adapter used by the
production workflow; it should not be mixed with historical detector calls in
the same process when comparing date conventions.

`windowed-cusum paper` and [`scripts/reproduce.py`](../scripts/reproduce.py) use
the historical `paper.py` detector path. Their synthetic example and capsule
replay remain useful checks, but do not run the current first-post/TDA/STP chain.

## Temporal Detection Aggregation

The current entry is
[`tda.py`](../tooling/diagnostics/harvest-inventory-20261104/tda.py), specifically
`aggregate_frames(units, membership, values, execution, nodes, output)`.
It consumes accepted pixel detections; it does not rerun CUSUM.

| Input | Required meaning |
|---|---|
| `units` | Native pixels identified by `unit_id`, in the row order of `values` and `execution` |
| `membership` | Fixed pixel-to-stand membership, using `unit_id` and `sample_id` |
| `values` | The pixel-by-month matrix on the frozen 126-month calendar |
| `execution` | Per-pixel `eligible_year_mask` in the same order as `units` |
| `nodes` | Unique accepted `node_id`, stand/pixel IDs, `month_index`, `response_month`, and original `window_start`/`window_end` |
| `output` | Destination directory for the generated Parquet tables |

TDA prepares monthly activity and phases using
[`run_episode_experiment.py`](../tooling/diagnostics/balsa-harvest-episode-20261101/run_episode_experiment.py)
and [`episode_relations.py`](../tooling/diagnostics/balsa-harvest-episode-20261101/episode_relations.py).
It assembles reports through
[`baselines.py`](../tooling/diagnostics/balsa-sts-v4_1-20261102/baselines.py)
and [`episode_assembly.py`](../tooling/diagnostics/balsa-harvest-episode-20261101/episode_assembly.py).
The current call uses `G=1`, a 108-month span limit, and a descriptive flag for
spans over 12 months. Unknown observation gaps are distinguished from observed
months without candidates. Formal reports require distinct responding pixels
to meet the fixed stand denominator's 20% support threshold. Repeated responses
from a pixel remain distinct source nodes but do not inflate distinct-pixel
support. Representative dates use source-node month indices with half-month
rounding upward.

The return value includes monthly activity, phases, phase templates and their
members, temporal relations, report components, accepted reports, source-node
ownership, edge decisions and progress tables. The same tables are written to
`output`. Legacy method IDs such as `R2_TIME` remain in stored products to preserve
their links to frozen evidence.

The file's `run()`/command-line path additionally loads author-local national
inputs and historical tables for exact parity checks. Those private data are not
distributed. The supplied-table `aggregate_frames` interface can be exercised
without them, as demonstrated by
[`test_harvest_inventory_tda.py`](../tests/test_harvest_inventory_tda.py).

## Spatial Timing Partitioning

The numerical implementation is
[`star_segmentation.py`](../src/lr_cusum/star_segmentation.py).
Its retained module name and `STARS_V1` identifiers predate the current STP name.
`segment(reports, members, units, Config(...))` operates on supplied tables:

- `reports` supplies `component_id`, `sample_id`, `prediction_id` and the fixed
  `total_pixels` denominator.
- `members` supplies accepted `component_id`, `unit_id`, unique `node_id` and
  `month_index` records.
- `units` supplies `unit_id`, `track_key`, `native_row` and `native_col` so that
  adjacency is defined on the original grid.

The implementation tests spatial organization of response dates while preserving
pixel positions, adjacency and the multiset of dates. Frozen defaults use 999
permutations and seed `20261102`. A parent is partitioned only after the
admissibility, significance, zone-size and parent-support conditions pass.
Returned tables are `stands`, `zones`, `rounds`, `node_ownership` and `tests`.
The reported BH-adjusted values are descriptive; the retained decision rule uses
the original `p_value <= 0.05` and support gates. Spatial child reports must not
be counted directly as additional stand harvest cycles.

[`stp.py`](../tooling/diagnostics/harvest-inventory-20261104/stp.py) is the frozen
inventory adapter. Its `run_stp` reuses the original calibration, verifies the
overlapping cohort and advances the original random stream before processing
additional reports. Consequently it also needs private historical reports,
calibration tables and parity outputs. Calling the standalone `segment` with a
fresh seed is a supported application of the numerical method, but does not
reconstruct that historical cached-prefix random stream by itself.

[`octa.py`](../tooling/diagnostics/harvest-inventory-20261104/octa.py) and
[`cstr.py`](../tooling/diagnostics/harvest-inventory-20261104/cstr.py) retain the
older entry names. They execute the current source rather than maintaining a
second algorithm. The optional
[`products.py`](../tooling/diagnostics/harvest-inventory-20261104/products.py)
combines frozen TDA/STP tables with author-local inventory geometry to create
spatial products; it is not a new detector or a public data download command.

## Verification and figure boundaries

The existing synthetic tests exercise the actual temporal relation, assembly,
TDA and STP implementations. See [`tests/README.md`](../tests/README.md) for the
test command and disposable output location. The STP test named
`test_frozen_cohort_actual_artifact_preserves_original_full_chain` additionally
requires private frozen outputs; that check is separate from the synthetic tests.

The seven approved manuscript figure PDFs accompany the public TeX and readable
main/supplementary PDFs. Compiling TeX also requires the official CAS template
described in the [release guide](PUBLIC_RELEASE.md). Current renderer and input mappings are recorded in
[`source_spec.json`](../provenance/figures/source_spec.json). Rendering from
private frozen evidence is a separate operation from compiling TeX with the
approved PDFs. `../scripts/figures/build_all.py` (author-local)
is the historical eight-figure diagnostic builder and does not regenerate the
current seven-figure manuscript.
