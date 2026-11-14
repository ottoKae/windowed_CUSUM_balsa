# V1.0.0 data dictionary

All tables use UTF-8. IDs are strings, never spreadsheet row numbers. `schema_inventory.csv` lists all actual columns and storage types, including frozen intermediate fields.

| File or field | Meaning |
|---|---|
| `inventory/stands.csv: sample_id` | Stable study stand identifier shared by all tables |
| `fc001_objectid` | Official source feature ID; distinct from study row numbers embedded in sample IDs |
| `species`, `validation_status` | Retained BALSA / VALIDADO attributes |
| `area_ha` | Inventory polygon area in hectares |
| `native_cells` | Fixed whole native cells assigned by centre membership |
| `in_inference`, `in_dated_subset` | Membership of the supported and dated subsets |
| `native_support_status` | `supported` or `no_native_cell_centres` |
| `inventory/stands.gpkg` | Fixed study geometries, not a newly edited cadastral product; CRS is embedded in GeoPackage |
| `references: event_id` | Unique reference record ID |
| `reference_month` | YYYY-MM reference date, preserved as recorded |
| `timing_type`, `date_basis` | Original timing classification and observation meaning |
| `interval_lower`, `interval_upper`, `bound_semantics` | Original reference bounds and their meaning; blank if absent; ±1-month evaluation tolerance is not inserted as a measured interval |
| `units: unit_id` | Native observation-unit ID used in source ledgers |
| `pixel_track_id` | Original pixel/track identity, also the detector seed input |
| `track_key`, `tile_id`, `grid_id` | Original archive group and spatial grid IDs; not interchangeable with satellite pass direction |
| `native_row`, `native_col` | Zero-based row/column within that native grid |
| `x`, `y` | Projected pixel-centre coordinates in the corresponding grid CRS |
| `lon`, `lat` | Geographic pixel-centre coordinates in degrees |
| `grids.json` | CRS, affine transform and monthly archive-presence mask for each grid/track |
| `membership: source_row_original` | Frozen inventory row identity for case traceability; not an owner ID |
| `calendar: column_index`, `month` | Zero-based array column and its YYYY-MM date |
| `monthly_vv.npy`, `monthly_vh.npy` | Float64 pixel-by-month dB matrices; invalid observations stored as NaN |
| `valid_mask.npy` | Boolean frozen joint VV/VH validity mask |
| `observation_count.npy` | UInt16 original cache n_obs counts; not a guarantee that a value passed validity checks; use with valid_mask |
| `pixel_detections: change_index`, `change_time` | Zero-based month index from 2016-01 and assigned candidate date |
| `peak_marker_month` | CuSum maximum month; may precede the assigned candidate date by more than one calendar month if observations are missing |
| `cusum_amplitude`, `null_p_value` | CuSum range Q and its window-specific permutation score |
| `pixel_execution: eligible_year_mask` | Nine-bit mask, bit zero for 2017; indicates available detection opportunity |
| `pixel_execution: series_index` | Row of the original national array; join subset outputs by unit_id, not this archived index |
| `TDA: component_id`, `prediction_id` | Stable parent/report IDs |
| `first_response_month`, `last_response_month` | Earliest/latest contributing detection month |
| `response_months` | JSON list of actual contributing months; gaps inside the envelope are not extra detections |
| `representative_month` | Rounded median of all contributing pixel–month dates; half-month ties round later |
| `response_pixels`, `total_pixels`, `support_fraction` | Distinct contributing pixels, all fixed stand members and their ratio |
| `span_months` | Inclusive first–last span; not verified harvest duration |
| `source_node_ids`, `source_unit_ids` | JSON lists of source detections and distinct pixels |
| `left_censored`, `right_censored`, `history_has_unknown` | Boundary and observation-gap flags retained from aggregation |
| `legacy_*`, `method=R2_TIME` | Frozen historical identifiers; current display name is TDA |
| `STP: zone_id` | Initial spatial zone |
| `round_id`, `parent_prediction_id` | Final timing group and its TDA parent; final groups need not be spatially connected |
| `STP: p_value`, `q_value` | Spatial permutation p-value used for the split gate; descriptive family-specific BH q-value |
| `products/adjacent_start_intervals.csv` | Consecutive TDA parents in each stand, ordered by first detection; interval is next first month minus previous first month |
| `stand_histories: tda_report_count` | Number of TDA parents; missing for unsupported stands, zero for supported stands without reports |
| `first_month_interval_median` | Median adjacent-start interval per stand, undefined with fewer than two reports |
| `representative_month_interval_median` | Separate archived median interval based on representative dates |
| `reproduction/bootstrap_draws.npy` | 5,000 × 459 integer indices into `cohort_order.csv`, sampled with replacement |

All reference-subset arrays retain the relative national row order. National and reference products retain the same IDs. Report counts are algorithm outputs; unmatched reports are not automatically field-confirmed false harvests.
