# Public data dictionary

## `demo/monthly_s1_synthetic.csv`

| Field | Type | Meaning |
|---|---|---|
| `ID` | string | Synthetic stand identifier (`DEMO_STAND_*`) |
| `time` | ISO date | First day of the calendar month |
| `VV_dB` | number or blank | Synthetic monthly VV backscatter in dB |
| `VH_dB` | number or blank | Synthetic monthly VH backscatter in dB |

## `demo/reference_events_synthetic.csv`

| Field | Type | Meaning |
|---|---|---|
| `ID` | string | Identifier matching the synthetic time series |
| `event_month` | ISO date | Invented reference month |
| `reference_type` | string | Always `synthetic` |

## Aggregate result tables

The files in `../results/` contain only cohort-level counts, metrics,
confidence intervals, or histogram bins. They contain no polygon identifiers,
coordinates, administrative labels, spatial-block identifiers, source-row
numbers, reviewer names, evidence filenames, or local filesystem paths.

Important units:

- event counts are counts of qualified detector rows for the stated route;
- `objects` is the number of evaluated reference polygons/stands;
- temporal errors and interval statistics are in calendar months;
- confidence intervals use resampling of complete spatial blocks where stated;
- percentages in histogram tables are on a 0--100 scale, while metric fractions
  such as precision and recall are on a 0--1 scale.

