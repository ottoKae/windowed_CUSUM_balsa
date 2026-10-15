# Interpretation sample collection

The collection consolidates existing evidence into one searchable local package.
It does not combine different sampling designs into one evaluation population.
No source workbook, reference event, detector output or score is changed.

## Entry points

- Public counts: [sample_collection_summary.json](../public/sample_collection_summary.json).
- Private package: `data/private/sample_collection_20261015_v1/`.
- Builder and search: [build_sample_catalog.py](../../tooling/catalogs/build_sample_catalog.py).
- Earlier inventory registry: `data/balsa_FC001_dated_event_subset/CURRENT.json`.
  This remains the original inventory/version authority; its current snapshot and
  live legacy review ledgers are included in the new collection.

The private package contains `samples.sqlite`, `source_catalog.csv`, `gaps.csv`,
`receipt.json` and byte-identical source copies in `objects/`. All these files must
remain ignored. The public schema and aggregate counts contain no exact locations,
stand identifiers, raw interpretation notes or workbook answers.

## Which records mean what

| Collection | Records | Interpretation |
|---|---:|---|
| Frozen W36+B20 report frame | 1,266 | Detection-report population; retains all fixed spatial and temporal support fields |
| Selected probability reports | 307 | Subset of the frame; original inclusion probabilities and weights are preserved |
| Current verified precision events | 300 | Event identities used for report verification, with current adjudicated timing |
| Current report-to-event links | 363 | Identity and spatial correspondence evidence; not additional events |
| Current report review states | 717 | Includes context reports; only selected probability reports enter the estimator |
| Original identity groups | 40 | Completed relationship decisions; separate from the later followups |
| Relationship followups | 28 | Additional interpretation operations, not 28 new independent samples |
| Repeat interpretations | 31 | Repeat readings with explicit event-alignment status; not independent truth |
| Continuous-history domains | 7 | Reviewed domains; unclear periods remain unclear |
| History observations | 11 | Observed harvest clues; not automatically 11 new independent events |
| Current pixel+B20 benchmark queue | 303 | Deferred probability queue, not completed validation |
| Historical native-method queue | 340 | Different method population; cannot substitute for the current queue |
| Selective catalogue source | 1,081 | All source records, including noneligible/unknown/interval records; **not** the 1,049 eligible monthly recall denominator |
| Historical references | 1,061 and 1,093 | Separate releases retained for provenance, not added together |
| Historical prospective reviews / candidates | 95 / 140 | Different review and candidate roles, with possible overlapping stands |
| Historical addition / pixel review queues | 29 / 145 | Original queues, with their actual completion fields retained |
| Official inventory identities | 3,890 | Stable polygon identity registry, not 3,890 verified harvest histories |

Counts overlap. There is no meaningful sum across this table. The record-level
collection is **not** a new ground-truth release. The frozen precision evidence
and the selective catalogue serve different estimands; neither unknown months nor
unreviewed areas become negative labels. The 7-domain review does not currently
provide a fully verified common domain for overall P/R/F1.

## Relational structure

| Table or view | Primary key / link | Meaning |
|---|---|---|
| `objects` | `sha256` | Original file bytes stored once, with relative snapshot path and size |
| `sources` | `source_id`; references `objects.sha256` | Original path, resolved path, role, availability and row count |
| `records` | `record_id`; references `objects.sha256` | Original sheet, original row number and complete `payload_json` |
| `identifiers` | `(record_id, field, value)` | Exact identifiers explicitly present in source fields |
| `source_records` | source-to-object join | All original versions and roles linked to their rows |
| `current_events` | event ID in payload | Frozen, adjudicated precision event table |
| `current_links` | report ID + verified event ID in payload | Explicit report/event correspondence |
| `current_reviews` | report ID in payload | Review state and preserved notes |
| `report_population` | report ID in payload | Entire original report frame |
| `formal_sample` | selected report IDs | Original 307-report sample with all weights |
| `history_observations` | original observation IDs | Separate continuous-history observation table |

`record_id` is the SHA-256 of `[source_file_sha256, sheet, original_row_number]`.
CSV row numbers start at 2 because row 1 is the header. Workbook rows retain their
sheet and cell sequence; header text is not guessed into scientific fields.
JSON input remains intact. For the same byte-identical file copied to two paths,
the records are stored once and two `sources` rows preserve their roles. A revised
file creates a new content identity. Events are never merged because their dates
or locations happen to agree.

The table's `payload_json` preserves all original fields without converting blank
review fields into confirmed negatives. Exact source IDs remain unchanged. A
query can retrieve the sample, its probability weight, its event links and the
original notes without asking the annotator to fill another workbook.

## Searching and rebuilding

From the repository root, search any known review, event or stand ID:

```bash
python tooling/catalogs/build_sample_catalog.py --query YOUR_REVIEW_ID
```

The query is read-only and uses a literal substring, with a 100-row display limit.
It reports original file, sheet and row. SQLite tools can inspect unrestricted
query results, for example:

```sql
SELECT source_id, role, path, sheet, row_number, payload_json
FROM source_records
WHERE source_id = 'current_verified_events';

SELECT json_extract(payload_json, '$.review_id') AS review_id,
       json_extract(payload_json, '$.prediction_id') AS report_id,
       json_extract(payload_json, '$.sampling_weight') AS sampling_weight
FROM formal_sample;
```

Build only on an authorized workstation with access to the source material:

```bash
python tooling/catalogs/build_sample_catalog.py --output data/private/sample_collection_NEW_VERSION
```

The builder refuses to overwrite a collection or place real records outside
`data/private/`. It uses the standard library plus the existing `openpyxl` IO
dependency for workbook preservation. It verifies current counts, event/report
foreign keys, positive sample weights, SQLite integrity and unchanged source
hashes. Missing registered sources are explicitly recorded; missing required
frozen evidence prevents successful validation. A successful `receipt.json`
identifies the completed collection. No detector, estimator or manuscript code
is called.

New discoveries are added to the source manifest as a new collection version.
An updated snapshot does not automatically change the frozen scoring release.
Original GUI/workbook write locations remain unchanged; the collection is a
read-only consolidation, so there are no competing editable master files.

## Scope and release boundary

This collection covers the registered historical inventory/review workflows and
the current probability, identity, repeated-reading and continuous-history
workflows. It indexes the older source/evidence catalogs rather than duplicating
image cubes and PDF libraries. It is not a claim that every historical scratch
file or backup is a distinct sample.

PlanetScope imagery, MAG geometries, reviewer notes, exact IDs and coordinates
remain local. Public upload consists of this schema, the builder and aggregate
counts. Releasing records or imagery requires a separate license/privacy check;
putting records under a Git repository does not grant redistribution rights.
