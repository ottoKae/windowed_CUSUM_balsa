# Near-decadal Sentinel-1 novelty audit — 2026-09-05

## Scope and reproducibility

This is a focused novelty check for a Windowed CUSUM manuscript, not an exhaustive systematic review and not proof of priority. Searches combined near-decadal/long-term Sentinel-1, forest disturbance, plantation harvesting, recurrence, CUSUM, monthly observations, and data cubes. The intended cutoff was 2026-09-05. Primary publisher, author, conference, and institutional sources were inspected; search-engine publication-age labels were not treated as verified publication dates.

- `query_protocol.json`: frozen initial scope and eight queries.
- `followup_queries.json`: eight year-specific follow-ups, recorded before execution.
- `metadata_followup_queries.json`: exact-title/DOI follow-up queries.
- `raw/`: unmodified serialized web-tool responses, including failed requests and source URLs. These are tool-response archives, not reconstructed publisher full text. Some later repository lookups are documented in these responses rather than a separate prospective query file.
- `selected_records.json`: manually selected bibliographic records with provenance links; selection is a bounded reading set, not a systematic inclusion decision for every search hit.
- `selected_normalized.jsonl`, `selected_deduplicated.csv`, `dedupe_receipt.json`: deterministic normalization and DOI-first deduplication of that selected set. Counts refer only to selected records, not all search hits.

The helper self-test and protocol validation passed. Searches do not establish that all competing studies have been located. Do not report a PRISMA count or state that no other near-decadal forest analysis exists.

## Important screening distinctions

1. The ESA 2025 Swiss poster is a close counterexample to a broad first-near-decadal claim, but its evidence type remains a conference poster. Indexed content and local PDF availability are recorded separately.
2. A presentation about ten years of research is not necessarily an analysis of a ten-year observation record.
3. A nine-year plantation age chronosequence is not longitudinal observation of the same polygon for nine years.
4. Crop harvesting, Portuguese `balsas` meaning river barges, soil-moisture products, and land-cover products with long nominal coverage are not automatically evidence about recurrent balsa stand disturbances.
5. Monthly mapping, multiple change points, and OPERA RTC use already have precedents; none alone supports a priority claim.
6. A preprint is not promoted to a peer-reviewed article without verification of the published version. A web-readable PDF is not marked locally downloaded unless it passed the acquisition audit.

## Interpretation boundary

Search and acquisition helpers handle discovery, provenance, download authorization, file identity and coverage. The manuscript interpretation is a separate synthesis in the updated optimization plan; it is not a conclusion produced by the acquisition software. Reading copies remain local unless redistribution rights have been checked individually.
