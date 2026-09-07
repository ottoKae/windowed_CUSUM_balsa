# Near-decadal Sentinel-1 and recurrent forest disturbance: reading set

Updated 2026-09-05. This folder supports the [manuscript optimization plan](../docs/timeliness_reading_20260905/03_Windowed_CUSUM_full_manuscript_optimization_plan.md). No detector, manuscript LaTeX, figure, or frozen result was changed by this literature task.

Final local coverage for this run: **5/9 selected records have validated PDFs** (four new downloads plus the existing user-provided Timeliness paper). Four records remain in the [manual acquisition queue](outputs/acquisition/manual/manual_queue.html). All five files passed identity/parse checks; the audit reported zero anomalies. These are document checks, not a scientific quality score.

## Main conclusion

Do not claim the first near-decadal Sentinel-1 forest analysis. The official indexed ESA 2025 Swiss poster reports a 2015–2024 national analysis; its local full text remains to be acquired. Continuous CuSum, multiple-change detection, monthly mapping, and OPERA RTC disturbance analysis also have precedents. A defensible research question connects an auditable longitudinal stand record to recurrent-event inference and explicitly bounded history reconstruction.

The plan distinguishes record extent **T**, local window **W**, displacement **S**, calendar support, and post-event support **h**. Longer records increase opportunities to observe recurrence, but do not automatically improve the accuracy of a fixed local detector.

Especially important: S05 Section 4.1.4.1 (PDF page 6) already describes sliding-window CuSum NRT with pre-/post-change image-count parameters. Windowing itself is therefore not a priority claim. The manuscript needs a rule-level comparison of image-count versus calendar support, central admission, consolidation, and history-oriented validation. S05 Section 4.1.3 also explicitly explains ReCuSum multiple breakpoints, supporting that distinction even though the S03 local PDF is still unavailable.

## Contents

| File/folder | Purpose |
|---|---|
| `bibliography.csv` | Frozen, selected nine-record bibliography; seven journal articles, one conference poster, one preprint |
| `papers_pdf/` | Validated local reading copies; not assumed redistributable |
| `approved_manifest.csv` | Verified OA/institutional URLs approved for this task |
| `approved_unaffected_hosts.csv`, `approved_arxiv.csv` | Bounded continuation queues excluding blocked hosts or unattempted records |
| `search_20260905/` | Queries, raw tool responses, provenance, normalized selected records and deduplication receipt |
| `outputs/acquisition/` | Input audit, download/import logs, coverage, file hashes and manual queue |
| `outputs/reading_text/` | Local `pdftotext -layout` derivatives for checking methods and source locations; not curated full-paper Markdown |
| `acquisition.local.toml` | Local acquisition configuration; credentials referenced by environment-variable name, not stored here |

## Reading priority and availability

| ID | Reading focus | Current local evidence |
|---|---|---|
| S01 Ruiz-Ramos 2020 | Continuous forest CuSum precedent | Institutional metadata and indexed content; PDF download blocked at Open University, Stirling page also returned 403 |
| S02 Pulella 2020 | Monthly backscatter/coherence monitoring | Validated PDF, 22 pages |
| S03 Ygorra 2023 | ReCuSum multiple-change precedent | Author-hosted PDF indexed in search; both author-hosted and subsequently verified HAL transfers failed, do not claim complete local reading |
| S04 van der Woude 2024 | Management disturbance, seasonality and geometry | Validated institutional PDF, 40 pages; Section 2.2.2 specifies January 2016–April 2022 |
| S05 Ygorra 2024 | CuSum NRT, initialization and confirmation | Validated publisher PDF, 12 pages |
| S06 Ruetschi 2025 | Close near-decadal national precedent | Official poster indexed text; local transfer failed and browser fetch returned 502; confirm the poster file before formal citation |
| S07 Welsink 2025 | Physical and reference evidence for fine-scale canopy gaps | Institutional PDF readable by the web tool; local transfer failed |
| S08 Hardiman-Mostow 2025 | OPERA RTC self-supervised disturbance mapping | Validated arXiv v2 PDF; retain preprint status |
| S09 Bullock 2022 | Timing/accuracy framework and interpretation | Validated copy imported from the user-provided PDF without altering the original |

Acquisition totals and SHA-256 values are authoritative in `outputs/acquisition/audit/summary.json`, `records.csv`, and the per-transfer JSONL logs. A download failure is not evidence that no lawful full text exists. Requests stopped on server blocks/rate limits or repeated connection failures. MDPI page requests returned 429; the independent Wageningen institutional copy was used for S04. No access controls were bypassed.

The official OA discovery helper subsequently checked the four missing records and returned candidate routes (including duplicates and formats that were not PDFs). The HAL alternative for ReCuSum was approved separately in `approved_hal_alternative.csv` but failed to transfer. Already blocked MDPI URLs were not retried. Keep `outputs/acquisition/oa_alternatives.csv` as candidate metadata, not as evidence of downloaded full text.

Document QA: UTF-8 decoding and all local links in the updated plan passed; selected-record JSON parsed; protocol validation and search-helper self-test passed; the selected bibliography deduplicated to nine records. No model training, detector evaluation, LaTeX compilation, commit, or push was performed.

## Reuse and publication

This is a focused novelty audit, not an exhaustive systematic review. Do not turn the selected-record count into a count of all eligible studies or claim that no competing method exists. The study-type and availability labels above are part of the evidence boundary.

The helper-generated `.gitignore` excludes PDF/XML reading copies, local configuration, extraction outputs and acquisition logs from ordinary Git adds. Open access is not automatically unrestricted redistribution: check each license before publishing a PDF collection. The bibliography, source links and synthesis can be shared subject to repository policy. This task does not commit or push anything.
