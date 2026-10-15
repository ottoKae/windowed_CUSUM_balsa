"""Collect existing interpretation evidence without changing references or scores.

Private SQLite/CSV output; public summaries contain aggregate counts only.
Run with the existing IO environment (openpyxl is used for source workbooks).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/private/sample_collection_20261015_v1"
V = "runs/validation/"
FINAL = V + "jag_reference_score_closeout_20261015_v1/"
SAMPLING = V + "b20_spatial_probability_review_20260928_v1/"
IDENTITY = V + "b20_month_tolerance_finalization_20260928_v1/"
HISTORY = V + "common_domain_prf_closure_20261008_v1/"
JAG = V + "jag_event_accuracy_recalculation_20260928_v1/"
BENCH = V + "core_pixel_b20_precision_alignment_20261004_v1/"
REGISTRY = "data/balsa_FC001_dated_event_subset/releases/registry_20260918_v1/"

# Roles are explicit: a queue, month observation and scored event are not synonyms.
SOURCES = [
    ("current_verified_events", FINAL + "evidence/verified_events.csv", "frozen_precision_evidence"),
    ("current_event_links", FINAL + "evidence/candidate_event_links.csv", "frozen_identity_and_space_links"),
    ("current_report_reviews", FINAL + "evidence/candidate_reviews.csv", "frozen_precision_review_state"),
    ("current_reference_changes", FINAL + "evidence/reference_changes.csv", "adjudication_history"),
    ("current_scores", FINAL + "scoring/report_scores.csv", "derived_scores_not_reference"),
    ("main_probability_frame", SAMPLING + "sampling_frame.csv", "frozen_report_population_and_probability_sample"),
    ("main_sampling_design", SAMPLING + "sampling_config.json", "frozen_sampling_design"),
    ("main_optical_months", SAMPLING + "optical_month_review.csv", "first_visible_month_observations"),
    ("identity40", IDENTITY + "identity_relation_answers.csv", "completed_identity_adjudication"),
    ("identity40_groups", IDENTITY + "identity_review_groups.csv", "identity_group_membership"),
    ("repeat31", IDENTITY + "repeat_comparison.csv", "repeat_interpretation_not_31_new_events"),
    ("repeat31_scope", V + "jag_final_accuracy_f1_20261002_v1/diagnostics/repeat31_comparability.csv", "repeat_event_alignment_status"),
    ("relations28", JAG + "precision_after_relations_20261002_v1/answer_import_audit.csv", "completed_identity_adjudication"),
    ("relations28_answers", JAG + "precision_after_relations_20261002_v1/answer_snapshot.json", "raw_adjudication_snapshot"),
    ("history7_domains", HISTORY + "reference_library_20261015_v1/domain_evidence_status.csv", "history_coverage_not_complete_truth"),
    ("history7_observations", HISTORY + "reference_library_20261015_v1/harvest_observations.csv", "observations_not_automatically_independent_events"),
    ("history7_answers", HISTORY + "reference_library_20261015_v1/raw_answers.json", "raw_history_answers"),
    ("history7_lineage", HISTORY + "reference_library_20261015_v1/observation_lineage.csv", "observation_lineage"),
    ("history7_spatial_flags", HISTORY + "reference_library_20261015_v1/prior_sample_spatial_scope_flags.csv", "spatial_uncertainty"),
    ("history7_readiness", HISTORY + "reference_library_20261015_v1/common_domain_scoring_readiness.csv", "complete_domain_eligibility"),
    ("history7_observation_alias", FINAL + "evidence/new_observations_preserved.csv", "frozen_copy_of_history_observations"),
    ("catalogue_R0", V + "existing_labels_w36_20260927_v1/inputs/harvest_events.csv", "frozen_selective_recall_catalogue"),
    ("old_native340", JAG + "core_probability_review_queue.csv", "historical_native_method_queue_not_current_benchmark"),
    ("old_native340_reviews", V + "jag_final_accuracy_f1_20261002_v1/core_review/core_review_status.csv", "historical_native_review_status"),
    ("current_benchmark303", BENCH + "formal_review_queue.csv", "deferred_pixel_B20_probability_queue"),
    ("benchmark303_answers", BENCH + "gui_answer_snapshot.json", "existing_answers_not_new_completed_queue"),
    ("benchmark303_evidence", BENCH + "existing_evidence_review/evidence_reuse.csv", "report_specific_evidence_applicability"),
    ("benchmark303_gaps", BENCH + "existing_evidence_review/gap_report_details.csv", "unresolved_applicability_not_false_reports"),
    ("inventory_registry", REGISTRY + "inventory/polygon_registry.csv", "inventory_identity_registry"),
    ("inventory_cohorts", REGISTRY + "inventory/cohort_membership.csv", "historical_cohort_membership"),
    ("historical_current1061", REGISTRY + "views/events_current_with_inventory_id.csv", "historical_reference_release_20260911"),
    ("historical_legacy1093", REGISTRY + "views/events_legacy_with_inventory_id.csv", "historical_reference_release_paper_v1"),
    ("prospective95", REGISTRY + "views/prospective_2026_review_index.csv", "historical_prospective_review"),
    ("prospective95_original", "data/local/paper_v1/prospective_2026/prior_blinded_review_unblinded.csv", "original_prospective_review_with_all_notes_and_dates"),
    ("prospective140", REGISTRY + "views/prospective_2026_n1_candidate_index.csv", "historical_prospective_candidate_queue"),
    ("gapfill29", REGISTRY + "views/gapfill_29_index.csv", "historical_addition_queue"),
    ("pixel_review145", REGISTRY + "views/pixelwise_145_review_index.csv", "historical_pixel_diagnostic_queue"),
    ("historical_source_index", REGISTRY + "sources.csv", "historical_source_locator"),
    ("historical_evidence_index", REGISTRY + "evidence_index.csv", "historical_evidence_locator"),
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def discover_sources():
    """Reuse the old registry, then register live ledgers and workbook originals."""
    specs = list(SOURCES)
    old = ROOT / REGISTRY / "sources.csv"
    if old.exists():
        for row in csv_rows(old):
            p = Path(row["source_path"])
            name = p.name
            is_review = ("manual_review_ledger" in name or "inherited_review" in name
                         or "review_template" in name or "event_entry_template" in name
                         or "blinded_review_queue" in name)
            if (is_review and p.suffix == ".csv") or (row["kind"] == "gapfill_source" and p.suffix == ".xlsx"):
                sid = "live_" + digest(str(p).encode())[:16]
                specs.append((sid, str(p), "live_legacy_ledger_not_automatically_frozen_truth"))
                copied = str(ROOT / REGISTRY / row["central_path"])
                specs.append((sid + "_snapshot", copied, "historical_ledger_snapshot"))
    # Workbook rows are preserved as cells, not guessed into event semantics.
    for folder in [SAMPLING, IDENTITY, HISTORY + "domains/", JAG, BENCH + "quick_review/"]:
        for p in sorted((ROOT / folder).glob("*.xlsx")):
            if not p.name.startswith("~$"):
                specs.append(("workbook_" + digest(str(p).encode())[:16], str(p), "raw_workbook_not_new_event_population"))
    return specs


def rows_for(path):
    if path.suffix.lower() == ".csv":
        return [("", i, row) for i, row in enumerate(csv_rows(path), 2)]
    if path.suffix.lower() == ".json":
        return [("", 1, json.loads(path.read_text(encoding="utf-8-sig"))) ]
    if path.suffix.lower() == ".xlsx":
        import openpyxl
        book = openpyxl.load_workbook(path, read_only=True, data_only=False)
        result = []
        for sheet in book:
            for i, values in enumerate(sheet.iter_rows(values_only=True), 1):
                if any(v is not None for v in values):
                    result.append((sheet.title, i, {"cells": list(values)}))
        book.close()
        return result
    raise ValueError(f"Unsupported collection source: {path}")


def identifiers(value):
    """Only index explicit identifiers; never infer identity from date/proximity."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key.lower().endswith("_id") or key == "ID") and isinstance(item, (str, int)) and str(item):
                found.add((key, str(item)))
            found.update(identifiers(item))
    elif isinstance(value, list):
        for item in value:
            found.update(identifiers(item))
    return found


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(output):
    output = output.resolve()
    private = (ROOT / "data/private").resolve()
    if not output.is_relative_to(private):
        raise ValueError("Real sample material must remain below data/private/")
    if output.exists():
        raise FileExistsError("Preserve snapshots: choose a new private output directory")
    output.mkdir(parents=True)
    (output / "objects").mkdir()
    db = sqlite3.connect(output / "samples.sqlite")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE objects(sha256 TEXT PRIMARY KEY, snapshot_path TEXT, bytes INTEGER);
    CREATE TABLE sources(source_id TEXT PRIMARY KEY, role TEXT, path TEXT, resolved_path TEXT,
      sha256 TEXT REFERENCES objects, status TEXT, record_count INTEGER);
    CREATE TABLE records(record_id TEXT PRIMARY KEY, sha256 TEXT REFERENCES objects,
      sheet TEXT, row_number INTEGER, payload_json TEXT);
    CREATE TABLE identifiers(record_id TEXT REFERENCES records, field TEXT, value TEXT,
      PRIMARY KEY(record_id,field,value));
    CREATE INDEX identifiers_value ON identifiers(value);
    CREATE INDEX records_object ON records(sha256);
    CREATE VIEW source_records AS SELECT s.source_id,s.role,s.path,r.*
      FROM sources s JOIN records r USING(sha256);
    """)
    sources, gaps, parsed, before = [], [], {}, {}
    for sid, relative, role in discover_sources():
        path = ROOT / relative
        display = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if not path.is_file():
            sources.append(dict(source_id=sid, role=role, path=display, resolved_path=str(path.resolve()),
                                sha256="", status="missing", record_count=0))
            gaps.append(dict(source_id=sid, reason="registered_source_missing", path=display))
            db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?)", (sid,role,display,str(path.resolve()),None,"missing",0))
            continue
        raw = path.read_bytes()
        sha = digest(raw)
        before[str(path)] = sha
        if sha not in parsed:
            rel = "objects/" + sha + path.suffix.lower()
            (output / rel).write_bytes(raw)
            db.execute("INSERT INTO objects VALUES(?,?,?)", (sha,rel,len(raw)))
            parsed[sha] = rows_for(path)
            for sheet, n, row in parsed[sha]:
                rid = digest(json_text([sha,sheet,n]).encode())
                db.execute("INSERT INTO records VALUES(?,?,?,?,?)", (rid,sha,sheet,n,json_text(row)))
                db.executemany("INSERT INTO identifiers VALUES(?,?,?)", [(rid,k,v) for k,v in sorted(identifiers(row))])
        nrows = len(parsed[sha])
        source = dict(source_id=sid,role=role,path=display,resolved_path=str(path.resolve()),
                      sha256=sha,status="available",record_count=nrows)
        sources.append(source)
        db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?)", tuple(source.values()))

    # Explicit current views; old releases stay queryable without being merged as truth.
    for view, sid in [("current_events","current_verified_events"),("current_links","current_event_links"),
                      ("current_reviews","current_report_reviews"),("report_population","main_probability_frame"),
                      ("history_observations","history7_observations")]:
        db.execute(f"CREATE VIEW {view} AS SELECT * FROM source_records WHERE source_id='{sid}'")
    db.execute("CREATE VIEW formal_sample AS SELECT * FROM report_population WHERE lower(json_extract(payload_json,'$.selected'))='true'")
    checks = validate(db)
    assert all(digest(Path(p).read_bytes()) == sha for p, sha in before.items()), "Source changed while collecting"
    db.commit()
    total_records = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    total_ids = db.execute("SELECT COUNT(*) FROM identifiers").fetchone()[0]
    db.close()
    write_csv(output / "source_catalog.csv", sources, list(sources[0]))
    write_csv(output / "gaps.csv", gaps, ["source_id","reason","path"])
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),source_paths=len(sources),
                   unique_source_objects=len(parsed),source_aliases=sum(s["status"]=="available" for s in sources)-len(parsed),
                   missing_sources=len(gaps),stored_source_rows=total_records,indexed_identifier_values=total_ids,
                   source_files_unchanged=True,scoring_performed=False,reference_edits_performed=False,checks=checks)
    (output / "receipt.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (output / "README.md").write_text(PRIVATE_README,encoding="utf-8")
    public = ROOT / "data/public/sample_collection_summary.json"
    public.write_text(json.dumps({"collection_version":"20261015_v1","counts":checks,
                                 "source_paths":len(sources),"unique_source_objects":len(parsed),"missing_sources":len(gaps),
                                 "statement":"Counts describe overlapping roles and versions, not one independent event population."},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


def validate(db):
    def rows(sid):
        return [json.loads(r[0]) for r in db.execute("SELECT payload_json FROM source_records WHERE source_id=? ORDER BY row_number",(sid,))]
    events, links, reviews = rows("current_verified_events"),rows("current_event_links"),rows("current_report_reviews")
    frame = rows("main_probability_frame")
    ids = {e["verified_event_id"] for e in events}
    predictions = {r["prediction_id"] for r in frame}
    assert len(ids)==len(events)==300
    assert len(predictions)==len(frame)==1266
    assert len(links)==363 and all(x["verified_event_id"] in ids and x["prediction_id"] in predictions for x in links)
    assert len(reviews)==717 and all(x["prediction_id"] in predictions for x in reviews)
    formal = [r for r in frame if r["selected"].lower()=="true"]
    assert len(formal)==307 and all(float(r["sampling_weight"])>0 for r in formal)
    assert len(rows("identity40"))==40 and len(rows("relations28"))==28 and len(rows("repeat31"))==31
    assert len(rows("history7_domains"))==7 and len(rows("history7_observations"))==11
    assert len(rows("current_benchmark303"))==303 and len(rows("old_native340"))==340
    assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    assert not db.execute("PRAGMA foreign_key_check").fetchall()
    return dict(frozen_verified_events=len(events),frozen_links=len(links),frozen_review_states=len(reviews),
                report_population=len(frame),formal_probability_sample=len(formal),identity_groups=40,relation_followups=28,
                repeat_readings=31,history_domains=7,history_observations=11,current_deferred_benchmark_reports=303,
                historical_native_benchmark_queue=340)


PRIVATE_README = """# Unified local interpretation collection

This is a read-only evidence snapshot, not a new truth table or a new writable GUI.
Use `source_catalog.csv` to locate every original and `samples.sqlite` to search.
`objects/` contains byte-identical source files stored once per SHA-256. The source
catalog retains every alias, version and role. Originals remain unchanged.

Quick lookup from the repository root:
```
python tooling/catalogs/build_sample_catalog.py --query YOUR_REVIEW_ID
python tooling/catalogs/build_sample_catalog.py --query YOUR_EVENT_ID
```
SQL views: `current_events`, `current_links`, `current_reviews`, `report_population`,
`formal_sample`, `history_observations`. `source_records` joins all roles and
versions to original rows; `identifiers` indexes explicit IDs. `payload_json`
preserves all columns and notes. Workbook records preserve sheet, row and cells.

The 300 precision evidence events are distinct from the selective recall catalogue.
The 307 selected reports retain their weights among 1,266 reports. The seven
history domains and their 11 observations are not complete-history ground truth.
31 repeat readings, 40 identity groups and 28 followups are evidence operations,
not extra independent samples. Historical and deferred benchmark queues stay
separate from completed review records. Never deduplicate events by month or
distance. Only byte-identical source artifacts share storage and record IDs.

This directory contains restricted IDs, exact locations, notes and raw workbooks.
Do not publish it. Public documentation is in data/schemas/sample_collection.md.
Changes to a source require a new snapshot; this tool refuses to overwrite one.
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--query",help="Read-only literal substring lookup in existing private collection")
    args=parser.parse_args()
    if args.query is not None:
        path=(args.output/"samples.sqlite").resolve()
        with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
            for row in db.execute("SELECT source_id,role,path,sheet,row_number,payload_json FROM source_records WHERE instr(lower(payload_json),lower(?))>0 LIMIT 100",(args.query,)):
                print(json_text(dict(zip(["source_id","role","path","sheet","row_number","record"],row))))
    else:
        build(args.output)


if __name__ == "__main__":
    main()
