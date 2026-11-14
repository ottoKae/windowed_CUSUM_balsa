"""Import optical review facts and score frozen B20 reports without inventing truth.

The sampling frame always contains the whole frozen population.  Only selected
reports enter the returned statistical table; all reports can compete for one
verified event.  Old catalogue classifications are deliberately never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from windowed_cusum.evaluation.b20_probability_estimation import tolerance_spec


REVIEW_COLUMNS = ["prediction_id", "review_status", "occurrence", "context_complete", "evidence_notes"]
EVENT_COLUMNS = ["verified_event_id", "sample_id", "timing_type", "reference_month",
                 "interval_start", "interval_end", "start_inclusive", "end_inclusive",
                 "date_basis", "evidence_notes"]
LINK_COLUMNS = ["prediction_id", "verified_event_id", "identity_status", "spatial_status", "evidence_notes"]
RESULT_COLUMNS = ["prediction_id", "sample_id", "tolerance_quarters", "review_status",
                  "judgement", "reason", "verified_event_id", "competing_prediction_id"]


def _text(value):
    return "" if value is None or (np.isscalar(value) and pd.isna(value)) else str(value).strip()


def _yes(value):
    return _text(value).lower() in {"true", "yes", "1", "是"}


def _month(value):
    value = _text(value)
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError(f"Scoring months must be YYYY-MM, got {value!r}")
    return pd.Period(value, freq="M").ordinal


def _quarter(month):
    return _month(month) // 3


def _table(frame, columns, key):
    frame = frame.copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{key}: missing columns {sorted(missing)}")
    frame = frame.loc[frame[key].map(_text).ne("")].copy()
    for column in columns:
        frame[column] = frame[column].map(_text)
    return frame


def _event_bounds(event):
    kind = event["timing_type"]
    if kind == "month":
        if event["interval_start"] or event["interval_end"]:
            raise ValueError("An event must have exactly one scoring time representation")
        return _month(event["reference_month"]), _month(event["reference_month"])
    if kind == "interval":
        if event["reference_month"]:
            raise ValueError("An event must have exactly one scoring time representation")
        if event["start_inclusive"].lower() not in {"true", "false", "yes", "no", "1", "0"}:
            raise ValueError("Explicit interval start inclusion is required")
        if event["end_inclusive"].lower() not in {"true", "false", "yes", "no", "1", "0"}:
            raise ValueError("Explicit interval end inclusion is required")
        lower = _month(event["interval_start"]) + int(not _yes(event["start_inclusive"]))
        upper = _month(event["interval_end"]) - int(not _yes(event["end_inclusive"]))
        if lower > upper:
            raise ValueError("Empty interval after applying the original open/closed monthly bounds")
        return lower, upper
    if kind == "unknown":
        return None
    raise ValueError(f"Invalid timing_type: {kind}")


def _response_months(prediction):
    months = prediction.get("response_months", "")
    if isinstance(months, str):
        months = json.loads(months)
    if not isinstance(months, (list, tuple, np.ndarray)) or not len(months):
        raise ValueError("Actual response_months are required for interval matching")
    return {_month(month) for month in months}


def _compatible(event, prediction, tolerance, tolerance_unit="quarter"):
    bounds = _event_bounds(event)
    if bounds is None:
        return None
    if event["timing_type"] == "month":
        predicted_month = prediction.get("representative_month", prediction.get("predicted_month", ""))
        ordinal = _month if tolerance_unit == "month" else _quarter
        error = abs(ordinal(event["reference_month"]) - ordinal(predicted_month))
        return (1, error) if error <= tolerance else None
    return (0, 0) if any(bounds[0] <= m <= bounds[1] for m in _response_months(prediction)) else None


def _canonical_matching(edges):
    """The existing four-level matcher, restricted to verified edges.

Maximize pairs, maximize month-reference pairs, minimize their date distance
in the explicitly selected unit (quarters or months),
then lexicographically minimize (verified_event_id, prediction_id) pairs.
"""
    if not edges:
        return {}
    event_ids = sorted({e for e, _ in edges})
    prediction_ids = sorted({p for _, p in edges})
    k = min(len(event_ids), len(prediction_ids))
    max_error = k * max(error for _, error in edges.values())
    month_bonus = max_error + 1
    count_bonus = k * month_bonus + max_error + 1

    def objective(pairs):
        return len(pairs), sum(edges[x][0] for x in pairs), -sum(edges[x][1] for x in pairs)

    def solve(fixed, forbidden):
        used_e, used_p = {e for e, _ in fixed}, {p for _, p in fixed}
        ee = [e for e in event_ids if e not in used_e]
        pp = [p for p in prediction_ids if p not in used_p]
        if not ee or not pp:
            return objective(fixed)
        costs = np.full((len(ee), len(pp) + len(ee)), count_bonus * (k + 1), dtype=np.int64)
        costs[:, len(pp):] = 0
        for i, e in enumerate(ee):
            for j, p in enumerate(pp):
                if (e, p) in edges and (e, p) not in forbidden:
                    monthly, error = edges[e, p]
                    costs[i, j] = -count_bonus - monthly * month_bonus + error
        rows, cols = linear_sum_assignment(costs)
        pairs = list(fixed) + [(ee[i], pp[j]) for i, j in zip(rows, cols)
                              if j < len(pp) and costs[i, j] < 0]
        return objective(pairs)

    target = solve([], set())
    fixed, forbidden, used_e, used_p = [], set(), set(), set()
    for edge in sorted(edges):
        e, p = edge
        if e in used_e or p in used_p:
            continue
        if solve(fixed + [edge], forbidden) == target:
            fixed.append(edge)
            used_e.add(e)
            used_p.add(p)
        else:
            forbidden.add(edge)
    if objective(fixed) != target:
        raise AssertionError("Verified matching did not retain its objective")
    return {p: e for e, p in fixed}


def score_review_tables(frame, candidate_reviews, verified_events, candidate_event_links,
                        *, tolerance_unit="quarter", tolerances=None):
    """Return selected reports x tolerances, preserving pending separately from U.

``context_complete=yes`` is the reviewer's evidence-based assertion that all
competitors able to change this report's credit have been considered.  Explicit
uncertain edges additionally suspend their whole connected competition component.
Candidate-event edges require independent identity AND spatial confirmation.
Quarter defaults (0, 1) preserve the existing schema. Monthly defaults (1, 2, 3)
compare fixed representative months, not any response month or window overlap.
Interval-reference rules remain unchanged and do not expand with the tolerance.
"""
    tolerance_column, tolerances = tolerance_spec(tolerance_unit, tolerances)
    required = {"prediction_id", "sample_id", "selected"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Full sampling frame requires {sorted(required)}")
    if frame.prediction_id.duplicated().any():
        raise ValueError("The full frame must have unique prediction_id")
    predictions = {_text(row["prediction_id"]): row for row in frame.to_dict("records")}
    reviews = _table(candidate_reviews, REVIEW_COLUMNS, "prediction_id")
    events = _table(verified_events, EVENT_COLUMNS, "verified_event_id")
    links = _table(candidate_event_links, LINK_COLUMNS, "prediction_id")
    if reviews.prediction_id.duplicated().any() or events.verified_event_id.duplicated().any():
        raise ValueError("Each candidate review and verified_event_id must be unique")
    if links.duplicated(["prediction_id", "verified_event_id"]).any():
        raise ValueError("Candidate-event links must be unique")
    if not set(reviews.prediction_id).issubset(predictions) or not set(links.prediction_id).issubset(predictions):
        raise ValueError("Review/link refers to a report outside the frozen frame")
    event_map = events.set_index("verified_event_id").to_dict("index")
    review_map = reviews.set_index("prediction_id").to_dict("index")
    for event_id, event in event_map.items():
        _event_bounds(event)
        if not event["sample_id"] or not event["date_basis"] or not event["evidence_notes"]:
            raise ValueError(f"Verified event {event_id} lacks stand/date basis/evidence")
    for prediction_id, review in review_map.items():
        if review["review_status"] not in {"pending", "reviewed"}:
            raise ValueError(f"Invalid review_status for {prediction_id}")
        allowed_occurrence = {"confirmed", "no_event", "unknown"}
        if review["review_status"] == "pending":
            allowed_occurrence.add("")
        if review["occurrence"] not in allowed_occurrence:
            raise ValueError(f"Invalid occurrence for {prediction_id}")
        if review["review_status"] == "reviewed" and not review["evidence_notes"]:
            raise ValueError(f"Reviewed report {prediction_id} requires optical evidence notes")

    # Connected components include uncertain edges: they must not be deleted
    # before assigning one-to-one credit to an apparently unopposed candidate.
    graph, uncertain_nodes, confirmed = {}, set(), {}
    for link in links.to_dict("records"):
        p, e = link["prediction_id"], link["verified_event_id"]
        if e not in event_map:
            raise ValueError(f"Link refers to absent verified_event_id {e}")
        if _text(predictions[p]["sample_id"]) != event_map[e]["sample_id"]:
            raise ValueError("A link cannot cross stand identities")
        if link["identity_status"] not in {"confirmed", "uncertain"} or link["spatial_status"] not in {"confirmed", "uncertain"}:
            raise ValueError("Link identity and spatial status must be confirmed or uncertain")
        if not link["evidence_notes"]:
            raise ValueError("Candidate-event links require supporting evidence")
        if review_map.get(p, {}).get("occurrence") == "no_event":
            raise ValueError("A no_event judgement conflicts with a possible or confirmed event link")
        pn, en = ("prediction", p), ("event", e)
        graph.setdefault(pn, set()).add(en)
        graph.setdefault(en, set()).add(pn)
        valid = (link["identity_status"] == link["spatial_status"] == "confirmed"
                 and event_map[e]["timing_type"] != "unknown")
        if valid:
            confirmed.setdefault(p, set()).add(e)
        else:
            uncertain_nodes.update([pn, en])
        # An unresolved competitor may acquire another event and thereby change
        # the globally optimal assignment.  Its known edges must not make the
        # rest of the component appear closed. Context candidates need the same
        # explicit completion assertion even though they are not statistical rows.
        linked_review = review_map.get(p, {})
        if (linked_review.get("review_status") != "reviewed"
                or not _yes(linked_review.get("context_complete", ""))
                or linked_review.get("occurrence") != "confirmed"):
            uncertain_nodes.add(pn)
    queue = list(uncertain_nodes)
    affected = set(queue)
    while queue:
        node = queue.pop()
        for other in graph.get(node, []):
            if other not in affected:
                affected.add(other)
                queue.append(other)

    output = []
    for tolerance in tolerances:
        edges = {}
        for p, event_ids in confirmed.items():
            if ("prediction", p) in affected:
                continue
            for e in event_ids:
                compatible = _compatible(event_map[e], predictions[p], tolerance, tolerance_unit)
                if compatible is not None:
                    edges[e, p] = compatible
        # Stands have no legal cross-stand edges, so their exact objectives are
        # separable. Avoid a large, mostly empty national assignment matrix.
        stand_edges = {}
        for (e, p), value in edges.items():
            stand_edges.setdefault(_text(predictions[p]["sample_id"]), {})[e, p] = value
        assigned = {}
        for sample_id in sorted(stand_edges):
            assigned.update(_canonical_matching(stand_edges[sample_id]))
        winner_by_event = {event: pred for pred, event in assigned.items()}
        for p, prediction in predictions.items():
            if not _yes(prediction["selected"]):
                continue
            review = review_map.get(p, {})
            status = review.get("review_status", "pending")
            row = dict(prediction_id=p, sample_id=prediction["sample_id"],
                       review_status=status, judgement="", reason="pending_optical_review",
                       verified_event_id="", competing_prediction_id="")
            row[tolerance_column] = tolerance
            if tolerance_unit == "month":
                row["tolerance_unit"] = tolerance_unit
            if status != "reviewed":
                output.append(row)
                continue
            row.update(judgement="U", reason="insufficient_optical_evidence")
            if review["occurrence"] == "unknown":
                pass
            elif not _yes(review["context_complete"]):
                row["reason"] = "competition_context_incomplete"
            elif ("prediction", p) in affected:
                row["reason"] = "unresolved_identity_space_time_or_competition"
            elif review["occurrence"] == "no_event":
                row.update(judgement="F", reason="no_event")
            elif p in assigned:
                row.update(judgement="T", reason="verified_one_to_one_match", verified_event_id=assigned[p])
            elif confirmed.get(p):
                legal_events = sorted(e for e in confirmed[p] if (e, p) in edges)
                if legal_events:
                    # Max-cardinality matching leaves every legal neighbour of
                    # an unmatched candidate occupied. Identity exclusivity is
                    # supplied by the complete-context review, not by proximity.
                    e = legal_events[0]
                    if e not in winner_by_event:
                        raise AssertionError("Unmatched candidate has a free legal event")
                    row.update(judgement="F", reason="duplicate_report", verified_event_id=e,
                               competing_prediction_id=winner_by_event[e])
                else:
                    row.update(judgement="F", reason="time_mismatch",
                               verified_event_id=";".join(sorted(confirmed[p])))
            else:
                row["reason"] = "no_confirmed_identity_and_spatial_link"
            output.append(row)
    result_columns = [tolerance_column if column == "tolerance_quarters" else column
                      for column in RESULT_COLUMNS]
    if tolerance_unit == "month":
        result_columns.insert(3, "tolerance_unit")
    return pd.DataFrame(output, columns=result_columns)


def verify_frozen_outputs(run_dir):
    """Fail before importing human work if frozen selection/support was altered."""
    run = Path(run_dir).resolve()
    config = json.loads((run / "sampling_config.json").read_text(encoding="utf-8-sig"))
    frozen = config.get("frozen_outputs", {})
    if not frozen or "sampling_frame.csv" not in frozen or "block_frame.csv" not in frozen:
        raise ValueError("sampling_config.json must identify the frozen sampling outputs")
    for relative_path, expected in frozen.items():
        path = (run / relative_path).resolve()
        if not path.is_relative_to(run):
            raise ValueError("A frozen output must stay inside the review run directory")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Frozen output hash mismatch: {relative_path}; do not alter or redraw the sample")


def publish_scored_results(run_dir, frame, tables, estimator=None):
    """Validate scoring AND estimation before replacing any derived results."""
    if estimator is None:
        from windowed_cusum.evaluation.b20_probability_estimation import summarize_probability_results
        estimator = summarize_probability_results
    result = score_review_tables(frame, **tables)
    summary = estimator(frame, result)
    # Serialize all tables before the first write, so an input/statistical/
    # serialization error cannot leave half-updated review results.
    outputs = {"candidate_review_results": result, "validation_summary": summary, **tables}
    serialized = {name: table.to_csv(index=False) for name, table in outputs.items()}
    run = Path(run_dir)
    for name, content in serialized.items():
        (run / f"{name}.csv").write_text(content, encoding="utf-8-sig", newline="")
    return result, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--from-workbook", action="store_true",
                        help="Read the filled workbook through the review-package exporter")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    verify_frozen_outputs(run)
    if args.from_workbook and (run / "month_only_config.json").exists():
        from b20_month_only_review import export
        export(run)
        return
    frame = pd.read_csv(run / "sampling_frame.csv", keep_default_na=False)
    if args.from_workbook:
        from build_b20_probability_review import export_review_tables
        tables = export_review_tables(run, write_csv=False)
    else:
        tables = {}
        for name, columns in (("candidate_reviews", REVIEW_COLUMNS), ("verified_events", EVENT_COLUMNS),
                              ("candidate_event_links", LINK_COLUMNS)):
            path = run / f"{name}.csv"
            tables[name] = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=columns)
    result, _ = publish_scored_results(run, frame, tables)
    counts = result.groupby(["tolerance_quarters", "review_status", "judgement"], dropna=False).size()
    print(counts.to_string())
    print("Only newly reviewed optical evidence is scored; pending rows are not reviewed U.")


if __name__ == "__main__":
    main()
