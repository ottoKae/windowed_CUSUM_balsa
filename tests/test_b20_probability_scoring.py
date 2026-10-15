"""Evidence identity, unknown competition and two independent time tolerances."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_b20_probability_review import (EVENT_COLUMNS, LINK_COLUMNS, REVIEW_COLUMNS,
                                         _canonical_matching, publish_scored_results,
                                         score_review_tables, verify_frozen_outputs)


def prediction(identifier, month="2021-04", selected=True, sample_id="s"):
    return {"prediction_id": identifier, "sample_id": sample_id, "selected": selected,
            "representative_month": month, "response_months": [month], "candidate_class": "C1"}


def review(identifier, **changes):
    return {"prediction_id": identifier, "review_status": "reviewed", "occurrence": "confirmed",
            "context_complete": "yes", "evidence_notes": "Optical observations a/b and complete competing identity review",
            **changes}


def event(identifier, month="2021-04", **changes):
    return {"verified_event_id": identifier, "sample_id": "s", "timing_type": "month",
            "reference_month": month, "interval_start": "", "interval_end": "",
            "start_inclusive": "", "end_inclusive": "", "date_basis": "optical_first_visible_month",
            "evidence_notes": "Optical tree cover followed by removal", **changes}


def link(prediction_id, event_id, **changes):
    return {"prediction_id": prediction_id, "verified_event_id": event_id,
            "identity_status": "confirmed", "spatial_status": "confirmed",
            "evidence_notes": "The actual response cells correspond to this optically verified event", **changes}


def score(predictions, reviews=(), events=(), links=(), **kwargs):
    return score_review_tables(pd.DataFrame(predictions), pd.DataFrame(reviews, columns=REVIEW_COLUMNS),
                               pd.DataFrame(events, columns=EVENT_COLUMNS), pd.DataFrame(links, columns=LINK_COLUMNS),
                               **kwargs)


def test_empty_review_is_pending_not_old_catalogue_truth_or_reviewed_unknown():
    result = score([prediction("p")])
    assert len(result) == 2
    assert set(result.review_status) == {"pending"}
    assert set(result.judgement) == {""}


def test_pending_workbook_defaults_accept_blank_occurrence():
    result = score([prediction("p")], [review("p", review_status="pending", occurrence="", evidence_notes="")])
    assert set(result.review_status) == {"pending"}
    assert set(result.judgement) == {""}


def test_matching_includes_unselected_context_and_detects_confirmed_duplicate():
    result = score([prediction("a_context", selected=False), prediction("z_selected")],
                   [review("a_context"), review("z_selected")], [event("e")],
                   [link("a_context", "e"), link("z_selected", "e")])
    assert len(result) == 2
    assert set(result.judgement) == {"F"}
    assert set(result.reason) == {"duplicate_report"}
    assert set(result.competing_prediction_id) == {"a_context"}


@pytest.mark.parametrize("tolerance_unit", ["quarter", "month"])
def test_unknown_competitor_suspends_entire_connected_component(tolerance_unit):
    result = score([prediction("a"), prediction("b", selected=False)],
                   [review("a"), review("b")], [event("e")],
                   [link("a", "e"), link("b", "e", identity_status="uncertain")],
                   tolerance_unit=tolerance_unit)
    assert set(result.judgement) == {"U"}
    assert set(result.reason) == {"unresolved_identity_space_time_or_competition"}


def test_open_context_on_other_candidate_suspends_competition():
    result = score([prediction("a"), prediction("b", selected=False)],
                   [review("a"), review("b", context_complete="no")], [event("e")],
                   [link("a", "e"), link("b", "e")])
    assert set(result.judgement) == {"U"}


def test_unsubmitted_context_cannot_claim_credit_even_with_temporary_filled_facts():
    result = score([prediction("z_selected"), prediction("a_context", selected=False)],
                   [review("z_selected"), review("a_context", review_status="pending")],
                   [event("e")], [link("z_selected", "e"), link("a_context", "e")])
    assert set(result.judgement) == {"U"}
    assert set(result.competing_prediction_id) == {""}


def test_quarter_tolerances_are_separately_matched():
    result = score([prediction("p", "2021-07")], [review("p")], [event("e", "2021-04")], [link("p", "e")])
    assert result.set_index("tolerance_quarters").judgement.to_dict() == {0: "F", 1: "T"}
    assert result.iloc[0].reason == "time_mismatch"


def test_time_coincidence_without_confirmed_spatial_identity_is_unknown():
    result = score([prediction("p")], [review("p")], [event("e")])
    assert set(result.judgement) == {"U"}
    assert set(result.reason) == {"no_confirmed_identity_and_spatial_link"}


def test_interval_uses_actual_members_and_original_open_closed_month_bounds():
    pred = prediction("p", "2021-04")
    pred["response_months"] = ["2021-03", "2021-04", "2021-05"]
    interval = event("e", "", timing_type="interval", interval_start="2021-03", interval_end="2021-05",
                     start_inclusive="no", end_inclusive="no")
    assert set(score([pred], [review("p")], [interval], [link("p", "e")]).judgement) == {"T"}
    pred["response_months"] = ["2021-03", "2021-05"]
    assert set(score([pred], [review("p")], [interval], [link("p", "e")]).judgement) == {"F"}


def test_date_conflict_represented_as_unknown_preserves_u():
    uncertain = event("e", "", timing_type="unknown", evidence_notes="Conflicting original dates retained; not resolved")
    result = score([prediction("p")], [review("p")], [uncertain], [link("p", "e")])
    assert set(result.judgement) == {"U"}


def test_no_event_requires_actual_review_and_evidence():
    result = score([prediction("p")], [review("p", occurrence="no_event")])
    assert set(result.judgement) == {"F"}
    assert set(result.reason) == {"no_event"}
    with pytest.raises(ValueError, match="evidence"):
        score([prediction("p")], [review("p", occurrence="no_event", evidence_notes="")])


def test_conflicting_time_representations_and_duplicate_event_identity_rejected():
    with pytest.raises(ValueError, match="one scoring"):
        score([prediction("p")], [review("p")], [event("e", interval_start="2021-03")])
    with pytest.raises(ValueError, match="unique"):
        score([prediction("p")], [review("p")], [event("e"), event("e", "2021-05")])


def test_cross_stand_edges_rejected():
    with pytest.raises(ValueError, match="cross stand"):
        score([prediction("p", sample_id="other")], [review("p")], [event("e")], [link("p", "e")])


def test_global_matching_maximizes_cardinality_before_error_and_lexicographic_tie():
    edges = {("e1", "a"): (1, 0), ("e1", "b"): (1, 1), ("e2", "a"): (1, 1)}
    assert _canonical_matching(edges) == {"b": "e1", "a": "e2"}
    assert _canonical_matching({("e", "z"): (1, 0), ("e", "a"): (1, 0)}) == {"a": "e"}
    assert _canonical_matching({("interval", "p"): (0, 0), ("month", "p"): (1, 1)}) == {"p": "month"}


def test_frozen_output_change_is_rejected_before_import(tmp_path):
    files = {"sampling_frame.csv": b"original frame", "block_frame.csv": b"original blocks"}
    frozen = {}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        frozen[name] = hashlib.sha256(data).hexdigest()
    (tmp_path / "sampling_config.json").write_text(json.dumps({"frozen_outputs": frozen}), encoding="utf-8")
    verify_frozen_outputs(tmp_path)
    (tmp_path / "sampling_frame.csv").write_bytes(b"changed selection")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_outputs(tmp_path)


def test_estimation_failure_preserves_all_prior_results(tmp_path):
    prior = tmp_path / "candidate_review_results.csv"
    prior.write_bytes(b"previous results")
    tables = {"candidate_reviews": pd.DataFrame(columns=REVIEW_COLUMNS),
              "verified_events": pd.DataFrame(columns=EVENT_COLUMNS),
              "candidate_event_links": pd.DataFrame(columns=LINK_COLUMNS)}

    def failed_estimation(*_args):
        raise ValueError("invalid sampling quota")

    with pytest.raises(ValueError, match="quota"):
        publish_scored_results(tmp_path, pd.DataFrame([prediction("p")]), tables, estimator=failed_estimation)
    assert prior.read_bytes() == b"previous results"
    assert not (tmp_path / "validation_summary.csv").exists()
    assert not (tmp_path / "candidate_reviews.csv").exists()


@pytest.mark.parametrize("reference, predicted, expected", [
    ("2021-12", "2022-01", ["T", "T", "T"]),
    ("2021-12", "2022-02", ["F", "T", "T"]),
    ("2021-12", "2022-03", ["F", "F", "T"]),
    ("2022-03", "2021-12", ["F", "F", "T"]),
    ("2021-01", "2021-06", ["F", "F", "F"]),
])
def test_monthly_distances_cross_year_and_are_not_quarter_distances(reference, predicted, expected):
    result = score([prediction("p", predicted)], [review("p")], [event("e", reference)],
                   [link("p", "e")], tolerance_unit="month")
    assert result.tolerance_months.tolist() == [1, 2, 3]
    assert result.tolerance_unit.eq("month").all()
    assert "tolerance_quarters" not in result
    assert result.judgement.tolist() == expected
    assert result.loc[result.judgement.eq("F"), "reason"].eq("time_mismatch").all()


def test_quarter_tolerance_can_accept_five_month_error_that_monthly_three_rejects():
    args = ([prediction("p", "2021-06")], [review("p")], [event("e", "2021-01")], [link("p", "e")])
    quarter = score(*args)
    monthly = score(*args, tolerance_unit="month")
    assert quarter.judgement.tolist() == ["F", "T"]
    assert monthly.judgement.tolist() == ["F", "F", "F"]
    assert "tolerance_unit" not in quarter  # Legacy default output schema stays intact.


def test_monthly_reference_uses_fixed_representative_not_any_response_month():
    pred = prediction("p", "2021-09")
    pred["response_months"] = ["2021-08", "2021-09", "2021-10"]
    result = score([pred], [review("p")], [event("e", "2021-06")], [link("p", "e")],
                   tolerance_unit="month")
    assert result.judgement.tolist() == ["F", "F", "T"]


def test_month_mode_rematches_all_competitors_with_one_credit_per_event():
    # At +/-1 the March report is outside both events; at +/-2 it acquires
    # the May event, while January keeps its unique January credit.
    result = score([prediction("march", "2021-03"), prediction("january", "2021-01")],
                   [review("march"), review("january")],
                   [event("jan", "2021-01"), event("may", "2021-05")],
                   [link("march", "jan"), link("march", "may"), link("january", "jan")],
                   tolerance_unit="month")
    assert result.groupby("tolerance_months").judgement.apply(lambda x: sum(x == "T")).tolist() == [1, 2, 2]
    for _, matches in result.loc[result.judgement.eq("T")].groupby("tolerance_months"):
        assert matches.verified_event_id.is_unique
    assert result.loc[result.tolerance_months.eq(2)].set_index("prediction_id").verified_event_id.to_dict() == {
        "march": "may", "january": "jan",
    }


def test_month_mode_context_wins_by_smaller_month_error_not_sample_weight():
    result = score([
        {**prediction("a_selected", "2021-07"), "sampling_weight": 100},
        {**prediction("z_context", "2021-06", selected=False), "sampling_weight": 1},
    ], [review("a_selected"), review("z_context")], [event("e", "2021-06")],
        [link("a_selected", "e"), link("z_context", "e")], tolerance_unit="month")
    assert len(result) == 3
    assert set(result.prediction_id) == {"a_selected"}
    assert result.judgement.eq("F").all()
    assert result.reason.eq("duplicate_report").all()
    assert result.competing_prediction_id.eq("z_context").all()


def test_month_mode_preserves_pending_unknown_and_identity_requirements():
    result = score([prediction("pending"), prediction("unknown"), prediction("unlinked")],
                   [review("unknown", occurrence="unknown"), review("unlinked")],
                   [event("e")], tolerance_unit="month")
    assert result.loc[result.prediction_id.eq("pending"), "review_status"].eq("pending").all()
    assert result.loc[result.prediction_id.eq("pending"), "judgement"].eq("").all()
    assert result.loc[result.prediction_id.eq("unknown"), "judgement"].eq("U").all()
    assert result.loc[result.prediction_id.eq("unlinked"), "judgement"].eq("U").all()


def test_month_mode_does_not_expand_interval_or_ignore_open_bounds():
    pred = prediction("p", "2021-04")
    pred["response_months"] = ["2021-03", "2021-05"]
    interval = event("e", "", timing_type="interval", interval_start="2021-03", interval_end="2021-05",
                     start_inclusive="no", end_inclusive="no")
    result = score([pred], [review("p")], [interval], [link("p", "e")], tolerance_unit="month")
    assert result.judgement.eq("F").all()
    pred["response_months"].append("2021-04")
    assert score([pred], [review("p")], [interval], [link("p", "e")],
                 tolerance_unit="month").judgement.eq("T").all()


def test_explicit_custom_tolerance_is_honoured_without_mutating_inputs():
    pred = prediction("p", "2021-05")
    result = score([pred], [review("p")], [event("e", "2021-04")], [link("p", "e")],
                   tolerance_unit="month", tolerances=(0, 2))
    assert result.tolerance_months.tolist() == [0, 2]
    assert result.judgement.tolist() == ["F", "T"]
    assert pred == prediction("p", "2021-05")


def test_monthly_scoring_to_estimator_preserves_unique_credit_and_unknown_denominator():
    from b20_probability_estimation import summarize_probability_results

    frame = pd.DataFrame([prediction("a", "2021-01"), prediction("b", "2021-02"), prediction("u")])
    frame = frame.assign(geo_stratum="H", block_id="B", M_h=1, m_h=1, N_bc=3, n_bc=3,
                         inclusion_probability=1.0, sampling_weight=1.0)
    result = score_review_tables(frame, pd.DataFrame([review("a"), review("b"), review("u", occurrence="unknown")]),
                                 pd.DataFrame([event("e", "2021-01")]),
                                 pd.DataFrame([link("a", "e"), link("b", "e")]), tolerance_unit="month")
    summary = summarize_probability_results(frame, result, population_size=3, tolerance_unit="month")
    assert summary.n_T.eq(1).all() and summary.n_F.eq(1).all() and summary.n_U.eq(1).all()
    assert summary.lower_hajek.to_numpy() == pytest.approx([1 / 3] * 3)
    assert summary.upper_hajek.to_numpy() == pytest.approx([2 / 3] * 3)
    assert summary.precision_hajek.isna().all()
    assert summary.ci_status.eq("population_census").all()
