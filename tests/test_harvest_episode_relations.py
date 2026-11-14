"""Mechanism checks for the isolated harvest-episode relation prototype."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "episode_relations",
    ROOT / "tooling/diagnostics/balsa-harvest-episode-20261101/episode_relations.py",
)
episode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(episode)
QUIET = "observed_no_candidate"
MISSING = "unobserved_or_ineligible"


def test_known_separator_survives_another_missing_month():
    result = episode.classify_gap([QUIET, QUIET, MISSING, QUIET, QUIET], 1)
    assert result == {"relation": "temporal_separate", "gap_months": 5,
                      "longest_quiet_run": 2, "has_unknown": True,
                      "quiet_runs": [(0, 1), (3, 4)], "unknown_offsets": [2]}
    result = episode.classify_gap([QUIET, MISSING, QUIET], 1)
    assert result["relation"] == "boundary_unknown"
    assert result["longest_quiet_run"] == 1 and result["has_unknown"]
    assert episode.classify_gap([MISSING, QUIET, QUIET], 1)["quiet_runs"] == [(1, 2)]


@pytest.mark.parametrize("states,g,expected", [
    ([QUIET], 1, "temporal_join"),
    ([QUIET, QUIET], 1, "temporal_separate"),
    ([MISSING], 1, "boundary_unknown"),
    (["outside_detection_scope"], 1, "boundary_unknown"),
    ([QUIET], 0, "temporal_separate"),
    ([MISSING], 0, "boundary_unknown"),
    ([MISSING, QUIET], 0, "temporal_separate"),
])
def test_gap_rules(states, g, expected):
    assert episode.classify_gap(states, g)["relation"] == expected


def test_one_bridge_node_removes_the_only_sts_opportunity():
    assert episode.active_phases([15, 20, 0, 18, 22]) == [(0, 1), (3, 4)]
    assert episode.active_phases([15, 20, 1, 18, 22]) == [(0, 4)]
    assert episode.active_phases([0, 0]) == []
    assert episode.active_phases([]) == []


def test_amplitude_and_baseline_are_removed_not_physical_identity_proven():
    step = np.r_[np.zeros(6), np.ones(7)]
    a = episode.build_phase_template(["a1", "a2"],
                                    [-7 - 2 * step, -10 - 5 * step])
    b = episode.build_phase_template(["b1", "b2"],
                                    [-12 - step, -4 - 3 * step])
    np.testing.assert_array_equal(a["baseline"], [-7, -10])
    np.testing.assert_array_equal(a["scale"], [2, 5])
    np.testing.assert_array_equal(a["template"], -step)
    assert episode.response_distance(a["template"], b["template"])["distance"] == 0


def test_contributors_can_change_and_cover_only_quarter_of_phase():
    step = np.r_[np.zeros(6), -np.ones(7)]
    rows = np.tile(step, (4, 1))
    rows[0, :2] = np.nan
    rows[1, 11:] = np.nan
    rows[2:] = np.nan
    result = episode.build_phase_template(["a", "b", "c", "d"], rows)
    assert result["computable"] and result["eligible_count"] == 2
    assert result["contributor_ids"][0] == ("b",)
    assert result["contributor_ids"][-1] == ("a",)
    assert result["contributor_changes"][2] == ("a",)
    assert result["contributor_changes"][11] == ("b",)
    assert result["contributor_fraction"][0] == .25
    assert result["contributor_fraction"][6] == .5
    np.testing.assert_array_equal(result["template"], step)
    np.testing.assert_array_equal(result["q25"], result["q75"])


def test_zero_scale_insufficient_months_and_single_member_are_unavailable():
    step = np.r_[np.zeros(6), -np.ones(7)]
    short = step.copy()
    short[:4] = np.nan
    result = episode.build_phase_template(["constant", "short", "valid"],
                                         [np.ones(13), short, step])
    assert result["eligible"].tolist() == [False, False, True]
    assert result["scale"][0] == 0
    assert not result["computable"] and np.isnan(result["template"]).all()
    assert not episode.build_phase_template(["one"], [step])["computable"]
    assert episode.apply_sts_veto("temporal_join", np.nan, .2) == {
        "relation": "temporal_join", "sts_status": "sts_unavailable"}


def test_template_and_contributor_identity_survive_input_reordering():
    step = np.r_[np.zeros(6), -np.ones(7)]
    values = np.stack([step, 3 * step + 4, step + .1 * np.arange(13)])
    values[0, 0] = np.nan
    original = episode.build_phase_template(["a", "b", "c"], values)
    reordered = episode.build_phase_template(["c", "a", "b"], values[[2, 0, 1]])
    np.testing.assert_array_equal(original["template"], reordered["template"])
    np.testing.assert_array_equal(original["contributor_count"], reordered["contributor_count"])
    assert [set(ids) for ids in original["contributor_ids"]] == [
        set(ids) for ids in reordered["contributor_ids"]]


def test_distance_uses_common_offsets_without_interpolating_missing_months():
    a = np.zeros(13)
    b = np.r_[np.zeros(6), np.ones(7)]
    a[[0, 8]] = np.nan
    b[[1, 9]] = np.nan
    result = episode.response_distance(a, b)
    assert result["available"]
    assert result["common_offsets"].tolist() == [-4, -3, -2, -1, 0, 1, 4, 5, 6]
    assert result["distance"] == pytest.approx(5 / 9)
    b[2:4] = np.nan
    result = episode.response_distance(a, b)
    assert not result["available"] and np.isnan(result["distance"])


def test_veto_cuts_only_join_and_infinite_threshold_recovers_b1():
    original = ["temporal_join", "temporal_separate", "boundary_unknown", "temporal_join"]
    distances = [.3, .9, .9, np.nan]
    actual = [episode.apply_sts_veto(r, d, .2)["relation"]
              for r, d in zip(original, distances)]
    assert actual == ["sts_separate", "temporal_separate", "boundary_unknown", "temporal_join"]
    disabled = [episode.apply_sts_veto(r, d, np.inf)["relation"]
                for r, d in zip(original, distances)]
    assert disabled == original
    assert episode.apply_sts_veto("temporal_join", .2, .2)["relation"] == "temporal_join"


def test_response_shape_veto_on_a_temporally_joinable_pair():
    phases = episode.active_phases([15, 20, 0, 18, 22])
    assert len(phases) == 2
    temporal = episode.classify_gap([QUIET], 1)
    persistent = np.r_[np.zeros(6), -np.ones(7)]
    recovery = np.r_[np.zeros(6), -np.ones(3), np.zeros(4)]
    a = episode.build_phase_template(["a1", "a2"], [persistent, persistent])
    b = episode.build_phase_template(["b1", "b2"], [recovery, recovery])
    comparison = episode.response_distance(a["template"], b["template"])
    assert comparison["distance"] == pytest.approx(4 / 13)
    assert episode.apply_sts_veto(temporal["relation"], comparison["distance"], .2) == {
        "relation": "sts_separate", "sts_status": "vetoed"}
    assert episode.apply_sts_veto(temporal["relation"], comparison["distance"], .4) == {
        "relation": "temporal_join", "sts_status": "retained"}


@pytest.mark.parametrize("states,g", [([], 1), (["active"], 1), (["typo"], 1),
                                      ([QUIET], -1), ([QUIET], 1.5)])
def test_invalid_gap_inputs_are_rejected(states, g):
    with pytest.raises(ValueError):
        episode.classify_gap(states, g)


def test_invalid_templates_and_duplicate_members_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        episode.build_phase_template(["a", "a"], np.ones((2, 13)))
    with pytest.raises(ValueError, match="shape"):
        episode.build_phase_template(["a"], np.ones((1, 12)))
    with pytest.raises(ValueError, match="shape"):
        episode.build_phase_template(["a"], np.full((1, 13), np.inf))
    with pytest.raises(ValueError, match="templates"):
        episode.response_distance(np.zeros(12), np.zeros(13))
    with pytest.raises(ValueError, match="nonnegative"):
        episode.apply_sts_veto("temporal_join", .1, -.1)
    with pytest.raises(ValueError, match="integer"):
        episode.active_phases([0, 1.2])
