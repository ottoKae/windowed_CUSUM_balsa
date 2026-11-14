"""A long detection envelope cannot retrieve two reference events."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/experiments/evaluate_reports.py"
SPEC = importlib.util.spec_from_file_location("public_report_evaluation", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_one_to_one_overlap_and_adjacent_contrast(tmp_path):
    refs = tmp_path / "refs.csv"
    pd.DataFrame(dict(event_id=["e1", "e2"], sample_id=["demo", "demo"],
                      reference_month=["2020-01", "2020-07"])).to_csv(refs, index=False)
    common = dict(sample_id=["demo", "demo"], prediction_id=["p1", "p2"],
                  first_response_month=["2020-02", "2020-06"], last_response_month=["2020-02", "2020-06"],
                  representative_month=["2020-02", "2020-06"])
    two = tmp_path / "two.csv"
    pd.DataFrame(common).to_csv(two, index=False)
    one = tmp_path / "one.csv"
    pd.DataFrame(dict(sample_id=["demo"], prediction_id=["long"], first_response_month=["2019-12"],
                      last_response_month=["2020-08"], representative_month=["2020-04"])).to_csv(one, index=False)
    result = module.score(refs, [("long", one), ("two", two)], tmp_path / "out")
    assert [r["n_matches"] for r in result] == [1, 2]
    assert [r["n_original_pairs_retained"] for r in result] == [0, 1]
    assert result[1]["catalogue_f1"] == 1


@pytest.mark.parametrize("first,last,representative", [
    ("2020-03", "2020-01", "2020-02"),
    (None, "2020-03", "2020-02"),
    ("2020-01", "2020-03", "2020-04"),
])
def test_invalid_report_dates_are_rejected(tmp_path, first, last, representative):
    refs = tmp_path / "refs.csv"
    pd.DataFrame([dict(event_id="e1", sample_id="demo", reference_month="2020-02")]).to_csv(refs, index=False)
    reports = tmp_path / "reports.csv"
    pd.DataFrame([dict(prediction_id="p1", sample_id="demo", first_response_month=first,
                      last_response_month=last, representative_month=representative)]).to_csv(reports, index=False)
    with pytest.raises(ValueError, match="Report"):
        module.score(refs, [("invalid", reports)], tmp_path / "out")


def test_empty_reports_can_be_scored(tmp_path):
    refs = tmp_path / "refs.csv"
    pd.DataFrame([dict(event_id="e1", sample_id="demo", reference_month="2020-02")]).to_csv(refs, index=False)
    reports = tmp_path / "reports.csv"
    pd.DataFrame(columns=["prediction_id", "sample_id", "first_response_month", "last_response_month",
                          "representative_month"]).to_csv(reports, index=False)
    result = module.score(refs, [("empty", reports)], tmp_path / "out")
    assert result[0]["n_reports"] == result[0]["n_matches"] == 0
    assert result[0]["recall"] == 0
