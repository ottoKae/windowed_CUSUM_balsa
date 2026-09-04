import numpy as np
import pandas as pd

from lr_cusum.dated_samples import temporal_reference_type


def test_temporal_reference_type_respects_locked_s1_start():
    assert temporal_reference_type(pd.Timestamp("2015-12-31")) == "pre_s1_left_censored"
    assert temporal_reference_type(pd.Timestamp("2016-01-01")) == "within_s1_reference"
    assert temporal_reference_type(pd.NaT) == "official_date_unavailable"


def test_missing_values_are_not_reinterpreted_as_dates():
    assert temporal_reference_type(None) == "official_date_unavailable"
    assert temporal_reference_type(np.nan) == "official_date_unavailable"

