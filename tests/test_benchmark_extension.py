"""Small offline regressions for the actual cloud response conventions."""
import numpy as np
import pandas as pd
import pytest
from windowed_cusum.benchmarks.structural_breakpoint_adapter import structural_events, YEAR_MS
from windowed_cusum.benchmarks.ewmacd_adapter import ewma_trace


def test_live_structural_calendar_and_jump():
    dates=pd.date_range('2016-01-01',periods=48,freq='MS')
    t=1970+dates.asi8/1e6/YEAR_MS
    record=dict(trajectory_index=0,tStart=[t[0],t[24]],tEnd=[t[23],t[-1]],
                VV_coefs=[[-8.,0.,0.,0.],[-12.,0.,0.,0.]])
    row=structural_events([record],dates).iloc[0]
    assert row.change_time==pd.Timestamp('2018-01-01')
    assert row.decrease and row.magnitude==-4
    record['tStart'][1]+=1/365.25
    with pytest.raises(ValueError,match='calendar date'):
        structural_events([record],dates)


def test_missing_ewma_months_are_not_compressed_calendar():
    dates=pd.date_range('2020-01-01',periods=4,freq='MS')
    values=np.array([[[-8.,np.nan,-12.,-12.]]])
    records=[dict(trajectory_index=0,ewma=[0,-1,-2],residuals=[0,-4,-4])]
    frame=ewma_trace(records,values,dates)
    assert list(frame.month)==list(dates[[0,2,3]])
    records[0]['ewma']=[0,0,-1,-2]
    with pytest.raises(ValueError,match='length/mask'):
        ewma_trace(records,values,dates)


def test_cloud_masked_fit_is_not_no_event():
    with pytest.raises(ValueError,match='Masked'):
        structural_events([dict(trajectory_index=0)],pd.date_range('2020-01-01',periods=3,freq='MS'))
