"""Official EWMACD; explicit finite-date mapping validated by cloud preflight."""
import numpy as np
import pandas as pd
from windowed_cusum.benchmarks.ccdc_adapter import virtual_collection, read_pixels


def run_ewmacd(values, dates, *, history_end, weight=.3, persistence=1):
    import ee
    if np.nanmin(values) <= -100:
        raise ValueError('VV input outside the declared non-vegetation-mask bound')
    collection, points, projection = virtual_collection(values, dates, ['VV'])
    result=ee.Algorithms.TemporalSegmentation.Ewmacd(**dict(
        timeSeries=collection,vegetationThreshold=-100,trainingStartYear=dates[0].year,
        trainingEndYear=history_end,harmonicCount=1,xBarLimit1=3,xBarLimit2=20,
        lambdasigs=3,rounding=True,persistence=persistence), **{'lambda':weight})
    return read_pixels(result,points,projection)


def ewma_trace(records, values, dates):
    dates=pd.DatetimeIndex(dates)
    rows=[]
    for r in records:
        i=int(r['trajectory_index']); mask=np.isfinite(values[i,0])
        scores=r.get('ewma') or []; residuals=r.get('residuals') or []
        if len(scores)!=int(mask.sum()) or len(residuals)!=len(scores):
            raise ValueError(f'EWMACD length/mask mismatch for trajectory {i}')
        rows.extend(dict(trajectory_index=i,month=d,score=s,residual=v)
                    for d,s,v in zip(dates[mask],scores,residuals))
    return pd.DataFrame(rows)
