"""Official GEE multiple-break fits; no interpolation or inferred cloud failures."""
import numpy as np
import pandas as pd

from .ccdc_adapter import virtual_collection, read_pixels

YEAR_MS = 365.25 * 86400000


def run_structural(values, dates, *, order=0, spacing=12, breaks=5):
    import ee
    collection, points, projection = virtual_collection(values, dates, ['VV'])
    result = ee.Algorithms.TemporalSegmentation.StructuralChangeBreakpoints(
        collection=collection, breakpointBand='VV', seasonalModelOrder=order,
        minSpacing=spacing, maxBreaks=breaks, dateFormat=1)
    return read_pixels(result, points, projection)


def structural_events(records, dates):
    """First observed month of the new segment; fitted jump at the SAME time.

    The live API returns 1970 + Unix milliseconds / 365.25-day years, despite
    its documentation saying years since 1970. This convention is tested.
    Snap only floating-point timestamp error, never across a missing month.
    """
    dates = pd.DatetimeIndex(dates)
    axis = 1970 + dates.asi8 / 1e6 / YEAR_MS
    rows = []
    for item in records:
        starts = np.asarray(item.get('tStart') or [], dtype=float)
        ends = np.asarray(item.get('tEnd') or [], dtype=float)
        coefs = np.asarray(item.get('VV_coefs') or [], dtype=float)
        if not len(starts):
            raise ValueError(f'Masked structural fit: {item["trajectory_index"]}')
        if len(starts) != len(ends) or coefs.ndim != 2 or len(coefs) != len(starts):
            raise ValueError('Malformed structural segments')
        for j in range(1, len(starts)):
            t = starts[j]
            k = int(np.argmin(abs(axis - t)))
            if abs(axis[k] - t) * 365.25 > .01:
                raise ValueError('Segment boundary does not map to an observed calendar date')
            basis = [1., t]
            for harmonic in range(1, (coefs.shape[1]-2)//2 + 1):
                basis.extend([np.cos(2*np.pi*harmonic*t), np.sin(2*np.pi*harmonic*t)])
            before, after = coefs[j-1] @ basis, coefs[j] @ basis
            rows.append(dict(trajectory_index=int(item['trajectory_index']), segment=j,
                             change_time=dates[k], magnitude=float(after-before),
                             decrease=bool(after < before), pre_fitted=float(before),
                             post_fitted=float(after), preceding_end=float(ends[j-1]),
                             raw_tStart=float(t), event_semantics='first_observed_new_segment'))
    return pd.DataFrame(rows, columns=['trajectory_index','segment','change_time','magnitude',
        'decrease','pre_fitted','post_fitted','preceding_end','raw_tStart','event_semantics'])
