"""Official annual LandTrendr, oriented to VV decreases; no pseudo-month dates.

API: https://developers.google.com/earth-engine/apidocs/ee-algorithms-temporalsegmentation-landtrendr
Raw fitted values are model outputs, never substituted for missing observations.
"""
import numpy as np
import pandas as pd

from .ccdc_adapter import virtual_collection, read_pixels


def run_landtrendr(annual_vv, years, parameters):
    import ee
    x = np.asarray(annual_vv, dtype=float)
    years = np.asarray(years, dtype=int)
    if x.ndim != 2 or x.shape[1] != len(years) or np.any(np.diff(years) <= 0):
        raise ValueError("Expected track x distinct ordered annual observations")
    dates = pd.to_datetime([f"{y}-07-01" for y in years])
    collection, points, projection = virtual_collection(-x[:, None, :], dates, ("negative_VV",))
    result = ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=collection, **parameters)
    return read_pixels(result, points, projection)


def parse_response(response, annual_vv, years):
    """Validate every response row before creating all segment/episode records.

Adjacent positive segments form one declining episode. Recovery or a flat segment
ends it. Short gaps are NOT converted to extra observations or exact month dates.
"""
    if sorted(int(r["trajectory_index"]) for r in response) != list(range(len(annual_vv))):
        raise ValueError("Incomplete/duplicate LandTrendr trajectory response")
    segments, episodes, support = [], [], []
    for r in response:
        i = int(r["trajectory_index"])
        observed = np.isfinite(annual_vv[i])
        matrix = r.get("LandTrendr")
        if matrix is None or len(matrix) == 0:
            if observed.sum() >= 6:
                raise ValueError(f"LandTrendr missing fit for eligible trajectory {i}")
            support.append({"trajectory_index": i, "fit_status": "insufficient_annual_observations"})
            continue
        a = np.asarray(matrix, dtype=float)
        if a.ndim != 2 or a.shape[0] != 4 or a.shape[1] < 2 or not np.isfinite(a).all():
            raise ValueError(f"Invalid LandTrendr matrix for {i}: {a.shape}")
        np.testing.assert_array_equal(a[0], np.asarray(years)[observed])
        np.testing.assert_allclose(a[1], -annual_vv[i, observed], rtol=1e-7, atol=1e-7)
        if not np.isin(a[3], [0, 1]).all():
            raise ValueError("Invalid vertex indicator")
        v = a[:, a[3] == 1]
        if v.shape[1] < 2:
            raise ValueError("Fitted trajectory has fewer than two vertices")
        support.append({"trajectory_index": i, "fit_status": "fitted", "n_obs": a.shape[1],
                        "n_vertices": v.shape[1], "rmse": r.get("rmse")})
        current = None
        for j in range(v.shape[1] - 1):
            start, end = int(v[0, j]), int(v[0, j + 1])
            magnitude = float(v[2, j + 1] - v[2, j])
            seg = {"trajectory_index": i, "segment": j, "start_year": start, "end_year": end,
                   "start_vv": -v[2, j], "end_vv": -v[2, j + 1],
                   "magnitude_db": magnitude, "duration_years": end - start,
                   "rate_db_per_year": magnitude / (end - start)}
            segments.append(seg)
            if magnitude > 1e-10:
                if current is None:
                    current = {**seg, "component_segments": [j], "event_year": start + 1}
                else:
                    current.update(end_year=end, end_vv=seg["end_vv"])
                    current["magnitude_db"] += magnitude
                    current["duration_years"] = end - current["start_year"]
                    current["component_segments"].append(j)
            elif current is not None:
                episodes.append(current)
                current = None
        if current is not None:
            episodes.append(current)
    columns = ["trajectory_index", "event_year", "magnitude_db", "duration_years", "component_segments"]
    ep = pd.DataFrame(episodes) if episodes else pd.DataFrame(columns=columns)
    if len(ep):
        ep["rate_db_per_year"] = ep.magnitude_db / ep.duration_years
    return pd.DataFrame(segments), ep, pd.DataFrame(support)


def episode_events(episodes, meta, magnitude, duration, report_years=(2018, 2025)):
    selected = episodes.loc[episodes.magnitude_db.ge(magnitude)
                            & episodes.duration_years.le(duration)
                            & episodes.event_year.between(*report_years)].copy()
    if selected.empty:
        return pd.DataFrame(columns=["sample_id", "event_year", "supporting_tracks", "n_tracks"])
    selected = selected.merge(meta.reset_index(names="trajectory_index")[["trajectory_index", "sample_id", "track_key"]],
                              on="trajectory_index", validate="many_to_one")
    return selected.groupby(["sample_id", "event_year"], as_index=False).agg(
        supporting_tracks=("track_key", lambda x: sorted(set(x))), n_tracks=("track_key", "nunique"))
