"""Official Earth Engine CCDC from Python, using isolated virtual pixels.

Each one-metre virtual pixel represents one frozen stand-track sequence.
No real geometry, new SAR product, interpolation, or zero-valued imputation.
The caller owns credentials, provenance, bounded batching and cached responses.
"""
import numpy as np
import pandas as pd


def virtual_collection(values, dates, bands):
    import ee
    values = np.asarray(values, dtype=np.float64)
    dates = pd.DatetimeIndex(dates)
    if values.ndim != 3 or values.shape[1:] != (len(bands), len(dates)):
        raise ValueError("expected trajectory x band x month")
    if len(set(bands)) != len(bands) or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("duplicate bands or unordered dates")
    if len(values) == 0 or len(dates) == 0:
        raise ValueError("empty input")
    projection = ee.Projection("EPSG:3857", [1, 0, 0, 0, -1, 0])
    coordinates = ee.Image.pixelCoordinates(projection).floor().toInt()
    x = coordinates.select("x")
    domain = coordinates.select("y").eq(0).And(x.gte(0)).And(x.lt(len(values)))
    indices = list(range(len(values)))
    images = []
    for t, date in enumerate(dates):
        layers = []
        for b, name in enumerate(bands):
            valid = np.isfinite(values[:, b, t])
            # The masked backing value is zero, but is NEVER a finite observation.
            v = np.where(valid, values[:, b, t], 0).tolist()
            layer = x.remap(indices, v, 0).toDouble().rename(name)
            mask = x.remap(indices, valid.astype(int).tolist(), 0).And(domain)
            layers.append(layer.updateMask(mask))
        image = ee.Image.cat(layers).setDefaultProjection(projection)
        images.append(image.set("system:time_start", int(date.value // 1_000_000)))
    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([i+0.5, -0.5], "EPSG:3857"), {"trajectory_index": i})
        for i in indices
    ])
    return ee.ImageCollection.fromImages(images), points, projection


def read_pixels(image, points, projection):
    """reduceRegions retains keys for entirely masked trajectories."""
    import ee
    response = image.reduceRegions(
        collection=points, reducer=ee.Reducer.first(), crs=projection, scale=1, tileScale=4,
    ).getInfo()
    rows = [f["properties"] for f in response["features"]]
    return sorted(rows, key=lambda r: r["trajectory_index"])


def run_ccdc(values, dates, bands=("VV",), *, chi=0.99, observations=6,
             regularization=0.0):
    import ee
    collection, points, projection = virtual_collection(values, dates, bands)
    result = ee.Algorithms.TemporalSegmentation.Ccdc(**{
        "collection": collection, "breakpointBands": list(bands),
        "minObservations": int(observations), "chiSquareProbability": float(chi),
        "minNumOfYearsScaler": 1.33, "dateFormat": 2,
        "lambda": float(regularization), "maxIterations": 25000,
    })
    return read_pixels(result, points, projection)


def segment_candidates(response, *, direction_band="VV", cutoff=None):
    """Keep all segments, including unconfirmed tails. Time format is Unix ms."""
    rows = []
    for record in response:
        breaks = record.get("tBreak") or []
        for j, value in enumerate(breaks):
            def field(name, default=np.nan):
                a = record.get(name)
                return a[j] if a is not None and j < len(a) else default
            probability = float(field("changeProb", 0))
            month = pd.to_datetime(value, unit="ms").to_period("M").to_timestamp() if value > 0 else pd.NaT
            magnitude = float(field(direction_band+"_magnitude"))
            rows.append({
                "trajectory_index": int(record["trajectory_index"]), "segment": j,
                "raw_tBreak": value, "change_time": month,
                "change_probability": probability, "magnitude": magnitude,
                "confirmed": bool(value > 0 and probability == 1.0),
                "decrease": bool(magnitude < 0), "n_obs": field("numObs"),
                "raw_tStart": field("tStart"), "raw_tEnd": field("tEnd"),
                "available_through_month": pd.Timestamp(cutoff) if cutoff is not None else pd.NaT,
            })
    return pd.DataFrame(rows)
