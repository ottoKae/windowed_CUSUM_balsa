"""Import-safe detector baselines reproduced from ``LR-CuSum.ipynb`` cell 118-122."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def _student_t_logpdf(x, df, loc, scale):
    z = (x - loc) / scale
    return (
        gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
        - 0.5 * np.log(df * np.pi) - np.log(scale)
        - 0.5 * (df + 1.0) * np.log1p(z * z / df)
    )


def fpol_bocd_detect(vv, vh, *, hazard=1 / 50, delta_m=10, min_obs=12):
    """Notebook f-pol BOCD with factorized VV/VH Student-t predictives."""
    vv = np.asarray(vv, dtype=float); vh = np.asarray(vh, dtype=float)
    valid = np.isfinite(vv) & np.isfinite(vh)
    if valid.sum() < 2 * min_obs:
        return []
    vv_clean, vh_clean = vv[valid], vh[valid]
    valid_index = np.flatnonzero(valid); length = len(vv_clean)
    maximum_run = length

    def arrays(first):
        mu = np.full(maximum_run + 1, np.nan)
        kappa = np.full(maximum_run + 1, np.nan)
        alpha = np.full(maximum_run + 1, np.nan)
        beta = np.full(maximum_run + 1, np.nan)
        mu[0], kappa[0], alpha[0], beta[0] = first, .01, .001, .01
        return mu, kappa, alpha, beta

    vv_state = arrays(vv_clean[0]); vh_state = arrays(vh_clean[0])
    posterior = np.zeros(maximum_run + 1); posterior[0] = 1.0
    log_hazard, log_growth = np.log(hazard), np.log(1.0 - hazard)
    previous_map = 0; changes = []
    for time in range(length):
        predictive = np.full(time + 2, -np.inf)
        active = np.flatnonzero(np.isfinite(vv_state[0][: time + 1]))
        if len(active):
            terms = []
            for state, value in ((vv_state, vv_clean[time]), (vh_state, vh_clean[time])):
                mu, kappa, alpha, beta = state
                degrees = 2.0 * alpha[active]
                scale = np.sqrt(beta[active] * (kappa[active] + 1.0)
                                / (alpha[active] * kappa[active]))
                terms.append(_student_t_logpdf(value, degrees, mu[active], scale))
            predictive[active] = terms[0] + terms[1]

        log_new = np.full(time + 2, -np.inf)
        # Preserve the notebook implementation exactly for reproducibility.
        log_new[0] = log_hazard + predictive[0]
        growing = np.flatnonzero(posterior[: time + 1] > 0)
        log_new[growing + 1] = log_growth + np.log(posterior[growing]) + predictive[growing]
        finite = np.isfinite(log_new)
        if not finite.any():
            break
        log_new -= np.logaddexp.reduce(log_new[finite])
        posterior[: time + 2] = np.exp(log_new)
        posterior[time + 2 :] = 0.0

        updated = []
        for state, value, prior in (
            (vv_state, vv_clean[time], vv_clean[0]),
            (vh_state, vh_clean[time], vh_clean[0]),
        ):
            mu, kappa, alpha, beta = state
            new_mu = np.full(maximum_run + 1, np.nan)
            new_kappa = np.full(maximum_run + 1, np.nan)
            new_alpha = np.full(maximum_run + 1, np.nan)
            new_beta = np.full(maximum_run + 1, np.nan)
            new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = prior, .01, .001, .01
            if len(active):
                old_kappa = kappa[active]; next_kappa = old_kappa + 1.0
                new_mu[active + 1] = (old_kappa * mu[active] + value) / next_kappa
                new_kappa[active + 1] = next_kappa
                new_alpha[active + 1] = alpha[active] + .5
                new_beta[active + 1] = (
                    beta[active] + old_kappa * (value - mu[active]) ** 2 / (2.0 * next_kappa)
                )
            updated.append((new_mu, new_kappa, new_alpha, new_beta))
        vv_state, vh_state = updated
        map_run = int(np.argmax(posterior[: time + 2]))
        if time >= min_obs and map_run < previous_map - delta_m:
            changes.append(int(valid_index[time - map_run]))
        previous_map = map_run
    return sorted(set(changes))


def _harmonic_matrix(time, period=12, order=1):
    columns = []
    for harmonic in range(1, order + 1):
        columns.extend((
            np.sin(2 * np.pi * harmonic * time / period),
            np.cos(2 * np.pi * harmonic * time / period),
        ))
    return np.column_stack(columns) if columns else np.empty((len(time), 0))


def _fit_seasonal_trend(time, values, period=12, order=1):
    design = np.column_stack((np.ones(len(time)), time, _harmonic_matrix(time, period, order)))
    try:
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, np.inf
    residuals = values - design @ coefficients
    rss = float(np.sum(residuals ** 2)); parameters = design.shape[1]
    bic = len(time) * np.log(rss / len(time)) + parameters * np.log(len(time)) if rss > 0 else -np.inf
    return residuals, bic


def _mosum_breakpoint(residuals, fraction=.15):
    length = len(residuals); width = max(int(length * fraction), 3)
    if width >= length // 2:
        return None
    scale = float(np.std(residuals, ddof=1))
    if scale < 1e-10:
        return None
    scores = np.asarray([
        abs(residuals[index : index + width].sum()) / (scale * np.sqrt(width))
        for index in range(length - 2 * width + 1)
    ])
    boundary = 1.96 * np.sqrt(1 + .5 * np.log(length / width))
    maximum = int(np.argmax(scores))
    return maximum + width if scores[maximum] > boundary else None


def bfast_detect_seasonal(values, *, fraction=.15, period=12, order=1, bic_improvement=2):
    """Notebook BFAST-style 0-break versus 1-break seasonal/trend detector."""
    values = np.asarray(values, dtype=float); valid = np.isfinite(values)
    if valid.sum() < 24:
        return None
    clean = values[valid]; original = np.flatnonzero(valid)
    time = np.arange(len(clean), dtype=float)
    residuals, bic_zero = _fit_seasonal_trend(time, clean, period, order)
    if residuals is None:
        return None
    point = _mosum_breakpoint(residuals, fraction)
    minimum = int(len(clean) * fraction)
    if point is None or point < minimum or point > len(clean) - minimum:
        return None
    _, bic_left = _fit_seasonal_trend(time[:point], clean[:point], period, order)
    _, bic_right = _fit_seasonal_trend(
        time[point:] - time[point], clean[point:], period, order
    )
    return int(original[point]) if bic_zero - (bic_left + bic_right) > bic_improvement else None


def cusum_single(values, *, confidence=.95, bootstrap=800, seed=20):
    """Notebook single-break absolute-CUSUM permutation test."""
    values = np.asarray(values, dtype=float); valid = np.isfinite(values)
    if valid.sum() < 12:
        return None, None
    clean = values[valid]; original = np.flatnonzero(valid)
    cumulative = np.cumsum(clean - clean.mean())
    local = int(np.argmax(np.abs(cumulative))); amplitude = float(abs(cumulative[local]))
    if amplitude < 1e-10:
        return None, None
    rng = np.random.RandomState(seed); below = 0
    for _ in range(bootstrap):
        shuffled = clean[rng.permutation(len(clean))]
        null = float(np.max(np.abs(np.cumsum(shuffled - shuffled.mean()))))
        below += amplitude > null
    level = below / bootstrap
    return (int(original[local]), level) if level >= confidence else (None, None)


def recusum_detect(
    values,
    *,
    confidence=.95,
    bootstrap=800,
    seed=20,
    minimum_segment=12,
    maximum_depth=10,
):
    """Notebook recursive CUSUM with bounded recursion and deterministic seeds."""
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).sum() < 2 * minimum_segment or maximum_depth <= 0:
        return []
    point, _ = cusum_single(values, confidence=confidence, bootstrap=bootstrap, seed=seed)
    if point is None:
        return []
    # The notebook permits edge splits. Guard only against a non-shrinking recursion.
    left = recusum_detect(
        values[:point], confidence=confidence, bootstrap=bootstrap, seed=seed + point,
        minimum_segment=minimum_segment, maximum_depth=maximum_depth - 1,
    ) if point > 0 else []
    right = recusum_detect(
        values[point:], confidence=confidence, bootstrap=bootstrap, seed=seed + point + 1,
        minimum_segment=minimum_segment, maximum_depth=maximum_depth - 1,
    ) if point < len(values) - 1 else []
    return sorted(set(left + [point] + [point + value for value in right]))


def is_decrease(values, index: int, radius: int = 6) -> bool:
    """Apply a common harvest-direction check around a candidate index."""
    values = np.asarray(values, dtype=float)
    before = values[max(0, index - radius) : index]
    after = values[index : min(len(values), index + radius)]
    if not np.isfinite(before).any() or not np.isfinite(after).any():
        return False
    return bool(np.nanmedian(after) < np.nanmedian(before))

