import numpy as np
import pandas as pd

_BAND_COLUMNS = ("u", "g", "r", "i", "z")
_PIVOT_WAVELENGTHS_ANGSTROM = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
_TRAIN_QUANTILE_LOW = 0.001
_TRAIN_QUANTILE_HIGH = 0.999
_FALLBACK_MAG_MIN = -1.0
_FALLBACK_MAG_MAX = 35.0
_XTX_COND_LIMIT = 1.0e12


def _sanitize_band(raw_series):
    values = pd.to_numeric(raw_series, errors="coerce").to_numpy(dtype=np.float64, copy=True)

    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]

    if finite_values.size == 0:
        lower, upper = _FALLBACK_MAG_MIN, _FALLBACK_MAG_MAX
        median = 0.5 * (lower + upper)
    else:
        finite_series = pd.Series(finite_values)
        median = float(finite_series.median())
        lower = float(finite_series.quantile(_TRAIN_QUANTILE_LOW))
        upper = float(finite_series.quantile(_TRAIN_QUANTILE_HIGH))
        if not np.isfinite(lower) or not np.isfinite(upper):
            lower, upper = _FALLBACK_MAG_MIN, _FALLBACK_MAG_MAX
        if lower > upper:
            lower, upper = _FALLBACK_MAG_MIN, _FALLBACK_MAG_MAX
        if not np.isfinite(median):
            median = 0.5 * (lower + upper)

    values[~finite_mask] = median
    return np.clip(values, lower, upper)


def add_observed_sed_continuum_moments(raw, deps, aux):
    index = raw.index
    n_rows = len(raw)

    slope = np.zeros(n_rows, dtype=np.float64)
    curvature = np.zeros(n_rows, dtype=np.float64)
    residual_rms = np.zeros(n_rows, dtype=np.float64)
    residual_mae = np.zeros(n_rows, dtype=np.float64)
    residual_max_abs = np.zeros(n_rows, dtype=np.float64)
    residual_u = np.zeros(n_rows, dtype=np.float64)
    residual_z = np.zeros(n_rows, dtype=np.float64)
    blue_red_break_delta = np.zeros(n_rows, dtype=np.float64)

    if n_rows == 0:
        return pd.DataFrame(
            {
                "sed_slope_quad": slope,
                "sed_curvature_quad": curvature,
                "sed_resid_rms": residual_rms,
                "sed_resid_mae": residual_mae,
                "sed_resid_max_abs": residual_max_abs,
                "sed_resid_u": residual_u,
                "sed_resid_z": residual_z,
                "sed_break_delta_blue_minus_red": blue_red_break_delta,
            },
            index=index,
        )

    band_matrix = np.column_stack([
        _sanitize_band(raw[column]) for column in _BAND_COLUMNS
    ])

    log_flux = -0.4 * band_matrix
    centered_flux = log_flux - log_flux.mean(axis=1, keepdims=True)

    pivots = np.array(_PIVOT_WAVELENGTHS_ANGSTROM, dtype=np.float64)
    t = np.log10(pivots)
    t = t - t.mean()
    t2 = t * t

    xTx = np.array([
        [5.0, np.sum(t), np.sum(t2)],
        [np.sum(t), np.sum(t2), np.sum(t * t2)],
        [np.sum(t2), np.sum(t * t2), np.sum(t2 * t2)],
    ], dtype=np.float64)

    cond_xTx = np.linalg.cond(xTx)
    if not np.isfinite(cond_xTx) or cond_xTx > _XTX_COND_LIMIT:
        return pd.DataFrame(
            {
                "sed_slope_quad": slope,
                "sed_curvature_quad": curvature,
                "sed_resid_rms": residual_rms,
                "sed_resid_mae": residual_mae,
                "sed_resid_max_abs": residual_max_abs,
                "sed_resid_u": residual_u,
                "sed_resid_z": residual_z,
                "sed_break_delta_blue_minus_red": blue_red_break_delta,
            },
            index=index,
        )

    inv_xTx = np.linalg.inv(xTx)

    moments0 = np.sum(centered_flux, axis=1)
    moments1 = centered_flux.dot(t)
    moments2 = centered_flux.dot(t2)
    rhs = np.column_stack((moments0, moments1, moments2))

    coef = rhs.dot(inv_xTx.T)
    beta0 = coef[:, 0]
    beta1 = coef[:, 1]
    beta2 = coef[:, 2]

    fit = beta0[:, None] + beta1[:, None] * t + beta2[:, None] * t2
    residuals = centered_flux - fit

    row_ok = np.isfinite(beta1) & np.isfinite(beta2)
    row_ok &= np.isfinite(residuals).all(axis=1)

    edge_delta = (centered_flux[:, 0] - centered_flux[:, 1]) / (t[0] - t[1])
    edge_delta -= (centered_flux[:, 2] - centered_flux[:, 3]) / (t[2] - t[3])

    row_ok &= np.isfinite(edge_delta)
    row_ok &= np.isfinite(beta0)

    if np.any(row_ok):
        slope[row_ok] = beta1[row_ok]
        curvature[row_ok] = beta2[row_ok]
        residual_row = residuals[row_ok]
        residual_rms[row_ok] = np.sqrt(np.mean(residual_row * residual_row, axis=1))
        residual_mae[row_ok] = np.mean(np.abs(residual_row), axis=1)
        residual_max_abs[row_ok] = np.max(np.abs(residual_row), axis=1)
        residual_u[row_ok] = residual_row[:, 0]
        residual_z[row_ok] = residual_row[:, 4]
        blue_red_break_delta[row_ok] = edge_delta[row_ok]

    return pd.DataFrame(
        {
            "sed_slope_quad": slope,
            "sed_curvature_quad": curvature,
            "sed_resid_rms": residual_rms,
            "sed_resid_mae": residual_mae,
            "sed_resid_max_abs": residual_max_abs,
            "sed_resid_u": residual_u,
            "sed_resid_z": residual_z,
            "sed_break_delta_blue_minus_red": blue_red_break_delta,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": "observed_sed_continuum_moments",
        "fn": add_observed_sed_continuum_moments,
        "depends_on": [],
        "description": "Compute robust relative SED shape descriptors by fitting a centered quadratic continuum over SDSS bands and summarizing quadratic slope/curvature and residual structure.",
    }
]