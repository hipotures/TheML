import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_WAVELENGTH_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_TEMP_MIN_K = 1500.0
_TEMP_MAX_K = 50000.0
_TEMP_POINTS = 180
_ROW_CHUNK_SIZE = 8192
_TEMP_CHUNK_SIZE = 45
_PLANCK_C2_ANGSTROM_K = 143877000.0
_EXP_MIN = 0.00000001
_EXP_MAX = 700.0
_MAG_FILL_VALUE = 30.0
_MIN_FLUX = 0.000000000001
_MAX_REDSHIFT = 7.5
_UNIFORM_SHAPE_VALUE = 0.2
_MARGIN_EXCLUDE_RADIUS = 2


def _shape_normalized_flux(magnitudes):
    values = np.array(magnitudes, dtype=np.float64, copy=True)
    np.nan_to_num(
        values,
        copy=False,
        nan=_MAG_FILL_VALUE,
        posinf=_MAG_FILL_VALUE,
        neginf=_MAG_FILL_VALUE,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        flux = np.power(10.0, -0.4 * values)

    flux = np.maximum(flux, _MIN_FLUX)
    sums = np.sum(flux, axis=1)

    shape = np.empty_like(flux)
    valid = np.isfinite(sums) & (sums > 0.0)
    shape[valid] = flux[valid] / sums[valid, None]
    shape[~valid] = _UNIFORM_SHAPE_VALUE
    return shape


def _safe_redshift(redshift):
    values = np.array(redshift, dtype=np.float64, copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=_MAX_REDSHIFT, neginf=0.0)
    return np.clip(values, 0.0, _MAX_REDSHIFT)


def _normalized_planck_grid(rest_wavelengths, temperatures):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        exponent = _PLANCK_C2_ANGSTROM_K / (
            rest_wavelengths[:, None, :] * temperatures[None, :, None]
        )
        np.clip(exponent, _EXP_MIN, _EXP_MAX, out=exponent)
        planck = np.power(rest_wavelengths, -5.0)[:, None, :] / np.expm1(exponent)

    sums = np.sum(planck, axis=2, keepdims=True)
    templates = np.full_like(planck, _UNIFORM_SHAPE_VALUE)
    valid = np.isfinite(sums) & (sums > 0.0)
    np.divide(planck, sums, out=templates, where=valid)
    return templates


def _normalized_planck_best(rest_wavelengths, temperatures):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        exponent = _PLANCK_C2_ANGSTROM_K / (rest_wavelengths * temperatures[:, None])
        np.clip(exponent, _EXP_MIN, _EXP_MAX, out=exponent)
        planck = np.power(rest_wavelengths, -5.0) / np.expm1(exponent)

    sums = np.sum(planck, axis=1, keepdims=True)
    templates = np.full_like(planck, _UNIFORM_SHAPE_VALUE)
    valid = np.isfinite(sums) & (sums > 0.0)
    np.divide(planck, sums, out=templates, where=valid)
    return templates


def add_blackbody_continuum_distance(raw, deps, aux):
    n_rows = len(raw)
    band_columns = list(_BANDS)
    wavelengths = np.asarray(_WAVELENGTH_ANGSTROM, dtype=np.float64)
    temperatures = np.geomspace(_TEMP_MIN_K, _TEMP_MAX_K, _TEMP_POINTS)
    log_temperatures = np.log10(temperatures)

    log_lambda = np.log(wavelengths)
    centered_log_lambda = log_lambda - np.mean(log_lambda)
    slope_weights = centered_log_lambda / np.sum(centered_log_lambda * centered_log_lambda)
    temperature_indices = np.arange(_TEMP_POINTS)

    best_log10_temperature = np.empty(n_rows, dtype=np.float64)
    ssr_min = np.empty(n_rows, dtype=np.float64)
    sqrt_ssr_min = np.empty(n_rows, dtype=np.float64)
    residual_values = np.empty((n_rows, len(_BANDS)), dtype=np.float64)
    max_abs_residual = np.empty(n_rows, dtype=np.float64)
    residual_log_wavelength_slope = np.empty(n_rows, dtype=np.float64)
    confidence_margin = np.empty(n_rows, dtype=np.float64)
    local_curvature = np.empty(n_rows, dtype=np.float64)

    raw_bands = raw.loc[:, band_columns]
    raw_redshift = raw["redshift"]

    for row_start in range(0, n_rows, _ROW_CHUNK_SIZE):
        row_end = min(row_start + _ROW_CHUNK_SIZE, n_rows)
        chunk_len = row_end - row_start

        shape = _shape_normalized_flux(
            raw_bands.iloc[row_start:row_end].to_numpy(dtype=np.float64, copy=True)
        )
        z_eff = _safe_redshift(
            raw_redshift.iloc[row_start:row_end].to_numpy(dtype=np.float64, copy=True)
        )
        rest_wavelengths = wavelengths[None, :] / (1.0 + z_eff[:, None])

        ssr_grid = np.empty((chunk_len, _TEMP_POINTS), dtype=np.float64)

        for temp_start in range(0, _TEMP_POINTS, _TEMP_CHUNK_SIZE):
            temp_end = min(temp_start + _TEMP_CHUNK_SIZE, _TEMP_POINTS)
            templates = _normalized_planck_grid(
                rest_wavelengths, temperatures[temp_start:temp_end]
            )
            templates -= shape[:, None, :]
            ssr_grid[:, temp_start:temp_end] = np.einsum(
                "ijk,ijk->ij", templates, templates, optimize=True
            )

        best_idx = np.argmin(ssr_grid, axis=1)
        row_idx = np.arange(chunk_len)
        best_ssr = ssr_grid[row_idx, best_idx]

        best_templates = _normalized_planck_best(rest_wavelengths, temperatures[best_idx])
        residuals = shape - best_templates

        excluded = (
            np.abs(temperature_indices[None, :] - best_idx[:, None])
            <= _MARGIN_EXCLUDE_RADIUS
        )
        outside_min = np.min(np.where(excluded, np.inf, ssr_grid), axis=1)
        margin = outside_min - best_ssr
        margin = np.where(np.isfinite(margin), np.maximum(margin, 0.0), 0.0)

        curvature = np.zeros(chunk_len, dtype=np.float64)
        interior = (best_idx > 0) & (best_idx < (_TEMP_POINTS - 1))
        interior_rows = row_idx[interior]
        interior_idx = best_idx[interior]
        curvature[interior] = (
            ssr_grid[interior_rows, interior_idx - 1]
            - 2.0 * ssr_grid[interior_rows, interior_idx]
            + ssr_grid[interior_rows, interior_idx + 1]
        )

        best_log10_temperature[row_start:row_end] = log_temperatures[best_idx]
        ssr_min[row_start:row_end] = best_ssr
        sqrt_ssr_min[row_start:row_end] = np.sqrt(best_ssr)
        residual_values[row_start:row_end, :] = residuals
        max_abs_residual[row_start:row_end] = np.max(np.abs(residuals), axis=1)
        residual_log_wavelength_slope[row_start:row_end] = residuals @ slope_weights
        confidence_margin[row_start:row_end] = margin
        local_curvature[row_start:row_end] = curvature

    return pd.DataFrame(
        {
            "best_log10_temperature": best_log10_temperature,
            "ssr_min": ssr_min,
            "sqrt_ssr_min": sqrt_ssr_min,
            "residual_u": residual_values[:, 0],
            "residual_g": residual_values[:, 1],
            "residual_r": residual_values[:, 2],
            "residual_i": residual_values[:, 3],
            "residual_z": residual_values[:, 4],
            "max_abs_residual": max_abs_residual,
            "residual_log_wavelength_slope": residual_log_wavelength_slope,
            "confidence_margin": confidence_margin,
            "local_curvature": local_curvature,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "blackbody_continuum_distance",
        "fn": add_blackbody_continuum_distance,
        "depends_on": [],
        "description": "Fits normalized ugriz continua to redshift-corrected blackbody templates and emits distance and residual-shape features.",
    }
]