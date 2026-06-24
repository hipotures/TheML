import numpy as np
import pandas as pd


BAND_COLUMNS = ("u", "g", "r", "i", "z")
WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
TEMP_COUNT = 180
MIN_TEMP = 1500.0
MAX_TEMP = 50000.0
PLANCK_COEFF = 14387.7
MAX_PLANCK_ARG = 700.0
MAG_FLOOR = 30.0
FLUX_SUM_GUARD = 1e-30
TEMP_CHUNK = 8


def _fit_planck_shape(log_t_values, shape_matrix, lam_prime, lam5_inv):
    t = np.power(10.0, log_t_values)
    arg = PLANCK_COEFF / (lam_prime * t[:, None])
    arg = np.clip(arg, a_min=None, a_max=MAX_PLANCK_ARG)
    blackbody = lam5_inv / np.expm1(arg)

    row_sum = blackbody.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    template = blackbody / row_sum

    numer = np.einsum("bi,bi->b", shape_matrix, template)
    denom = np.einsum("bi,bi->b", template, template)
    denom = np.where(denom == 0.0, 1.0, denom)

    scale = numer / denom
    residual = shape_matrix - template * scale[:, None]
    ssr = np.einsum("bi,bi->b", residual, residual)

    return template, scale, ssr


def add_blackbody_continuum_distance(raw, deps, aux):
    _ = deps, aux

    raw_mags = raw.loc[:, BAND_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    raw_mags = np.where(np.isfinite(raw_mags), raw_mags, MAG_FLOOR)
    flux = np.power(10.0, -0.4 * raw_mags)

    flux_sum = np.sum(flux, axis=1)
    shape = np.empty_like(flux)
    finite_mask = flux_sum > FLUX_SUM_GUARD
    shape[finite_mask] = flux[finite_mask] / flux_sum[finite_mask][:, None]
    if np.any(~finite_mask):
        shape[~finite_mask] = 0.2

    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    z_eff = np.where((~np.isfinite(redshift)) | (redshift <= 0.0), 0.0, redshift)

    wavelengths = np.asarray(WAVELENGTHS_ANGSTROM, dtype=np.float64)
    lam_prime = wavelengths[None, :] / (1.0 + z_eff[:, None])
    lam5_inv = np.power(lam_prime, -5.0)

    shape_sq = np.einsum("bi,bi->b", shape, shape)

    log_t_grid = np.linspace(np.log10(MIN_TEMP), np.log10(MAX_TEMP), TEMP_COUNT, dtype=np.float64)
    temp_grid = np.power(10.0, log_t_grid)

    n = raw.shape[0]
    best_ssr = np.full(n, np.inf, dtype=np.float64)
    best_idx = np.full(n, -1, dtype=np.int64)
    second_ssr = np.full(n, np.inf, dtype=np.float64)

    for start in range(0, TEMP_COUNT, TEMP_CHUNK):
        stop = min(TEMP_COUNT, start + TEMP_CHUNK)
        chunk = temp_grid[start:stop]
        arg = PLANCK_COEFF / (lam_prime[:, None, :] * chunk[None, :, :])
        arg = np.clip(arg, a_min=None, a_max=MAX_PLANCK_ARG)

        raw_template = lam5_inv[:, None, :] / np.expm1(arg)
        temp_sum = raw_template.sum(axis=2, keepdims=True)
        temp_sum = np.where(temp_sum == 0.0, 1.0, temp_sum)
        templates = raw_template / temp_sum

        template_sq = np.einsum("bci,bci->bc", templates, templates)
        template_sq = np.where(template_sq == 0.0, 1.0, template_sq)
        numer = np.einsum("bi,bci->bc", shape, templates)
        scale = numer / template_sq

        ssr = shape_sq[:, None] - 2.0 * scale * numer + (scale * scale) * template_sq
        ssr = np.where(np.isfinite(ssr), ssr, np.inf)

        local_min = ssr.min(axis=1)
        local_arg = np.argmin(ssr, axis=1).astype(np.int64)
        local_idx = start + local_arg
        if chunk.shape[0] > 1:
            local_second = np.partition(ssr, 1, axis=1)[:, 1]
        else:
            local_second = np.full(n, np.inf, dtype=np.float64)

        prev_best = best_ssr
        prev_second = second_ssr

        take_new = local_min < prev_best

        best_ssr = np.where(take_new, local_min, prev_best)
        best_idx = np.where(take_new, local_idx, best_idx)
        second_ssr = np.where(
            take_new,
            np.minimum(prev_best, local_second),
            np.minimum(prev_second, local_min),
        )

    if np.any(best_idx < 0):
        best_idx = np.where(best_idx < 0, 0, best_idx)
        best_ssr = np.where(best_ssr == np.inf, 0.0, best_ssr)

    best_log_t = log_t_grid[best_idx]
    dlog = log_t_grid[1] - log_t_grid[0]

    prev_log_t = best_log_t.copy()
    next_log_t = best_log_t.copy()
    interior = (best_idx > 0) & (best_idx < (TEMP_COUNT - 1))

    if np.any(interior):
        prev_log_t[interior] = log_t_grid[best_idx[interior] - 1]
        next_log_t[interior] = log_t_grid[best_idx[interior] + 1]

    _, _, prev_ssr = _fit_planck_shape(prev_log_t, shape, lam_prime, lam5_inv)
    _, _, next_ssr = _fit_planck_shape(next_log_t, shape, lam_prime, lam5_inv)

    curvature = np.full(n, np.nan, dtype=np.float64)
    curvature_valid = interior & np.isfinite(best_ssr) & np.isfinite(prev_ssr) & np.isfinite(next_ssr)
    curvature_num = prev_ssr - 2.0 * best_ssr + next_ssr
    curvature[curvature_valid] = np.where(
        np.abs(curvature_num[curvature_valid]) > 1e-18,
        curvature_num[curvature_valid] / (dlog * dlog),
        np.nan,
    )

    denom = prev_ssr - 2.0 * best_ssr + next_ssr
    refined_log_t = best_log_t.copy()
    refine_valid = interior & np.isfinite(denom) & (np.abs(denom) > 1e-18)
    refined_log_t[refine_valid] = np.clip(
        best_log_t[refine_valid] + 0.5 * (prev_ssr[refine_valid] - next_ssr[refine_valid]) / denom[refine_valid] * dlog,
        log_t_grid[0],
        log_t_grid[-1],
    )

    _, refined_scale, refined_ssr = _fit_planck_shape(refined_log_t, shape, lam_prime, lam5_inv)
    template_refined, _, _ = _fit_planck_shape(refined_log_t, shape, lam_prime, lam5_inv)
    residuals = shape - template_refined * refined_scale[:, None]

    features = {
        "bbcd_log10_T_star": refined_log_t,
        "bbcd_ssr_min": refined_ssr,
        "bbcd_ssr_rms": np.sqrt(np.maximum(refined_ssr, 0.0)),
        "bbcd_ssr_gap_to_second_best": second_ssr - refined_ssr,
        "bbcd_ssr_curvature_logT": curvature,
        "bbcd_residual_u": residuals[:, 0],
        "bbcd_residual_g": residuals[:, 1],
        "bbcd_residual_r": residuals[:, 2],
        "bbcd_residual_i": residuals[:, 3],
        "bbcd_residual_z": residuals[:, 4],
    }

    return pd.DataFrame(features, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "blackbody_continuum_distance",
        "fn": add_blackbody_continuum_distance,
        "depends_on": [],
        "description": "Fits a redshift-adjusted blackbody manifold in ugriz shape space and outputs fit distance diagnostics and residual shape features.",
    }
]