import numpy as np
import pandas as pd

_BANDS = ("u", "g", "r", "i", "z")
_BAND_CENTERS = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
_BAND_SPAN_EDGES = (3000.0, 4100.0, 5500.0, 6900.0, 8200.0, 9200.0)

_LINE_DEFINITIONS = (
    ("ca_k", 3933.7),
    ("ca_h", 3968.5),
    ("g_band", 4304.0),
    ("h_delta", 4102.0),
    ("h_gamma", 4341.0),
    ("h_beta", 4861.0),
    ("mg_b", 5175.0),
    ("na_d", 5893.0),
)

_BLUE_LINES = ("ca_k", "ca_h", "g_band", "h_delta", "h_gamma")
_RED_LINES = ("h_beta", "mg_b", "na_d")

def _sanitize_magnitudes(values, fallback_values=None):
    arr = np.asarray(values, dtype=float)
    invalid = (~np.isfinite(arr)) | (arr < -99.0) | (arr > 90.0)
    baseline = arr[~invalid]
    if fallback_values is not None:
        fallback = np.asarray(fallback_values, dtype=float)
        fallback_invalid = (~np.isfinite(fallback)) | (fallback < -99.0) | (fallback > 90.0)
        fallback_valid = fallback[~fallback_invalid]
        if fallback_valid.size:
            baseline = np.concatenate((baseline, fallback_valid)) if baseline.size else fallback_valid
    if baseline.size == 0:
        replacement = 0.0
    else:
        replacement = np.nanmedian(baseline)
    if not np.isfinite(replacement):
        replacement = 0.0
    return np.where(invalid, replacement, arr).astype(float)

def _robust_loglinear_predict(log_lam, log_flux_rows, log_lam_obs):
    log_lam = np.asarray(log_lam, dtype=float)
    y = np.asarray(log_flux_rows, dtype=float)
    n_rows = y.shape[0]
    n_pts = y.shape[1]

    sx = float(np.sum(log_lam))
    sx2 = float(np.sum(log_lam * log_lam))
    denom = n_pts * sx2 - sx * sx

    sy = np.nansum(y, axis=1)
    sxy = np.nansum(y * log_lam, axis=1)

    slope = np.divide(n_pts * sxy - sx * sy, denom, out=np.zeros(n_rows), where=np.abs(denom) > 1e-18)
    intercept = np.divide(sy - slope * sx, n_pts, out=np.zeros(n_rows), where=n_pts > 0)

    fit = slope[:, None] * log_lam + intercept[:, None]
    resid = y - fit
    mad = np.nanmedian(np.abs(resid), axis=1)
    scale = 1.4826 * mad
    has_scale = np.isfinite(scale) & (scale > 0.0)

    if np.any(has_scale):
        t = np.zeros_like(resid)
        t[has_scale] = resid[has_scale] / (4.685 * scale[has_scale, None])
        w = (1.0 - t * t) ** 2
        w = np.where(np.abs(t) < 1.0, w, 0.0)
        w = np.where(np.isnan(w), 0.0, w)
        w = np.where(has_scale[:, None], w, 1.0)

        sw = np.sum(w, axis=1)
        sxw = np.sum(w * log_lam, axis=1)
        sx2w = np.sum(w * log_lam * log_lam, axis=1)
        syw = np.sum(w * y, axis=1)
        sxyw = np.sum(w * y * log_lam, axis=1)

        denom_w = sw * sx2w - sxw * sxw
        slope_w = np.divide(sw * sxyw - sxw * syw, denom_w, out=slope, where=np.abs(denom_w) > 1e-12)
        intercept_w = np.divide(syw - slope_w * sxw, sw, out=intercept, where=np.abs(sw) > 1e-12)

        slope = np.where(has_scale, slope_w, slope)
        intercept = np.where(has_scale, intercept_w, intercept)

    return slope * log_lam_obs + intercept

def _estimate_continuum(log_lam, log_flux_rows, lam_obs, band_idx):
    log_lam_obs = np.log(np.clip(lam_obs, 1e-300, None))
    cont = _robust_loglinear_predict(log_lam, log_flux_rows, log_lam_obs)

    interior = (band_idx >= 1) & (band_idx <= (len(log_lam) - 2))
    if np.any(interior):
        rows = np.nonzero(interior)[0]
        left = band_idx[rows] - 1
        right = band_idx[rows] + 1

        x_obs = log_lam_obs[rows]
        x_left = log_lam[left]
        x_right = log_lam[right]
        y_left = log_flux_rows[rows, left]
        y_right = log_flux_rows[rows, right]

        cont[rows] = y_left + (x_obs - x_left) * (y_right - y_left) / (x_right - x_left)

    return cont

def add_redshifted_absorption_trough_residuals(raw, deps, aux):
    raw_index = raw.index
    n = len(raw)
    if n == 0:
        return pd.DataFrame(index=raw_index)

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    redshift = np.where(np.isfinite(redshift), redshift, 0.0)

    aux_bands = {}
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        for band in _BANDS:
            if band in aux.columns:
                aux_bands[band] = pd.to_numeric(aux[band], errors="coerce").to_numpy(dtype=float)

    band_flux_cols = []
    for band in _BANDS:
        band_flux_cols.append(np.power(10.0, -0.4 * _sanitize_magnitudes(raw[band], aux_bands.get(band))))
    flux_matrix = np.column_stack(band_flux_cols)
    log_flux_matrix = np.log(np.clip(flux_matrix, 1e-300, None))

    band_centers = np.asarray(_BAND_CENTERS, dtype=float)
    log_band_centers = np.log(band_centers)
    band_edges = np.asarray(_BAND_SPAN_EDGES, dtype=float)

    line_deficits = {}
    line_excess = {}
    line_visible = {}

    for line_name, rest_wave in _LINE_DEFINITIONS:
        lam_obs = rest_wave * (1.0 + redshift)
        lam_obs = np.where(np.isfinite(lam_obs), lam_obs, 0.0)

        band_idx = np.digitize(lam_obs, band_edges, right=True) - 1
        visible = (band_idx >= 0) & (band_idx < len(band_centers))
        band_idx_safe = np.where(visible, band_idx, 0)

        log_cont = _estimate_continuum(log_band_centers, log_flux_matrix, lam_obs, band_idx_safe)

        band_flux = np.zeros(n, dtype=float)
        if np.any(visible):
            row_idx = np.arange(n, dtype=int)
            band_flux[visible] = flux_matrix[row_idx[visible], band_idx_safe[visible]]

        residual = (np.exp(log_cont) - band_flux) / (band_flux + 1e-12)
        residual = np.where(visible, residual, 0.0)

        line_visible[line_name] = visible.astype(np.int8)
        line_deficits[line_name] = np.where(residual > 0, residual, 0.0)
        line_excess[line_name] = np.where(residual < 0, -residual, 0.0)

    all_line_names = tuple(name for name, _ in _LINE_DEFINITIONS)
    all_def_mat = np.column_stack([line_deficits[name] for name in all_line_names])
    all_ex_mat = np.column_stack([line_excess[name] for name in all_line_names])

    blue_def_mat = np.column_stack([line_deficits[name] for name in _BLUE_LINES])
    blue_vis_mat = np.column_stack([line_visible[name].astype(float) for name in _BLUE_LINES])
    red_def_mat = np.column_stack([line_deficits[name] for name in _RED_LINES])
    red_vis_mat = np.column_stack([line_visible[name].astype(float) for name in _RED_LINES])

    blue_count = blue_vis_mat.sum(axis=1)
    red_count = red_vis_mat.sum(axis=1)

    total_blue_abs_deficit = np.divide(
        blue_def_mat.sum(axis=1),
        blue_count,
        out=np.zeros(n, dtype=float),
        where=blue_count > 0,
    )
    total_red_abs_deficit = np.divide(
        red_def_mat.sum(axis=1),
        red_count,
        out=np.zeros(n, dtype=float),
        where=red_count > 0,
    )

    total_deficit = all_def_mat.sum(axis=1)
    total_excess = all_ex_mat.sum(axis=1)
    absorption_excess_ratio = np.divide(
        total_excess,
        total_deficit + 1e-6,
        out=np.zeros(n, dtype=float),
        where=total_deficit > 0,
    )

    metal_trough_balance = total_blue_abs_deficit - total_red_abs_deficit
    line_family_contrast = (
        (line_deficits["ca_k"] + line_deficits["ca_h"]) -
        (line_deficits["h_delta"] + line_deficits["h_gamma"] + line_deficits["g_band"])
    )

    regimes = (
        ("lt_0p4", redshift < 0.4),
        ("ge_0p4_lt_1p2", (redshift >= 0.4) & (redshift < 1.2),
        ),
        ("ge_1p2_lt_2p0", (redshift >= 1.2) & (redshift < 2.0)),
        ("ge_2p0", redshift >= 2.0),
    )

    features = {}

    for line_name in all_line_names:
        features[f"absorption_{line_name}_visible"] = line_visible[line_name]
        features[f"absorption_{line_name}_deficit"] = line_deficits[line_name]
        features[f"absorption_{line_name}_excess"] = line_excess[line_name]

    features["total_blue_abs_deficit"] = total_blue_abs_deficit
    features["total_red_abs_deficit"] = total_red_abs_deficit
    features["absorption_excess_ratio"] = absorption_excess_ratio
    features["metal_trough_balance"] = metal_trough_balance
    features["line_family_contrast"] = line_family_contrast

    for label, mask in regimes:
        features[f"z_regime_{label}"] = mask.astype(np.int8)
        m = mask.astype(float)

        blue_sum_r = (blue_def_mat * m[:, None]).sum(axis=1)
        red_sum_r = (red_def_mat * m[:, None]).sum(axis=1)
        blue_count_r = (blue_vis_mat * m[:, None]).sum(axis=1)
        red_count_r = (red_vis_mat * m[:, None]).sum(axis=1)

        blue_r = np.divide(
            blue_sum_r,
            blue_count_r,
            out=np.zeros(n, dtype=float),
            where=blue_count_r > 0,
        )
        red_r = np.divide(
            red_sum_r,
            red_count_r,
            out=np.zeros(n, dtype=float),
            where=red_count_r > 0,
        )

        ex_sum_r = total_excess * m
        def_sum_r = total_deficit * m
        ratio_r = np.divide(
            ex_sum_r,
            def_sum_r + 1e-6,
            out=np.zeros(n, dtype=float),
            where=def_sum_r > 0,
        )

        family_r = (
            (line_deficits["ca_k"] + line_deficits["ca_h"] -
             line_deficits["h_delta"] - line_deficits["h_gamma"] - line_deficits["g_band"]) * m
        )
        balance_r = blue_r - red_r

        features[f"regime_blue_abs_deficit_{label}"] = blue_r
        features[f"regime_red_abs_deficit_{label}"] = red_r
        features[f"regime_absorption_excess_ratio_{label}"] = ratio_r
        features[f"regime_line_family_contrast_{label}"] = family_r
        features[f"regime_metal_trough_balance_{label}"] = balance_r

    return pd.DataFrame(features, index=raw_index)

FEATURE_GROUPS = [
    {
        "name": "redshifted_absorption_trough_residuals",
        "fn": add_redshifted_absorption_trough_residuals,
        "depends_on": [],
        "description": "Generate redshift-aware broadband residual features around known absorption-sensitive wavelengths in ugriz space.",
    },
]