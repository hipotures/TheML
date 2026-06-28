import numpy as np
import pandas as pd


BAND_COLUMNS = ("u", "g", "r", "i", "z")
BAND_WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
LYMAN_LIMIT_ANGSTROM = 912.0
LYMAN_ALPHA_ANGSTROM = 1216.0
EPSILON = 1.0e-12


def _weighted_polyfit_metrics(x0, y0, weights, degree):
    n_rows = y0.shape[0]
    cols = [np.ones_like(x0), x0]
    if degree >= 2:
        cols.append(x0 * x0)
    design = np.stack(cols, axis=2)

    valid = np.isfinite(y0) & np.isfinite(x0) & np.isfinite(weights) & (weights > EPSILON)
    w = np.where(valid, weights, 0.0)
    yw = np.where(valid, y0, 0.0)

    xtw = np.transpose(design * w[:, :, None], (0, 2, 1))
    xtwx = np.matmul(xtw, design)
    xtwy = np.sum(design * (w * yw)[:, :, None], axis=1)

    coef = np.full((n_rows, degree + 1), np.nan, dtype=float)
    fitted = np.full_like(y0, np.nan, dtype=float)
    feasible = np.zeros(n_rows, dtype=bool)

    for row_idx in range(n_rows):
        rank = np.linalg.matrix_rank(xtwx[row_idx], tol=1.0e-10)
        if rank >= degree + 1:
            try:
                coef[row_idx] = np.linalg.solve(xtwx[row_idx], xtwy[row_idx])
                feasible[row_idx] = True
            except np.linalg.LinAlgError:
                coef[row_idx] = np.linalg.lstsq(xtwx[row_idx], xtwy[row_idx], rcond=None)[0]
                feasible[row_idx] = True

    if feasible.any():
        fitted[feasible] = np.sum(design[feasible] * coef[feasible, None, :], axis=2)

    residual = np.where(valid, y0 - fitted, np.nan)
    rss = np.sum(np.where(valid, w * residual * residual, 0.0), axis=1)

    weight_sum = np.sum(w, axis=1)
    y_weighted_mean = np.divide(
        np.sum(w * yw, axis=1),
        weight_sum,
        out=np.full(n_rows, np.nan, dtype=float),
        where=weight_sum > EPSILON,
    )
    tss = np.sum(np.where(valid, w * (yw - y_weighted_mean[:, None]) ** 2, 0.0), axis=1)
    r2 = np.where(tss > EPSILON, 1.0 - rss / np.maximum(tss, EPSILON), np.nan)

    support = np.sum(valid, axis=1)
    dof = np.maximum(support - (degree + 1), 1)
    resid_std = np.sqrt(rss / dof)

    standardized = np.divide(
        np.abs(residual),
        np.maximum(resid_std[:, None], EPSILON),
        out=np.full_like(residual, np.nan),
        where=np.isfinite(residual),
    )
    max_abs_std_resid = np.nanmax(np.where(valid, standardized, np.nan), axis=1)

    slope = coef[:, 1]
    curvature = coef[:, 2] if degree >= 2 else np.full(n_rows, np.nan, dtype=float)

    rss = np.where(feasible, rss, np.nan)
    r2 = np.where(feasible, r2, np.nan)
    resid_std = np.where(feasible, resid_std, np.nan)
    max_abs_std_resid = np.where(feasible, max_abs_std_resid, np.nan)
    slope = np.where(feasible, slope, np.nan)
    curvature = np.where(feasible, curvature, np.nan)

    return {
        "rss": rss,
        "r2": r2,
        "resid_std": resid_std,
        "max_abs_std_resid": max_abs_std_resid,
        "slope": slope,
        "curvature": curvature,
        "feasible": feasible,
    }


def _safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPSILON),
    )


def _shape_contrasts(y0, weights):
    pair_support = {
        "ug": weights[:, 0] + weights[:, 1],
        "gr": weights[:, 1] + weights[:, 2],
        "ri": weights[:, 2] + weights[:, 3],
        "iz": weights[:, 3] + weights[:, 4],
    }

    ug_ok = pair_support["ug"] > EPSILON
    gr_ok = pair_support["gr"] > EPSILON
    ri_ok = pair_support["ri"] > EPSILON
    iz_ok = pair_support["iz"] > EPSILON

    blue_slope = np.where(ug_ok, y0[:, 0] - y0[:, 1], np.nan)
    red_slope = np.where(iz_ok, y0[:, 3] - y0[:, 4], np.nan)
    endpoint_slope_contrast = np.where(ug_ok & iz_ok, blue_slope - red_slope, np.nan)
    middle_curvature_contrast = np.where(
        gr_ok & ri_ok,
        (y0[:, 1] - y0[:, 2]) - (y0[:, 2] - y0[:, 3]),
        np.nan,
    )

    return {
        "blue_slope": blue_slope,
        "red_slope": red_slope,
        "endpoint_slope_contrast": endpoint_slope_contrast,
        "middle_curvature_contrast": middle_curvature_contrast,
        "blue_slope_support": ug_ok,
        "red_slope_support": iz_ok,
        "endpoint_slope_support": ug_ok & iz_ok,
        "middle_curvature_support": gr_ok & ri_ok,
    }


def _add_branch_features(out, branch_name, x0, y0, weights):
    linear = _weighted_polyfit_metrics(x0, y0, weights, 1)
    quadratic = _weighted_polyfit_metrics(x0, y0, weights, 2)
    shape = _shape_contrasts(y0, weights)

    gain = _safe_ratio(linear["rss"] - quadratic["rss"], np.maximum(linear["rss"], EPSILON))
    gain = np.clip(gain, -1.0, 1.0)

    out[f"{branch_name}_linear_rss"] = linear["rss"]
    out[f"{branch_name}_quadratic_rss"] = quadratic["rss"]
    out[f"{branch_name}_linear_r2"] = linear["r2"]
    out[f"{branch_name}_quadratic_r2"] = quadratic["r2"]
    out[f"{branch_name}_linear_slope"] = linear["slope"]
    out[f"{branch_name}_quadratic_slope"] = quadratic["slope"]
    out[f"{branch_name}_curvature"] = quadratic["curvature"]
    out[f"{branch_name}_abs_curvature"] = np.abs(quadratic["curvature"])
    out[f"{branch_name}_signed_curvature"] = quadratic["curvature"]
    out[f"{branch_name}_curvature_gain"] = gain
    out[f"{branch_name}_linear_resid_std"] = linear["resid_std"]
    out[f"{branch_name}_quadratic_resid_std"] = quadratic["resid_std"]
    out[f"{branch_name}_linear_max_abs_std_resid"] = linear["max_abs_std_resid"]
    out[f"{branch_name}_quadratic_max_abs_std_resid"] = quadratic["max_abs_std_resid"]
    out[f"{branch_name}_linear_feasible"] = linear["feasible"]
    out[f"{branch_name}_quadratic_feasible"] = quadratic["feasible"]

    for key, value in shape.items():
        out[f"{branch_name}_{key}"] = value

    return linear, quadratic, gain, shape


def add_restframe_sed_family_fit(raw, deps, aux):
    index = raw.index
    mags = raw.loc[:, BAND_COLUMNS].astype(float).to_numpy(copy=True)
    wavelengths = np.asarray(BAND_WAVELENGTHS_ANGSTROM, dtype=float)

    redshift = raw["redshift"].astype(float).to_numpy(copy=True)
    zc = np.maximum(redshift, 0.0)
    z_low_flag = redshift < 0.0

    rest_wavelengths = wavelengths[None, :] / (1.0 + zc[:, None])
    x = np.log10(rest_wavelengths)
    x0 = x - np.mean(x, axis=1, keepdims=True)

    y = -0.4 * mags
    y0 = y - np.mean(y, axis=1, keepdims=True)

    base_weights = np.ones_like(y0, dtype=float)
    lyman_weights = np.where(
        rest_wavelengths > LYMAN_ALPHA_ANGSTROM,
        1.0,
        np.where(rest_wavelengths > LYMAN_LIMIT_ANGSTROM, 0.5, 0.15),
    )

    no_uv_downweight = np.all(lyman_weights == 1.0, axis=1)

    out = pd.DataFrame(index=index)
    out["z_low_flag"] = z_low_flag
    out["no_uv_downweight"] = no_uv_downweight
    out["min_rest_wavelength"] = np.min(rest_wavelengths, axis=1)
    out["max_rest_wavelength"] = np.max(rest_wavelengths, axis=1)
    out["n_limited_lyman_alpha_bands"] = np.sum(rest_wavelengths <= LYMAN_ALPHA_ANGSTROM, axis=1)
    out["n_limited_lyman_limit_bands"] = np.sum(rest_wavelengths <= LYMAN_LIMIT_ANGSTROM, axis=1)
    out["mean_lyman_weight"] = np.mean(lyman_weights, axis=1)

    base_linear, base_quadratic, base_gain, base_shape = _add_branch_features(
        out, "base", x0, y0, base_weights
    )
    lyman_linear, lyman_quadratic, lyman_gain, lyman_shape = _add_branch_features(
        out, "lyman", x0, y0, lyman_weights
    )

    out["lyman_instability"] = (
        (~lyman_linear["feasible"])
        | (~lyman_quadratic["feasible"])
        | (np.sum(lyman_weights > EPSILON, axis=1) < 4)
        | (np.min(lyman_weights, axis=1) <= 0.15)
    )

    out["diff_quadratic_r2"] = lyman_quadratic["r2"] - base_quadratic["r2"]
    out["diff_linear_r2"] = lyman_linear["r2"] - base_linear["r2"]
    out["diff_quadratic_rss_ratio"] = _safe_ratio(lyman_quadratic["rss"], base_quadratic["rss"])
    out["diff_linear_rss_ratio"] = _safe_ratio(lyman_linear["rss"], base_linear["rss"])
    out["diff_quadratic_slope"] = lyman_quadratic["slope"] - base_quadratic["slope"]
    out["diff_linear_slope"] = lyman_linear["slope"] - base_linear["slope"]
    out["diff_curvature"] = lyman_quadratic["curvature"] - base_quadratic["curvature"]
    out["diff_abs_curvature"] = np.abs(lyman_quadratic["curvature"]) - np.abs(base_quadratic["curvature"])
    out["diff_curvature_gain"] = lyman_gain - base_gain
    out["diff_endpoint_slope_contrast"] = (
        lyman_shape["endpoint_slope_contrast"] - base_shape["endpoint_slope_contrast"]
    )
    out["diff_quadratic_max_abs_std_resid"] = (
        lyman_quadratic["max_abs_std_resid"] - base_quadratic["max_abs_std_resid"]
    )
    out["diff_linear_max_abs_std_resid"] = (
        lyman_linear["max_abs_std_resid"] - base_linear["max_abs_std_resid"]
    )

    return out


FEATURE_GROUPS = [
    {
        "name": "restframe_sed_family_fit",
        "fn": add_restframe_sed_family_fit,
        "depends_on": [],
        "description": "Fits unweighted and Lyman-aware rest-frame ugriz continuum families and returns shape residual diagnostics.",
    }
]