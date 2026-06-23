import numpy as np
import pandas as pd

WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
REST_BREAK_WAVELENGTHS = (912.0, 1216.0)


def _safe_ratio(numerator, denominator, default=0.0):
    out = np.full_like(denominator, default, dtype=np.float64)
    valid = denominator != 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _fit_linear_model(x, y, w):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    wsum = np.sum(w, axis=1)
    sx = np.sum(w * x, axis=1)
    sx2 = np.sum(w * x * x, axis=1)
    sy = np.sum(w * y, axis=1)
    sxy = np.sum(w * x * y, axis=1)

    denom = wsum * sx2 - sx * sx
    slope = np.full(x.shape[0], np.nan, dtype=np.float64)
    intercept = np.full(x.shape[0], np.nan, dtype=np.float64)

    valid = (denom != 0.0) & (wsum != 0.0)
    if np.any(valid):
        slope[valid] = (wsum[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
        intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / wsum[valid]

    y_hat = intercept[:, None] + slope[:, None] * x
    y_mean = np.divide(sy, wsum, out=np.zeros_like(sy), where=wsum != 0.0)
    sse = np.sum(w * (y - y_hat) ** 2, axis=1)
    ss_tot = np.sum(w * (y - y_mean[:, None]) ** 2, axis=1)
    r2 = np.ones(x.shape[0], dtype=np.float64)
    r2 = np.divide(1.0 - sse / ss_tot, 1.0, out=r2, where=ss_tot != 0.0)

    return sse, r2, slope, intercept


def _fit_quadratic_model(x, y, w):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2

    s0 = np.sum(w, axis=1)
    s1 = np.sum(w * x, axis=1)
    s2 = np.sum(w * x2, axis=1)
    s3 = np.sum(w * x3, axis=1)
    s4 = np.sum(w * x4, axis=1)
    t0 = np.sum(w * y, axis=1)
    t1 = np.sum(w * x * y, axis=1)
    t2 = np.sum(w * x2 * y, axis=1)

    den = (
        s0 * (s2 * s4 - s3 * s3)
        - s1 * (s1 * s4 - s2 * s3)
        + s2 * (s1 * s3 - s2 * s2)
    )

    intercept = np.full(x.shape[0], np.nan, dtype=np.float64)
    slope = np.full(x.shape[0], np.nan, dtype=np.float64)
    curvature = np.full(x.shape[0], np.nan, dtype=np.float64)

    valid = (den != 0.0) & (s0 != 0.0)
    if np.any(valid):
        a = t0 * (s2 * s4 - s3 * s3) - s1 * (t1 * s4 - s2 * t2) + s2 * (t1 * s3 - s2 * t2)
        b = s0 * (t1 * s4 - s2 * t2) - t0 * (s1 * s4 - s2 * s3) + s2 * (s1 * t2 - s2 * t1)
        c = s0 * (s2 * t2 - t1 * s3) - s1 * (s1 * t2 - t0 * s3) + t0 * (s1 * s3 - s2 * s2)

        intercept[valid] = a[valid] / den[valid]
        slope[valid] = b[valid] / den[valid]
        curvature[valid] = c[valid] / den[valid]

    y_hat = intercept[:, None] + slope[:, None] * x + curvature[:, None] * x2
    y_mean = np.divide(
        t0,
        s0,
        out=np.zeros_like(t0, dtype=np.float64),
        where=s0 != 0.0,
    )
    sse = np.sum(w * (y - y_hat) ** 2, axis=1)
    ss_tot = np.sum(w * (y - y_mean[:, None]) ** 2, axis=1)
    r2 = np.ones(x.shape[0], dtype=np.float64)
    r2 = np.divide(1.0 - sse / ss_tot, 1.0, out=r2, where=ss_tot != 0.0)

    return sse, r2, intercept, slope, curvature


def add_restframe_sed_family_fit(raw, deps, aux):
    mags = raw[["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float64)
    redshift = raw["redshift"].to_numpy(dtype=np.float64)

    z_plus = np.maximum(redshift, 0.0)
    wavelengths = np.asarray(WAVELENGTHS_ANGSTROM, dtype=np.float64)
    rest_lambda = wavelengths[None, :] / (1.0 + z_plus[:, None])

    x = np.log10(rest_lambda)
    y = -0.4 * mags

    weights = np.ones_like(x, dtype=np.float64)
    weights[rest_lambda <= REST_BREAK_WAVELENGTHS[1]] = 0.5
    weights[rest_lambda <= REST_BREAK_WAVELENGTHS[0]] = 0.2

    linear_sse, linear_r2, linear_slope, linear_intercept = _fit_linear_model(x, y, np.ones_like(y, dtype=np.float64))
    quad_sse, quad_r2, quad_intercept, quad_slope, quad_curv = _fit_quadratic_model(
        x, y, np.ones_like(y, dtype=np.float64)
    )
    linear_gain = _safe_ratio(linear_sse - quad_sse, linear_sse, default=0.0)

    wx_linear_sse, wx_linear_r2, wx_linear_slope, wx_linear_intercept = _fit_linear_model(x, y, weights)
    wx_quad_sse, wx_quad_r2, wx_quad_intercept, wx_quad_slope, wx_quad_curv = _fit_quadratic_model(x, y, weights)
    weighted_gain = _safe_ratio(wx_linear_sse - wx_quad_sse, wx_linear_sse, default=0.0)

    slope_contrast = (y[:, 0] - y[:, 1]) / (x[:, 0] - x[:, 1]) - (y[:, 3] - y[:, 4]) / (x[:, 3] - x[:, 4])

    weighted_fallback = np.all(weights == 1.0, axis=1)

    return pd.DataFrame(
        {
            "restframe_sed_linear_sse": linear_sse,
            "restframe_sed_linear_r2": linear_r2,
            "restframe_sed_linear_slope": linear_slope,
            "restframe_sed_linear_intercept": linear_intercept,
            "restframe_sed_quad_sse": quad_sse,
            "restframe_sed_quad_r2": quad_r2,
            "restframe_sed_quad_curvature": quad_curv,
            "restframe_sed_quad_slope": quad_slope,
            "restframe_sed_quad_intercept": quad_intercept,
            "restframe_sed_curvature_gain": linear_gain,
            "restframe_sed_endpoint_slope_contrast": slope_contrast,
            "restframe_sed_weighted_linear_sse": wx_linear_sse,
            "restframe_sed_weighted_linear_r2": wx_linear_r2,
            "restframe_sed_weighted_linear_slope": wx_linear_slope,
            "restframe_sed_weighted_linear_intercept": wx_linear_intercept,
            "restframe_sed_weighted_quad_sse": wx_quad_sse,
            "restframe_sed_weighted_quad_r2": wx_quad_r2,
            "restframe_sed_weighted_quad_curvature": wx_quad_curv,
            "restframe_sed_weighted_quad_slope": wx_quad_slope,
            "restframe_sed_weighted_quad_intercept": wx_quad_intercept,
            "restframe_sed_weighted_curvature_gain": weighted_gain,
            "restframe_sed_weighted_fallback": weighted_fallback,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "restframe_sed_family_fit",
        "fn": add_restframe_sed_family_fit,
        "depends_on": [],
        "description": "Fits linear and quadratic rest-frame log-flux continuua in ugriz to derive curvature, fit residual, and slope-contrast diagnostics with an Lyman-weighted variant.",
    }
]