import numpy as np
import pandas as pd


PIVOT_WAVELENGTHS_ANGSTROM = {
    "u": 3562.0,
    "g": 4686.0,
    "r": 6165.0,
    "i": 7481.0,
    "z": 8931.0,
}


def _numeric_series(raw, column):
    return pd.to_numeric(raw[column], errors="coerce")


def _mask_nonfinite(values, *operands):
    finite = np.ones(len(values), dtype=bool)
    for operand in operands:
        finite &= np.isfinite(operand.to_numpy(dtype="float64", copy=False))
    return pd.Series(np.where(finite, values, np.nan), index=operands[0].index)


def _safe_diff(left, right):
    values = left.to_numpy(dtype="float64", copy=False) - right.to_numpy(dtype="float64", copy=False)
    return _mask_nonfinite(values, left, right)


def _safe_second_diff(left, middle, right):
    values = (
        left.to_numpy(dtype="float64", copy=False)
        - 2.0 * middle.to_numpy(dtype="float64", copy=False)
        + right.to_numpy(dtype="float64", copy=False)
    )
    return _mask_nonfinite(values, left, middle, right)


def _safe_slope(color, left, right, spacing):
    values = color.to_numpy(dtype="float64", copy=False) / spacing
    return _mask_nonfinite(values, left, right)


def _safe_curvature(right_slope, left_slope, left, middle, right, spacing):
    values = (
        right_slope.to_numpy(dtype="float64", copy=False)
        - left_slope.to_numpy(dtype="float64", copy=False)
    ) / spacing
    return _mask_nonfinite(values, left, middle, right)


def add_broadband_color_shape(raw, deps, aux):
    u = _numeric_series(raw, "u")
    g = _numeric_series(raw, "g")
    r = _numeric_series(raw, "r")
    i = _numeric_series(raw, "i")
    z = _numeric_series(raw, "z")

    log_u = np.log(PIVOT_WAVELENGTHS_ANGSTROM["u"])
    log_g = np.log(PIVOT_WAVELENGTHS_ANGSTROM["g"])
    log_r = np.log(PIVOT_WAVELENGTHS_ANGSTROM["r"])
    log_i = np.log(PIVOT_WAVELENGTHS_ANGSTROM["i"])
    log_z = np.log(PIVOT_WAVELENGTHS_ANGSTROM["z"])

    color_u_g = _safe_diff(u, g)
    color_g_r = _safe_diff(g, r)
    color_r_i = _safe_diff(r, i)
    color_i_z = _safe_diff(i, z)

    slope_u_g = _safe_slope(color_u_g, u, g, log_g - log_u)
    slope_g_r = _safe_slope(color_g_r, g, r, log_r - log_g)
    slope_r_i = _safe_slope(color_r_i, r, i, log_i - log_r)
    slope_i_z = _safe_slope(color_i_z, i, z, log_z - log_i)

    features = pd.DataFrame(index=raw.index)

    features["color_u_g"] = color_u_g
    features["color_g_r"] = color_g_r
    features["color_r_i"] = color_r_i
    features["color_i_z"] = color_i_z
    features["color_u_r"] = _safe_diff(u, r)
    features["color_u_i"] = _safe_diff(u, i)
    features["color_u_z"] = _safe_diff(u, z)
    features["color_g_i"] = _safe_diff(g, i)
    features["color_g_z"] = _safe_diff(g, z)
    features["color_r_z"] = _safe_diff(r, z)

    features["slope_u_g"] = slope_u_g
    features["slope_g_r"] = slope_g_r
    features["slope_r_i"] = slope_r_i
    features["slope_i_z"] = slope_i_z

    features["curve_u_g_r"] = _safe_second_diff(u, g, r)
    features["curve_g_r_i"] = _safe_second_diff(g, r, i)
    features["curve_r_i_z"] = _safe_second_diff(r, i, z)

    features["k_u_g_r"] = _safe_curvature(
        slope_g_r,
        slope_u_g,
        u,
        g,
        r,
        0.5 * (log_r - log_u),
    )
    features["k_g_r_i"] = _safe_curvature(
        slope_r_i,
        slope_g_r,
        g,
        r,
        i,
        0.5 * (log_i - log_g),
    )
    features["k_r_i_z"] = _safe_curvature(
        slope_i_z,
        slope_r_i,
        r,
        i,
        z,
        0.5 * (log_z - log_r),
    )

    features["curve_u_r_z"] = _safe_second_diff(u, r, z)

    return features


FEATURE_GROUPS = [
    {
        "name": "broadband_color_shape",
        "fn": add_broadband_color_shape,
        "depends_on": [],
        "description": "Observed-frame ugriz color, slope, and curvature descriptors of broadband SED shape.",
    }
]