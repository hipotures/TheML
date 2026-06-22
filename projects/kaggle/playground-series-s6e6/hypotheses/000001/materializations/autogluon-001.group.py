import numpy as np
import pandas as pd


def _safe_subtract(raw: pd.DataFrame, left: str, right: str) -> pd.Series:
    left_col = raw[left]
    right_col = raw[right]
    valid = np.isfinite(left_col) & np.isfinite(right_col)
    result = pd.Series(np.nan, index=raw.index, dtype=float)
    result.loc[valid] = left_col.loc[valid] - right_col.loc[valid]
    return result


def add_broadband_color_shape(raw, deps, aux):
    u = raw["u"]
    g = raw["g"]
    r = raw["r"]
    i = raw["i"]
    z = raw["z"]

    color_u_g = _safe_subtract(raw, "u", "g")
    color_g_r = _safe_subtract(raw, "g", "r")
    color_r_i = _safe_subtract(raw, "r", "i")
    color_i_z = _safe_subtract(raw, "i", "z")
    color_u_r = _safe_subtract(raw, "u", "r")
    color_g_i = _safe_subtract(raw, "g", "i")
    color_r_z = _safe_subtract(raw, "r", "z")
    color_u_z = _safe_subtract(raw, "u", "z")

    curve_ugr = pd.Series(np.nan, index=raw.index, dtype=float)
    curve_ugr_valid = np.isfinite(u) & np.isfinite(g) & np.isfinite(r)
    curve_ugr.loc[curve_ugr_valid] = u.loc[curve_ugr_valid] - 2.0 * g.loc[curve_ugr_valid] + r.loc[curve_ugr_valid]

    curve_gri = pd.Series(np.nan, index=raw.index, dtype=float)
    curve_gri_valid = np.isfinite(g) & np.isfinite(r) & np.isfinite(i)
    curve_gri.loc[curve_gri_valid] = g.loc[curve_gri_valid] - 2.0 * r.loc[curve_gri_valid] + i.loc[curve_gri_valid]

    curve_riz = pd.Series(np.nan, index=raw.index, dtype=float)
    curve_riz_valid = np.isfinite(r) & np.isfinite(i) & np.isfinite(z)
    curve_riz.loc[curve_riz_valid] = r.loc[curve_riz_valid] - 2.0 * i.loc[curve_riz_valid] + z.loc[curve_riz_valid]

    return pd.DataFrame(
        {
            "color_u_g": color_u_g,
            "color_g_r": color_g_r,
            "color_r_i": color_r_i,
            "color_i_z": color_i_z,
            "color_u_r": color_u_r,
            "color_g_i": color_g_i,
            "color_r_z": color_r_z,
            "color_u_z": color_u_z,
            "curve_ugr": curve_ugr,
            "curve_gri": curve_gri,
            "curve_riz": curve_riz,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "broadband_color_shape",
        "fn": add_broadband_color_shape,
        "depends_on": [],
        "description": "Create broadband color and curvature features from ugriz magnitudes.",
    }
]
