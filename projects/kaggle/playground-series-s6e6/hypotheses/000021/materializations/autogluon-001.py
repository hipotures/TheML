from __future__ import annotations

import numpy as np
import pandas as pd


MAGS = ("u", "g", "r", "i", "z")
COLOR_PAIRS = (("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "i"), ("u", "r"), ("g", "z"), ("r", "z"))


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def aide_broadband_flux_ratios(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    eps = 1e-6

    mags = {col: _num(raw, col) for col in MAGS}
    for left, right in COLOR_PAIRS:
        color = mags[left] - mags[right]
        out[f"aide_abs_{left}_{right}"] = color.abs()
        out[f"aide_ratio_{left}_over_{right}"] = mags[left] / (mags[right].abs() + eps)

    for col, values in mags.items():
        clipped = values.clip(-50, 50)
        out[f"aide_log_flux_{col}"] = -0.4 * np.log(10.0) * values
        out[f"aide_exp_flux_{col}"] = np.exp(-0.4 * np.log(10.0) * clipped)

    redshift = _num(raw, "redshift")
    clipped_z = redshift.clip(lower=0)
    out["aide_redshift_clipped"] = clipped_z
    out["aide_redshift_log1p"] = np.log1p(clipped_z)
    out["aide_redshift_sq"] = clipped_z ** 2
    out["aide_redshift_cube"] = clipped_z ** 3
    out["aide_u_r_i_curv"] = mags["u"] - 2.0 * mags["r"] + mags["i"]
    out["aide_g_i_z_curv"] = mags["g"] - 2.0 * mags["i"] + mags["z"]
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_broadband_flux_ratios",
        "fn": aide_broadband_flux_ratios,
        "depends_on": [],
        "description": "AIDE top-5 broadband ratios, absolute colors, pseudo-flux values, and redshift powers.",
    }
]
