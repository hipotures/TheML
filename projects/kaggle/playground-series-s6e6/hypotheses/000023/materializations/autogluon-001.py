from __future__ import annotations

import numpy as np
import pandas as pd


MAGS = ("u", "g", "r", "i", "z")


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def _colors(raw: pd.DataFrame) -> pd.DataFrame:
    u, g, r, i, z = (_num(raw, col) for col in MAGS)
    return pd.DataFrame(
        {
            "u_g": u - g,
            "g_r": g - r,
            "r_i": r - i,
            "i_z": i - z,
            "u_i": u - i,
            "u_r": u - r,
            "g_z": g - z,
            "r_z": r - z,
            "u_r_i_curv": u - 2.0 * r + i,
            "g_i_z_curv": g - 2.0 * i + z,
        },
        index=raw.index,
    )


def aide_redshift_bin_color_residuals(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    redshift = _num(raw, "redshift").clip(lower=0)
    bins = pd.qcut(redshift.rank(method="first"), q=24, duplicates="drop", labels=False)
    bins = pd.Series(bins, index=raw.index)
    colors = _colors(raw)

    residual_l2 = pd.Series(0.0, index=raw.index)
    residual_l1 = pd.Series(0.0, index=raw.index)
    z_max = pd.Series(0.0, index=raw.index)
    deltas = []
    for col in colors.columns:
        mean = colors[col].groupby(bins).transform("mean")
        std = colors[col].groupby(bins).transform("std").replace(0, np.nan)
        delta = colors[col] - mean
        z_score = delta / std
        out[f"aide_{col}_redshift_bin_delta"] = delta
        out[f"aide_{col}_redshift_bin_z"] = z_score
        residual_l2 = residual_l2 + delta.fillna(0.0) ** 2
        residual_l1 = residual_l1 + delta.abs().fillna(0.0)
        z_max = np.maximum(z_max, z_score.abs().fillna(0.0))
        deltas.append(delta)

    delta_frame = pd.concat(deltas, axis=1)
    out["aide_redshift_color_residual_l2"] = np.sqrt(residual_l2)
    out["aide_redshift_color_residual_abs_l1"] = residual_l1
    out["aide_redshift_color_delta_mean"] = delta_frame.mean(axis=1)
    out["aide_redshift_color_delta_std"] = delta_frame.std(axis=1)
    out["aide_redshift_color_z_max"] = z_max
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_redshift_bin_color_residuals",
        "fn": aide_redshift_bin_color_residuals,
        "depends_on": [],
        "description": "AIDE qcut-redshift color residuals and aggregate residual scores.",
    }
]
