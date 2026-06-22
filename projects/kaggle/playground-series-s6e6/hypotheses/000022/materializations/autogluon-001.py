from __future__ import annotations

import numpy as np
import pandas as pd


MAGS = ("u", "g", "r", "i", "z")


def _num(raw: pd.DataFrame, col: str) -> pd.Series:
    if col not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    return pd.to_numeric(raw[col], errors="coerce").astype("float64")


def _sky_key(raw: pd.DataFrame) -> pd.Series:
    alpha = _num(raw, "alpha").fillna(0.0)
    delta = _num(raw, "delta").fillna(0.0)
    a_bin = np.floor(alpha * 2.0).astype("int64")
    d_bin = np.floor((delta + 90.0) * 2.0).astype("int64")
    return pd.Series(a_bin.astype(str) + "_" + d_bin.astype(str), index=raw.index)


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


def aide_sky_cell_local_residuals(raw: pd.DataFrame, deps: dict[str, pd.DataFrame] | None = None, aux: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    key = _sky_key(raw)
    counts = key.map(key.value_counts()).astype("float64")
    out["aide_sky_cell_count"] = counts

    parts = key.str.split("_", expand=True).astype("int64")
    a_bin = parts[0]
    d_bin = parts[1]
    count_map = key.value_counts().to_dict()
    neighbor_counts = []
    for a_val, d_val in zip(a_bin.to_numpy(), d_bin.to_numpy()):
        total = 0
        for da in (-1, 0, 1):
            for dd in (-1, 0, 1):
                total += count_map.get(f"{a_val + da}_{d_val + dd}", 0)
        neighbor_counts.append(total)
    out["aide_sky_cell_3x3_count"] = np.asarray(neighbor_counts, dtype="float64")
    out["aide_sky_concentration"] = out["aide_sky_cell_count"] / (out["aide_sky_cell_3x3_count"] + 1.0)

    values = pd.DataFrame({col: _num(raw, col) for col in MAGS}, index=raw.index)
    values["redshift_clipped"] = _num(raw, "redshift").clip(lower=0)
    residual_l2 = pd.Series(0.0, index=raw.index)
    residual_l1 = pd.Series(0.0, index=raw.index)
    z_max = pd.Series(0.0, index=raw.index)

    for col in values.columns:
        mean = values[col].groupby(key).transform("mean")
        std = values[col].groupby(key).transform("std").replace(0, np.nan)
        delta = values[col] - mean
        z_score = delta / std
        out[f"aide_{col}_sky_cell_delta"] = delta
        out[f"aide_{col}_sky_cell_z"] = z_score
        residual_l2 = residual_l2 + delta.fillna(0.0) ** 2
        residual_l1 = residual_l1 + delta.abs().fillna(0.0)
        z_max = np.maximum(z_max, z_score.abs().fillna(0.0))

    out["aide_sky_cell_residual_l2"] = np.sqrt(residual_l2)
    out["aide_sky_cell_residual_abs_l1"] = residual_l1
    out["aide_sky_cell_z_max"] = z_max

    color_values = _colors(raw)
    color_l2 = pd.Series(0.0, index=raw.index)
    color_l1 = pd.Series(0.0, index=raw.index)
    color_z_max = pd.Series(0.0, index=raw.index)
    color_deltas = []
    for col in color_values.columns:
        mean = color_values[col].groupby(key).transform("mean")
        std = color_values[col].groupby(key).transform("std").replace(0, np.nan)
        delta = color_values[col] - mean
        z_score = delta / std
        out[f"aide_{col}_sky_color_cell_delta"] = delta
        out[f"aide_{col}_sky_color_cell_z"] = z_score
        color_l2 = color_l2 + delta.fillna(0.0) ** 2
        color_l1 = color_l1 + delta.abs().fillna(0.0)
        color_z_max = np.maximum(color_z_max, z_score.abs().fillna(0.0))
        color_deltas.append(delta)

    delta_frame = pd.concat(color_deltas, axis=1)
    out["aide_sky_color_cell_residual_l2"] = np.sqrt(color_l2)
    out["aide_sky_color_cell_residual_abs_l1"] = color_l1
    out["aide_sky_color_cell_delta_mean"] = delta_frame.mean(axis=1)
    out["aide_sky_color_cell_delta_std"] = delta_frame.std(axis=1)
    out["aide_sky_color_cell_z_max"] = color_z_max
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_GROUPS = [
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "AIDE half-degree sky-cell densities and local residual/z-score features.",
    }
]
