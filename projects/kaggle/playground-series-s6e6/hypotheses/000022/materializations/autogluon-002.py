import numpy as np
import pandas as pd

_AIDE_CELLS_PER_DEGREE = 2
_AIDE_RA_BINS = 360 * _AIDE_CELLS_PER_DEGREE
_AIDE_DEC_BINS = 180 * _AIDE_CELLS_PER_DEGREE

_AIDE_BASE_FEATURES = ("u", "g", "r", "i", "z", "redshift_clipped")
_AIDE_COLOR_FEATURES = (
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "u_i",
    "u_r",
    "g_z",
    "r_z",
    "u_2r_i",
    "g_2i_z",
)

def _compute_sky_cells(alpha_series, delta_series):
    alpha = np.asarray(alpha_series, dtype=np.float64)
    delta = np.asarray(delta_series, dtype=np.float64)

    alpha_wrapped = np.mod(alpha, 360.0)
    ra_bin = np.floor(alpha_wrapped * float(_AIDE_CELLS_PER_DEGREE)).astype(np.int64)
    ra_bin = np.clip(ra_bin, 0, _AIDE_RA_BINS - 1)

    delta_shift = np.clip(delta + 90.0, 0.0, 180.0 - 1e-12)
    dec_bin = np.floor(delta_shift * float(_AIDE_CELLS_PER_DEGREE)).astype(np.int64)
    dec_bin = np.clip(dec_bin, 0, _AIDE_DEC_BINS - 1)

    cell_key = dec_bin * _AIDE_RA_BINS + ra_bin
    return ra_bin, dec_bin, cell_key

def _compute_cell_neighbors(ra_bin, dec_bin):
    cell_counts = np.zeros((_AIDE_DEC_BINS, _AIDE_RA_BINS), dtype=np.int64)
    np.add.at(cell_counts, (dec_bin, ra_bin), 1)

    cell_count = cell_counts[dec_bin, ra_bin]

    # Wrap longitude with neighbors across 0/360 edges.
    wrapped_ra = np.concatenate(
        [cell_counts[:, -1:], cell_counts, cell_counts[:, :1]],
        axis=1
    )
    padded = np.pad(wrapped_ra, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    ra_idx = ra_bin + 2
    dec_idx = dec_bin + 1

    neighbor_count = (
        padded[dec_idx - 1, ra_idx - 1] + padded[dec_idx - 1, ra_idx] + padded[dec_idx - 1, ra_idx + 1] +
        padded[dec_idx, ra_idx - 1] + padded[dec_idx, ra_idx] + padded[dec_idx, ra_idx + 1] +
        padded[dec_idx + 1, ra_idx - 1] + padded[dec_idx + 1, ra_idx] + padded[dec_idx + 1, ra_idx + 1]
    )

    concentration = np.zeros_like(cell_count, dtype=np.float64)
    np.divide(
        cell_count,
        neighbor_count,
        out=concentration,
        where=neighbor_count > 0
    )
    return cell_count, neighbor_count, concentration

def _safe_zscore(residuals, scales):
    z = residuals / scales
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def add_aide_sky_cell_local_residuals(raw, deps, aux):
    alpha = raw["alpha"]
    delta = raw["delta"]

    ra_bin, dec_bin, cell_key = _compute_sky_cells(alpha, delta)
    cell_count, neighbor_count, concentration = _compute_cell_neighbors(ra_bin, dec_bin)

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)
    redshift_clipped = np.clip(raw["redshift"].to_numpy(dtype=np.float64), 0.0, 7.0)

    color_table = {
        "u_g": u - g,
        "g_r": g - r,
        "r_i": r - i,
        "i_z": i - z,
        "u_i": u - i,
        "u_r": u - r,
        "g_z": g - z,
        "r_z": r - z,
        "u_2r_i": u - 2.0 * r + i,
        "g_2i_z": g - 2.0 * i + z,
    }

    all_features = pd.DataFrame(
        {
            "cell_key": cell_key,
            "u": u,
            "g": g,
            "r": r,
            "i": i,
            "z": z,
            "redshift_clipped": redshift_clipped,
            "u_g": color_table["u_g"],
            "g_r": color_table["g_r"],
            "r_i": color_table["r_i"],
            "i_z": color_table["i_z"],
            "u_i": color_table["u_i"],
            "u_r": color_table["u_r"],
            "g_z": color_table["g_z"],
            "r_z": color_table["r_z"],
            "u_2r_i": color_table["u_2r_i"],
            "g_2i_z": color_table["g_2i_z"],
        },
        index=raw.index,
    )

    grouped = all_features.groupby("cell_key", sort=False)

    cell_means = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("mean")
    cell_stds = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("std").replace(0.0, np.nan)

    residuals = all_features[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES] - cell_means
    zscores = _safe_zscore(residuals, cell_stds)

    color_residuals = residuals[_AIDE_COLOR_FEATURES].to_numpy(dtype=np.float64)
    color_zscores = zscores[_AIDE_COLOR_FEATURES].to_numpy(dtype=np.float64)

    out = pd.DataFrame(index=raw.index)
    out["aide_sky_cell_count"] = cell_count
    out["aide_sky_cell_3x3_count"] = neighbor_count
    out["aide_sky_cell_concentration"] = concentration

    for feat in _AIDE_BASE_FEATURES:
        out[f"aide_sky_cell_residual_{feat}"] = residuals[feat]
        out[f"aide_sky_cell_zscore_{feat}"] = zscores[feat]

    for feat in _AIDE_COLOR_FEATURES:
        out[f"aide_sky_cell_residual_{feat}"] = residuals[feat]
        out[f"aide_sky_cell_zscore_{feat}"] = zscores[feat]

    out["aide_sky_cell_color_residual_l2"] = np.sqrt((color_residuals ** 2).sum(axis=1))
    out["aide_sky_cell_color_residual_abs_l1"] = np.abs(color_residuals).sum(axis=1)
    out["aide_sky_cell_color_residual_mean_signed"] = color_residuals.mean(axis=1)
    out["aide_sky_cell_color_residual_std"] = color_residuals.std(axis=1, ddof=0)
    out["aide_sky_cell_color_max_abs_zscore"] = np.abs(color_zscores).max(axis=1)

    return out

FEATURE_GROUPS = [
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": add_aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "Build deterministic half-degree sky-cell features with local neighbor density and within-cell residual statistics over magnitudes, clipped redshift, and color-shape features.",
    },
]