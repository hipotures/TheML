import numpy as np
import pandas as pd
from collections import deque

_AIDE_RA_BINS = 720
_AIDE_DEC_BINS = 338
_AIDE_SHRINKAGE_K = 20.0
_AIDE_FALLBACK_THRESHOLD = 20
_AIDE_REDSHIFT_MIN = 0.0
_AIDE_REDSHIFT_MAX = 7.0
_AIDE_EPS = 1e-6

_AIDE_FEATURE_ORDER = (
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift_clipped",
    "u_minus_g",
    "g_minus_r",
    "r_minus_i",
    "i_minus_z",
    "u_minus_i",
    "u_minus_r",
    "g_minus_z",
    "r_minus_z",
    "u_minus_2r_plus_i",
    "g_minus_2i_plus_z",
)

_AIDE_COLOR_SHAPE_FEATURES = (
    "u_minus_g",
    "g_minus_r",
    "r_minus_i",
    "i_minus_z",
    "u_minus_i",
    "u_minus_r",
    "g_minus_z",
    "r_minus_z",
    "u_minus_2r_plus_i",
    "g_minus_2i_plus_z",
)


def _safe_divide(numerator, denominator):
    out = np.full_like(numerator, np.nan, dtype=np.float64)
    mask = denominator > 0
    out[mask] = numerator[mask] / denominator[mask]
    return out


def _build_cell_index(alpha, delta):
    alpha_clean = np.where(np.isfinite(alpha), alpha, 0.0)
    delta_clean = np.where(np.isfinite(delta), delta, -90.0)

    ra_bin = np.floor(np.mod(alpha_clean, 360.0) * 2.0).astype(np.int32)
    ra_bin = np.clip(ra_bin, 0, _AIDE_RA_BINS - 1)

    dec_bin = np.floor((delta_clean + 90.0) * 2.0).astype(np.int32)
    dec_bin = np.clip(dec_bin, 0, _AIDE_DEC_BINS - 1)

    cell_idx = ra_bin * _AIDE_DEC_BINS + dec_bin
    return ra_bin, dec_bin, cell_idx


def _nearest_nonempty_lookup(cell_counts, ra_bins, dec_bins):
    n_cells = ra_bins * dec_bins
    nearest = np.full(n_cells, -1, dtype=np.int32)
    dist = np.full(n_cells, -1, dtype=np.int32)

    sources = np.flatnonzero(cell_counts > 0)
    if sources.size == 0:
        return nearest, dist

    q = deque()
    for src in sources:
        nearest[src] = src
        dist[src] = 0
        q.append(src)

    while q:
        cur = q.popleft()
        cur_ra = cur // dec_bins
        cur_dec = cur - cur_ra * dec_bins
        nd = dist[cur] + 1

        ra_minus = cur_ra - 1
        if ra_minus < 0:
            ra_minus = ra_bins - 1
        nb = ra_minus * dec_bins + cur_dec
        if dist[nb] < 0:
            dist[nb] = nd
            nearest[nb] = nearest[cur]
            q.append(nb)

        ra_plus = cur_ra + 1
        if ra_plus >= ra_bins:
            ra_plus = 0
        nb = ra_plus * dec_bins + cur_dec
        if dist[nb] < 0:
            dist[nb] = nd
            nearest[nb] = nearest[cur]
            q.append(nb)

        if cur_dec > 0:
            nb = cur_ra * dec_bins + (cur_dec - 1)
            if dist[nb] < 0:
                dist[nb] = nd
                nearest[nb] = nearest[cur]
                q.append(nb)

        if cur_dec + 1 < dec_bins:
            nb = cur_ra * dec_bins + (cur_dec + 1)
            if dist[nb] < 0:
                dist[nb] = nd
                nearest[nb] = nearest[cur]
                q.append(nb)

    return nearest, dist


def _compute_cell_stats(values, cell_idx, n_cells):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    cell_mean = np.full(n_cells, np.nan, dtype=np.float64)
    cell_std = np.full(n_cells, np.nan, dtype=np.float64)

    if not np.any(finite):
        return cell_mean, cell_std, np.nan, np.nan, _AIDE_EPS

    idx = cell_idx[finite]
    vals = values[finite]

    counts = np.bincount(idx, weights=np.ones(vals.shape[0], dtype=np.float64), minlength=n_cells).astype(np.int32)
    sums = np.bincount(idx, weights=vals, minlength=n_cells)
    sumsq = np.bincount(idx, weights=vals * vals, minlength=n_cells)

    valid = counts > 0
    cell_mean[valid] = sums[valid] / counts[valid]

    var = np.zeros(n_cells, dtype=np.float64)
    var[valid] = np.maximum(sumsq[valid] / counts[valid] - cell_mean[valid] * cell_mean[valid], 0.0)
    cell_std[valid] = np.sqrt(var[valid])

    global_mean = np.mean(vals)
    global_std = np.sqrt(np.mean((vals - global_mean) ** 2))

    if np.any(valid):
        cell_std_vals = cell_std[valid]
        sigma_floor = np.nanpercentile(cell_std_vals, 0.1)
        if not np.isfinite(sigma_floor) or sigma_floor <= _AIDE_EPS:
            pos = cell_std_vals[cell_std_vals > _AIDE_EPS]
            if pos.size > 0:
                sigma_floor = np.nanpercentile(pos, 0.1)
            elif np.isfinite(global_std) and global_std > _AIDE_EPS:
                sigma_floor = global_std
            else:
                sigma_floor = _AIDE_EPS
    else:
        sigma_floor = _AIDE_EPS

    if not np.isfinite(sigma_floor) or sigma_floor <= 0:
        sigma_floor = _AIDE_EPS

    return cell_mean, cell_std, global_mean, global_std, sigma_floor


def add_aide_sky_cell_local_residuals(raw, deps, aux):
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=np.float64)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=np.float64)
    idx = raw.index
    n = len(raw)

    if n == 0:
        return pd.DataFrame(index=idx)

    ra_bin, dec_bin, cell_idx = _build_cell_index(alpha, delta)
    n_cells = _AIDE_RA_BINS * _AIDE_DEC_BINS

    cell_counts = np.bincount(cell_idx, minlength=n_cells).astype(np.int32)
    n_cell = cell_counts[cell_idx].astype(np.float64)

    ra_minus = (ra_bin - 1) % _AIDE_RA_BINS
    ra_plus = (ra_bin + 1) % _AIDE_RA_BINS
    dec_minus = dec_bin - 1
    dec_plus = dec_bin + 1

    counts_grid = cell_counts.reshape(_AIDE_RA_BINS, _AIDE_DEC_BINS)

    n_neigh = (
        counts_grid[ra_bin, dec_bin]
        + counts_grid[ra_minus, dec_bin]
        + counts_grid[ra_plus, dec_bin]
    ).astype(np.float64)

    mid_mask = dec_bin > 0
    if np.any(mid_mask):
        n_neigh[mid_mask] += (
            counts_grid[ra_minus[mid_mask], dec_minus[mid_mask]]
            + counts_grid[ra_bin[mid_mask], dec_minus[mid_mask]]
            + counts_grid[ra_plus[mid_mask], dec_minus[mid_mask]]
        )

    plus_mask = dec_plus < _AIDE_DEC_BINS
    if np.any(plus_mask):
        n_neigh[plus_mask] += (
            counts_grid[ra_minus[plus_mask], dec_plus[plus_mask]]
            + counts_grid[ra_bin[plus_mask], dec_plus[plus_mask]]
            + counts_grid[ra_plus[plus_mask], dec_plus[plus_mask]]
        )

    concentration = n_cell / (n_neigh + _AIDE_EPS)

    nearest_idx, nearest_dist = _nearest_nonempty_lookup(cell_counts, _AIDE_RA_BINS, _AIDE_DEC_BINS)
    fallback_idx = nearest_idx[cell_idx].astype(np.int32)
    fallback_dist = nearest_dist[cell_idx].astype(np.float64)

    neighbor_cells = np.stack(
        (
            ra_minus * _AIDE_DEC_BINS + dec_minus,
            ra_bin * _AIDE_DEC_BINS + dec_minus,
            ra_plus * _AIDE_DEC_BINS + dec_minus,
            ra_minus * _AIDE_DEC_BINS + dec_bin,
            ra_plus * _AIDE_DEC_BINS + dec_bin,
            ra_minus * _AIDE_DEC_BINS + dec_plus,
            ra_bin * _AIDE_DEC_BINS + dec_plus,
            ra_plus * _AIDE_DEC_BINS + dec_plus,
        ),
        axis=1,
    )
    neighbor_cells = np.where(
        (neighbor_cells == cell_idx[:, None]) | (neighbor_cells < 0),
        -1,
        neighbor_cells,
    )
    neighbor_counts = np.where(neighbor_cells >= 0, cell_counts[neighbor_cells], -1)
    neighbor_has = neighbor_counts > 0
    neighbor_any = np.any(neighbor_has, axis=1)
    neighbor_first = np.argmax(neighbor_has.astype(np.int8), axis=1)
    neighbor_best = neighbor_cells[np.arange(n), neighbor_first]
    neighbor_best = np.where(neighbor_any, neighbor_best, cell_idx)

    sparse_mask = n_cell < float(_AIDE_FALLBACK_THRESHOLD)
    use_alt = sparse_mask & (fallback_dist == 0.0) & neighbor_any

    fallback_idx = np.where(use_alt, neighbor_best, fallback_idx)
    fallback_dist = np.where(use_alt, 1.0, fallback_dist)

    fallback_n_cell = cell_counts[fallback_idx].astype(np.float64)
    fallback_w = fallback_n_cell / (fallback_n_cell + _AIDE_SHRINKAGE_K)
    use_fallback = sparse_mask

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=np.float64)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=np.float64)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=np.float64)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=np.float64)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=np.float64)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64)
    redshift_clipped = np.clip(redshift, _AIDE_REDSHIFT_MIN, _AIDE_REDSHIFT_MAX)

    feature_vals = {
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "redshift_clipped": redshift_clipped,
        "u_minus_g": u - g,
        "g_minus_r": g - r,
        "r_minus_i": r - i,
        "i_minus_z": i - z,
        "u_minus_i": u - i,
        "u_minus_r": u - r,
        "g_minus_z": g - z,
        "r_minus_z": r - z,
        "u_minus_2r_plus_i": u - 2.0 * r + i,
        "g_minus_2i_plus_z": g - 2.0 * i + z,
    }

    color_set = set(_AIDE_COLOR_SHAPE_FEATURES)

    local_delta_sum = np.zeros(n, dtype=np.float64)
    local_delta_abs_sum = np.zeros(n, dtype=np.float64)
    local_delta_sq_sum = np.zeros(n, dtype=np.float64)
    local_z_sum = np.zeros(n, dtype=np.float64)
    local_z_abs_max = np.full(n, -np.inf, dtype=np.float64)
    local_valid = np.zeros(n, dtype=np.int32)

    local_color_delta_sum = np.zeros(n, dtype=np.float64)
    local_color_delta_sq_sum = np.zeros(n, dtype=np.float64)
    local_color_valid = np.zeros(n, dtype=np.int32)

    fallback_delta_sum = np.zeros(n, dtype=np.float64)
    fallback_delta_abs_sum = np.zeros(n, dtype=np.float64)
    fallback_delta_sq_sum = np.zeros(n, dtype=np.float64)
    fallback_z_sum = np.zeros(n, dtype=np.float64)
    fallback_z_abs_max = np.full(n, -np.inf, dtype=np.float64)
    fallback_valid = np.zeros(n, dtype=np.int32)

    fallback_color_delta_sum = np.zeros(n, dtype=np.float64)
    fallback_color_delta_sq_sum = np.zeros(n, dtype=np.float64)
    fallback_color_valid = np.zeros(n, dtype=np.int32)

    new_features = {}

    for feat in _AIDE_FEATURE_ORDER:
        vals = feature_vals[feat]
        finite = np.isfinite(vals)

        cell_mean, cell_std, global_mean, global_std, sigma_floor = _compute_cell_stats(vals, cell_idx, n_cells)

        mu_local = (
            n_cell * cell_mean[cell_idx] / (n_cell + _AIDE_SHRINKAGE_K)
            + (1.0 - n_cell / (n_cell + _AIDE_SHRINKAGE_K)) * global_mean
        )
        sigma_local = (
            n_cell * cell_std[cell_idx] / (n_cell + _AIDE_SHRINKAGE_K)
            + (1.0 - n_cell / (n_cell + _AIDE_SHRINKAGE_K)) * global_std
        )
        sigma_local = np.where(np.isfinite(sigma_local), sigma_local, sigma_floor)
        sigma_local = np.where(sigma_local < sigma_floor, sigma_floor, sigma_local)

        local_delta = np.full(n, np.nan, dtype=np.float64)
        local_z = np.full(n, np.nan, dtype=np.float64)

        valid_local = finite & np.isfinite(mu_local) & np.isfinite(sigma_local)
        if np.any(valid_local):
            local_delta[valid_local] = vals[valid_local] - mu_local[valid_local]
            local_z[valid_local] = local_delta[valid_local] / (sigma_local[valid_local] + _AIDE_EPS)

            d = local_delta[valid_local]
            zv = local_z[valid_local]

            local_delta_sum[valid_local] += d
            local_delta_abs_sum[valid_local] += np.abs(d)
            local_delta_sq_sum[valid_local] += d * d
            local_z_sum[valid_local] += zv
            local_z_abs_max[valid_local] = np.maximum(local_z_abs_max[valid_local], np.abs(zv))
            local_valid[valid_local] += 1

            if feat in color_set:
                local_color_delta_sum[valid_local] += d
                local_color_delta_sq_sum[valid_local] += d * d
                local_color_valid[valid_local] += 1

        mu_fb = (
            fallback_w * cell_mean[fallback_idx]
            + (1.0 - fallback_w) * global_mean
        )
        sigma_fb = (
            fallback_w * cell_std[fallback_idx]
            + (1.0 - fallback_w) * global_std
        )
        sigma_fb = np.where(np.isfinite(sigma_fb), sigma_fb, sigma_floor)
        sigma_fb = np.where(sigma_fb < sigma_floor, sigma_floor, sigma_fb)

        fallback_delta = np.full(n, np.nan, dtype=np.float64)
        fallback_z = np.full(n, np.nan, dtype=np.float64)

        valid_fb = use_fallback & finite & np.isfinite(mu_fb) & np.isfinite(sigma_fb)
        if np.any(valid_fb):
            fallback_delta[valid_fb] = vals[valid_fb] - mu_fb[valid_fb]
            fallback_z[valid_fb] = fallback_delta[valid_fb] / (sigma_fb[valid_fb] + _AIDE_EPS)

            d_fb = fallback_delta[valid_fb]
            z_fb = fallback_z[valid_fb]

            fallback_delta_sum[valid_fb] += d_fb
            fallback_delta_abs_sum[valid_fb] += np.abs(d_fb)
            fallback_delta_sq_sum[valid_fb] += d_fb * d_fb
            fallback_z_sum[valid_fb] += z_fb
            fallback_z_abs_max[valid_fb] = np.maximum(fallback_z_abs_max[valid_fb], np.abs(z_fb))
            fallback_valid[valid_fb] += 1

            if feat in color_set:
                fallback_color_delta_sum[valid_fb] += d_fb
                fallback_color_delta_sq_sum[valid_fb] += d_fb * d_fb
                fallback_color_valid[valid_fb] += 1

        new_features[f"{feat}_cell_delta"] = local_delta
        new_features[f"{feat}_cell_z"] = local_z

    local_delta_mean = _safe_divide(local_delta_sum, local_valid.astype(np.float64))
    local_delta_abs_mean = _safe_divide(local_delta_abs_sum, local_valid.astype(np.float64))
    local_z_mean = _safe_divide(local_z_sum, local_valid.astype(np.float64))
    local_delta_l2 = np.sqrt(local_delta_sq_sum)
    local_z_abs_max = np.where(local_valid > 0, local_z_abs_max, np.nan)

    local_color_mean = _safe_divide(local_color_delta_sum, local_color_valid.astype(np.float64))
    local_color_var = _safe_divide(local_color_delta_sq_sum, local_color_valid.astype(np.float64)) - local_color_mean * local_color_mean
    local_color_var = np.where(local_color_var < 0.0, 0.0, local_color_var)
    local_shape_resid_std = np.full(n, np.nan, dtype=np.float64)
    color_mask = local_color_valid > 0
    local_shape_resid_std[color_mask] = np.sqrt(local_color_var[color_mask])

    fallback_delta_mean = _safe_divide(fallback_delta_sum, fallback_valid.astype(np.float64))
    fallback_delta_abs_mean = _safe_divide(fallback_delta_abs_sum, fallback_valid.astype(np.float64))
    fallback_z_mean = _safe_divide(fallback_z_sum, fallback_valid.astype(np.float64))
    fallback_delta_l2 = np.sqrt(fallback_delta_sq_sum)
    fallback_z_abs_max = np.where(fallback_valid > 0, fallback_z_abs_max, np.nan)

    fallback_color_mean = _safe_divide(fallback_color_delta_sum, fallback_color_valid.astype(np.float64))
    fallback_color_var = _safe_divide(fallback_color_delta_sq_sum, fallback_color_valid.astype(np.float64)) - fallback_color_mean * fallback_color_mean
    fallback_color_var = np.where(fallback_color_var < 0.0, 0.0, fallback_color_var)
    fallback_shape_resid_std = np.full(n, np.nan, dtype=np.float64)
    fb_color_mask = fallback_color_valid > 0
    fallback_shape_resid_std[fb_color_mask] = np.sqrt(fallback_color_var[fb_color_mask])

    cell_alpha_center = (ra_bin.astype(np.float64) + 0.5) * 0.5
    cell_delta_center = (dec_bin.astype(np.float64) + 0.5) * 0.5 - 90.0

    new_features.update(
        {
            "sky_cell_ra_bin": ra_bin.astype(np.float64),
            "sky_cell_dec_bin": dec_bin.astype(np.float64),
            "sky_cell_ra_center": cell_alpha_center,
            "sky_cell_dec_center": cell_delta_center,
            "sky_cell_n": n_cell,
            "sky_cell_n_neigh": n_neigh,
            "sky_cell_concentration": concentration,
            "sky_cell_is_sparse": sparse_mask.astype(np.int8),
            "sky_cell_fallback_used": use_fallback.astype(np.int8),
            "sky_cell_fallback_dist": np.where(use_fallback, fallback_dist, np.nan),
            "sky_cell_fallback_cell_n": np.where(use_fallback, fallback_n_cell, np.nan),
            "sky_cell_local_delta_mean": local_delta_mean,
            "sky_cell_local_delta_abs_mean": local_delta_abs_mean,
            "sky_cell_local_delta_l1": local_delta_abs_sum,
            "sky_cell_local_delta_l2": local_delta_l2,
            "sky_cell_local_z_mean": local_z_mean,
            "sky_cell_local_abs_z_max": local_z_abs_max,
            "sky_cell_local_shape_resid_std": local_shape_resid_std,
            "sky_cell_local_n_features": local_valid.astype(np.float64),
            "sky_cell_fallback_delta_mean": fallback_delta_mean,
            "sky_cell_fallback_delta_abs_mean": fallback_delta_abs_mean,
            "sky_cell_fallback_delta_l1": fallback_delta_abs_sum,
            "sky_cell_fallback_delta_l2": fallback_delta_l2,
            "sky_cell_fallback_z_mean": fallback_z_mean,
            "sky_cell_fallback_abs_z_max": fallback_z_abs_max,
            "sky_cell_fallback_shape_resid_std": fallback_shape_resid_std,
            "sky_cell_fallback_n_features": fallback_valid.astype(np.float64),
        }
    )

    return pd.DataFrame(new_features, index=idx)


FEATURE_GROUPS = [
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": add_aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "Spatial sky-cell residualization with shrinkage to local cell/global statistics and sparse-cell neighborhood fallback features.",
    }
]