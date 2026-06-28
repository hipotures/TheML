import numpy as np
import pandas as pd


RA_BINS = 720
DEC_BINS = 360
TOTAL_SKY_CELLS = 259200
CELL_SHRINKAGE = 30.0
STD_FLOOR_MIN = 0.001
CELL_WIDTH_DEGREES = 0.5
HALF_CELL_WIDTH_DEGREES = 0.25

MAG_DESCRIPTOR_NAMES = ("u", "g", "r", "i", "z", "redshift_clip")
COLOR_DESCRIPTOR_NAMES = ("u_g", "g_r", "r_i", "i_z", "u_i", "u_r", "g_z", "r_z")
SHAPE_DESCRIPTOR_NAMES = ("u_2r_i", "g_2i_z")


def _as_float_array(raw, column):
    values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(values)
    if not finite.all():
        if finite.any():
            fill_value = float(np.median(values[finite]))
        else:
            fill_value = 0.0
        values[~finite] = fill_value
    return values


def _neighbor_sum_grid(grid):
    out = np.zeros_like(grid, dtype=np.float64)

    dec_lower = np.zeros_like(grid, dtype=np.float64)
    dec_lower[1:, :] = grid[:-1, :]

    dec_upper = np.zeros_like(grid, dtype=np.float64)
    dec_upper[:-1, :] = grid[1:, :]

    for dec_part in (dec_lower, grid, dec_upper):
        out += dec_part
        out += np.roll(dec_part, 1, axis=1)
        out += np.roll(dec_part, -1, axis=1)

    return out


def _global_mean_std(values):
    finite = np.isfinite(values)
    if finite.any():
        clean = values[finite]
        mean = float(np.mean(clean))
        std = float(np.std(clean))
    else:
        mean = 0.0
        std = 1.0

    if not np.isfinite(std) or std < STD_FLOOR_MIN:
        std = STD_FLOOR_MIN
    return mean, std


def _sigma_floor(sigma, counts):
    valid = (counts > 1) & np.isfinite(sigma) & (sigma > 0.0)
    if valid.any():
        floor = float(np.percentile(sigma[valid], 1.0))
        if not np.isfinite(floor):
            floor = STD_FLOOR_MIN
    else:
        floor = STD_FLOOR_MIN
    return max(floor, STD_FLOOR_MIN)


def _shrunk_mean_sigma(sums, sumsq, counts, global_mean, global_std):
    mean = np.full(counts.shape, global_mean, dtype=np.float64)
    has_count = counts > 0
    mean[has_count] = sums[has_count] / counts[has_count]

    variance = np.zeros(counts.shape, dtype=np.float64)
    has_variance = counts > 1
    variance[has_variance] = (sumsq[has_variance] / counts[has_variance]) - (
        mean[has_variance] * mean[has_variance]
    )
    variance = np.maximum(variance, 0.0)

    sigma = np.sqrt(variance)
    floor = _sigma_floor(sigma, counts)
    sigma = np.maximum(sigma, floor)

    weight = counts / (counts + CELL_SHRINKAGE)
    shrunk_mean = (weight * mean) + ((1.0 - weight) * global_mean)
    shrunk_sigma = (weight * sigma) + ((1.0 - weight) * global_std)
    shrunk_sigma = np.maximum(shrunk_sigma, floor)

    return shrunk_mean, shrunk_sigma


def _new_summary_state(n_rows):
    return {
        "count": np.zeros(n_rows, dtype=np.int16),
        "sum": np.zeros(n_rows, dtype=np.float64),
        "sum_abs": np.zeros(n_rows, dtype=np.float64),
        "sum_sq": np.zeros(n_rows, dtype=np.float64),
        "max_abs_z": np.zeros(n_rows, dtype=np.float64),
    }


def _update_summary_state(state, residual, zscore):
    valid = np.isfinite(residual) & np.isfinite(zscore)
    clean_residual = np.where(valid, residual, 0.0)
    state["count"] += valid.astype(np.int16)
    state["sum"] += clean_residual
    state["sum_abs"] += np.abs(clean_residual)
    state["sum_sq"] += clean_residual * clean_residual
    state["max_abs_z"] = np.maximum(state["max_abs_z"], np.where(valid, np.abs(zscore), 0.0))


def _finish_summary(features, state, prefix):
    count_float = state["count"].astype(np.float64)
    nonzero = count_float > 0.0

    mean = np.zeros_like(count_float, dtype=np.float64)
    mean_abs = np.zeros_like(count_float, dtype=np.float64)
    np.divide(state["sum"], count_float, out=mean, where=nonzero)
    np.divide(state["sum_abs"], count_float, out=mean_abs, where=nonzero)

    variance = np.zeros_like(count_float, dtype=np.float64)
    np.divide(state["sum_sq"], count_float, out=variance, where=nonzero)
    variance = np.maximum(variance - (mean * mean), 0.0)

    features[prefix + "_mean_resid"] = mean.astype(np.float32)
    features[prefix + "_mean_abs_resid"] = mean_abs.astype(np.float32)
    features[prefix + "_l1_resid"] = state["sum_abs"].astype(np.float32)
    features[prefix + "_l2_resid"] = np.sqrt(state["sum_sq"]).astype(np.float32)
    features[prefix + "_std_resid"] = np.sqrt(variance).astype(np.float32)
    features[prefix + "_max_abs_z"] = state["max_abs_z"].astype(np.float32)
    features[prefix + "_valid_count"] = state["count"]


def _descriptor_groups(raw):
    u = _as_float_array(raw, "u")
    g = _as_float_array(raw, "g")
    r = _as_float_array(raw, "r")
    i = _as_float_array(raw, "i")
    z = _as_float_array(raw, "z")
    redshift = np.clip(_as_float_array(raw, "redshift"), -0.02, 7.05)

    return (
        ("u", "mag", u),
        ("g", "mag", g),
        ("r", "mag", r),
        ("i", "mag", i),
        ("z", "mag", z),
        ("redshift_clip", "mag", redshift),
        ("u_g", "color", u - g),
        ("g_r", "color", g - r),
        ("r_i", "color", r - i),
        ("i_z", "color", i - z),
        ("u_i", "color", u - i),
        ("u_r", "color", u - r),
        ("g_z", "color", g - z),
        ("r_z", "color", r - z),
        ("u_2r_i", "shape", u - (2.0 * r) + i),
        ("g_2i_z", "shape", g - (2.0 * i) + z),
    )


def add_aide_sky_cell_local_residuals(raw, deps, aux):
    n_rows = len(raw)
    index = raw.index

    alpha = _as_float_array(raw, "alpha")
    delta = _as_float_array(raw, "delta")

    alpha_mod = np.mod(alpha, 360.0)
    ra_bin = np.floor(alpha_mod * 2.0).astype(np.int32)
    ra_bin = np.mod(ra_bin, RA_BINS).astype(np.int32)

    dec_scaled = np.floor((delta + 90.0) * 2.0)
    dec_bin = np.clip(dec_scaled, 0, DEC_BINS - 1).astype(np.int32)

    cell_id = (dec_bin * RA_BINS + ra_bin).astype(np.int32)
    cell_counts = np.bincount(cell_id, minlength=TOTAL_SKY_CELLS).astype(np.float64)
    cell_count_grid = cell_counts.reshape(DEC_BINS, RA_BINS)
    neigh_counts = _neighbor_sum_grid(cell_count_grid).reshape(TOTAL_SKY_CELLS)

    row_cell_count = cell_counts[cell_id]
    row_neigh_count = neigh_counts[cell_id]

    ra_center = (ra_bin.astype(np.float64) * CELL_WIDTH_DEGREES) + HALF_CELL_WIDTH_DEGREES
    dec_center = (dec_bin.astype(np.float64) * CELL_WIDTH_DEGREES) - 90.0 + HALF_CELL_WIDTH_DEGREES

    features = {
        "sky_ra_bin": ra_bin.astype(np.int16),
        "sky_dec_bin": dec_bin.astype(np.int16),
        "sky_cell_id": cell_id.astype(np.int32),
        "sky_cell_count": row_cell_count.astype(np.int32),
        "sky_neigh_count": row_neigh_count.astype(np.int32),
        "sky_log1p_cell_count": np.log1p(row_cell_count).astype(np.float32),
        "sky_log1p_neigh_count": np.log1p(row_neigh_count).astype(np.float32),
        "sky_cell_concentration": (row_cell_count / (row_neigh_count + 1.0e-6)).astype(np.float32),
        "sky_ra_cell_frac_offset": ((alpha_mod - ra_center) / HALF_CELL_WIDTH_DEGREES).astype(np.float32),
        "sky_dec_cell_frac_offset": ((delta - dec_center) / HALF_CELL_WIDTH_DEGREES).astype(np.float32),
        "sky_cell_support_level": np.where(row_cell_count >= 3.0, 0, np.where(row_neigh_count >= 3.0, 1, 2)).astype(np.int8),
        "sky_neigh_support_level": np.where(row_neigh_count > 0.0, 0, 2).astype(np.int8),
    }

    summary_states = {
        ("cell", "mag"): _new_summary_state(n_rows),
        ("cell", "color"): _new_summary_state(n_rows),
        ("cell", "shape"): _new_summary_state(n_rows),
        ("neigh", "mag"): _new_summary_state(n_rows),
        ("neigh", "color"): _new_summary_state(n_rows),
        ("neigh", "shape"): _new_summary_state(n_rows),
    }

    for name, subset, values in _descriptor_groups(raw):
        global_mean, global_std = _global_mean_std(values)

        sums = np.bincount(cell_id, weights=values, minlength=TOTAL_SKY_CELLS)
        sumsq = np.bincount(cell_id, weights=values * values, minlength=TOTAL_SKY_CELLS)

        cell_mean, cell_sigma = _shrunk_mean_sigma(sums, sumsq, cell_counts, global_mean, global_std)

        neigh_sums = _neighbor_sum_grid(sums.reshape(DEC_BINS, RA_BINS)).reshape(TOTAL_SKY_CELLS)
        neigh_sumsq = _neighbor_sum_grid(sumsq.reshape(DEC_BINS, RA_BINS)).reshape(TOTAL_SKY_CELLS)
        neigh_mean, neigh_sigma = _shrunk_mean_sigma(
            neigh_sums,
            neigh_sumsq,
            neigh_counts,
            global_mean,
            global_std,
        )

        cell_resid = values - cell_mean[cell_id]
        cell_z = cell_resid / cell_sigma[cell_id]
        neigh_resid = values - neigh_mean[cell_id]
        neigh_z = neigh_resid / neigh_sigma[cell_id]

        features["cell_resid_" + name] = cell_resid.astype(np.float32)
        features["cell_z_" + name] = cell_z.astype(np.float32)
        features["neigh_resid_" + name] = neigh_resid.astype(np.float32)
        features["neigh_z_" + name] = neigh_z.astype(np.float32)

        _update_summary_state(summary_states[("cell", subset)], cell_resid, cell_z)
        _update_summary_state(summary_states[("neigh", subset)], neigh_resid, neigh_z)

    for reference in ("cell", "neigh"):
        for subset in ("mag", "color", "shape"):
            _finish_summary(features, summary_states[(reference, subset)], reference + "_" + subset)

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": add_aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "Smoothed sky-cell density and local photometric residual features from unsupervised coordinate neighborhoods.",
    }
]