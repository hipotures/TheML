import numpy as np
import pandas as pd

_COLOR_SPECS = (
    ("u", "g", "u_g"),
    ("g", "r", "g_r"),
    ("r", "i", "r_i"),
    ("i", "z", "i_z"),
    ("u", "r", "u_r"),
    ("g", "i", "g_i"),
    ("r", "z", "r_z"),
)

_POPULATION_VALUES = ("Red_Sequence", "Blue_Cloud")

_TRAIN_ID_CUTOFF = 577347
_N_BINS = 24
_N_MIN = 300.0
_MAD_SCALE = 1.4826
_SCALE_FLOOR = 0.0001
_RESIDUAL_CLIP = 8.0
_CORR_CLIP = 0.98
_RIDGE = 0.001
_EPSILON = 0.000000001
_CHUNK_SIZE = 65536


def _median_finite(values, default):
    finite = np.isfinite(values)
    if finite.any():
        return float(np.median(values[finite]))
    return float(default)


def _as_numeric_array(raw, column, default):
    if column in raw.columns:
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype="float64", copy=True)
    else:
        values = np.empty(len(raw), dtype=np.float64)
        values.fill(default)

    finite = np.isfinite(values)
    if not finite.all():
        values[~finite] = _median_finite(values, default)
    return values


def _infer_train_mask(raw):
    n_rows = len(raw)
    if "id" not in raw.columns:
        return np.ones(n_rows, dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype="float64", copy=True)
    train_mask = np.isfinite(ids) & (ids < _TRAIN_ID_CUTOFF)

    if int(train_mask.sum()) < max(100, _N_BINS * 2):
        return np.ones(n_rows, dtype=bool)
    return train_mask


def _population_codes(raw):
    n_rows = len(raw)
    codes = np.ones(n_rows, dtype=np.intp)

    if "galaxy_population" not in raw.columns:
        return codes

    values = raw["galaxy_population"].astype(str).to_numpy(copy=False)
    codes[values == _POPULATION_VALUES[0]] = 0
    codes[values == _POPULATION_VALUES[1]] = 1
    return codes


def _color_matrix(raw):
    magnitudes = {}
    for left, right, _ in _COLOR_SPECS:
        if left not in magnitudes:
            magnitudes[left] = _as_numeric_array(raw, left, 0.0)
        if right not in magnitudes:
            magnitudes[right] = _as_numeric_array(raw, right, 0.0)

    colors = np.empty((len(raw), len(_COLOR_SPECS)), dtype=np.float64)
    for idx, (left, right, _) in enumerate(_COLOR_SPECS):
        colors[:, idx] = magnitudes[left] - magnitudes[right]
    return colors


def _redshift_coordinate(raw):
    redshift = _as_numeric_array(raw, "redshift", 0.0)
    redshift = np.maximum(redshift, -0.009999)
    return np.log1p(redshift)


def _redshift_bins(t, train_mask):
    valid_train = train_mask & np.isfinite(t)
    if int(valid_train.sum()) < 2:
        valid_train = np.isfinite(t)

    if not valid_train.any():
        return np.linspace(0.0, 1.0, _N_BINS + 1)

    train_t = t[valid_train]
    quantiles = np.linspace(0.0, 1.0, _N_BINS + 1)
    edges = np.quantile(train_t, quantiles)

    if (not np.isfinite(edges).all()) or float(edges[-1] - edges[0]) <= _EPSILON:
        center = _median_finite(train_t, 0.0)
        return np.linspace(center - 0.000001, center + 0.000001, _N_BINS + 1)

    return edges


def _assign_edge_bins(t, edges):
    filled = np.asarray(t, dtype=np.float64).copy()
    finite = np.isfinite(filled)
    if not finite.all():
        filled[~finite] = 0.5 * (float(edges[0]) + float(edges[-1]))

    clipped = np.clip(filled, float(edges[0]), float(edges[-1]))
    return np.searchsorted(edges[1:-1], clipped, side="right").astype(np.int16)


def _center_interpolation(t, centers):
    n_rows = len(t)
    filled = np.asarray(t, dtype=np.float64).copy()
    finite = np.isfinite(filled)
    if not finite.all():
        filled[~finite] = 0.5 * (float(centers[0]) + float(centers[-1]))

    high = np.searchsorted(centers, filled, side="left").astype(np.int16)
    high = np.clip(high, 1, len(centers) - 1)
    low = (high - 1).astype(np.int16)

    denom = centers[high] - centers[low]
    weight = np.zeros(n_rows, dtype=np.float64)
    usable = np.abs(denom) > _EPSILON
    weight[usable] = (filled[usable] - centers[low[usable]]) / denom[usable]
    weight = np.clip(weight, 0.0, 1.0)

    low_extreme = filled <= centers[0]
    high_extreme = filled >= centers[-1]

    low[low_extreme] = 0
    high[low_extreme] = 0
    weight[low_extreme] = 0.0

    low[high_extreme] = len(centers) - 1
    high[high_extreme] = len(centers) - 1
    weight[high_extreme] = 0.0

    return low, high, weight


def _robust_loc_scale(matrix, fallback_loc, fallback_scale):
    fallback_loc = np.asarray(fallback_loc, dtype=np.float64)
    fallback_scale = np.maximum(np.asarray(fallback_scale, dtype=np.float64), _SCALE_FLOOR)

    if matrix.shape[0] == 0:
        return fallback_loc.copy(), fallback_scale.copy()

    loc = np.median(matrix, axis=0)
    scale = _MAD_SCALE * np.median(np.abs(matrix - loc), axis=0)

    loc_bad = ~np.isfinite(loc)
    if loc_bad.any():
        loc[loc_bad] = fallback_loc[loc_bad]

    scale_bad = (~np.isfinite(scale)) | (scale < _SCALE_FLOOR)
    if scale_bad.any():
        scale[scale_bad] = fallback_scale[scale_bad]

    scale = np.maximum(scale, _SCALE_FLOOR)
    return loc, scale


def _build_manifold_stats(colors, edge_bins, train_mask, pop_codes):
    n_features = colors.shape[1]
    zeros = np.zeros(n_features, dtype=np.float64)
    ones = np.ones(n_features, dtype=np.float64)
    identity = np.eye(n_features, dtype=np.float64)

    global_loc, global_scale = _robust_loc_scale(colors[train_mask], zeros, ones)

    pop_locs = np.empty((2, n_features), dtype=np.float64)
    pop_scales = np.empty((2, n_features), dtype=np.float64)

    for pop_idx in range(2):
        pop_mask = train_mask & (pop_codes == pop_idx)
        pop_n = float(pop_mask.sum())
        raw_loc, raw_scale = _robust_loc_scale(colors[pop_mask], global_loc, global_scale)
        shrink = _N_MIN / (pop_n + _N_MIN)
        pop_locs[pop_idx] = (1.0 - shrink) * raw_loc + shrink * global_loc
        pop_scales[pop_idx] = (1.0 - shrink) * raw_scale + shrink * global_scale
        pop_scales[pop_idx] = np.maximum(pop_scales[pop_idx], _SCALE_FLOOR)

    locs = np.empty((2, _N_BINS, n_features), dtype=np.float64)
    scales = np.empty((2, _N_BINS, n_features), dtype=np.float64)
    counts = np.zeros((2, _N_BINS), dtype=np.float64)
    shrink_weights = np.empty((2, _N_BINS), dtype=np.float64)
    inv_corrs = np.empty((2, _N_BINS, n_features, n_features), dtype=np.float64)

    for pop_idx in range(2):
        for bin_idx in range(_N_BINS):
            cell_mask = train_mask & (pop_codes == pop_idx) & (edge_bins == bin_idx)
            cell_n = int(cell_mask.sum())
            counts[pop_idx, bin_idx] = float(cell_n)

            if cell_n > 0:
                raw_loc, raw_scale = _robust_loc_scale(colors[cell_mask], pop_locs[pop_idx], pop_scales[pop_idx])
            else:
                raw_loc = pop_locs[pop_idx].copy()
                raw_scale = pop_scales[pop_idx].copy()

            shrink = _N_MIN / (float(cell_n) + _N_MIN)
            shrink_weights[pop_idx, bin_idx] = shrink

            locs[pop_idx, bin_idx] = (1.0 - shrink) * raw_loc + shrink * pop_locs[pop_idx]
            scales[pop_idx, bin_idx] = (1.0 - shrink) * raw_scale + shrink * pop_scales[pop_idx]
            scales[pop_idx, bin_idx] = np.maximum(scales[pop_idx, bin_idx], _SCALE_FLOOR)

            corr = identity.copy()
            if cell_n >= int(_N_MIN):
                standardized = (colors[cell_mask] - locs[pop_idx, bin_idx]) / scales[pop_idx, bin_idx]
                standardized = np.clip(standardized, -_RESIDUAL_CLIP, _RESIDUAL_CLIP)
                candidate = np.corrcoef(standardized, rowvar=False)

                if candidate.shape == corr.shape:
                    candidate = np.asarray(candidate, dtype=np.float64)
                    candidate[~np.isfinite(candidate)] = identity[~np.isfinite(candidate)]
                    candidate = np.clip(candidate, -_CORR_CLIP, _CORR_CLIP)
                    candidate = 0.5 * (candidate + candidate.T)
                    np.fill_diagonal(candidate, 1.0)
                    corr = candidate

            corr = (1.0 - shrink) * corr + shrink * identity
            np.fill_diagonal(corr, 1.0 + _RIDGE)

            try:
                inv_corrs[pop_idx, bin_idx] = np.linalg.pinv(corr, rcond=0.000001)
            except np.linalg.LinAlgError:
                inv_corrs[pop_idx, bin_idx] = identity / (1.0 + _RIDGE)

    return locs, scales, inv_corrs, counts, shrink_weights


def _interpolate_grid(grid, pop_codes, low_bins, high_bins, weights):
    lower = grid[pop_codes, low_bins]
    upper = grid[pop_codes, high_bins]

    if lower.ndim == 1:
        return lower * (1.0 - weights) + upper * weights

    weight_shape = (weights.shape[0],) + (1,) * (lower.ndim - 1)
    shaped_weights = weights.reshape(weight_shape)
    return lower * (1.0 - shaped_weights) + upper * shaped_weights


def _mahalanobis_sq(standardized, inv_corrs, pop_codes, low_bins, high_bins, weights):
    n_rows = standardized.shape[0]
    out = np.empty(n_rows, dtype=np.float64)

    for start in range(0, n_rows, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n_rows)
        lower = inv_corrs[pop_codes[start:end], low_bins[start:end]]
        upper = inv_corrs[pop_codes[start:end], high_bins[start:end]]
        shaped_weights = weights[start:end].reshape((end - start, 1, 1))
        inv_corr = lower * (1.0 - shaped_weights) + upper * shaped_weights

        x = standardized[start:end]
        values = np.einsum("ij,ijk,ik->i", x, inv_corr, x)
        bad = ~np.isfinite(values)
        if bad.any():
            fallback = np.sum(x * x, axis=1)
            values[bad] = fallback[bad]

        out[start:end] = np.maximum(values, 0.0)

    return out


def _float32(values):
    return np.asarray(values).astype(np.float32, copy=False)


def _add_distance_features(feature_data, prefix, standardized, inv_corrs, pop_codes, low_bins, high_bins, weights):
    abs_std = np.abs(standardized)
    diag_sq = np.sum(standardized * standardized, axis=1)
    diag_l2 = np.sqrt(np.maximum(diag_sq, 0.0))
    maha_sq = _mahalanobis_sq(standardized, inv_corrs, pop_codes, low_bins, high_bins, weights)
    maha = np.sqrt(np.maximum(maha_sq, 0.0))

    feature_data[prefix + "_diag_mean_abs_z"] = _float32(np.mean(abs_std, axis=1))
    feature_data[prefix + "_diag_max_abs_z"] = _float32(np.max(abs_std, axis=1))
    feature_data[prefix + "_diag_l2_distance"] = _float32(diag_l2)
    feature_data[prefix + "_diag_chi2_distance"] = _float32(diag_sq)
    feature_data[prefix + "_mahalanobis_distance"] = _float32(maha)
    feature_data[prefix + "_mahalanobis_sq_distance"] = _float32(maha_sq)

    return diag_l2, maha


def add_population_color_manifold_drifts(raw, deps, aux):
    _ = deps, aux

    n_rows = len(raw)
    colors = _color_matrix(raw)
    train_mask = _infer_train_mask(raw)
    pop_codes = _population_codes(raw)

    redshift_t = _redshift_coordinate(raw)
    edges = _redshift_bins(redshift_t, train_mask)
    centers = 0.5 * (edges[:-1] + edges[1:])

    edge_bins = _assign_edge_bins(redshift_t, edges)
    low_bins, high_bins, interp_weight = _center_interpolation(redshift_t, centers)

    locs, scales, inv_corrs, counts, shrink_weights = _build_manifold_stats(
        colors, edge_bins, train_mask, pop_codes
    )

    feature_data = {}

    for color_idx, (_, _, color_name) in enumerate(_COLOR_SPECS):
        feature_data["color_" + color_name] = _float32(colors[:, color_idx])

    feature_data["redshift_edge_bin"] = edge_bins
    feature_data["redshift_center_low_bin"] = low_bins
    feature_data["redshift_center_high_bin"] = high_bins
    feature_data["redshift_interp_weight"] = _float32(interp_weight)

    assigned_codes = pop_codes
    opposite_codes = 1 - pop_codes
    red_codes = np.zeros(n_rows, dtype=np.intp)
    blue_codes = np.ones(n_rows, dtype=np.intp)

    assigned_count = _interpolate_grid(counts, assigned_codes, low_bins, high_bins, interp_weight)
    opposite_count = _interpolate_grid(counts, opposite_codes, low_bins, high_bins, interp_weight)
    red_count = _interpolate_grid(counts, red_codes, low_bins, high_bins, interp_weight)
    blue_count = _interpolate_grid(counts, blue_codes, low_bins, high_bins, interp_weight)

    assigned_shrink = _interpolate_grid(shrink_weights, assigned_codes, low_bins, high_bins, interp_weight)
    opposite_shrink = _interpolate_grid(shrink_weights, opposite_codes, low_bins, high_bins, interp_weight)
    red_shrink = _interpolate_grid(shrink_weights, red_codes, low_bins, high_bins, interp_weight)
    blue_shrink = _interpolate_grid(shrink_weights, blue_codes, low_bins, high_bins, interp_weight)

    feature_data["assigned_effective_cell_count"] = _float32(assigned_count)
    feature_data["opposite_effective_cell_count"] = _float32(opposite_count)
    feature_data["red_sequence_effective_cell_count"] = _float32(red_count)
    feature_data["blue_cloud_effective_cell_count"] = _float32(blue_count)
    feature_data["assigned_log1p_effective_cell_count"] = _float32(np.log1p(assigned_count))
    feature_data["opposite_log1p_effective_cell_count"] = _float32(np.log1p(opposite_count))
    feature_data["assigned_shrinkage_weight"] = _float32(assigned_shrink)
    feature_data["opposite_shrinkage_weight"] = _float32(opposite_shrink)
    feature_data["red_sequence_shrinkage_weight"] = _float32(red_shrink)
    feature_data["blue_cloud_shrinkage_weight"] = _float32(blue_shrink)

    assigned_loc = _interpolate_grid(locs, assigned_codes, low_bins, high_bins, interp_weight)
    assigned_scale = _interpolate_grid(scales, assigned_codes, low_bins, high_bins, interp_weight)
    assigned_std = np.clip((colors - assigned_loc) / assigned_scale, -_RESIDUAL_CLIP, _RESIDUAL_CLIP)
    assigned_diag, assigned_maha = _add_distance_features(
        feature_data,
        "assigned",
        assigned_std,
        inv_corrs,
        assigned_codes,
        low_bins,
        high_bins,
        interp_weight,
    )

    opposite_loc = _interpolate_grid(locs, opposite_codes, low_bins, high_bins, interp_weight)
    opposite_scale = _interpolate_grid(scales, opposite_codes, low_bins, high_bins, interp_weight)
    opposite_std = np.clip((colors - opposite_loc) / opposite_scale, -_RESIDUAL_CLIP, _RESIDUAL_CLIP)
    opposite_diag, opposite_maha = _add_distance_features(
        feature_data,
        "opposite",
        opposite_std,
        inv_corrs,
        opposite_codes,
        low_bins,
        high_bins,
        interp_weight,
    )

    feature_data["diag_l2_log_diff_assigned_minus_opposite"] = _float32(
        np.log1p(assigned_diag) - np.log1p(opposite_diag)
    )
    feature_data["diag_l2_ratio_assigned_to_opposite"] = _float32(
        (assigned_diag + _EPSILON) / (opposite_diag + _EPSILON)
    )
    feature_data["diag_l2_min_population_distance"] = _float32(np.minimum(assigned_diag, opposite_diag))
    feature_data["diag_l2_distance_gap_assigned_minus_opposite"] = _float32(assigned_diag - opposite_diag)

    feature_data["mahalanobis_log_diff_assigned_minus_opposite"] = _float32(
        np.log1p(assigned_maha) - np.log1p(opposite_maha)
    )
    feature_data["mahalanobis_ratio_assigned_to_opposite"] = _float32(
        (assigned_maha + _EPSILON) / (opposite_maha + _EPSILON)
    )
    feature_data["mahalanobis_min_population_distance"] = _float32(np.minimum(assigned_maha, opposite_maha))
    feature_data["mahalanobis_distance_gap_assigned_minus_opposite"] = _float32(assigned_maha - opposite_maha)

    red_loc = _interpolate_grid(locs, red_codes, low_bins, high_bins, interp_weight)
    blue_loc = _interpolate_grid(locs, blue_codes, low_bins, high_bins, interp_weight)
    red_scale = _interpolate_grid(scales, red_codes, low_bins, high_bins, interp_weight)
    blue_scale = _interpolate_grid(scales, blue_codes, low_bins, high_bins, interp_weight)

    assigned_side = np.where(pop_codes == 0, 1.0, -1.0)
    tag_aligned_sum = np.zeros(n_rows, dtype=np.float64)
    separator_abs_sum = np.zeros(n_rows, dtype=np.float64)

    for color_idx, (_, _, color_name) in enumerate(_COLOR_SPECS):
        red_median = red_loc[:, color_idx]
        blue_median = blue_loc[:, color_idx]
        color_values = colors[:, color_idx]
        median_gap = red_median - blue_median
        gap_sign = np.sign(median_gap)

        separator_margin = (color_values - 0.5 * (red_median + blue_median)) * gap_sign
        tag_aligned = separator_margin * assigned_side

        red_abs_z = np.minimum(np.abs((color_values - red_median) / red_scale[:, color_idx]), _RESIDUAL_CLIP)
        blue_abs_z = np.minimum(np.abs((color_values - blue_median) / blue_scale[:, color_idx]), _RESIDUAL_CLIP)

        feature_data["separator_margin_" + color_name] = _float32(separator_margin)
        feature_data["tag_aligned_separator_margin_" + color_name] = _float32(tag_aligned)
        feature_data["abs_population_median_gap_" + color_name] = _float32(np.abs(median_gap))
        feature_data["red_sequence_abs_z_resid_" + color_name] = _float32(red_abs_z)
        feature_data["blue_cloud_abs_z_resid_" + color_name] = _float32(blue_abs_z)

        tag_aligned_sum += tag_aligned
        separator_abs_sum += np.abs(separator_margin)

    feature_data["tag_aligned_separator_margin_mean"] = _float32(tag_aligned_sum / float(len(_COLOR_SPECS)))
    feature_data["separator_margin_mean_abs"] = _float32(separator_abs_sum / float(len(_COLOR_SPECS)))

    return pd.DataFrame(feature_data, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "population_color_manifold_drifts",
        "fn": add_population_color_manifold_drifts,
        "depends_on": [],
        "description": "Redshift-local, population-conditioned color manifold residuals, separators, distances, and reliability measures.",
    }
]