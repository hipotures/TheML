import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
SOFTENING_BY_BAND = {
    "u": 1.4e-10,
    "g": 0.9e-10,
    "r": 1.2e-10,
    "i": 1.8e-10,
    "z": 7.4e-10,
}
REDSHIFT_BIN_EDGES = (-1.0e300, 0.02, 0.1, 0.4, 0.8, 1.5, 3.0, 1.0e300)
CENTERED_BAND_COORDS = (-2.0, -1.0, 0.0, 1.0, 2.0)
TRAIN_ID_MAX = 577346
RIDGE_PENALTY = 1.0e-6
MIN_EFFECTIVE_WEIGHT = 1.0e-4
EPS = 1.0e-12


def _finite_median(values, default):
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    value = float(np.median(arr))
    if not np.isfinite(value):
        return float(default)
    return value


def _training_mask(raw):
    n_rows = len(raw)
    if "id" not in raw.columns:
        return np.ones(n_rows, dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(ids) & (ids <= TRAIN_ID_MAX)
    if np.any(mask):
        return mask
    return np.ones(n_rows, dtype=bool)


def _redshift_bins(raw):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    internal_edges = np.asarray(REDSHIFT_BIN_EDGES[1:-1], dtype=np.float64)
    bins = np.searchsorted(internal_edges, redshift, side="left")
    bins = np.clip(bins, 0, len(REDSHIFT_BIN_EDGES) - 2)
    bins = np.where(np.isfinite(redshift), bins, -1)
    return bins.astype(np.int16, copy=False)


def _asinh_flux_and_weight(magnitudes, softening):
    log_coeff = np.log(10.0) / 2.5
    a = -(log_coeff * magnitudes) - np.log(softening)
    a = np.clip(a, -35.0, 35.0)

    x = np.sinh(a)
    q = 2.0 * softening * x
    weight = np.abs(x) / np.sqrt(1.0 + x * x)

    q = np.where(np.isfinite(q), q, np.nan)
    weight = np.where(np.isfinite(weight), weight, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    return q, weight


def _weights_from_flux(q_matrix):
    weights = np.empty_like(q_matrix, dtype=np.float64)
    for band_index, band in enumerate(BANDS):
        softening = SOFTENING_BY_BAND[band]
        x = q_matrix[:, band_index] / (2.0 * softening)
        weight = np.abs(x) / np.sqrt(1.0 + x * x)
        weight = np.where(np.isfinite(weight), weight, 0.0)
        weights[:, band_index] = np.clip(weight, 0.0, 1.0)
    return weights


def _weighted_median_5(values, weights):
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)

    total_weight = np.sum(sorted_weights, axis=1)
    cumulative_weight = np.cumsum(sorted_weights, axis=1)
    cutoff = 0.5 * total_weight

    median_index = np.argmax(cumulative_weight >= cutoff[:, None], axis=1)
    medians = sorted_values[np.arange(values.shape[0]), median_index]
    medians = np.where(total_weight > 0.0, medians, np.nan)
    return medians


def _fallback_q_values(raw, q_matrix, redshift_bins, stats_mask, global_medians):
    spectral_type = raw["spectral_type"].astype("object")
    spectral_type = spectral_type.where(spectral_type.notna(), "__missing__")
    galaxy_population = raw["galaxy_population"].astype("object")
    galaxy_population = galaxy_population.where(galaxy_population.notna(), "__missing__")

    meta = pd.DataFrame(
        {
            "redshift_bin": redshift_bins,
            "spectral_type": spectral_type.to_numpy(),
            "galaxy_population": galaxy_population.to_numpy(),
        },
        index=raw.index,
    )

    train_frame = meta.loc[stats_mask].copy()
    for band_index, band in enumerate(BANDS):
        train_frame[band] = q_matrix[stats_mask, band_index]

    group_keys = ["redshift_bin", "spectral_type", "galaxy_population"]
    cell_medians = train_frame.groupby(group_keys, observed=True, dropna=False)[list(BANDS)].median()
    bin_medians = train_frame.groupby("redshift_bin", observed=True, dropna=False)[list(BANDS)].median()

    cell_lookup = meta.join(cell_medians, on=group_keys)
    bin_lookup = meta[["redshift_bin"]].join(bin_medians, on="redshift_bin")

    fallback = cell_lookup.loc[:, list(BANDS)].to_numpy(dtype=np.float64, copy=True)
    bin_values = bin_lookup.loc[:, list(BANDS)].to_numpy(dtype=np.float64, copy=True)

    for band_index in range(len(BANDS)):
        missing = ~np.isfinite(fallback[:, band_index])
        fallback[missing, band_index] = bin_values[missing, band_index]

        missing = ~np.isfinite(fallback[:, band_index])
        fallback[missing, band_index] = global_medians[band_index]

    return fallback


def _finite_feature(values):
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, 0.0)


def add_asinh_jacobian_weighted_shape(raw, deps, aux):
    n_rows = len(raw)

    q_matrix = np.empty((n_rows, len(BANDS)), dtype=np.float64)
    initial_weights = np.empty((n_rows, len(BANDS)), dtype=np.float64)

    for band_index, band in enumerate(BANDS):
        magnitudes = pd.to_numeric(raw[band], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        q_values, weights = _asinh_flux_and_weight(magnitudes, SOFTENING_BY_BAND[band])
        q_matrix[:, band_index] = q_values
        initial_weights[:, band_index] = weights

    finite_q_rows = np.all(np.isfinite(q_matrix), axis=1)
    initial_n_eff = np.sum(initial_weights, axis=1)
    low_confidence = (~finite_q_rows) | (initial_n_eff < MIN_EFFECTIVE_WEIGHT)

    train_mask = _training_mask(raw)
    stats_mask = train_mask & finite_q_rows
    if not np.any(stats_mask):
        stats_mask = finite_q_rows
    if not np.any(stats_mask):
        stats_mask = np.ones(n_rows, dtype=bool)

    global_medians = np.empty(len(BANDS), dtype=np.float64)
    for band_index in range(len(BANDS)):
        global_medians[band_index] = _finite_median(q_matrix[stats_mask, band_index], 0.0)

    global_abs_q_median = _finite_median(np.abs(q_matrix[stats_mask]), EPS)
    global_abs_q_median = max(global_abs_q_median, EPS)

    q_final = q_matrix.copy()
    if np.any(low_confidence):
        redshift_bins = _redshift_bins(raw)
        fallback_q = _fallback_q_values(raw, q_matrix, redshift_bins, stats_mask, global_medians)
        q_final[low_confidence, :] = fallback_q[low_confidence, :]

    q_final = np.where(np.isfinite(q_final), q_final, global_medians[None, :])
    shape_weights = _weights_from_flux(q_final)

    shape_n_eff = np.sum(shape_weights, axis=1)
    q_level = np.sum(shape_weights * q_final, axis=1) / np.maximum(shape_n_eff, 1.0e-6)

    abs_deviation = np.abs(q_final - q_level[:, None])
    weighted_abs_deviation = _weighted_median_5(abs_deviation, shape_weights)
    weighted_abs_deviation = np.where(np.isfinite(weighted_abs_deviation), weighted_abs_deviation, 0.0)
    q_scale = np.maximum(np.maximum(weighted_abs_deviation, global_abs_q_median), EPS)

    y = (q_final - q_level[:, None]) / q_scale[:, None]

    t = np.asarray(CENTERED_BAND_COORDS, dtype=np.float64)
    design = np.column_stack((np.ones(len(BANDS), dtype=np.float64), t, t * t - 2.0))

    lhs = np.einsum("nb,bp,bq->npq", shape_weights, design, design, optimize=True)
    lhs[:, 0, 0] += RIDGE_PENALTY
    lhs[:, 1, 1] += RIDGE_PENALTY
    lhs[:, 2, 2] += RIDGE_PENALTY

    rhs = np.einsum("nb,bp,nb->np", shape_weights, design, y, optimize=True)
    coeffs = np.linalg.solve(lhs, rhs[:, :, None])[:, :, 0]

    fitted = coeffs @ design.T
    residuals = y - fitted
    resid_rms = np.sqrt(np.sum(shape_weights * residuals * residuals, axis=1) / np.maximum(shape_n_eff, 1.0e-6))
    resid_mae = np.sum(shape_weights * np.abs(residuals), axis=1) / np.maximum(shape_n_eff, 1.0e-6)

    pair_left = (0, 1, 2, 3)
    pair_right = (1, 2, 3, 4)
    adjacent_contrasts = []
    adjacent_abs_unweighted = []
    adjacent_reliable = []

    for left, right in zip(pair_left, pair_right):
        pair_weight = np.sqrt(
            shape_weights[:, left] * shape_weights[:, right]
            / np.maximum(shape_weights[:, left] + shape_weights[:, right], 1.0e-6)
        )
        contrast = pair_weight * (q_final[:, left] - q_final[:, right]) / q_scale
        adjacent_contrasts.append(contrast)

        descriptor_pair_ok = (initial_weights[:, left] > 0.1) & (initial_weights[:, right] > 0.1)
        adjacent_reliable.append(descriptor_pair_ok)
        adjacent_abs_unweighted.append(np.abs((q_final[:, left] - q_final[:, right]) / q_scale))

    reliable_sum = np.zeros(n_rows, dtype=np.float64)
    reliable_count = np.zeros(n_rows, dtype=np.float64)
    for pair_values, pair_ok in zip(adjacent_abs_unweighted, adjacent_reliable):
        reliable_sum += np.where(pair_ok, pair_values, 0.0)
        reliable_count += pair_ok.astype(np.float64)
    mean_reliable_adj_abs_contrast = np.where(reliable_count > 0.0, reliable_sum / reliable_count, 0.0)

    curvature_contrasts = []
    for left, mid, right in ((0, 1, 2), (1, 2, 3), (2, 3, 4)):
        endpoint_weight = np.sqrt(
            shape_weights[:, left] * shape_weights[:, right]
            / np.maximum(shape_weights[:, left] + shape_weights[:, right], 1.0e-6)
        )
        curvature = endpoint_weight * (q_final[:, left] - 2.0 * q_final[:, mid] + q_final[:, right]) / q_scale
        curvature_contrasts.append(curvature)

    descriptor_weights = np.where(np.isfinite(initial_weights), initial_weights, 0.0)

    return pd.DataFrame(
        {
            "s0": _finite_feature(coeffs[:, 0]),
            "s1": _finite_feature(coeffs[:, 1]),
            "s2": _finite_feature(coeffs[:, 2]),
            "resid_rms": _finite_feature(resid_rms),
            "resid_mae": _finite_feature(resid_mae),
            "contrast_ug": _finite_feature(adjacent_contrasts[0]),
            "contrast_gr": _finite_feature(adjacent_contrasts[1]),
            "contrast_ri": _finite_feature(adjacent_contrasts[2]),
            "contrast_iz": _finite_feature(adjacent_contrasts[3]),
            "curvature_ugr": _finite_feature(curvature_contrasts[0]),
            "curvature_gri": _finite_feature(curvature_contrasts[1]),
            "curvature_riz": _finite_feature(curvature_contrasts[2]),
            "n_eff": _finite_feature(initial_n_eff),
            "frac_hi": np.mean(descriptor_weights > 0.6, axis=1),
            "frac_mid": np.mean(descriptor_weights > 0.2, axis=1),
            "min_w": np.min(descriptor_weights, axis=1),
            "max_w": np.max(descriptor_weights, axis=1),
            "mean_reliable_adj_abs_contrast": _finite_feature(mean_reliable_adj_abs_contrast),
            "low_confidence": low_confidence.astype(np.int8),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "asinh_jacobian_weighted_shape",
        "fn": add_asinh_jacobian_weighted_shape,
        "depends_on": [],
        "description": "Jacobian-weighted asinh flux shape features that summarize reliable ugriz continuum slope, curvature, contrast, and low-confidence fallbacks.",
    }
]