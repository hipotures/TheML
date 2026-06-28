import numpy as np
import pandas as pd


BANDS = ("u", "g", "r", "i", "z")
NUMERIC_COLUMNS = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
WAVELENGTH_ANGSTROM = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
EPSILON = 1.0e-9


def _safe_numeric_series(raw, column_name, index):
    if column_name in raw.columns:
        return pd.to_numeric(raw[column_name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=index, dtype="float64")


def _safe_scale(value, eps):
    if not np.isfinite(value) or abs(value) < eps:
        return eps
    return float(value)


def _robust_center_scale(values, eps):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        q25, q75 = np.nanquantile(finite, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < eps:
        scale = float(np.nanstd(finite))
    return center, _safe_scale(scale, eps)


def _support_percentile(values):
    series = pd.Series(values)
    rank_pct = series.rank(method="average", pct=True, na_option="keep").to_numpy(dtype="float64")
    support = np.minimum(rank_pct, 1.0 - rank_pct)
    return np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)


def _tail_features(values, prefix, eps):
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        q001 = q01 = q99 = q999 = center = 0.0
        scale = 1.0
    else:
        q001, q01, q99, q999 = np.nanquantile(finite, [0.001, 0.01, 0.99, 0.999])
        center, scale = _robust_center_scale(finite, eps)

    lower_denom = _safe_scale(float(q01 - q001), eps)
    upper_denom = _safe_scale(float(q999 - q99), eps)

    clean = np.nan_to_num(arr, nan=center, posinf=q999, neginf=q001)
    lower_tail = np.maximum(0.0, (float(q001) - clean) / lower_denom)
    upper_tail = np.maximum(0.0, (clean - float(q999)) / upper_denom)
    two_sided_tail = np.maximum(lower_tail, upper_tail)
    robust_z = (clean - center) / scale
    support = _support_percentile(clean)

    return {
        prefix + "_lower_tail_margin": np.clip(lower_tail, 0.0, 10.0),
        prefix + "_upper_tail_margin": np.clip(upper_tail, 0.0, 10.0),
        prefix + "_two_sided_tail_margin": np.clip(two_sided_tail, 0.0, 10.0),
        prefix + "_support_edge_pct": np.clip(support, 0.0, 0.5),
        prefix + "_robust_center_z": np.clip(robust_z, -10.0, 10.0),
        prefix + "_beyond_q001_q999": (lower_tail > 0.0) | (upper_tail > 0.0),
    }


def _continuum_residual_features(band_matrix, eps):
    n_rows, n_bands = band_matrix.shape
    log_wave = np.log(np.asarray(WAVELENGTH_ANGSTROM, dtype="float64"))
    x = log_wave - log_wave.mean()
    denom = _safe_scale(float(np.sum(x * x)), eps)

    row_mean = np.nanmean(band_matrix, axis=1)
    centered_y = band_matrix - row_mean[:, None]
    slopes = np.nansum(centered_y * x[None, :], axis=1) / denom
    fitted = row_mean[:, None] + slopes[:, None] * x[None, :]
    residuals = band_matrix - fitted

    band_centers = np.nanmedian(residuals, axis=0)
    band_scales = np.nanmedian(np.abs(residuals - band_centers[None, :]), axis=0) * 1.4826
    band_scales = np.where(np.isfinite(band_scales) & (band_scales >= eps), band_scales, 1.0)

    standardized = (residuals - band_centers[None, :]) / band_scales[None, :]
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    abs_standardized = np.abs(standardized)
    max_band_index = np.argmax(abs_standardized, axis=1)

    features = {
        "continuum_max_isolated_residual": np.clip(np.max(abs_standardized, axis=1), 0.0, 10.0),
        "continuum_residual_dispersion": np.clip(np.nanstd(standardized, axis=1), 0.0, 10.0),
        "continuum_bands_over_2sigma": np.sum(abs_standardized > 2.0, axis=1).astype("int16"),
        "continuum_bands_over_4sigma": np.sum(abs_standardized > 4.0, axis=1).astype("int16"),
        "continuum_blue_u_signed_residual": np.clip(standardized[:, 0], -10.0, 10.0),
        "continuum_red_z_signed_residual": np.clip(standardized[:, 4], -10.0, 10.0),
    }

    for idx, band in enumerate(BANDS):
        features["continuum_pathological_band_is_" + band] = max_band_index == idx
        features["continuum_" + band + "_signed_residual"] = np.clip(standardized[:, idx], -10.0, 10.0)

    return features


def add_surrogate_photometry_flag_margins(raw, deps, aux):
    index = raw.index
    eps = EPSILON
    out = {}

    base = {}
    for column_name in NUMERIC_COLUMNS:
        base[column_name] = _safe_numeric_series(raw, column_name, index).to_numpy(dtype="float64")

    tail_columns = {}
    for column_name in NUMERIC_COLUMNS:
        tail_columns[column_name] = base[column_name]

    for left, right in zip(BANDS[:-1], BANDS[1:]):
        tail_columns[left + "_minus_" + right] = base[left] - base[right]

    for left_index, left in enumerate(BANDS):
        for right in BANDS[left_index + 1:]:
            tail_columns[left + "_minus_" + right] = base[left] - base[right]

    numeric_tail_arrays = []
    numeric_extreme_arrays = []
    color_tail_arrays = []
    color_extreme_arrays = []

    for feature_name, values in tail_columns.items():
        local = _tail_features(values, feature_name, eps)
        tail = local[feature_name + "_two_sided_tail_margin"]
        extreme = local[feature_name + "_beyond_q001_q999"]

        if "_minus_" in feature_name:
            color_tail_arrays.append(tail)
            color_extreme_arrays.append(extreme)
        else:
            numeric_tail_arrays.append(tail)
            numeric_extreme_arrays.append(extreme)

        for output_name, output_values in local.items():
            out[output_name] = output_values

    band_matrix = np.column_stack([base[band] for band in BANDS])
    out.update(_continuum_residual_features(band_matrix, eps))

    if numeric_tail_arrays:
        numeric_tails = np.column_stack(numeric_tail_arrays)
        numeric_extremes = np.column_stack(numeric_extreme_arrays)
        out["max_numeric_tail_margin"] = np.clip(np.max(numeric_tails, axis=1), 0.0, 10.0)
        out["count_numeric_beyond_support"] = np.sum(numeric_extremes, axis=1).astype("int16")

    if color_tail_arrays:
        color_tails = np.column_stack(color_tail_arrays)
        color_extremes = np.column_stack(color_extreme_arrays)
        out["max_color_tail_margin"] = np.clip(np.max(color_tails, axis=1), 0.0, 10.0)
        out["count_extreme_colors"] = np.sum(color_extremes, axis=1).astype("int16")

    alpha = np.nan_to_num(base["alpha"], nan=180.0, posinf=360.0, neginf=0.0)
    out["alpha_wrap_edge_distance"] = np.clip(np.minimum(alpha, 360.0 - alpha), 0.0, 180.0)

    delta = base["delta"]
    finite_delta = delta[np.isfinite(delta)]
    if finite_delta.size:
        delta_min = float(np.nanmin(finite_delta))
        delta_max = float(np.nanmax(finite_delta))
    else:
        delta_min = -90.0
        delta_max = 90.0
    clean_delta = np.nan_to_num(delta, nan=(delta_min + delta_max) * 0.5, posinf=delta_max, neginf=delta_min)
    out["delta_survey_edge_distance"] = np.clip(np.minimum(clean_delta - delta_min, delta_max - clean_delta), 0.0, None)

    redshift = np.nan_to_num(base["redshift"], nan=0.0, posinf=7.01, neginf=-0.01)
    out["redshift_low_cap_proximity"] = np.clip(0.05 - np.abs(redshift - -0.01), 0.0, 0.05) / 0.05
    out["redshift_high_cap_proximity"] = np.clip(0.05 - np.abs(7.01 - redshift), 0.0, 0.05) / 0.05
    out["redshift_near_any_cap"] = np.maximum(out["redshift_low_cap_proximity"], out["redshift_high_cap_proximity"])

    result = pd.DataFrame(out, index=index)
    return result


FEATURE_GROUPS = [
    {
        "name": "surrogate_photometry_flag_margins",
        "fn": add_surrogate_photometry_flag_margins,
        "depends_on": [],
        "description": "Surrogate photometry quality and support-boundary margins from magnitudes, colors, coordinates, redshift, and continuum residuals.",
    }
]