import numpy as np
import pandas as pd

REFERENCE_NUMERIC_COLUMNS = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
PHOTOMETRY_COLUMNS = ("u", "g", "r", "i", "z")
MARGINAL_COLUMNS = ("sin_alpha", "cos_alpha", "delta", "u", "g", "r", "i", "z", "redshift")


def _clean_reference_frame(frame, fallback_index=None):
    if frame is None:
        return pd.DataFrame(index=fallback_index)

    cleaned = pd.DataFrame(index=frame.index)

    if "alpha" in frame.columns:
        alpha = pd.to_numeric(frame["alpha"], errors="coerce").to_numpy(dtype="float64", copy=True)
        finite = np.isfinite(alpha)
        wrapped = np.full(alpha.shape[0], np.nan, dtype="float64")
        wrapped[finite] = np.mod(alpha[finite], 360.0)
        angle = 2.0 * np.pi * wrapped / 360.0
        cleaned["sin_alpha"] = np.sin(angle)
        cleaned["cos_alpha"] = np.cos(angle)

    for col in ("delta", "u", "g", "r", "i", "z", "redshift"):
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64", copy=True)
        if col in PHOTOMETRY_COLUMNS:
            values[values <= -9000.0] = np.nan
        if col == "redshift":
            finite = np.isfinite(values)
            values[finite & (values < 0.0)] = 0.0
        cleaned[col] = values

    return cleaned


def _positive_finite(value):
    return bool(np.isfinite(value) and value > 0.0)


def _fit_auxiliary_statistics(aux_clean, candidate_columns):
    medians = {}
    initial_scales = {}
    sorted_values = {}
    positive_scales = []

    for col in candidate_columns:
        values = aux_clean[col].to_numpy(dtype="float64", copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        values = np.sort(values)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = 1.4826 * mad

        if not _positive_finite(scale):
            q25, q75 = np.percentile(values, (25.0, 75.0))
            scale = float((q75 - q25) / 1.349)

        if _positive_finite(scale):
            positive_scales.append(scale)

        medians[col] = median
        initial_scales[col] = scale
        sorted_values[col] = values

    fallback_scale = np.nan
    if positive_scales:
        fallback_scale = float(np.median(np.asarray(positive_scales, dtype="float64")))

    usable_columns = []
    scales = {}
    for col in candidate_columns:
        if col not in medians:
            continue
        scale = initial_scales[col]
        if not _positive_finite(scale):
            scale = fallback_scale
        if _positive_finite(scale):
            usable_columns.append(col)
            scales[col] = float(scale)

    return {
        "usable_columns": usable_columns,
        "medians": medians,
        "scales": scales,
        "sorted_values": sorted_values,
    }


def _assign_neutral_marginal_features(out, raw_clean):
    n_rows = len(out.index)

    for col in MARGINAL_COLUMNS:
        if col in raw_clean.columns:
            values = raw_clean[col].to_numpy(dtype="float64", copy=False)
            missing = ~np.isfinite(values)
        else:
            missing = np.ones(n_rows, dtype=bool)

        out["missing_" + col] = missing.astype("int8")
        out["robust_z_" + col] = np.zeros(n_rows, dtype="float32")
        out["abs_robust_z_" + col] = np.zeros(n_rows, dtype="float32")
        out["signed_median_delta_" + col] = np.zeros(n_rows, dtype="float32")
        out["abs_median_delta_" + col] = np.zeros(n_rows, dtype="float32")
        out["empirical_cdf_" + col] = np.full(n_rows, 0.5, dtype="float32")


def _auxiliary_z_matrix(aux_clean, usable_columns, medians, scales, clip_value):
    columns = []
    n_rows = len(aux_clean.index)

    for col in usable_columns:
        values = aux_clean[col].to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(values)
        z_values = np.full(n_rows, np.nan, dtype="float64")
        z_values[valid] = np.clip((values[valid] - medians[col]) / scales[col], -clip_value, clip_value)
        columns.append(z_values)

    if not columns:
        return np.empty((n_rows, 0), dtype="float64")

    return np.column_stack(columns)


def _diagonal_variance(aux_z):
    variances = []
    for idx in range(aux_z.shape[1]):
        values = aux_z[:, idx]
        values = values[np.isfinite(values)]
        if values.size >= 2:
            variance = float(np.var(values, ddof=1))
        else:
            variance = 1.0
        if not _positive_finite(variance):
            variance = 1.0
        variances.append(max(variance, 1.0e-8))
    return np.asarray(variances, dtype="float64")


def _regularized_covariance(aux_z):
    d_full = aux_z.shape[1]
    if d_full < 2:
        return None, False

    complete = np.isfinite(aux_z).all(axis=1)
    complete_z = aux_z[complete]
    if complete_z.shape[0] < max(d_full + 2, 10):
        return None, False

    covariance = np.cov(complete_z, rowvar=False)
    if covariance.shape != (d_full, d_full):
        return None, False
    if not np.isfinite(covariance).all():
        return None, False
    if np.linalg.matrix_rank(covariance) < d_full:
        return None, False

    trace = float(np.trace(covariance))
    if not np.isfinite(trace) or trace <= 0.0:
        return None, False

    ridge = max(1.0e-6 * trace / float(d_full), 1.0e-8)
    covariance = covariance + np.eye(d_full, dtype="float64") * ridge

    if not np.isfinite(covariance).all():
        return None, False

    return covariance, True


def _mahalanobis_squared(z_matrix, valid_matrix, covariance, diag_variance, use_full_covariance):
    n_rows, d_full = z_matrix.shape
    result = np.zeros(n_rows, dtype="float64")
    if d_full == 0 or n_rows == 0:
        return result

    bit_weights = np.asarray([1 << idx for idx in range(d_full)], dtype="uint16")
    mask_codes = valid_matrix.astype("uint16").dot(bit_weights)

    for code in np.unique(mask_codes):
        code_int = int(code)
        if code_int == 0:
            continue

        row_mask = mask_codes == code
        obs_idx = np.flatnonzero((code_int & bit_weights) != 0)
        if obs_idx.size == 0:
            continue

        z_sub = z_matrix[row_mask][:, obs_idx]

        if use_full_covariance:
            sub_cov = covariance[np.ix_(obs_idx, obs_idx)]
            try:
                inv_cov = np.linalg.inv(sub_cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(sub_cov)
            distances = np.einsum("ij,jk,ik->i", z_sub, inv_cov, z_sub)
        else:
            distances = np.sum((z_sub * z_sub) / diag_variance[obs_idx], axis=1)

        distances = distances * (float(d_full) / float(obs_idx.size))
        result[row_mask] = distances

    result[~np.isfinite(result)] = 0.0
    result[result < 0.0] = 0.0
    return result


def add_aide_aux_reference_distribution_distance(raw, deps, aux):
    raw_clean = _clean_reference_frame(raw)
    aux_clean = _clean_reference_frame(aux, fallback_index=None)

    out = pd.DataFrame(index=raw.index)
    aux_count = len(aux_clean.index)
    candidate_columns = [col for col in MARGINAL_COLUMNS if col in raw_clean.columns and col in aux_clean.columns]

    stats = _fit_auxiliary_statistics(aux_clean, candidate_columns)
    usable_columns = stats["usable_columns"]
    reference_available = aux_count > 0 and len(usable_columns) >= 3

    _assign_neutral_marginal_features(out, raw_clean)

    out["reference_available"] = bool(reference_available)
    out["reference_unavailable"] = bool(not reference_available)
    out["auxiliary_reference_count"] = np.int32(aux_count)
    out["usable_marginal_count"] = np.int16(len(usable_columns))
    out["covariance_full_used"] = False

    n_rows = len(raw.index)

    if not reference_available:
        out["missing_count"] = np.zeros(n_rows, dtype="int16")
        out["available_ratio"] = np.zeros(n_rows, dtype="float32")
        out["mean_abs_z"] = np.zeros(n_rows, dtype="float32")
        out["max_abs_z"] = np.zeros(n_rows, dtype="float32")
        out["rms_abs_z"] = np.zeros(n_rows, dtype="float32")
        out["mean_rank_shift"] = np.zeros(n_rows, dtype="float32")
        out["max_rank_shift"] = np.zeros(n_rows, dtype="float32")
        out["mean_signed_z"] = np.zeros(n_rows, dtype="float32")
        out["mahalanobis_sq"] = np.zeros(n_rows, dtype="float32")
        out["mahalanobis"] = np.zeros(n_rows, dtype="float32")
        out["log1p_mahalanobis_sq"] = np.zeros(n_rows, dtype="float32")
        out["joint_available_count"] = np.zeros(n_rows, dtype="int16")
        out["joint_available_ratio"] = np.zeros(n_rows, dtype="float32")
        return out

    medians = stats["medians"]
    scales = stats["scales"]
    sorted_values = stats["sorted_values"]

    z_columns = []
    cdf_columns = []
    valid_columns = []

    for col in usable_columns:
        values = raw_clean[col].to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(values)
        median = medians[col]
        scale = scales[col]
        delta = values - median

        robust_z = np.zeros(n_rows, dtype="float64")
        signed_delta = np.zeros(n_rows, dtype="float64")
        empirical_cdf = np.full(n_rows, 0.5, dtype="float64")

        robust_z[valid] = np.clip(delta[valid] / scale, -8.0, 8.0)
        signed_delta[valid] = delta[valid]

        ref_values = sorted_values[col]
        query = values[valid]
        count_less = np.searchsorted(ref_values, query, side="left")
        count_leq = np.searchsorted(ref_values, query, side="right")
        count_equal = count_leq - count_less
        positions = (count_less.astype("float64") + 0.5 * count_equal.astype("float64") + 0.5) / (float(ref_values.size) + 1.0)
        empirical_cdf[valid] = np.clip(positions, 1.0e-6, 1.0 - 1.0e-6)

        out["missing_" + col] = (~valid).astype("int8")
        out["robust_z_" + col] = robust_z.astype("float32")
        out["abs_robust_z_" + col] = np.abs(robust_z).astype("float32")
        out["signed_median_delta_" + col] = signed_delta.astype("float32")
        out["abs_median_delta_" + col] = np.abs(signed_delta).astype("float32")
        out["empirical_cdf_" + col] = empirical_cdf.astype("float32")

        z_columns.append(robust_z)
        cdf_columns.append(empirical_cdf)
        valid_columns.append(valid)

    z_matrix = np.column_stack(z_columns)
    cdf_matrix = np.column_stack(cdf_columns)
    valid_matrix = np.column_stack(valid_columns)

    available_count = valid_matrix.sum(axis=1).astype("int16")
    d_full = len(usable_columns)
    missing_count = (d_full - available_count).astype("int16")
    available_ratio = available_count.astype("float64") / float(d_full)

    abs_z = np.abs(z_matrix)
    rank_shift = np.abs(cdf_matrix - 0.5)
    has_available = available_count > 0

    mean_abs_z = np.divide(abs_z.sum(axis=1), available_count, out=np.zeros(n_rows, dtype="float64"), where=has_available)
    rms_abs_z = np.sqrt(np.divide((abs_z * abs_z).sum(axis=1), available_count, out=np.zeros(n_rows, dtype="float64"), where=has_available))
    mean_rank_shift = np.divide(rank_shift.sum(axis=1), available_count, out=np.zeros(n_rows, dtype="float64"), where=has_available)
    mean_signed_z = np.divide(z_matrix.sum(axis=1), available_count, out=np.zeros(n_rows, dtype="float64"), where=has_available)

    out["missing_count"] = missing_count
    out["available_ratio"] = available_ratio.astype("float32")
    out["mean_abs_z"] = mean_abs_z.astype("float32")
    out["max_abs_z"] = abs_z.max(axis=1).astype("float32")
    out["rms_abs_z"] = rms_abs_z.astype("float32")
    out["mean_rank_shift"] = mean_rank_shift.astype("float32")
    out["max_rank_shift"] = rank_shift.max(axis=1).astype("float32")
    out["mean_signed_z"] = mean_signed_z.astype("float32")

    aux_z = _auxiliary_z_matrix(aux_clean, usable_columns, medians, scales, 6.0)
    diag_variance = _diagonal_variance(aux_z)
    covariance, use_full_covariance = _regularized_covariance(aux_z)
    mahalanobis_sq = _mahalanobis_squared(z_matrix, valid_matrix, covariance, diag_variance, use_full_covariance)

    out["covariance_full_used"] = bool(use_full_covariance)
    out["mahalanobis_sq"] = mahalanobis_sq.astype("float32")
    out["mahalanobis"] = np.sqrt(mahalanobis_sq).astype("float32")
    out["log1p_mahalanobis_sq"] = np.log1p(mahalanobis_sq).astype("float32")
    out["joint_available_count"] = available_count
    out["joint_available_ratio"] = available_ratio.astype("float32")

    return out


FEATURE_GROUPS = [
    {
        "name": "aide_aux_reference_distribution_distance",
        "fn": add_aide_aux_reference_distribution_distance,
        "depends_on": [],
        "description": "Robust auxiliary-reference distances measuring marginal and joint typicality of sky position, photometry, and redshift.",
    }
]