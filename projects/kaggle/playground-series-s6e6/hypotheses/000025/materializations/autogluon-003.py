import numpy as np
import pandas as pd

OVERLAP_COLUMNS = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
EXTREME_NEGATIVE_COLUMNS = ("u", "g", "r", "i", "z")
JOINT_COLUMNS = ("sin_alpha", "cos_alpha", "delta", "u", "g", "r", "i", "z", "redshift")

REDSHIFT_FLOOR = 0.0001
EXTREME_NEGATIVE_VALUE = -9000.0
ROBUST_Z_CLIP = 8.0
JOINT_Z_CLIP = 6.0
MIN_OVERLAP_AVAILABLE = 3
MIN_JOINT_DIM = 3
COV_RIDGE_FACTOR = 1e-6


def _to_float(values, length):
    if values is None:
        return np.full(length, np.nan, dtype=np.float64)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    arr = arr.astype(np.float64, copy=False)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _clean_overlap_column(values, column_name):
    arr = values.copy()
    if column_name in EXTREME_NEGATIVE_COLUMNS:
        arr[arr <= EXTREME_NEGATIVE_VALUE] = np.nan
    if column_name == "redshift":
        low = np.isfinite(arr) & (arr < REDSHIFT_FLOOR)
        arr[low] = REDSHIFT_FLOOR
    return arr


def _robust_location_scale(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0, 0.0, 0
    v = values[finite]
    median = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - median)))
    if not np.isfinite(mad) or mad <= 0.0:
        q25 = float(np.nanpercentile(v, 25.0))
        q75 = float(np.nanpercentile(v, 75.0))
        mad = (q75 - q25) / 1.348
    return median, mad, v.size


def _apply_positive_scale_fallback(scales):
    positive = [s for s in scales.values() if np.isfinite(s) and s > 0.0]
    fallback = float(np.nanmedian(positive)) if positive else 1.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    for k, s in scales.items():
        if not np.isfinite(s) or s <= 0.0:
            scales[k] = fallback
    return fallback


def _ecdf_rank_probs(reference_values, query_values):
    ref = reference_values[np.isfinite(reference_values)]
    probs = np.full_like(query_values, np.nan, dtype=np.float64)
    n_ref = ref.size
    if n_ref == 0:
        return probs
    ref_sorted = np.sort(ref)
    finite = np.isfinite(query_values)
    ranks = np.searchsorted(ref_sorted, query_values[finite], side="right")
    p = (ranks + 1.0) / (n_ref + 1.0)
    lo = 1.0 / (n_ref + 2.0)
    hi = 1.0 - lo
    probs[finite] = np.clip(p, lo, hi)
    return probs


def _encode_mask(mask_matrix):
    n_rows, n_cols = mask_matrix.shape
    codes = np.zeros(n_rows, dtype=np.int16)
    for i in range(n_cols):
        codes |= (mask_matrix[:, i].astype(np.int16) << i)
    return codes


def _mask_to_indices(mask, n_dims):
    idx = []
    bit = 0
    while mask and bit < n_dims:
        if mask & 1:
            idx.append(bit)
        mask >>= 1
        bit += 1
    return idx


def add_aide_aux_reference_distribution_distance(raw, deps, aux):
    idx = raw.index
    n = len(raw)

    has_aux = isinstance(aux, pd.DataFrame) and len(aux) > 0

    raw_overlap = {}
    raw_missing = {}
    for col in OVERLAP_COLUMNS:
        if col in raw.columns:
            arr = _to_float(raw[col], n)
        else:
            arr = np.full(n, np.nan, dtype=np.float64)
        arr = _clean_overlap_column(arr, col)
        raw_overlap[col] = arr
        raw_missing[col] = ~np.isfinite(arr)

    overlap_available = np.column_stack([~raw_missing[col] for col in OVERLAP_COLUMNS]).astype(float)
    overlap_missing_count = overlap_available.sum(axis=1).astype(np.int16)
    overlap_available_ratio = overlap_missing_count / float(len(OVERLAP_COLUMNS))

    if has_aux:
        aux_n = len(aux)
        ref_overlap = {}
        for col in OVERLAP_COLUMNS:
            if col in aux.columns:
                arr = _to_float(aux[col], aux_n)
            else:
                arr = np.full(aux_n, np.nan, dtype=np.float64)
            ref_overlap[col] = _clean_overlap_column(arr, col)

        overlap_median = {}
        overlap_scale = {}
        overlap_n_ref = {}
        for col in OVERLAP_COLUMNS:
            med, sc, n_ref = _robust_location_scale(ref_overlap[col])
            overlap_median[col] = med
            overlap_scale[col] = sc
            overlap_n_ref[col] = n_ref
        _apply_positive_scale_fallback(overlap_scale)
    else:
        overlap_median = {col: 0.0 for col in OVERLAP_COLUMNS}
        overlap_scale = {col: 1.0 for col in OVERLAP_COLUMNS}
        overlap_n_ref = {col: 0 for col in OVERLAP_COLUMNS}

    overlap_abs_z_mat = []
    overlap_rank_shift_mat = []
    overlap_abs_med_delta_mat = []

    if has_aux:
        for col in OVERLAP_COLUMNS:
            values = raw_overlap[col]
            med = overlap_median[col]
            sc = overlap_scale[col]
            z = (values - med) / sc
            miss = ~np.isfinite(values)
            z[miss] = np.nan
            z = np.clip(z, -ROBUST_Z_CLIP, ROBUST_Z_CLIP)

            p = _ecdf_rank_probs(ref_overlap[col], values)
            overlap_abs_z_mat.append(np.abs(z))
            overlap_rank_shift_mat.append(np.abs(p - 0.5))
            delta = np.abs(values - med)
            delta[miss] = np.nan
            overlap_abs_med_delta_mat.append(delta)
        overlap_mean_abs_z = np.nanmean(np.column_stack(overlap_abs_z_mat), axis=1)
        overlap_max_abs_z = np.nanmax(np.column_stack(overlap_abs_z_mat), axis=1)
        overlap_mean_rank_shift = np.nanmean(np.column_stack(overlap_rank_shift_mat), axis=1)
        overlap_mean_abs_median_delta = np.nanmean(np.column_stack(overlap_abs_med_delta_mat), axis=1)
    else:
        overlap_mean_abs_z = np.zeros(n, dtype=np.float64)
        overlap_max_abs_z = np.zeros(n, dtype=np.float64)
        overlap_mean_rank_shift = np.zeros(n, dtype=np.float64)
        overlap_mean_abs_median_delta = np.zeros(n, dtype=np.float64)

    overlap_mean_abs_z = np.nan_to_num(overlap_mean_abs_z, nan=0.0)
    overlap_max_abs_z = np.nan_to_num(overlap_max_abs_z, nan=0.0)
    overlap_mean_rank_shift = np.nan_to_num(overlap_mean_rank_shift, nan=0.0)
    overlap_mean_abs_median_delta = np.nan_to_num(overlap_mean_abs_median_delta, nan=0.0)

    alpha = raw_overlap["alpha"]
    raw_sin = np.sin(np.deg2rad(alpha))
    raw_cos = np.cos(np.deg2rad(alpha))

    if has_aux:
        if "alpha" in aux.columns:
            ref_alpha = _to_float(aux["alpha"], len(aux))
        else:
            ref_alpha = np.full(len(aux), np.nan, dtype=np.float64)
        ref_alpha = ref_alpha.copy()
        ref_alpha[~np.isfinite(ref_alpha)] = np.nan
        ref_sin = np.sin(np.deg2rad(ref_alpha))
        ref_cos = np.cos(np.deg2rad(ref_alpha))

        ref_joint_raw = {
            "sin_alpha": ref_sin,
            "cos_alpha": ref_cos,
            "delta": ref_overlap["delta"],
            "u": ref_overlap["u"],
            "g": ref_overlap["g"],
            "r": ref_overlap["r"],
            "i": ref_overlap["i"],
            "z": ref_overlap["z"],
            "redshift": ref_overlap["redshift"],
        }
        joint_median = {}
        joint_scale = {}
        for col in JOINT_COLUMNS:
            med, sc, _ = _robust_location_scale(ref_joint_raw[col])
            if np.isfinite(med):
                joint_median[col] = med
            else:
                joint_median[col] = overlap_median.get(col, 0.0)
            joint_scale[col] = sc
        # keep overlap columns on same robust scale baseline when possible
        for col in OVERLAP_COLUMNS:
            if col in overlap_median:
                joint_median[col] = overlap_median[col]
                joint_scale[col] = overlap_scale[col]
        _apply_positive_scale_fallback(joint_scale)

        ref_joint_std_cols = []
        for col in JOINT_COLUMNS:
            vals = ref_joint_raw[col]
            med = joint_median[col]
            sc = joint_scale[col]
            z = (vals - med) / sc
            z[~np.isfinite(vals)] = np.nan
            z = np.clip(z, -JOINT_Z_CLIP, JOINT_Z_CLIP)
            ref_joint_std_cols.append(z)
        ref_joint_std = np.column_stack(ref_joint_std_cols)
    else:
        joint_median = {col: 0.0 for col in JOINT_COLUMNS}
        joint_scale = {col: 1.0 for col in JOINT_COLUMNS}
        ref_joint_std = None

    raw_joint_std_cols = []
    for col in JOINT_COLUMNS:
        if col == "sin_alpha":
            vals = raw_sin
        elif col == "cos_alpha":
            vals = raw_cos
        else:
            vals = raw_overlap[col]
        med = joint_median[col]
        sc = joint_scale[col]
        z = (vals - med) / sc
        z[~np.isfinite(vals)] = np.nan
        z = np.clip(z, -JOINT_Z_CLIP, JOINT_Z_CLIP)
        raw_joint_std_cols.append(z)
    raw_joint_std = np.column_stack(raw_joint_std_cols)

    if has_aux and ref_joint_std is not None:
        ref_complete = np.all(np.isfinite(ref_joint_std), axis=1)
        ref_joint_complete = ref_joint_std[ref_complete]
        has_joint_reference = ref_joint_complete.shape[0] > 0
        if has_joint_reference:
            ref_mean = np.mean(ref_joint_complete, axis=0)
            centered = ref_joint_complete - ref_mean
            if centered.shape[0] > 1:
                joint_cov = centered.T.dot(centered) / float(centered.shape[0] - 1)
            else:
                joint_cov = np.zeros((len(JOINT_COLUMNS), len(JOINT_COLUMNS)), dtype=np.float64)
            joint_rank = np.linalg.matrix_rank(joint_cov, tol=1e-12)
            if len(JOINT_COLUMNS) >= 2 or joint_rank < len(JOINT_COLUMNS):
                trace = float(np.trace(joint_cov))
                tau = COV_RIDGE_FACTOR * (trace / len(JOINT_COLUMNS) if len(JOINT_COLUMNS) > 0 else 1.0)
                if not np.isfinite(tau) or tau <= 0.0:
                    tau = COV_RIDGE_FACTOR
                joint_cov = joint_cov + np.eye(len(JOINT_COLUMNS)) * tau
        else:
            joint_cov = None
    else:
        ref_mean = np.zeros(len(JOINT_COLUMNS), dtype=np.float64)
        has_joint_reference = False
        joint_cov = None

    row_centered = raw_joint_std - ref_mean
    row_centered[~np.isfinite(row_centered)] = np.nan
    row_joint_mask = np.isfinite(row_centered)
    joint_available_count = row_joint_mask.sum(axis=1).astype(np.int16)

    joint_mahalanobis2 = np.zeros(n, dtype=np.float64)
    joint_mahalanobis = np.zeros(n, dtype=np.float64)
    joint_inlier_score = np.zeros(n, dtype=np.float64)
    joint_fallback_used = np.zeros(n, dtype=bool)

    if has_aux and has_joint_reference:
        codes = _encode_mask(row_joint_mask)
        unique_codes = np.unique(codes)
        inv_cache = {}
        for code in unique_codes:
            rows = np.where(codes == code)[0]
            if rows.size == 0:
                continue
            dim_idx = _mask_to_indices(int(code), len(JOINT_COLUMNS))
            dim_k = len(dim_idx)

            if dim_k == 0:
                joint_fallback_used[rows] = True
                continue

            sub = row_centered[rows][:, dim_idx]

            if dim_k < MIN_JOINT_DIM:
                fallback = np.nanmean(np.abs(sub), axis=1)
                fallback = np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)
                joint_mahalanobis[rows] = fallback
                joint_mahalanobis2[rows] = fallback * fallback
                joint_inlier_score[rows] = np.log1p(joint_mahalanobis2[rows])
                joint_fallback_used[rows] = True
                continue

            inv = inv_cache.get(code)
            if inv is None:
                sub_cov = joint_cov[np.ix_(dim_idx, dim_idx)]
                sub_rank = np.linalg.matrix_rank(sub_cov, tol=1e-12)
                if dim_k >= 2 or sub_rank < dim_k:
                    tr = float(np.trace(sub_cov))
                    tau = COV_RIDGE_FACTOR * (tr / dim_k if dim_k > 0 else 1.0)
                    if not np.isfinite(tau) or tau <= 0.0:
                        tau = COV_RIDGE_FACTOR
                    sub_cov = sub_cov + np.eye(dim_k) * tau
                inv = np.linalg.pinv(sub_cov)
                inv_cache[code] = inv

            d2 = np.einsum("ij,jk,ik->i", sub, inv, sub)
            d2 = np.maximum(d2, 0.0)
            joint_mahalanobis2[rows] = d2
            joint_mahalanobis[rows] = np.sqrt(d2)
            joint_inlier_score[rows] = np.log1p(d2)
            joint_fallback_used[rows] = False

    effective_overlap_ok = overlap_missing_count >= MIN_OVERLAP_AVAILABLE
    active_rows = effective_overlap_ok if has_aux else np.zeros(n, dtype=bool)
    inactive_rows = ~active_rows

    if not has_aux or not np.any(active_rows):
        overlap_mean_abs_z = np.zeros(n, dtype=np.float64)
        overlap_max_abs_z = np.zeros(n, dtype=np.float64)
        overlap_mean_rank_shift = np.zeros(n, dtype=np.float64)
        overlap_mean_abs_median_delta = np.zeros(n, dtype=np.float64)
        joint_mahalanobis2[:] = 0.0
        joint_mahalanobis[:] = 0.0
        joint_inlier_score[:] = 0.0
        joint_fallback_used[:] = True
    else:
        overlap_mean_abs_z = np.where(active_rows, overlap_mean_abs_z, 0.0)
        overlap_max_abs_z = np.where(active_rows, overlap_max_abs_z, 0.0)
        overlap_mean_rank_shift = np.where(active_rows, overlap_mean_rank_shift, 0.0)
        overlap_mean_abs_median_delta = np.where(active_rows, overlap_mean_abs_median_delta, 0.0)
        if not has_joint_reference:
            joint_mahalanobis2[:] = 0.0
            joint_mahalanobis[:] = 0.0
            joint_inlier_score[:] = 0.0
            joint_fallback_used[:] = True
        else:
            joint_mahalanobis2[inactive_rows] = 0.0
            joint_mahalanobis[inactive_rows] = 0.0
            joint_inlier_score[inactive_rows] = 0.0
            joint_fallback_used[inactive_rows] = True

    new_features = pd.DataFrame(index=idx)
    for col in OVERLAP_COLUMNS:
        new_features[f"overlap_missing_{col}"] = raw_missing[col]
    new_features["overlap_missing_count"] = overlap_missing_count
    new_features["overlap_available_ratio"] = overlap_available_ratio.astype(np.float64)
    new_features["overlap_mean_abs_z"] = overlap_mean_abs_z
    new_features["overlap_max_abs_z"] = overlap_max_abs_z
    new_features["overlap_mean_rank_shift"] = overlap_mean_rank_shift
    new_features["overlap_mean_abs_median_delta"] = overlap_mean_abs_median_delta
    new_features["joint_available_count"] = joint_available_count.astype(np.float64)
    new_features["joint_available_ratio"] = (joint_available_count / float(len(JOINT_COLUMNS))).astype(np.float64)
    new_features["joint_mahalanobis2"] = joint_mahalanobis2
    new_features["joint_mahalanobis"] = joint_mahalanobis
    new_features["joint_inlier_score"] = joint_inlier_score
    new_features["joint_fallback_used"] = joint_fallback_used
    new_features["reference_available"] = np.full(n, bool(has_aux), dtype=bool)
    new_features["overlap_effective_ok"] = active_rows

    return new_features


FEATURE_GROUPS = [
    {
        "name": "aide_aux_reference_distribution_distance",
        "fn": add_aide_aux_reference_distribution_distance,
        "depends_on": [],
        "description": "Builds robust marginal and joint auxiliary-reference distance features using cleaned overlap predictors and fallback-safe Mahalanobis estimation.",
    }
]