import numpy as np
import pandas as pd

AUX_REFERENCE_COLUMNS = ("alpha", "delta", "u", "g", "r", "i", "z", "redshift")
PHOTOMETRY_COLUMNS = ("u", "g", "r", "i", "z")
FEATURE_GROUP_NAME = "aide_aux_reference_distribution_distance"
MAD_TO_STD = 1.482602218505602
MAHALANOBIS_INLIER_QUANTILE = 0.975


def _to_float_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=True)


def _prepare_reference_series(aux, column):
    values = _to_float_array(aux[column])
    if column in PHOTOMETRY_COLUMNS:
        values[values <= -9000.0] = np.nan
    if column == "redshift":
        finite = np.isfinite(values)
        values[finite] = np.maximum(values[finite], 0.0)
    return values


def _prepare_raw_series(raw, column):
    values = _to_float_array(raw[column])
    if column in PHOTOMETRY_COLUMNS:
        values[values <= -9000.0] = np.nan
    if column == "redshift":
        finite = np.isfinite(values)
        values[finite] = np.maximum(values[finite], 0.0)
    return values


def _robust_center_and_scale(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan, np.nan
    center = np.median(values[finite])
    mad = np.median(np.abs(values[finite] - center))
    if not np.isfinite(mad) or mad <= 0.0:
        return center, np.nan
    return center, mad * MAD_TO_STD


def _robust_z(values, center, scale):
    z = np.full(len(values), np.nan, dtype=np.float64)
    if np.isfinite(center) and np.isfinite(scale) and scale > 0.0:
        finite = np.isfinite(values)
        z[finite] = (values[finite] - center) / scale
    return z


def _empirical_cdf_position(reference_values, query_values):
    finite_reference = np.isfinite(reference_values)
    ref = np.sort(reference_values[finite_reference])
    out = np.full(len(query_values), np.nan, dtype=np.float64)
    if ref.size == 0:
        return out
    finite_query = np.isfinite(query_values)
    if not np.any(finite_query):
        return out
    pos = np.searchsorted(ref, query_values[finite_query], side="right")
    out[finite_query] = pos.astype(np.float64) / float(ref.size)
    return out


def _mahalanobis_from_standardized(ref_standardized):
    complete = np.isfinite(ref_standardized).all(axis=1)
    ref_complete = ref_standardized[complete]
    if ref_complete.shape[0] < 2 or ref_complete.shape[1] < 1:
        return None, ref_complete
    cov = np.cov(ref_complete, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    return np.linalg.pinv(cov), ref_complete


def add_aide_aux_reference_distribution_distance(raw, deps, aux):
    _ = deps
    idx = raw.index
    n_rows = len(raw)

    feature_data = {}

    overlapping_columns = [col for col in AUX_REFERENCE_COLUMNS if col in raw.columns]
    has_aux = isinstance(aux, pd.DataFrame) and aux.shape[0] > 0

    ref_standardized_parts = []
    raw_standardized_parts = []
    z_components_for_l2 = []

    for column in overlapping_columns:
        raw_values = _prepare_raw_series(raw, column)
        cdf_col = np.full(n_rows, 0.5, dtype=np.float64)
        z_col = np.zeros(n_rows, dtype=np.float64)
        delta_col = np.zeros(n_rows, dtype=np.float64)

        if has_aux and column in aux.columns:
            ref_values = _prepare_reference_series(aux, column)
            center, scale = _robust_center_and_scale(ref_values)

            if np.isfinite(center):
                cdf_col = _empirical_cdf_position(ref_values, raw_values)
                z_col = _robust_z(raw_values, center, scale)
                finite = np.isfinite(raw_values)
                delta_col = np.zeros(n_rows, dtype=np.float64)
                delta_col[finite] = np.abs(raw_values[finite] - center)

                if np.isfinite(scale) and scale > 0.0:
                    ref_standardized = _robust_z(ref_values, center, scale)
                    raw_standardized = _robust_z(raw_values, center, scale)
                    ref_standardized_parts.append(ref_standardized[:, None])
                    raw_standardized_parts.append(raw_standardized[:, None])

        feature_data[f"cdf_{column}"] = cdf_col
        feature_data[f"robust_z_{column}"] = z_col
        feature_data[f"abs_delta_from_ref_median_{column}"] = delta_col
        z_components_for_l2.append(z_col)

    if z_components_for_l2:
        z_stack = np.column_stack(z_components_for_l2)
        feature_data["joint_l2_robust_z"] = np.sqrt(np.nansum(np.square(z_stack), axis=1))
    else:
        feature_data["joint_l2_robust_z"] = np.zeros(n_rows, dtype=np.float64)

    mahal2 = np.zeros(n_rows, dtype=np.float64)
    mahal = np.zeros(n_rows, dtype=np.float64)
    mahal_inlier = np.ones(n_rows, dtype=np.int8)

    if ref_standardized_parts:
        ref_matrix = np.column_stack(ref_standardized_parts)
        raw_matrix = np.column_stack(raw_standardized_parts)
        inv_cov, ref_complete = _mahalanobis_from_standardized(ref_matrix)
        if inv_cov is not None and ref_complete.shape[0] > 0:
            aux_tmp = np.dot(ref_complete, inv_cov)
            aux_mahal2 = np.einsum("ij,ij->i", aux_tmp, ref_complete)
            aux_mahal2 = np.maximum(aux_mahal2, 0.0)
            finite_aux = np.isfinite(aux_mahal2)
            if np.any(finite_aux):
                cutoff = np.quantile(aux_mahal2[finite_aux], MAHALANOBIS_INLIER_QUANTILE)
                raw_complete = np.isfinite(raw_matrix).all(axis=1)
                if np.any(raw_complete):
                    raw_tmp = np.dot(raw_matrix[raw_complete], inv_cov)
                    raw_mahal2 = np.einsum("ij,ij->i", raw_tmp, raw_matrix[raw_complete])
                    raw_mahal2 = np.maximum(raw_mahal2, 0.0)
                    mahal2[raw_complete] = raw_mahal2
                    mahal[raw_complete] = np.sqrt(raw_mahal2)
                    mahal_inlier[raw_complete] = (raw_mahal2 <= cutoff).astype(np.int8)

    feature_data["mahalanobis2_reference"] = mahal2
    feature_data["mahalanobis_reference"] = mahal
    feature_data["mahalanobis_inlier_reference"] = mahal_inlier

    return pd.DataFrame(feature_data, index=idx)


FEATURE_GROUPS = [
    {
        "name": FEATURE_GROUP_NAME,
        "fn": add_aide_aux_reference_distribution_distance,
        "depends_on": [],
        "description": "Compare each row against auxiliary reference columns with robust CDF position, robust z-score, median-delta, joint robust-z L2, and robust Mahalanobis distance features.",
    }
]