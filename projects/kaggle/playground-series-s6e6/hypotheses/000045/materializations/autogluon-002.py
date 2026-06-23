import numpy as np
import pandas as pd

PAIRWISE_COLOR_DEFS = (
    ("color_u_g", "u", "g"),
    ("color_u_r", "u", "r"),
    ("color_u_i", "u", "i"),
    ("color_u_z", "u", "z"),
    ("color_g_r", "g", "r"),
    ("color_g_i", "g", "i"),
    ("color_g_z", "g", "z"),
    ("color_r_i", "r", "i"),
    ("color_r_z", "r", "z"),
    ("color_i_z", "i", "z"),
)

REDSHIFT_BIN_COUNT = 32
MIN_BIN_ROWS = 120
STD_RESID_CLIP = 8.0
COV_REGULARIZATION = 1e-6
TOP_PCA_COMPONENTS = 3
COV_EIGEN_EPS = 1e-12


def _safe_float_array(raw, column):
    if column not in raw.columns:
        return np.zeros(len(raw), dtype=float)
    return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)


def _build_pairwise_colors(raw):
    data = {}
    for feature_name, left, right in PAIRWISE_COLOR_DEFS:
        left_values = _safe_float_array(raw, left)
        right_values = _safe_float_array(raw, right)
        data[feature_name] = left_values - right_values
    return pd.DataFrame(data, index=raw.index)


def _bin_statistics(color_block):
    block = np.asarray(color_block, dtype=float)
    if block.size == 0:
        return None

    center = np.nanmedian(block, axis=0)
    mad = np.nanmedian(np.abs(block - center), axis=0)
    mad = np.where(np.isfinite(mad) & (mad > 0.0), mad, 1.0)

    standardized = (block - center) / mad
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    standardized = np.clip(standardized, -STD_RESID_CLIP, STD_RESID_CLIP)

    n_rows, n_features = block.shape
    if n_rows <= 1:
        covariance = np.eye(n_features, dtype=float)
    else:
        covariance = np.cov(standardized, rowvar=False, bias=True)
        if covariance.shape != (n_features, n_features):
            covariance = np.eye(n_features, dtype=float)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

    covariance = (covariance + covariance.T) * 0.5
    covariance = covariance + COV_REGULARIZATION * np.eye(covariance.shape[0])

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        n_features = block.shape[1]
        eigenvalues = np.ones(n_features, dtype=float)
        eigenvectors = np.eye(n_features, dtype=float)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    eigenvalues = np.where(np.isfinite(eigenvalues) & (eigenvalues > 0.0), eigenvalues, COV_EIGEN_EPS)

    try:
        inv_covariance = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inv_covariance = np.linalg.pinv(covariance)

    return center, mad, inv_covariance, eigenvectors, eigenvalues


def _coerce_bin_labels_to_int(bins_like):
    if isinstance(bins_like, pd.Categorical):
        bins = np.asarray(bins_like.codes, dtype=np.int64)
        bins = np.where(bins < 0, 0, bins)
        return bins

    if isinstance(bins_like, pd.Series):
        return bins_like.fillna(0).astype(np.int64).to_numpy()

    bins = np.asarray(bins_like)
    if np.issubdtype(bins.dtype, np.integer):
        return bins.astype(np.int64, copy=True)

    bins = pd.to_numeric(pd.Series(bins), errors="coerce").fillna(0).astype(np.int64).to_numpy()
    return bins


def _compute_bin_assignments(redshift_values):
    n = len(redshift_values)
    redshift_for_bins = np.asarray(redshift_values, dtype=float)

    finite = redshift_for_bins[np.isfinite(redshift_for_bins)]
    if finite.size == 0:
        return np.zeros(n, dtype=int), 1

    redshift_for_bins = np.where(redshift_for_bins < 0.0, 0.0, redshift_for_bins)
    finite = redshift_for_bins[np.isfinite(redshift_for_bins)]

    if np.unique(finite).size <= 1:
        z0 = float(np.nanmin(finite))
        edges = np.array([z0, z0 + 1.0], dtype=float)
    else:
        requested_bins = int(min(REDSHIFT_BIN_COUNT, finite.size))
        probs = np.linspace(0.0, 1.0, requested_bins + 1)
        edges = np.quantile(finite, probs)
        edges = np.unique(edges)
        if edges.size < 2:
            z0 = float(np.nanmin(finite))
            edges = np.array([z0, z0 + 1.0], dtype=float)

    labels = pd.cut(redshift_for_bins, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    bins = _coerce_bin_labels_to_int(labels)
    num_bins = int(np.nanmax(bins)) + 1
    if num_bins < 1:
        num_bins = 1
        bins[:] = 0
    return bins, num_bins


def _resolve_bins(bin_ids, num_bins):
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=num_bins).astype(np.int64)
    resolved = np.arange(num_bins, dtype=np.int64)

    if MIN_BIN_ROWS <= 0:
        return resolved, counts

    good_bins = np.flatnonzero(counts >= MIN_BIN_ROWS)
    if good_bins.size == 0:
        resolved[:] = -1
        return resolved, counts

    populated = np.flatnonzero(counts > 0)
    for b in range(num_bins):
        if counts[b] >= MIN_BIN_ROWS:
            continue

        replacement = -1
        for step in range(1, num_bins + 1):
            left = b - step
            if left >= 0 and counts[left] >= MIN_BIN_ROWS:
                replacement = left
                break
            right = b + step
            if right < num_bins and counts[right] >= MIN_BIN_ROWS:
                replacement = right
                break

        if replacement == -1 and populated.size > 0:
            replacement = int(populated[np.argmin(np.abs(populated - b))])

        resolved[b] = replacement

    return resolved, counts


def add_pairwise_color_lattice_residuals(raw, deps, aux):
    color_df = _build_pairwise_colors(raw)
    color_names = [name for name, _, _ in PAIRWISE_COLOR_DEFS]
    colors = np.asarray(color_df.to_numpy(dtype=float))
    n_rows, n_features = colors.shape

    redshift_values = _safe_float_array(raw, "redshift")
    raw_bin_ids, num_bins = _compute_bin_assignments(redshift_values)
    resolved_bins, _ = _resolve_bins(raw_bin_ids, num_bins)

    global_stats = _bin_statistics(colors)

    if num_bins <= 0:
        mapped_bin_ids = np.full(n_rows, -1, dtype=np.int64)
    else:
        mapped_bin_ids = resolved_bins[np.clip(raw_bin_ids, 0, num_bins - 1)]

    stats_by_source = {-1: global_stats}

    for source_bin in np.unique(mapped_bin_ids):
        if source_bin in stats_by_source:
            continue
        source_idx = np.flatnonzero(raw_bin_ids == int(source_bin))
        if source_idx.size == 0:
            stats_by_source[int(source_bin)] = global_stats
            continue
        source_stats = _bin_statistics(colors[source_idx])
        if source_stats is None:
            source_stats = global_stats
        stats_by_source[int(source_bin)] = source_stats

    residual_features = np.zeros((n_rows, n_features), dtype=float)
    mahalanobis = np.zeros(n_rows, dtype=float)
    principal_scores = np.zeros((n_rows, TOP_PCA_COMPONENTS), dtype=float)

    for source_bin in np.unique(mapped_bin_ids):
        row_idx = np.flatnonzero(mapped_bin_ids == source_bin)
        if row_idx.size == 0:
            continue

        center, mad, inv_covariance, eigvecs, eigvals = stats_by_source[int(source_bin)]

        block = colors[row_idx]
        block_std = (block - center) / mad
        block_std = np.nan_to_num(block_std, nan=0.0, posinf=0.0, neginf=0.0)
        block_std = np.clip(block_std, -STD_RESID_CLIP, STD_RESID_CLIP)

        residual_features[row_idx, :] = block_std

        rotated = block_std @ inv_covariance
        mahalanobis[row_idx] = np.sqrt(np.einsum("ij,ij->i", rotated, block_std))

        n_components = min(TOP_PCA_COMPONENTS, n_features)
        proj = block_std @ eigvecs[:, :n_components]
        proj = proj / np.sqrt(eigvals[:n_components])
        principal_scores[row_idx, :n_components] = proj

    residual_columns = [f"{name}_zbin_residual" for name in color_names]
    feature_df = pd.DataFrame(residual_features, index=raw.index, columns=residual_columns)
    feature_df["pairwise_mahalanobis"] = mahalanobis
    feature_df["pairwise_pc1"] = principal_scores[:, 0]
    feature_df["pairwise_pc2"] = principal_scores[:, 1]
    feature_df["pairwise_pc3"] = principal_scores[:, 2]

    return feature_df


FEATURE_GROUPS = [
    {
        "name": "pairwise_color_lattice_residuals",
        "fn": add_pairwise_color_lattice_residuals,
        "depends_on": [],
        "description": "Builds redshift-binned pairwise ugriz color residual geometry using robust median/MAD normalization, Mahalanobis distance, and bin-conditioned PCA residual scores.",
    }
]