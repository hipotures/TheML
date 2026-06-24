import numpy as np
import pandas as pd

EPSILON = 1e-12
SVD_TOLERANCE = 1e-15
RESIDUAL_BAND_CAP = 0.2
MIN_BIN_FRACTION = 0.004
MIN_BIN_ABSOLUTE = 4000
BANDS = ("u", "g", "r", "i", "z")
REDSHIFT_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

def _build_redshift_edges(redshift_values):
    redshift = np.asarray(redshift_values, dtype=np.float64)
    quantiles = np.asarray(REDSHIFT_QUANTILES, dtype=np.float64)
    edges = np.quantile(redshift, quantiles)
    edges[0] = float(np.min(redshift))
    edges[-1] = float(np.max(redshift))

    dedup = []
    for value in edges:
        if not dedup or value > dedup[-1] + EPSILON:
            dedup.append(float(value))

    if len(dedup) < 2:
        center = float(redshift[0] if redshift.size else 0.0)
        pad = max(1e-6, abs(center) * 1e-6 + 1e-4)
        return np.array([center - pad, center + pad], dtype=np.float64)

    return np.asarray(dedup, dtype=np.float64)

def _merge_bin_edges(edges, counts, min_size):
    edges = np.asarray(edges, dtype=np.float64).copy()
    counts = np.asarray(counts, dtype=np.int64).copy()

    if edges.size < 2 or counts.size == 0:
        return edges, counts

    while counts.size > 1 and counts.min() < min_size:
        idx = int(np.argmin(counts))
        if counts[idx] >= min_size:
            break

        if idx == 0:
            counts[idx] += counts[idx + 1]
            counts = np.delete(counts, idx + 1)
            edges = np.delete(edges, idx + 1)
        elif idx == counts.size - 1:
            counts[idx - 1] += counts[idx]
            counts = np.delete(counts, idx)
            edges = np.delete(edges, idx)
        else:
            if counts[idx - 1] <= counts[idx + 1]:
                counts[idx - 1] += counts[idx]
                counts = np.delete(counts, idx)
                edges = np.delete(edges, idx)
            else:
                counts[idx] += counts[idx + 1]
                counts = np.delete(counts, idx + 1)
                edges = np.delete(edges, idx + 1)

    return edges, counts

def _fit_standardized_pca_residual(p_block):
    block = np.asarray(p_block, dtype=np.float64)
    mu = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    std = np.maximum(std, EPSILON)
    standardized = (block - mu) / std
    standardized = standardized - standardized.mean(axis=0, keepdims=True)

    if standardized.shape[0] <= 1:
        loadings = np.zeros((block.shape[1], 0), dtype=np.float64)
        residual_std = standardized
        return mu, std, loadings, residual_std

    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    k = int(np.sum(singular_values > SVD_TOLERANCE))
    if k == 0:
        loadings = np.zeros((block.shape[1], 0), dtype=np.float64)
        residual_std = standardized
        return mu, std, loadings, residual_std

    k = min(3, k)
    loadings = vt[:k].T
    residual_std = standardized - (standardized @ loadings) @ loadings.T
    return mu, std, loadings, residual_std

def _rank_pct(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if arr.size == 1:
        return np.array([0.5], dtype=np.float64)
    return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float64)

def add_redshift_partitioned_sed_pca_residuals(raw, deps, aux):
    del deps, aux

    index = raw.index
    redshift = raw["redshift"].to_numpy(dtype=np.float64)
    mags = raw.loc[:, BANDS].to_numpy(dtype=np.float64)
    n_rows = redshift.shape[0]

    if n_rows == 0:
        return pd.DataFrame(index=index)

    flux = np.power(10.0, -0.4 * mags)
    shape = flux / (flux.sum(axis=1, keepdims=True) + EPSILON)

    # Global fallback PCA on all rows.
    g_mu, g_std, g_loadings, g_res_std = _fit_standardized_pca_residual(shape)
    g_res = np.clip(g_res_std * g_std, -RESIDUAL_BAND_CAP, RESIDUAL_BAND_CAP)
    g_norm = np.linalg.norm(g_res, axis=1)
    g_rel = g_norm / (np.linalg.norm(shape, axis=1) + EPSILON)
    g_frac = (g_res * g_res).sum(axis=1) / ((shape * shape).sum(axis=1) + EPSILON)
    g_orth = (g_res_std * g_res_std).sum(axis=1)

    g_dir = np.zeros((n_rows, 3), dtype=np.float64)
    if g_loadings.shape[1] > 0:
        g_scores = g_res_std @ g_loadings
        g_n = min(3, g_scores.shape[1])
        g_dir[:, :g_n] = g_scores[:, :g_n]

    g_rank = _rank_pct(g_norm)

    # Start from global values, then replace by local-bin values when available.
    residual = g_res.copy()
    residual_norm = g_norm.copy()
    residual_rel = g_rel.copy()
    residual_energy = g_frac.copy()
    residual_orth = g_orth.copy()
    residual_dir = g_dir.copy()
    residual_bin_rank = g_rank.copy()

    # Build redshift bins from quantiles and merge until minimum support constraint.
    edges = _build_redshift_edges(redshift)
    initial_bin = pd.cut(redshift, edges, labels=False, include_lowest=True, right=True, duplicates="drop")
    initial_bin_series = pd.Series(initial_bin, index=index)
    if initial_bin_series.notna().any():
        valid_bins = initial_bin_series.dropna().astype(np.int64).to_numpy()
        init_counts = np.bincount(valid_bins, minlength=int(valid_bins.max()) + 1)
    else:
        init_counts = np.array([], dtype=np.int64)

    init_bin_count = max(1, init_counts.size)
    min_bin_size = max(MIN_BIN_ABSOLUTE, int(np.floor(MIN_BIN_FRACTION * float(n_rows) / init_bin_count)))
    merged_edges, merged_counts = _merge_bin_edges(edges, init_counts, min_bin_size)

    local_possible = not (
        merged_counts.size == 0 or (merged_counts.size == 1 and merged_counts[0] < min_bin_size)
    )

    if local_possible:
        clipped_redshift = np.clip(redshift, merged_edges[0], merged_edges[-1])
        local_bin = pd.cut(
            clipped_redshift,
            merged_edges,
            labels=False,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )
        local_bin_ids = local_bin.to_numpy(dtype=np.float64)
        local_mask = ~np.isnan(local_bin_ids)
        assigned = np.full(n_rows, -1, dtype=np.int64)
        assigned[local_mask] = local_bin_ids[local_mask].astype(np.int64)

        if (~local_mask).any():
            midpoints = 0.5 * (merged_edges[:-1] + merged_edges[1:])
            nearest = np.argmin(np.abs(clipped_redshift[:, None] - midpoints[None, :]), axis=1)
            assigned[~local_mask] = nearest[~local_mask]
            local_mask[~local_mask] = True

        for b in range(merged_counts.size):
            row_idx = np.flatnonzero(assigned == b)
            if row_idx.size == 0:
                continue

            block = shape[row_idx]
            b_mu, b_std, b_loadings, b_res_std = _fit_standardized_pca_residual(block)
            b_res = np.clip(b_res_std * b_std, -RESIDUAL_BAND_CAP, RESIDUAL_BAND_CAP)

            b_norm = np.linalg.norm(b_res, axis=1)
            b_rel = b_norm / (np.linalg.norm(block, axis=1) + EPSILON)
            b_frac = (b_res * b_res).sum(axis=1) / ((block * block).sum(axis=1) + EPSILON)
            b_orth = (b_res_std * b_res_std).sum(axis=1)

            b_dir = np.zeros((row_idx.size, 3), dtype=np.float64)
            if b_loadings.shape[1] > 0:
                b_scores = b_res_std @ b_loadings
                b_n = min(3, b_scores.shape[1])
                b_dir[:, :b_n] = b_scores[:, :b_n]

            b_rank = _rank_pct(b_norm)

            residual[row_idx] = b_res
            residual_norm[row_idx] = b_norm
            residual_rel[row_idx] = b_rel
            residual_energy[row_idx] = b_frac
            residual_orth[row_idx] = b_orth
            residual_dir[row_idx] = b_dir
            residual_bin_rank[row_idx] = b_rank

    features = {
        "sed_pca_residual_norm_l2": residual_norm,
        "sed_pca_residual_relative_norm": residual_rel,
        "sed_pca_residual_energy_fraction": residual_energy,
        "sed_pca_residual_outside_pc_energy": residual_orth,
        "sed_pca_residual_dir_pc1": residual_dir[:, 0],
        "sed_pca_residual_dir_pc2": residual_dir[:, 1],
        "sed_pca_residual_dir_pc3": residual_dir[:, 2],
        "sed_pca_residual_band_u": residual[:, 0],
        "sed_pca_residual_band_g": residual[:, 1],
        "sed_pca_residual_band_r": residual[:, 2],
        "sed_pca_residual_band_i": residual[:, 3],
        "sed_pca_residual_band_z": residual[:, 4],
        "sed_pca_recon_error_bin_quantile": residual_bin_rank,
    }

    return pd.DataFrame(features, index=index)

FEATURE_GROUPS = [
    {
        "name": "redshift_partitioned_sed_pca_residuals",
        "fn": add_redshift_partitioned_sed_pca_residuals,
        "depends_on": [],
        "description": "Build redshift-partitioned PCA residual spectral-manifold features from normalized ugriz shape vectors.",
    },
]