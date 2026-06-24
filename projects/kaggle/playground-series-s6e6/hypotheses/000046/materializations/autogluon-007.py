import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_K_NEIGHBORS = 64
_MIN_STABLE_K = 6
_EPSILON = 1e-12
_MAD_CLIP = 1e-6
_SPACE_PREFIX = "manifold_codim"

_FEATURE_KEYS = (
    "r2",
    "residual2",
    "a12",
    "a23",
    "tail",
    "dim95",
    "rk",
    "r_med",
    "r_mean",
    "density_proxy",
    "r_ratio",
    "frac_in_box",
    "asym_mad",
    "asym_mad_ratio",
    "degenerate",
    "n_neighbors",
)

_SPARSE_REPLACE_KEYS = (
    "r2",
    "residual2",
    "a12",
    "a23",
    "tail",
    "dim95",
    "rk",
    "r_med",
    "r_mean",
    "density_proxy",
    "r_ratio",
)


def _coerce_train_mask_from_series(series, positive_is_train=True):
    s = pd.Series(series)

    if s.empty:
        return None

    if pd.api.types.is_bool_dtype(s.dtype):
        values = s.fillna(False).astype(bool).to_numpy(dtype=bool)
        return values if positive_is_train else ~values

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_non_na = numeric.dropna()
    if len(numeric_non_na) > 0:
        uniq = pd.unique(numeric_non_na)
        if len(uniq) <= 3 and np.all(np.isin(uniq, [0, 1])):
            values = numeric.to_numpy(dtype=float)
            values = np.where(np.isnan(values), 0.0, values)
            train_mask = values > 0.5
            return train_mask.astype(bool) if positive_is_train else ~train_mask.astype(bool)

    strings = s.astype("string").str.lower().str.strip().fillna("")
    uniq = set(strings.unique())
    if not uniq:
        return None

    train_tokens = {"train", "tr", "training", "valid", "validation", "val", "1", "true", "t", "yes", "y"}
    test_tokens = {"test", "te", "0", "false", "f", "no", "n"}
    all_tokens = train_tokens | test_tokens

    if uniq.issubset(all_tokens):
        base_mask = strings.isin(train_tokens).to_numpy(dtype=bool)
        return base_mask if positive_is_train else ~base_mask

    return None


def _extract_train_mask(raw, aux):
    n = len(raw)

    candidate_frames = []
    if isinstance(aux, pd.DataFrame) and len(aux) == n and not aux.empty:
        candidate_frames.append(aux)
    if isinstance(raw, pd.DataFrame) and len(raw) == n:
        candidate_frames.append(raw)

    explicit_train_cols = {
        "is_train",
        "train",
        "is_train_row",
        "train_row",
        "is_train_mask",
        "train_mask",
        "training",
    }
    explicit_test_cols = {
        "is_test",
        "test",
        "is_test_row",
        "test_row",
        "is_test_mask",
        "test_mask",
        "testing",
    }
    split_cols = {"split", "data_split", "partition", "subset", "phase", "group"}

    for frame in candidate_frames:
        for col in frame.columns:
            key = str(col).lower()
            if key in explicit_train_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=True)
                if parsed is not None:
                    return parsed
            if key in explicit_test_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=False)
                if parsed is not None:
                    return parsed
            if key in split_cols:
                parsed = _coerce_train_mask_from_series(frame[col], positive_is_train=True)
                if parsed is not None:
                    return parsed

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        if ids.notna().all():
            ids_sorted = np.sort(ids.to_numpy(dtype=np.float64))
            if ids_sorted.size >= 2:
                gaps = np.diff(ids_sorted)
                jumps = np.flatnonzero(gaps > 1)
                if jumps.size > 0:
                    cutoff = ids_sorted[jumps[0]] + 1.0
                    return (ids.to_numpy(dtype=np.float64) < cutoff).astype(bool)

    return np.ones(n, dtype=bool)


def _category_codes(raw):
    n = len(raw)
    if "spectral_type" not in raw.columns or "galaxy_population" not in raw.columns:
        return np.zeros(n, dtype=np.int64)

    spectral = raw["spectral_type"].astype("string").fillna("__NA__").astype(str).str.strip()
    population = raw["galaxy_population"].astype("string").fillna("__NA__").astype(str).str.strip()
    labels = (spectral + "|" + population).to_numpy()
    codes, _ = pd.factorize(labels)
    return codes.astype(np.int64)


def _global_eigen_stats(X_train):
    d = X_train.shape[1]
    if X_train.shape[0] == 0:
        mean = np.zeros(d, dtype=np.float64)
        cov = np.eye(d, dtype=np.float64)
    else:
        mean = np.nanmean(X_train, axis=0)
        centered = X_train - mean
        if X_train.shape[0] > 1:
            cov = (centered.T @ centered) / float(X_train.shape[0] - 1)
        else:
            cov = np.zeros((d, d), dtype=np.float64)

    cov = 0.5 * (cov + cov.T) + _EPSILON * np.eye(d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return mean, eigvals[order], eigvecs[:, order]


def _compute_row_metrics(row_vec, neighbors, neighbor_dists, global_stats):
    global_mean, global_eigvals, global_eigvecs = global_stats
    d = row_vec.shape[0]

    metrics = {
        "r2": np.nan,
        "residual2": np.nan,
        "a12": np.nan,
        "a23": np.nan,
        "tail": np.nan,
        "dim95": np.nan,
        "rk": np.nan,
        "r_med": np.nan,
        "r_mean": np.nan,
        "density_proxy": np.nan,
        "r_ratio": np.nan,
        "frac_in_box": np.nan,
        "asym_mad": np.nan,
        "asym_mad_ratio": np.nan,
        "degenerate": 1.0,
        "n_neighbors": 0.0,
    }

    if neighbors is None or neighbors.shape[0] == 0:
        # No local neighbors available: fallback to global geometric stats.
        eigvals = global_eigvals
        eigvecs = global_eigvecs
        centroid = global_mean
        metrics["n_neighbors"] = 0.0
        metrics["degenerate"] = 1.0
    else:
        metrics["n_neighbors"] = float(neighbors.shape[0])
        centroid = np.nanmean(neighbors, axis=0)
        centered = neighbors - centroid

        if neighbors.shape[0] > 1:
            cov = (centered.T @ centered) / float(neighbors.shape[0] - 1)
        else:
            cov = np.zeros((d, d), dtype=np.float64)

        cov = 0.5 * (cov + cov.T) + _EPSILON * np.eye(d, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigen_rank = np.linalg.matrix_rank(cov, tol=1e-12)
        eigen_trace = float(np.trace(cov))
        if eigen_rank < d or eigen_trace <= _EPSILON:
            eigvals = global_eigvals
            eigvecs = global_eigvecs
            centroid = global_mean
            metrics["degenerate"] = 1.0
        else:
            metrics["degenerate"] = 0.0

    # Eigen descriptors
    eig_sum = float(np.sum(eigvals))
    if eig_sum <= _EPSILON:
        metrics["r2"] = np.nan
        metrics["tail"] = np.nan
        metrics["a12"] = np.nan
        metrics["a23"] = np.nan
        metrics["dim95"] = np.nan
        metrics["residual2"] = np.nan
    else:
        metrics["r2"] = float((eigvals[0] + eigvals[1]) / eig_sum)
        metrics["tail"] = float(eigvals[-1] / eig_sum)
        denom_a23 = max(eigvals[2], _EPSILON)
        metrics["a12"] = float(np.log((eigvals[0] + _EPSILON) / (eigvals[1] + _EPSILON)))
        metrics["a23"] = float(np.log((eigvals[1] + _EPSILON) / (denom_a23 + _EPSILON)))
        cum = np.cumsum(eigvals / eig_sum)
        metrics["dim95"] = float(np.searchsorted(cum, 0.95, side="left") + 1)

        row_delta = row_vec - centroid
        v2 = eigvecs[:, :2]
        orth = row_delta - (v2 @ (v2.T @ row_delta))
        residual_denom = float(np.sum(eigvals[2:]) + _EPSILON)
        metrics["residual2"] = float((orth @ orth) / residual_denom)

    # Neighborhood support
    if neighbor_dists is not None and neighbor_dists.size > 0:
        metrics["rk"] = float(np.max(neighbor_dists))
        metrics["r_med"] = float(np.median(neighbor_dists))
        metrics["r_mean"] = float(np.mean(neighbor_dists))
        metrics["density_proxy"] = float(1.0 / (metrics["r_med"] + _EPSILON))
        metrics["r_ratio"] = float(metrics["rk"] / (metrics["r_med"] + _EPSILON))

        q05 = np.quantile(neighbors, 0.05, axis=0)
        q95 = np.quantile(neighbors, 0.95, axis=0)
        within_box = (row_vec >= q05) & (row_vec <= q95)
        metrics["frac_in_box"] = float(np.mean(within_box.astype(np.float64)))
    else:
        metrics["r_ratio"] = np.nan

    # Asymmetry / skewed support signal
    if neighbors is not None and neighbors.shape[0] > 0:
        nbr_median = np.nanmedian(neighbors, axis=0)
        nbr_mad = np.nanmedian(np.abs(neighbors - nbr_median), axis=0)
        metrics["asym_mad"] = float(np.median(np.abs(row_vec - nbr_median)))
        metrics["asym_mad_ratio"] = float(metrics["asym_mad"] / (np.median(nbr_mad) + _EPSILON))

    return metrics


def _query_space_metrics(X_scaled, train_indices, query_indices, k_requested, remove_self, global_stats):
    query_indices = np.asarray(query_indices, dtype=np.int64)
    k_requested = np.asarray(k_requested, dtype=np.int64)
    remove_self = np.asarray(remove_self, dtype=bool)

    nq = len(query_indices)
    out = {key: np.full(nq, np.nan, dtype=np.float64) for key in _FEATURE_KEYS}

    if nq == 0 or len(train_indices) == 0:
        out["n_neighbors"] = np.zeros(nq, dtype=np.float64)
        return out

    max_k = int(np.max(k_requested))
    if max_k <= 0:
        out["n_neighbors"] = np.zeros(nq, dtype=np.float64)
        return out

    nn_neighbors = min(len(train_indices), max_k + 1)
    nn = NearestNeighbors(
        n_neighbors=nn_neighbors,
        algorithm="kd_tree",
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X_scaled[train_indices])

    dists, idx = nn.kneighbors(X_scaled[query_indices], n_neighbors=nn_neighbors, return_distance=True)

    for pos, row_idx in enumerate(query_indices):
        k = int(k_requested[pos])
        if k <= 0:
            continue

        cand_pos = idx[pos]
        cand_rows = train_indices[cand_pos]
        cand_d = dists[pos]

        if remove_self[pos]:
            keep = cand_rows != row_idx
            cand_rows = cand_rows[keep]
            cand_d = cand_d[keep]

        if cand_rows.size == 0:
            continue

        keep_k = min(k, int(cand_rows.size))
        cand_rows = cand_rows[:keep_k]
        cand_d = cand_d[:keep_k]

        metrics = _compute_row_metrics(
            X_scaled[row_idx],
            X_scaled[cand_rows],
            cand_d,
            global_stats,
        )

        for key in _FEATURE_KEYS:
            out[key][pos] = metrics[key]

    return out


def _compute_space_features(raw, feature_matrix, train_mask, space_name):
    n = feature_matrix.shape[0]
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.size != n:
        train_mask = np.ones(n, dtype=bool)

    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        train_indices = np.arange(n, dtype=np.int64)
        train_mask = np.ones(n, dtype=bool)

    X = np.asarray(feature_matrix, dtype=np.float64)
    train_X = X[train_indices]

    med = np.nanmedian(train_X, axis=0)
    mad = np.nanmedian(np.abs(train_X - med), axis=0)
    mad = np.where(mad < _MAD_CLIP, _MAD_CLIP, mad)
    X_scaled = (X - med) / mad

    global_stats = _global_eigen_stats(X_scaled[train_indices])

    all_indices = np.arange(n, dtype=np.int64)
    full_k = np.where(
        train_mask,
        np.minimum(_K_NEIGHBORS, np.maximum(train_indices.size - 1, 0)),
        np.minimum(_K_NEIGHBORS, train_indices.size),
    )
    global_metrics = _query_space_metrics(X_scaled, train_indices, all_indices, full_k, train_mask, global_stats)

    space_features = {key: np.full(n, np.nan, dtype=np.float64) for key in _FEATURE_KEYS}
    sparse_mask = np.zeros(n, dtype=bool)
    k_local = np.zeros(n, dtype=np.int64)

    cat_codes = _category_codes(raw)
    for code in np.unique(cat_codes):
        q_rows = np.flatnonzero(cat_codes == code)
        in_train_q = train_mask[q_rows]

        cat_train_rows = np.flatnonzero(train_mask & (cat_codes == code))
        cat_count = int(cat_train_rows.size)

        if cat_count == 0:
            local_mask = np.zeros(q_rows.size, dtype=bool)
            fallback_mask = np.ones(q_rows.size, dtype=bool)
            k_local_block = np.zeros(q_rows.size, dtype=np.int64)
            sparse_block = np.ones(q_rows.size, dtype=bool)
        else:
            k_train = int(min(_K_NEIGHBORS, max(cat_count - 1, 0)))
            k_test = int(min(_K_NEIGHBORS, cat_count))
            k_local_block = np.where(in_train_q, k_train, k_test).astype(np.int64)
            sparse_block = k_local_block < _MIN_STABLE_K
            fallback_mask = (k_local_block == 0) | (cat_count < (2 * k_local_block))
            local_mask = ~fallback_mask

        k_local[q_rows] = k_local_block
        sparse_mask[q_rows] = sparse_block

        local_rows = q_rows[local_mask]
        if local_rows.size > 0:
            local_k = k_local_block[local_mask]
            local_remove = train_mask[local_rows]
            local_metrics = _query_space_metrics(
                X_scaled,
                cat_train_rows,
                local_rows,
                local_k,
                local_remove,
                global_stats,
            )
            for key in _FEATURE_KEYS:
                space_features[key][local_rows] = local_metrics[key]

        fallback_rows = q_rows[fallback_mask]
        if fallback_rows.size > 0:
            for key in _FEATURE_KEYS:
                space_features[key][fallback_rows] = global_metrics[key][fallback_rows]

    for key in _SPARSE_REPLACE_KEYS:
        if key in _FEATURE_KEYS:
            space_features[key][sparse_mask] = global_metrics[key][sparse_mask]

    space_features["sparse_flag"] = sparse_mask.astype(np.float64)
    return space_features


def add_local_manifold_codimension(raw, deps, aux):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(index=pd.Index([]) if raw is None else raw.index)

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)
    redshift = raw["redshift"].to_numpy(dtype=np.float64)

    # S1 = [c1, c2, c3, c4, redshift]
    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z
    X_s1 = np.column_stack((c1, c2, c3, c4, redshift))

    # S2 = [u, g, r, i, z]
    X_s2 = np.column_stack((u, g, r, i, z))

    train_mask = _extract_train_mask(raw, aux)

    s1 = _compute_space_features(raw, X_s1, train_mask, "s1")
    s2 = _compute_space_features(raw, X_s2, train_mask, "s2")

    out = pd.DataFrame(index=raw.index)

    for key in _FEATURE_KEYS:
        out[f"{_SPACE_PREFIX}_s1_{key}"] = s1[key]
        out[f"{_SPACE_PREFIX}_s2_{key}"] = s2[key]

    out[f"{_SPACE_PREFIX}_s1_sparse_flag"] = s1["sparse_flag"]
    out[f"{_SPACE_PREFIX}_s2_sparse_flag"] = s2["sparse_flag"]

    out[f"{_SPACE_PREFIX}_abs_delta_r2"] = (out[f"{_SPACE_PREFIX}_s1_r2"] - out[f"{_SPACE_PREFIX}_s2_r2"]).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_residual2"] = (
        out[f"{_SPACE_PREFIX}_s1_residual2"] - out[f"{_SPACE_PREFIX}_s2_residual2"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_density_proxy"] = (
        out[f"{_SPACE_PREFIX}_s1_density_proxy"] - out[f"{_SPACE_PREFIX}_s2_density_proxy"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_r_ratio"] = (
        out[f"{_SPACE_PREFIX}_s1_r_ratio"] - out[f"{_SPACE_PREFIX}_s2_r_ratio"]
    ).abs()
    out[f"{_SPACE_PREFIX}_abs_delta_frac_in_box"] = (
        out[f"{_SPACE_PREFIX}_s1_frac_in_box"] - out[f"{_SPACE_PREFIX}_s2_frac_in_box"]
    ).abs()

    return out


FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Builds local-manifold curvature, codimension, and neighborhood support descriptors from two scaled photometric spaces with category-conditioned neighborhoods and sparse-region fallbacks.",
    }
]