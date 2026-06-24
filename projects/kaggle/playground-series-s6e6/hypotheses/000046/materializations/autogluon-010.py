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


def _linear_quantile_bounds(neighbors):
    n, d = neighbors.shape

    if n == 0:
        nan = np.full(d, np.nan, dtype=np.float64)
        return nan, nan

    if n == 1:
        vec = neighbors[0].astype(np.float64)
        return vec, vec.copy()

    pos_low = 0.05 * (n - 1)
    lo_low = int(np.floor(pos_low))
    hi_low = int(np.ceil(pos_low))
    w_low = pos_low - lo_low

    pos_high = 0.95 * (n - 1)
    lo_high = int(np.floor(pos_high))
    hi_high = int(np.ceil(pos_high))
    w_high = pos_high - lo_high

    nth = sorted(set((lo_low, hi_low, lo_high, hi_high)))
    partitioned = np.partition(neighbors, nth, axis=0)

    q05 = partitioned[lo_low]
    if hi_low != lo_low:
        q05 = q05 + (partitioned[hi_low] - q05) * w_low

    q95 = partitioned[lo_high]
    if hi_high != lo_high:
        q95 = q95 + (partitioned[hi_high] - q95) * w_high

    return q05, q95


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
        eigvals = global_eigvals
        eigvecs = global_eigvecs
        centroid = global_mean
        metrics["n_neighbors"] = 0.0
        metrics["degenerate"] = 1.0
    else:
        metrics["n_neighbors"] = float(neighbors.shape[0])
        centroid = np.mean(neighbors, axis=0)
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
        metrics["a12"] = float(np.log((eigvals[0] + _EPSILON) / (eigvals[1] + _EPSILON)))
        metrics["a23"] = float(np.log((eigvals[1] + _EPSILON) / (max(eigvals[2], _EPSILON) + _EPSILON)))
        cum = np.cumsum(eigvals / eig_sum)
        metrics["dim95"] = float(np.searchsorted(cum, 0.95, side="left") + 1)

        row_delta = row_vec - centroid
        v2 = eigvecs[:, :2]
        orth = row_delta - (v2 @ (v2.T @ row_delta))
        residual_denom = float(np.sum(eigvals[2:]) + _EPSILON)
        metrics["residual2"] = float((orth @ orth) / residual_denom)

    if neighbor_dists is not None and neighbor_dists.size > 0:
        metrics["rk"] = float(np.max(neighbor_dists))
        metrics["r_med"] = float(np.median(neighbor_dists))
        metrics["r_mean"] = float(np.mean(neighbor_dists))
        metrics["density_proxy"] = float(1.0 / (metrics["r_med"] + _EPSILON))
        metrics["r_ratio"] = float(metrics["rk"] / (metrics["r_med"] + _EPSILON))

        q05, q95 = _linear_quantile_bounds(neighbors)
        within_box = (row_vec >= q05) & (row_vec <= q95)
        metrics["frac_in_box"] = float(np.mean(within_box.astype(np.float64)))
    else:
        metrics["r_ratio"] = np.nan

    if neighbors is not None and neighbors.shape[0] > 0:
        nbr_median = np.median(neighbors, axis=0)
        nbr_mad = np.median(np.abs(neighbors - nbr_median), axis=0)
        metrics["asym_mad"] = float(np.median(np.abs(row_vec - nbr_median)))
        metrics["asym_mad_ratio"] = float(metrics["asym_mad"] / (np.median(nbr_mad) + _EPSILON))

    return metrics


def _write_metrics(output, pos, values):
    for key in _FEATURE_KEYS:
        output[key][pos] = values[key]


def _compute_space_features(raw, feature_matrix, train_mask, category_codes):
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

    n_train = train_indices.size
    nn_neighbors = int(min(n_train, _K_NEIGHBORS + 1))
    if nn_neighbors <= 0:
        all_idx = np.empty((n, 0), dtype=np.int32)
        all_dist = np.empty((n, 0), dtype=np.float32)
    else:
        nn = NearestNeighbors(
            n_neighbors=nn_neighbors,
            algorithm="kd_tree",
            metric="euclidean",
            n_jobs=-1,
        )
        nn.fit(X_scaled[train_indices])
        all_dist, all_idx = nn.kneighbors(X_scaled[all_indices], n_neighbors=nn_neighbors, return_distance=True)
        all_dist = np.asarray(all_dist, dtype=np.float32)
        all_idx = np.asarray(all_idx, dtype=np.int32)

    global_k = np.where(
        train_mask,
        np.minimum(_K_NEIGHBORS, np.maximum(n_train - 1, 0)),
        np.minimum(_K_NEIGHBORS, n_train),
    ).astype(np.int64)

    cat_codes = np.asarray(category_codes, dtype=np.int64)
    if cat_codes.size != n:
        cat_codes = np.zeros(n, dtype=np.int64)

    max_code = int(cat_codes.max(initial=0))
    train_cat_codes = cat_codes[train_mask]
    if train_cat_codes.size > 0:
        cat_counts = np.bincount(train_cat_codes, minlength=max_code + 1)
    else:
        cat_counts = np.array([0], dtype=np.int64)

    cat_counts_per_row = cat_counts[cat_codes] if cat_counts.size > 0 else np.zeros(n, dtype=np.int64)

    local_k = np.where(train_mask, cat_counts_per_row - 1, cat_counts_per_row).astype(np.int64)
    local_k = np.clip(local_k, 0, _K_NEIGHBORS)
    sparse_mask = local_k < _MIN_STABLE_K
    fallback_mask = (local_k == 0) | (cat_counts_per_row < (2 * local_k))

    space_features = {key: np.full(n, np.nan, dtype=np.float64) for key in _FEATURE_KEYS}

    for pos in range(n):
        row_vec = X_scaled[pos]
        row_code = cat_codes[pos]
        row_local_k = int(local_k[pos])

        row_idx = all_idx[pos]
        row_dist = all_dist[pos]

        if train_mask[pos] and row_idx.size > 0 and row_idx[0] == pos:
            row_idx = row_idx[1:]
            row_dist = row_dist[1:]
        elif train_mask[pos] and row_idx.size > 0:
            keep = row_idx != pos
            row_idx = row_idx[keep]
            row_dist = row_dist[keep]

        use_global = True
        sparse = bool(sparse_mask[pos])

        if not fallback_mask[pos] and row_local_k > 0 and row_idx.size >= row_local_k:
            same_cat = cat_codes[row_idx] == row_code
            local_candidates = row_idx[same_cat]
            local_dist = row_dist[same_cat]
            if local_candidates.size >= row_local_k:
                local_metrics = _compute_row_metrics(
                    row_vec,
                    X_scaled[local_candidates[:row_local_k]],
                    local_dist[:row_local_k],
                    global_stats,
                )
                _write_metrics(space_features, pos, local_metrics)
                use_global = sparse
                if not sparse:
                    continue
            else:
                use_global = True

        if use_global:
            g_k = int(global_k[pos])
            g_count = int(min(g_k, row_idx.size))
            if g_k <= 0 or g_count <= 0:
                global_metrics = _compute_row_metrics(row_vec, None, None, global_stats)
            else:
                global_metrics = _compute_row_metrics(
                    row_vec,
                    X_scaled[row_idx[:g_count]],
                    row_dist[:g_count],
                    global_stats,
                )

            if fallback_mask[pos] or sparse is True:
                if fallback_mask[pos]:
                    _write_metrics(space_features, pos, global_metrics)
                else:
                    for key in _SPARSE_REPLACE_KEYS:
                        space_features[key][pos] = global_metrics[key]
            else:
                _write_metrics(space_features, pos, global_metrics)
                for key in _SPARSE_REPLACE_KEYS:
                    space_features[key][pos] = global_metrics[key]

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

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z
    X_s1 = np.column_stack((c1, c2, c3, c4, redshift))
    X_s2 = np.column_stack((u, g, r, i, z))

    train_mask = _extract_train_mask(raw, aux)
    category_codes = _category_codes(raw)

    s1 = _compute_space_features(raw, X_s1, train_mask, category_codes)
    s2 = _compute_space_features(raw, X_s2, train_mask, category_codes)

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