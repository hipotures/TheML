import numpy as np
import pandas as pd


_NEIGHBOR_K = 64
_MIN_VALID_NEIGHBORS = 16
_CONDITION_MIN_TRAIN = 128
_SHRINKAGE_RHO = 0.05
_EPS = 1.0e-12
_SCALE_FLOOR = 1.0e-6
_CHUNK_SIZE = 4096
_DEFAULT_TRAIN_ID_UPPER = 577346

_TRAIN_MASK_COLUMNS = (
    "is_train",
    "__is_train__",
    "_is_train",
    "train",
    "is_training",
    "in_train",
    "is_train_row",
)
_SPLIT_COLUMNS = (
    "split",
    "dataset",
    "source",
    "__split__",
    "_split",
    "row_type",
)
_TRAIN_STRINGS = ("train", "training", "tr", "fit", "true", "1", "yes", "y")
_TEST_STRINGS = ("test", "testing", "te", "predict", "submission", "false", "0", "no", "n")
_CONTRAST_METRICS = (
    "top2_share",
    "effective_dim",
    "residual2",
    "density_log",
    "frac_in_envelope",
)


def _mask_from_attrs(obj, n_rows):
    attrs = getattr(obj, "attrs", None)
    if not attrs:
        return None

    for key in ("train_n", "n_train", "num_train", "train_rows"):
        value = attrs.get(key)
        if isinstance(value, (int, np.integer)) and 0 < int(value) <= n_rows:
            mask = np.zeros(n_rows, dtype=bool)
            mask[: int(value)] = True
            return mask

    for key in ("train_mask", "is_train", "__is_train__"):
        value = attrs.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape[0] == n_rows:
            return arr.astype(bool, copy=False)

    return None


def _series_to_train_mask(series, n_rows):
    if len(series) != n_rows:
        return None

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool, copy=False)

    if pd.api.types.is_numeric_dtype(series):
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(arr)
        if finite.any():
            lo = np.nanmin(arr[finite])
            hi = np.nanmax(arr[finite])
            if lo >= 0.0 and hi <= 1.0:
                return np.nan_to_num(arr, nan=0.0) > 0.5

    values = series.astype("string").str.lower().str.strip()
    train = values.isin(_TRAIN_STRINGS)
    test = values.isin(_TEST_STRINGS)
    if bool(train.any()) or bool(test.any()):
        return train.fillna(False).to_numpy(dtype=bool, copy=False)

    return None


def _infer_train_mask(raw, aux):
    n_rows = len(raw)

    for obj in (aux, raw):
        mask = _mask_from_attrs(obj, n_rows)
        if mask is not None and mask.shape[0] == n_rows and mask.any():
            return mask.astype(bool, copy=False)

    if isinstance(aux, pd.DataFrame) and len(aux) == n_rows:
        for col in _TRAIN_MASK_COLUMNS:
            if col in aux.columns:
                mask = _series_to_train_mask(aux[col], n_rows)
                if mask is not None and mask.any():
                    return mask
        for col in _SPLIT_COLUMNS:
            if col in aux.columns:
                mask = _series_to_train_mask(aux[col], n_rows)
                if mask is not None and mask.any():
                    return mask

    for col in _TRAIN_MASK_COLUMNS:
        if col in raw.columns:
            mask = _series_to_train_mask(raw[col], n_rows)
            if mask is not None and mask.any():
                return mask

    for col in _SPLIT_COLUMNS:
        if col in raw.columns:
            mask = _series_to_train_mask(raw[col], n_rows)
            if mask is not None and mask.any():
                return mask

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(ids)
        if finite.any():
            has_train_ids = np.any(ids[finite] <= _DEFAULT_TRAIN_ID_UPPER)
            has_test_ids = np.any(ids[finite] > _DEFAULT_TRAIN_ID_UPPER)
            if has_train_ids and has_test_ids:
                return ids <= _DEFAULT_TRAIN_ID_UPPER
            if has_train_ids:
                return np.ones(n_rows, dtype=bool)

    return np.ones(n_rows, dtype=bool)


def _make_spaces(raw):
    required = ("u", "g", "r", "i", "z", "redshift")
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise KeyError("Missing columns required for local manifold features: " + ", ".join(missing))

    u = raw["u"].to_numpy(dtype=np.float64, copy=False)
    g = raw["g"].to_numpy(dtype=np.float64, copy=False)
    r = raw["r"].to_numpy(dtype=np.float64, copy=False)
    i = raw["i"].to_numpy(dtype=np.float64, copy=False)
    z = raw["z"].to_numpy(dtype=np.float64, copy=False)
    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    color_z = np.column_stack((c_ug, c_gr, c_ri, c_iz, redshift))
    color = np.column_stack((c_ug, c_gr, c_ri, c_iz))
    mag = np.column_stack((u, g, r, i, z))

    return (
        ("color_z", color_z),
        ("color", color),
        ("mag", mag),
    )


def _robust_scale(values, train_mask):
    train_values = values[train_mask]
    if train_values.shape[0] == 0:
        train_values = values

    med = np.nanmedian(train_values, axis=0)
    quantiles = np.nanpercentile(train_values, (25.0, 75.0), axis=0)
    scale = quantiles[1] - quantiles[0]

    med = np.where(np.isfinite(med), med, 0.0)
    scale = np.where(np.isfinite(scale) & (np.abs(scale) >= _SCALE_FLOOR), scale, 1.0)

    scaled = (values - med) / scale
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled.astype(np.float32, copy=False)


def _category_codes(raw):
    if "spectral_type" not in raw.columns or "galaxy_population" not in raw.columns:
        return np.zeros(len(raw), dtype=np.int32)

    spectral = raw["spectral_type"].astype("string").fillna("__missing_spectral__")
    population = raw["galaxy_population"].astype("string").fillna("__missing_population__")
    multi = pd.MultiIndex.from_arrays((spectral, population))
    codes = pd.factorize(multi, sort=False)[0]
    return codes.astype(np.int32, copy=False)


def _metric_names_for_dim(d):
    names = [
        "top2_share",
        "effective_dim",
        "dim95",
        "log_anisotropy_12",
        "log_anisotropy_23",
        "tail_share",
        "mahalanobis_shrunk",
        "residual2",
        "signed_top1_projection",
        "d_1",
        "d_8",
        "d_32",
        "d_64",
        "median_distance",
        "mean_distance",
        "density_log",
        "radius_ratio",
        "neighbor_distance_z",
        "frac_in_envelope",
        "mean_percentile_rank",
        "max_outside_envelope_excess",
        "sparse_flag",
    ]
    if d >= 5:
        names.insert(1, "top3_share")
        names.insert(9, "residual3")
    return names


def _fallback_metric_names_for_dim(d):
    names = [
        "top2_share",
        "effective_dim",
        "dim95",
        "log_anisotropy_12",
        "log_anisotropy_23",
        "tail_share",
        "mahalanobis_shrunk",
        "residual2",
        "signed_top1_projection",
    ]
    if d >= 5:
        names.extend(("top3_share", "residual3"))
    return names


def _allocate_metrics(length, d):
    metrics = {}
    for name in _metric_names_for_dim(d):
        if name == "sparse_flag":
            metrics[name] = np.ones(length, dtype=np.int8)
        else:
            metrics[name] = np.full(length, np.nan, dtype=np.float32)
    return metrics


def _fit_neighbor_model(x_train):
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(
        n_neighbors=min(_NEIGHBOR_K + 1, x_train.shape[0]),
        algorithm="kd_tree",
        leaf_size=40,
        metric="euclidean",
        n_jobs=-1,
    )
    return model.fit(x_train)


def _select_neighbors(indices, distances, self_local):
    n_rows = indices.shape[0]
    raw_k = indices.shape[1]
    out_idx = np.full((n_rows, _NEIGHBOR_K), -1, dtype=np.int32)
    out_dist = np.full((n_rows, _NEIGHBOR_K), np.nan, dtype=np.float64)

    if raw_k > _NEIGHBOR_K:
        keep = indices != self_local[:, None]
        rank = np.cumsum(keep, axis=1)
        take = keep & (rank <= _NEIGHBOR_K)
        counts = take.sum(axis=1)
        if np.all(counts == _NEIGHBOR_K):
            out_idx[:, :] = indices[take].reshape(n_rows, _NEIGHBOR_K)
            out_dist[:, :] = distances[take].reshape(n_rows, _NEIGHBOR_K)
            return out_idx, out_dist

    for row_idx in range(n_rows):
        if self_local[row_idx] >= 0:
            keep = indices[row_idx] != self_local[row_idx]
        else:
            keep = np.ones(raw_k, dtype=bool)

        chosen_idx = indices[row_idx, keep][:_NEIGHBOR_K]
        chosen_dist = distances[row_idx, keep][:_NEIGHBOR_K]
        count = chosen_idx.shape[0]
        out_idx[row_idx, :count] = chosen_idx
        out_dist[row_idx, :count] = chosen_dist

    return out_idx, out_dist


def _distance_at(sorted_distances, counts, nth):
    result = np.full(sorted_distances.shape[0], np.nan, dtype=np.float64)
    has_neighbors = counts > 0
    if np.any(has_neighbors):
        rows = np.flatnonzero(has_neighbors)
        positions = np.minimum(counts[has_neighbors], nth) - 1
        result[has_neighbors] = sorted_distances[rows, positions]
    return result


def _median_from_sorted(sorted_values, counts):
    result = np.full(sorted_values.shape[0], np.nan, dtype=np.float64)
    has_neighbors = counts > 0
    if np.any(has_neighbors):
        rows = np.flatnonzero(has_neighbors)
        lo = (counts[has_neighbors] - 1) // 2
        hi = counts[has_neighbors] // 2
        result[has_neighbors] = 0.5 * (sorted_values[rows, lo] + sorted_values[rows, hi])
    return result


def _median_unsorted(values, counts):
    ordered = np.sort(values, axis=1)
    return _median_from_sorted(ordered, counts)


def _percentile_bounds(neighbors, valid, counts):
    n_rows, k_count, d = neighbors.shape
    p5 = np.full((n_rows, d), np.nan, dtype=np.float64)
    p95 = np.full((n_rows, d), np.nan, dtype=np.float64)

    if np.all(counts == k_count):
        bounds = np.percentile(neighbors, (5.0, 95.0), axis=1)
        return bounds[0], bounds[1]

    full = counts == k_count
    if np.any(full):
        bounds = np.percentile(neighbors[full], (5.0, 95.0), axis=1)
        p5[full] = bounds[0]
        p95[full] = bounds[1]

    partial = np.flatnonzero((counts > 0) & (~full))
    for row_idx in partial:
        vals = neighbors[row_idx, valid[row_idx]]
        bounds = np.percentile(vals, (5.0, 95.0), axis=0)
        p5[row_idx] = bounds[0]
        p95[row_idx] = bounds[1]

    return p5, p95


def _put_metric(metrics, name, target, values):
    metrics[name][target] = np.asarray(values, dtype=np.float32)


def _fill_metrics_chunk(metrics, target_start, rows, x_all, x_train, neighbor_idx, neighbor_dist):
    n_rows = len(rows)
    d = x_all.shape[1]
    target = slice(target_start, target_start + n_rows)

    valid = neighbor_idx >= 0
    counts = valid.sum(axis=1).astype(np.int32, copy=False)
    sparse = counts < _MIN_VALID_NEIGHBORS
    metrics["sparse_flag"][target] = sparse.astype(np.int8, copy=False)

    idx_safe = np.where(valid, neighbor_idx, 0)
    neighbors = x_train[idx_safe].astype(np.float64, copy=False)
    valid3 = valid[:, :, None]

    denom = np.maximum(counts, 1).astype(np.float64, copy=False)
    masked_neighbors = np.where(valid3, neighbors, 0.0)
    centroid = masked_neighbors.sum(axis=1) / denom[:, None]

    x_query = x_all[rows].astype(np.float64, copy=False)
    offset = x_query - centroid
    centered = (neighbors - centroid[:, None, :]) * valid3

    cov_denom = np.maximum(counts - 1, 1).astype(np.float64, copy=False)
    cov = np.einsum("nkd,nke->nde", centered, centered) / cov_denom[:, None, None]
    trace = np.trace(cov, axis1=1, axis2=2)
    eye = np.eye(d, dtype=np.float64)
    sigma = (1.0 - _SHRINKAGE_RHO) * cov
    sigma = sigma + (_SHRINKAGE_RHO * trace / float(d))[:, None, None] * eye[None, :, :]
    sigma = sigma + _EPS * eye[None, :, :]

    try:
        eigvals, eigvecs = np.linalg.eigh(sigma)
    except np.linalg.LinAlgError:
        sigma = sigma + 1.0e-8 * eye[None, :, :]
        eigvals, eigvecs = np.linalg.eigh(sigma)

    eigvals = np.maximum(eigvals[:, ::-1], 0.0)
    eigvecs = eigvecs[:, :, ::-1]
    eig_sum = eigvals.sum(axis=1)
    eig_sum_safe = eig_sum + _EPS

    top2_share = (eigvals[:, 0] + eigvals[:, 1]) / eig_sum_safe
    _put_metric(metrics, "top2_share", target, top2_share)

    if d >= 5:
        top3_share = (eigvals[:, 0] + eigvals[:, 1] + eigvals[:, 2]) / eig_sum_safe
        _put_metric(metrics, "top3_share", target, top3_share)

    effective_dim = (eig_sum * eig_sum) / (np.sum(eigvals * eigvals, axis=1) + _EPS)
    _put_metric(metrics, "effective_dim", target, effective_dim)

    cumulative = np.cumsum(eigvals, axis=1) / eig_sum_safe[:, None]
    dim95 = np.argmax(cumulative >= 0.95, axis=1) + 1
    _put_metric(metrics, "dim95", target, dim95)

    _put_metric(metrics, "log_anisotropy_12", target, np.log((eigvals[:, 0] + _EPS) / (eigvals[:, 1] + _EPS)))
    _put_metric(metrics, "log_anisotropy_23", target, np.log((eigvals[:, 1] + _EPS) / (eigvals[:, 2] + _EPS)))
    _put_metric(metrics, "tail_share", target, eigvals[:, -1] / eig_sum_safe)

    solved = np.linalg.solve(sigma, offset[:, :, None])[:, :, 0]
    mahalanobis = np.maximum(np.einsum("nd,nd->n", offset, solved), 0.0)
    _put_metric(metrics, "mahalanobis_shrunk", target, mahalanobis)

    top2_vecs = eigvecs[:, :, :2]
    coeff2 = np.einsum("ndk,nd->nk", top2_vecs, offset)
    proj2 = np.einsum("ndk,nk->nd", top2_vecs, coeff2)
    resid2_vec = offset - proj2
    residual2 = np.einsum("nd,nd->n", resid2_vec, resid2_vec) / (eigvals[:, 2:].sum(axis=1) + _EPS)
    _put_metric(metrics, "residual2", target, residual2)

    if d >= 5:
        top3_vecs = eigvecs[:, :, :3]
        coeff3 = np.einsum("ndk,nd->nk", top3_vecs, offset)
        proj3 = np.einsum("ndk,nk->nd", top3_vecs, coeff3)
        resid3_vec = offset - proj3
        residual3 = np.einsum("nd,nd->n", resid3_vec, resid3_vec) / (eigvals[:, 3:].sum(axis=1) + _EPS)
        _put_metric(metrics, "residual3", target, residual3)

    top1_vec = eigvecs[:, :, 0].copy()
    anchor_dim = np.argmax(np.abs(top1_vec), axis=1)
    signs = np.sign(top1_vec[np.arange(n_rows), anchor_dim])
    signs = np.where(signs == 0.0, 1.0, signs)
    top1_vec *= signs[:, None]
    signed_projection = np.einsum("nd,nd->n", offset, top1_vec) / np.sqrt(eigvals[:, 0] + _EPS)
    _put_metric(metrics, "signed_top1_projection", target, signed_projection)

    _put_metric(metrics, "d_1", target, _distance_at(neighbor_dist, counts, 1))
    _put_metric(metrics, "d_8", target, _distance_at(neighbor_dist, counts, 8))
    _put_metric(metrics, "d_32", target, _distance_at(neighbor_dist, counts, 32))
    d64 = _distance_at(neighbor_dist, counts, 64)
    _put_metric(metrics, "d_64", target, d64)

    median_distance = _median_from_sorted(neighbor_dist, counts)
    mean_distance = np.sum(np.where(valid, neighbor_dist, 0.0), axis=1) / denom
    mean_distance = np.where(counts > 0, mean_distance, np.nan)

    _put_metric(metrics, "median_distance", target, median_distance)
    _put_metric(metrics, "mean_distance", target, mean_distance)
    _put_metric(metrics, "density_log", target, -np.log(median_distance + _EPS))
    _put_metric(metrics, "radius_ratio", target, d64 / (median_distance + _EPS))

    neighbor_centroid_dist = np.sqrt(np.einsum("nkd,nkd->nk", centered, centered))
    centroid_dist_work = np.where(valid, neighbor_centroid_dist, np.inf)
    median_neighbor_centroid = _median_unsorted(centroid_dist_work, counts)
    mad_work = np.where(valid, np.abs(neighbor_centroid_dist - median_neighbor_centroid[:, None]), np.inf)
    mad_neighbor_centroid = _median_unsorted(mad_work, counts)
    distance_to_centroid = np.sqrt(np.einsum("nd,nd->n", offset, offset))
    neighbor_distance_z = (distance_to_centroid - median_neighbor_centroid) / (mad_neighbor_centroid + 1.0e-6)
    _put_metric(metrics, "neighbor_distance_z", target, neighbor_distance_z)

    p5, p95 = _percentile_bounds(neighbors, valid, counts)
    inside = (x_query >= p5) & (x_query <= p95)
    frac_inside = inside.mean(axis=1)
    frac_inside = np.where(counts > 0, frac_inside, np.nan)
    _put_metric(metrics, "frac_in_envelope", target, frac_inside)

    rank_counts = np.sum((neighbors <= x_query[:, None, :]) & valid3, axis=1)
    percentile_rank = rank_counts / denom[:, None]
    mean_percentile_rank = percentile_rank.mean(axis=1)
    mean_percentile_rank = np.where(counts > 0, mean_percentile_rank, np.nan)
    _put_metric(metrics, "mean_percentile_rank", target, mean_percentile_rank)

    width = p95 - p5
    below = np.maximum(p5 - x_query, 0.0)
    above = np.maximum(x_query - p95, 0.0)
    outside = np.maximum(below, above) / (width + 1.0e-6)
    outside = np.nan_to_num(outside, nan=0.0, posinf=0.0, neginf=0.0)
    max_excess = outside.max(axis=1)
    max_excess = np.where(counts > 0, max_excess, np.nan)
    _put_metric(metrics, "max_outside_envelope_excess", target, max_excess)

    if np.any(sparse):
        for name in _fallback_metric_names_for_dim(d):
            segment = metrics[name][target]
            segment[sparse] = np.nan


def _apply_sparse_fallback(metrics, train_query_mask, d):
    sparse = metrics["sparse_flag"].astype(bool, copy=False)
    if not np.any(sparse):
        return

    train_valid = train_query_mask & (~sparse)
    any_valid = ~sparse

    for name in _fallback_metric_names_for_dim(d):
        arr = metrics[name]
        source = train_valid
        if not np.any(np.isfinite(arr[source])):
            source = any_valid

        if np.any(np.isfinite(arr[source])):
            fill_value = np.nanmedian(arr[source])
        else:
            fill_value = 0.0

        if not np.isfinite(fill_value):
            fill_value = 0.0

        arr[sparse] = fill_value


def _finalize_metrics(metrics):
    for name, arr in metrics.items():
        if np.issubdtype(arr.dtype, np.floating):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_neighbor_metrics_for_rows(x_all, x_train, query_rows, row_to_local):
    query_rows = np.asarray(query_rows, dtype=np.int64)
    d = x_all.shape[1]
    metrics = _allocate_metrics(len(query_rows), d)

    if len(query_rows) == 0 or x_train.shape[0] == 0:
        _finalize_metrics(metrics)
        return metrics

    model = _fit_neighbor_model(x_train)
    n_query_neighbors = min(_NEIGHBOR_K + 1, x_train.shape[0])

    for start in range(0, len(query_rows), _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, len(query_rows))
        rows = query_rows[start:end]
        distances, indices = model.kneighbors(
            x_all[rows],
            n_neighbors=n_query_neighbors,
            return_distance=True,
        )
        neighbor_idx, neighbor_dist = _select_neighbors(
            indices.astype(np.int32, copy=False),
            distances.astype(np.float64, copy=False),
            row_to_local[rows],
        )
        _fill_metrics_chunk(metrics, start, rows, x_all, x_train, neighbor_idx, neighbor_dist)

    train_query_mask = row_to_local[query_rows] >= 0
    _apply_sparse_fallback(metrics, train_query_mask, d)
    _finalize_metrics(metrics)
    return metrics


def _store_metrics(out, prefix, metrics):
    for name, values in metrics.items():
        out[prefix + "_" + name] = values


def _full_pool_metrics(x_scaled, train_mask):
    n_rows = x_scaled.shape[0]
    train_rows = np.flatnonzero(train_mask)
    row_to_local = np.full(n_rows, -1, dtype=np.int32)
    row_to_local[train_rows] = np.arange(train_rows.shape[0], dtype=np.int32)
    query_rows = np.arange(n_rows, dtype=np.int64)
    return _compute_neighbor_metrics_for_rows(x_scaled, x_scaled[train_rows], query_rows, row_to_local)


def _conditioned_metrics(x_scaled, train_mask, category_codes, full_metrics):
    n_rows = x_scaled.shape[0]
    conditioned = {name: values.copy() for name, values in full_metrics.items()}
    conditioned_missing = np.ones(n_rows, dtype=np.int8)

    for code in np.unique(category_codes):
        rows = np.flatnonzero(category_codes == code)
        train_rows = rows[train_mask[rows]]

        if train_rows.shape[0] < _CONDITION_MIN_TRAIN:
            continue

        row_to_local = np.full(n_rows, -1, dtype=np.int32)
        row_to_local[train_rows] = np.arange(train_rows.shape[0], dtype=np.int32)

        group_metrics = _compute_neighbor_metrics_for_rows(
            x_scaled,
            x_scaled[train_rows],
            rows,
            row_to_local,
        )

        for name, values in group_metrics.items():
            conditioned[name][rows] = values

        conditioned_missing[rows] = 0

    conditioned["conditioned_missing"] = conditioned_missing
    return conditioned


def add_local_manifold_codimension(raw, deps, aux):
    train_mask = _infer_train_mask(raw, aux)
    if train_mask.shape[0] != len(raw) or not train_mask.any():
        train_mask = np.ones(len(raw), dtype=bool)

    category_codes = _category_codes(raw)
    out = {}

    for space_name, values in _make_spaces(raw):
        x_scaled = _robust_scale(values, train_mask)

        full_metrics = _full_pool_metrics(x_scaled, train_mask)
        _store_metrics(out, space_name + "_full", full_metrics)

        conditioned = _conditioned_metrics(x_scaled, train_mask, category_codes, full_metrics)
        _store_metrics(out, space_name + "_conditioned", conditioned)

    for metric in _CONTRAST_METRICS:
        left = out["color_z_full_" + metric]
        color = out["color_full_" + metric]
        mag = out["mag_full_" + metric]
        out["contrast_color_z_color_full_absdiff_" + metric] = np.abs(left - color).astype(np.float32, copy=False)
        out["contrast_color_z_mag_full_absdiff_" + metric] = np.abs(left - mag).astype(np.float32, copy=False)

    return pd.DataFrame(out, index=raw.index, copy=False)


FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Train-anchored local manifold geometry and support diagnostics across color, color-redshift, and magnitude spaces.",
    }
]