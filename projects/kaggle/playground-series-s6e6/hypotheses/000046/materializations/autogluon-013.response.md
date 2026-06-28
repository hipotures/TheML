import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


K_NEIGHBORS = 64
MIN_VALID_NEIGHBORS = 16
CATEGORY_MIN_TRAIN = 128
SHRINKAGE_RHO = 0.05
EPS = 1e-12
SCALE_FLOOR = 1e-6
BATCH_SIZE = 4096


def _as_float_array(frame, columns):
    return frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)


def _infer_train_mask(raw, aux):
    n = len(raw)

    for col in ("is_train", "_is_train", "__is_train__", "train", "_train"):
        if aux is not None and col in aux.columns:
            return aux[col].to_numpy(dtype=bool, copy=False)

    for col in ("split", "_split", "__split__"):
        if aux is not None and col in aux.columns:
            values = aux[col].astype(str).str.lower()
            return values.isin(("train", "tr", "training")).to_numpy(dtype=bool, copy=False)

    if "id" in raw.columns:
        ids = raw["id"].to_numpy()
        return ids < 577347

    mask = np.zeros(n, dtype=bool)
    mask[: max(1, int(round(n * 0.7)))] = True
    return mask


def _robust_scale(train_values, all_values):
    med = np.nanmedian(train_values, axis=0)
    q75 = np.nanpercentile(train_values, 75, axis=0)
    q25 = np.nanpercentile(train_values, 25, axis=0)
    scale = np.maximum(q75 - q25, SCALE_FLOOR)
    return (all_values - med) / scale


def _nearest_model(x_train, k):
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for local_manifold_codimension features")
    model = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean", n_jobs=-1)
    model.fit(x_train)
    return model


def _safe_eig(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    return vals, vecs


def _row_metrics(x, neigh, distances):
    d = neigh.shape[1]
    centroid = neigh.mean(axis=0)
    centered = neigh - centroid
    denom = max(neigh.shape[0] - 1, 1)
    cov = centered.T @ centered / denom
    trace = float(np.trace(cov))
    cov = (1.0 - SHRINKAGE_RHO) * cov + SHRINKAGE_RHO * trace / max(d, 1) * np.eye(d)

    vals, vecs = _safe_eig(cov)
    total = float(vals.sum()) + EPS
    csum = np.cumsum(vals) / total

    offset = x - centroid
    inv_vals = 1.0 / np.maximum(vals, EPS)
    rotated = vecs.T @ offset
    mahal = float(np.sum((rotated * rotated) * inv_vals))

    residual2_vec = offset - vecs[:, : min(2, d)] @ rotated[: min(2, d)]
    residual2_denom = float(vals[2:].sum()) + EPS if d > 2 else EPS
    residual2 = float(np.dot(residual2_vec, residual2_vec) / residual2_denom)

    if d >= 5:
        residual3_vec = offset - vecs[:, :3] @ rotated[:3]
        residual3 = float(np.dot(residual3_vec, residual3_vec) / (float(vals[3:].sum()) + EPS))
        top3_share = float(vals[:3].sum() / total)
    else:
        residual3 = np.nan
        top3_share = np.nan

    sorted_dist = np.sort(distances)
    med_dist = float(np.median(sorted_dist))
    neigh_centroid_dist = np.linalg.norm(neigh - centroid, axis=1)
    nc_med = float(np.median(neigh_centroid_dist))
    nc_mad = float(np.median(np.abs(neigh_centroid_dist - nc_med)))
    point_centroid_dist = float(np.linalg.norm(offset))

    p05 = np.percentile(neigh, 5, axis=0)
    p95 = np.percentile(neigh, 95, axis=0)
    inside = (x >= p05) & (x <= p95)
    lower_scale = np.maximum(np.abs(p05), 1.0)
    upper_scale = np.maximum(np.abs(p95), 1.0)
    lower_excess = np.maximum(p05 - x, 0.0) / lower_scale
    upper_excess = np.maximum(x - p95, 0.0) / upper_scale

    ranks = np.mean(neigh <= x, axis=0)

    return {
        "top2_share": float(vals[: min(2, d)].sum() / total),
        "top3_share": top3_share,
        "effective_dim": float((total * total) / (float(np.sum(vals * vals)) + EPS)),
        "dim95": float(np.searchsorted(csum, 0.95) + 1),
        "log_anisotropy_12": float(np.log((vals[0] + EPS) / (vals[1] + EPS))) if d >= 2 else 0.0,
        "log_anisotropy_23": float(np.log((vals[1] + EPS) / (vals[2] + EPS))) if d >= 3 else np.nan,
        "tail_share": float(vals[-1] / total),
        "mahalanobis_shrunk": mahal,
        "residual2": residual2,
        "residual3": residual3,
        "signed_top1_projection": float(rotated[0] / np.sqrt(vals[0] + EPS)),
        "d_1": float(sorted_dist[min(0, len(sorted_dist) - 1)]),
        "d_8": float(sorted_dist[min(7, len(sorted_dist) - 1)]),
        "d_32": float(sorted_dist[min(31, len(sorted_dist) - 1)]),
        "d_64": float(sorted_dist[min(63, len(sorted_dist) - 1)]),
        "median_distance": med_dist,
        "mean_distance": float(np.mean(sorted_dist)),
        "density_log": float(-np.log(med_dist + EPS)),
        "radius_ratio": float(sorted_dist[-1] / (med_dist + EPS)),
        "neighbor_distance_z": float((point_centroid_dist - nc_med) / (nc_mad + 1e-6)),
        "fraction_in_envelope": float(np.mean(inside)),
        "mean_percentile_rank": float(np.mean(ranks)),
        "max_outside_excess": float(np.max(np.maximum(lower_excess, upper_excess))),
    }


def _metric_names(d):
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
        "fraction_in_envelope",
        "mean_percentile_rank",
        "max_outside_excess",
        "sparse_flag",
        "conditioned_missing",
    ]
    if d >= 5:
        names.insert(1, "top3_share")
        names.insert(9, "residual3")
    return names


def _fallback_values(x_train):
    model = _nearest_model(x_train, min(K_NEIGHBORS + 1, len(x_train)))
    sample_n = min(len(x_train), 4096)
    sample_idx = np.linspace(0, len(x_train) - 1, sample_n).astype(int)
    distances, indices = model.kneighbors(x_train[sample_idx], return_distance=True)

    rows = []
    for pos, train_pos in enumerate(sample_idx):
        idx = indices[pos]
        dist = distances[pos]
        keep = idx != train_pos
        idx = idx[keep][:K_NEIGHBORS]
        dist = dist[keep][:K_NEIGHBORS]
        if len(idx) >= MIN_VALID_NEIGHBORS:
            rows.append(_row_metrics(x_train[train_pos], x_train[idx], dist))

    names = _metric_names(x_train.shape[1])
    if not rows:
        return {name: 0.0 for name in names if name not in ("sparse_flag", "conditioned_missing")}

    return {
        name: float(np.nanmedian([row.get(name, np.nan) for row in rows]))
        for name in names
        if name not in ("sparse_flag", "conditioned_missing")
    }


def _compute_mode_features(x_all, x_train, train_mask, query_indices, model, train_lookup, prefix, conditioned_missing, fallback):
    d = x_all.shape[1]
    metric_names = _metric_names(d)
    out = {prefix + "_" + name: np.empty(len(query_indices), dtype=np.float64) for name in metric_names}

    need = min(K_NEIGHBORS + 1, len(x_train))
    for start in range(0, len(query_indices), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(query_indices))
        row_idx = query_indices[start:stop]
        distances, indices = model.kneighbors(x_all[row_idx], n_neighbors=need, return_distance=True)

        for local_pos, global_idx in enumerate(row_idx):
            idx = indices[local_pos]
            dist = distances[local_pos]

            mapped_train_pos = train_lookup.get(int(global_idx), -1)
            if mapped_train_pos >= 0:
                keep = idx != mapped_train_pos
                idx = idx[keep]
                dist = dist[keep]

            idx = idx[:K_NEIGHBORS]
            dist = dist[:K_NEIGHBORS]

            if len(idx) < MIN_VALID_NEIGHBORS:
                row = dict(fallback)
                row["sparse_flag"] = 1.0
            else:
                row = _row_metrics(x_all[global_idx], x_train[idx], dist)
                row["sparse_flag"] = 0.0

            row["conditioned_missing"] = float(conditioned_missing)

            for name in metric_names:
                out[prefix + "_" + name][start + local_pos] = row.get(name, np.nan)

    return pd.DataFrame(out, index=query_indices)


def _compute_space(raw, x_all, train_mask, space_name):
    n = len(raw)
    x_train = x_all[train_mask]
    train_indices = np.flatnonzero(train_mask)
    train_lookup = {int(global_idx): int(pos) for pos, global_idx in enumerate(train_indices)}
    fallback = _fallback_values(x_train)

    result_parts = []
    full_model = _nearest_model(x_train, min(K_NEIGHBORS + 1, len(x_train)))
    all_indices = np.arange(n)
    result_parts.append(
        _compute_mode_features(
            x_all,
            x_train,
            train_mask,
            all_indices,
            full_model,
            train_lookup,
            space_name + "_full",
            0.0,
            fallback,
        )
    )

    conditioned_frames = []
    cond_prefix = space_name + "_conditioned"
    categories = raw["spectral_type"].astype(str) + "__" + raw["galaxy_population"].astype(str)

    for category in categories.unique():
        query_global = np.flatnonzero((categories == category).to_numpy())
        subset_train_global = query_global[train_mask[query_global]]

        if len(subset_train_global) < CATEGORY_MIN_TRAIN:
            conditioned_frames.append(
                _compute_mode_features(
                    x_all,
                    x_train,
                    train_mask,
                    query_global,
                    full_model,
                    train_lookup,
                    cond_prefix,
                    1.0,
                    fallback,
                )
            )
            continue

        subset_train_pos = np.searchsorted(train_indices, subset_train_global)
        subset_x_train = x_train[subset_train_pos]
        subset_lookup = {int(global_idx): int(pos) for pos, global_idx in enumerate(subset_train_global)}
        subset_model = _nearest_model(subset_x_train, min(K_NEIGHBORS + 1, len(subset_x_train)))

        conditioned_frames.append(
            _compute_mode_features(
                x_all,
                subset_x_train,
                train_mask,
                query_global,
                subset_model,
                subset_lookup,
                cond_prefix,
                0.0,
                fallback,
            )
        )

    result_parts.append(pd.concat(conditioned_frames, axis=0).sort_index())
    return pd.concat(result_parts, axis=1)


def _add_cross_space_contrasts(features):
    pairs = (("color_z", "color"), ("color_z", "mag"))
    metrics = ("top2_share", "effective_dim", "residual2", "density_log", "fraction_in_envelope")

    additions = {}
    for left, right in pairs:
        for metric in metrics:
            left_col = left + "_full_" + metric
            right_col = right + "_full_" + metric
            if left_col in features.columns and right_col in features.columns:
                additions["contrast_abs_" + left + "_vs_" + right + "_" + metric] = (
                    features[left_col] - features[right_col]
                ).abs()

    if additions:
        return pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)
    return features


def add_local_manifold_codimension(raw, deps, aux):
    train_mask = _infer_train_mask(raw, aux)
    raw_local = raw.copy(deep=False)

    c_ug = raw_local["u"].to_numpy(dtype=np.float64) - raw_local["g"].to_numpy(dtype=np.float64)
    c_gr = raw_local["g"].to_numpy(dtype=np.float64) - raw_local["r"].to_numpy(dtype=np.float64)
    c_ri = raw_local["r"].to_numpy(dtype=np.float64) - raw_local["i"].to_numpy(dtype=np.float64)
    c_iz = raw_local["i"].to_numpy(dtype=np.float64) - raw_local["z"].to_numpy(dtype=np.float64)

    color = np.column_stack([c_ug, c_gr, c_ri, c_iz])
    color_z = np.column_stack([c_ug, c_gr, c_ri, c_iz, raw_local["redshift"].to_numpy(dtype=np.float64)])
    mag = _as_float_array(raw_local, ["u", "g", "r", "i", "z"])

    spaces = {
        "color_z": _robust_scale(color_z[train_mask], color_z),
        "color": _robust_scale(color[train_mask], color),
        "mag": _robust_scale(mag[train_mask], mag),
    }

    frames = []
    for space_name, values in spaces.items():
        frames.append(_compute_space(raw_local, values, train_mask, space_name))

    features = pd.concat(frames, axis=1)
    features = _add_cross_space_contrasts(features)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features.index = raw.index
    return features


FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Train-anchored local neighborhood geometry, manifold residual, support, and cross-space contrast diagnostics.",
    }
]