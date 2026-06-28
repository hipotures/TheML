import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


K_NEIGHBORS = 64
SELF_QUERY_K = 65
CONDITION_MIN_ROWS = 128
SPARSE_MIN_NEIGHBORS = 16
SHRINKAGE_RHO = 0.05
EPS = 1e-12
SCALE_FLOOR = 1e-6
MAD_FLOOR = 1e-6


def _resolve_train_mask(raw, aux):
    n_rows = len(raw)

    if aux is not None and len(aux) == n_rows:
        for col in ("is_train", "_is_train", "train", "split"):
            if col in aux.columns:
                values = aux[col]
                if col == "split":
                    lower = values.astype(str).str.lower()
                    if lower.isin(("train", "test")).any():
                        return lower.eq("train").to_numpy(dtype=bool)
                return values.astype(bool).to_numpy()

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        valid = ids.notna()
        if valid.all():
            id_values = ids.to_numpy()
            diffs = np.diff(np.sort(id_values))
            if len(diffs) and np.nanmax(diffs) <= 1.0:
                return id_values <= np.nanmedian(id_values)

            likely_train = id_values < 577347
            if likely_train.any() and (~likely_train).any():
                return likely_train

    return np.ones(n_rows, dtype=bool)


def _robust_scale(train_values, all_values):
    med = np.nanmedian(train_values, axis=0)
    q75 = np.nanpercentile(train_values, 75, axis=0)
    q25 = np.nanpercentile(train_values, 25, axis=0)
    scale = np.maximum(q75 - q25, SCALE_FLOOR)
    return (all_values - med) / scale


def _build_spaces(raw):
    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float)
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    c_ug = u - g
    c_gr = g - r
    c_ri = r - i
    c_iz = i - z

    return {
        "color_z": np.column_stack((c_ug, c_gr, c_ri, c_iz, redshift)),
        "color": np.column_stack((c_ug, c_gr, c_ri, c_iz)),
        "mag": np.column_stack((u, g, r, i, z)),
    }


def _nearest_neighbors(train_x, query_x, k, query_is_train=False):
    if len(train_x) == 0:
        return np.empty((len(query_x), 0), dtype=int), np.empty((len(query_x), 0), dtype=float)

    query_k = min(k + 1 if query_is_train else k, len(train_x))

    if NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=query_k, algorithm="auto", metric="euclidean")
        nn.fit(train_x)
        distances, indices = nn.kneighbors(query_x, return_distance=True)
    else:
        indices = np.empty((len(query_x), query_k), dtype=int)
        distances = np.empty((len(query_x), query_k), dtype=float)
        chunk = 2048
        for start in range(0, len(query_x), chunk):
            stop = min(start + chunk, len(query_x))
            d2 = ((query_x[start:stop, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
            part = np.argpartition(d2, query_k - 1, axis=1)[:, :query_k]
            part_d2 = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(part_d2, axis=1)
            indices[start:stop] = np.take_along_axis(part, order, axis=1)
            distances[start:stop] = np.sqrt(np.take_along_axis(part_d2, order, axis=1))

    if not query_is_train:
        return indices[:, :k], distances[:, :k]

    out_idx = np.empty((len(query_x), min(k, max(len(train_x) - 1, 0))), dtype=int)
    out_dst = np.empty((len(query_x), out_idx.shape[1]), dtype=float)
    for row in range(len(query_x)):
        keep = indices[row] != row
        kept_idx = indices[row][keep][: out_idx.shape[1]]
        kept_dst = distances[row][keep][: out_dst.shape[1]]
        out_idx[row, : len(kept_idx)] = kept_idx
        out_dst[row, : len(kept_dst)] = kept_dst
    return out_idx, out_dst


def _safe_percentile_rank(values, x):
    return np.mean(values <= x)


def _compute_metric_names(prefix, d):
    names = [
        "top2_share",
        "effective_dim",
        "dim95",
        "log_anisotropy_12",
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
        "max_outside_envelope_excess",
        "sparse_flag",
    ]
    if d >= 3:
        names.append("log_anisotropy_23")
    if d >= 5:
        names.extend(["top3_share", "residual3"])
    return [prefix + "_" + name for name in names]


def _row_metrics(x, neigh_x, neigh_dist, global_medians):
    d = x.shape[0]
    n = len(neigh_x)
    names = _compute_metric_names("", d)
    names = [name[1:] for name in names]

    if n < SPARSE_MIN_NEIGHBORS:
        out = dict(global_medians)
        out["sparse_flag"] = 1.0
        return out

    mu = np.mean(neigh_x, axis=0)
    centered = neigh_x - mu
    cov = np.cov(centered, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    trace = float(np.trace(cov))
    sigma = (1.0 - SHRINKAGE_RHO) * cov + SHRINKAGE_RHO * trace / max(d, 1) * np.eye(d)
    sigma = sigma + EPS * np.eye(d)

    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    total = float(np.sum(eigvals))
    total_eps = total + EPS
    cumulative = np.cumsum(eigvals) / total_eps

    v = x - mu
    inv = np.linalg.pinv(sigma)
    mahal = float(v @ inv @ v)

    proj2 = eigvecs[:, :2] @ (eigvecs[:, :2].T @ v)
    tail2 = float(np.sum(eigvals[2:])) + EPS
    residual2 = float(np.sum((v - proj2) ** 2) / tail2)

    distance_to_centroid = float(np.linalg.norm(v))
    neigh_centroid_dist = np.linalg.norm(centered, axis=1)
    centroid_med = float(np.median(neigh_centroid_dist))
    centroid_mad = float(np.median(np.abs(neigh_centroid_dist - centroid_med)))
    neighbor_distance_z = (distance_to_centroid - centroid_med) / (centroid_mad + MAD_FLOOR)

    q05 = np.percentile(neigh_x, 5, axis=0)
    q95 = np.percentile(neigh_x, 95, axis=0)
    spread = np.maximum(q95 - q05, SCALE_FLOOR)
    below = np.maximum(q05 - x, 0.0) / spread
    above = np.maximum(x - q95, 0.0) / spread
    outside = np.maximum(below, above)

    out = {
        "top2_share": float(np.sum(eigvals[:2]) / total_eps),
        "effective_dim": float((total * total) / (np.sum(eigvals * eigvals) + EPS)),
        "dim95": float(np.searchsorted(cumulative, 0.95) + 1),
        "log_anisotropy_12": float(np.log((eigvals[0] + EPS) / (eigvals[1] + EPS))) if d >= 2 else 0.0,
        "tail_share": float(eigvals[-1] / total_eps),
        "mahalanobis_shrunk": mahal,
        "residual2": residual2,
        "signed_top1_projection": float((v @ eigvecs[:, 0]) / np.sqrt(eigvals[0] + EPS)),
        "d_1": float(neigh_dist[min(0, len(neigh_dist) - 1)]),
        "d_8": float(neigh_dist[min(7, len(neigh_dist) - 1)]),
        "d_32": float(neigh_dist[min(31, len(neigh_dist) - 1)]),
        "d_64": float(neigh_dist[min(63, len(neigh_dist) - 1)]),
        "median_distance": float(np.median(neigh_dist)),
        "mean_distance": float(np.mean(neigh_dist)),
        "density_log": float(-np.log(np.median(neigh_dist) + EPS)),
        "radius_ratio": float(neigh_dist[min(63, len(neigh_dist) - 1)] / (np.median(neigh_dist) + EPS)),
        "neighbor_distance_z": float(neighbor_distance_z),
        "fraction_in_envelope": float(np.mean((x >= q05) & (x <= q95))),
        "mean_percentile_rank": float(np.mean([_safe_percentile_rank(neigh_x[:, j], x[j]) for j in range(d)])),
        "max_outside_envelope_excess": float(np.max(outside)),
        "sparse_flag": 0.0,
    }

    if d >= 3:
        out["log_anisotropy_23"] = float(np.log((eigvals[1] + EPS) / (eigvals[2] + EPS)))
    if d >= 5:
        proj3 = eigvecs[:, :3] @ (eigvecs[:, :3].T @ v)
        tail3 = float(np.sum(eigvals[3:])) + EPS
        out["top3_share"] = float(np.sum(eigvals[:3]) / total_eps)
        out["residual3"] = float(np.sum((v - proj3) ** 2) / tail3)

    for name in names:
        if name not in out:
            out[name] = 0.0

    return out


def _global_fallback_metrics(train_x):
    d = train_x.shape[1]
    if len(train_x) < SPARSE_MIN_NEIGHBORS:
        return {name[1:]: 0.0 for name in _compute_metric_names("", d)}

    sample_n = min(len(train_x), 512)
    sample = train_x[:sample_n]
    idx, dst = _nearest_neighbors(train_x, sample, K_NEIGHBORS, query_is_train=True)
    rows = []
    empty = {name[1:]: 0.0 for name in _compute_metric_names("", d)}
    for row in range(sample_n):
        metrics = _row_metrics(sample[row], train_x[idx[row]], dst[row], empty)
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    med = frame.median(axis=0, numeric_only=True).to_dict()
    for name in empty:
        med.setdefault(name, 0.0)
    med["sparse_flag"] = 1.0
    return med


def _compute_mode_features(space_name, mode_name, all_x, train_x, train_mask, group_keys=None):
    d = all_x.shape[1]
    col_names = _compute_metric_names(space_name + "_" + mode_name, d)
    local_names = [name.replace(space_name + "_" + mode_name + "_", "", 1) for name in col_names]
    result = np.zeros((len(all_x), len(col_names)), dtype=float)
    global_medians = _global_fallback_metrics(train_x)

    if group_keys is None:
        train_idx, train_dst = _nearest_neighbors(train_x, train_x, K_NEIGHBORS, query_is_train=True)
        test_positions = np.where(~train_mask)[0]
        if len(test_positions):
            test_idx, test_dst = _nearest_neighbors(train_x, all_x[test_positions], K_NEIGHBORS, query_is_train=False)
        else:
            test_idx = np.empty((0, K_NEIGHBORS), dtype=int)
            test_dst = np.empty((0, K_NEIGHBORS), dtype=float)

        test_counter = 0
        train_counter = 0
        for pos in range(len(all_x)):
            if train_mask[pos]:
                idx = train_idx[train_counter]
                dst = train_dst[train_counter]
                train_counter += 1
            else:
                idx = test_idx[test_counter]
                dst = test_dst[test_counter]
                test_counter += 1
            metrics = _row_metrics(all_x[pos], train_x[idx], dst, global_medians)
            result[pos] = [metrics[name] for name in local_names]
        return pd.DataFrame(result, columns=col_names)

    train_groups = group_keys[train_mask]
    all_groups = group_keys
    conditioned_missing = np.zeros(len(all_x), dtype=float)
    full_fallback = _compute_mode_features(space_name, "full_for_conditioned_fallback", all_x, train_x, train_mask)
    fallback_values = full_fallback.to_numpy(dtype=float)
    fallback_names = [name.replace(space_name + "_full_for_conditioned_fallback_", "", 1) for name in full_fallback.columns]
    fallback_lookup = {name: i for i, name in enumerate(fallback_names)}

    for key in pd.unique(pd.Series(all_groups)):
        row_positions = np.where(all_groups == key)[0]
        train_positions = np.where(train_groups == key)[0]

        if len(train_positions) < CONDITION_MIN_ROWS:
            conditioned_missing[row_positions] = 1.0
            for out_i, local_name in enumerate(local_names):
                if local_name in fallback_lookup:
                    result[row_positions, out_i] = fallback_values[row_positions, fallback_lookup[local_name]]
            continue

        sub_train_x = train_x[train_positions]
        sub_train_global_positions = np.where(train_mask)[0][train_positions]
        query_x = all_x[row_positions]
        query_is_train = np.isin(row_positions, sub_train_global_positions)

        train_row_positions = row_positions[query_is_train]
        test_row_positions = row_positions[~query_is_train]

        if len(train_row_positions):
            local_order = np.searchsorted(sub_train_global_positions, train_row_positions)
            idx, dst = _nearest_neighbors(sub_train_x, sub_train_x[local_order], K_NEIGHBORS, query_is_train=False)
            for j, pos in enumerate(train_row_positions):
                keep = train_positions[idx[j]] != local_order[j]
                use_idx = idx[j][keep][:K_NEIGHBORS]
                use_dst = dst[j][keep][:K_NEIGHBORS]
                metrics = _row_metrics(all_x[pos], sub_train_x[use_idx], use_dst, global_medians)
                result[pos] = [metrics[name] for name in local_names]

        if len(test_row_positions):
            idx, dst = _nearest_neighbors(sub_train_x, all_x[test_row_positions], K_NEIGHBORS, query_is_train=False)
            for j, pos in enumerate(test_row_positions):
                metrics = _row_metrics(all_x[pos], sub_train_x[idx[j]], dst[j], global_medians)
                result[pos] = [metrics[name] for name in local_names]

    frame = pd.DataFrame(result, columns=col_names)
    frame[space_name + "_" + mode_name + "_conditioned_missing"] = conditioned_missing
    return frame


def add_local_manifold_codimension(raw, deps, aux):
    train_mask = _resolve_train_mask(raw, aux)
    spaces = _build_spaces(raw)
    frames = []

    spectral = raw["spectral_type"].astype(str).to_numpy() if "spectral_type" in raw.columns else np.repeat("missing", len(raw))
    population = raw["galaxy_population"].astype(str).to_numpy() if "galaxy_population" in raw.columns else np.repeat("missing", len(raw))
    group_keys = np.char.add(np.char.add(spectral.astype(str), "||"), population.astype(str))

    full_metric_frames = {}

    for space_name, values in spaces.items():
        values = np.asarray(values, dtype=float)
        values = np.where(np.isfinite(values), values, np.nan)
        col_medians = np.nanmedian(values[train_mask], axis=0)
        values = np.where(np.isfinite(values), values, col_medians)
        scaled = _robust_scale(values[train_mask], values)
        train_scaled = scaled[train_mask]

        full_frame = _compute_mode_features(space_name, "full", scaled, train_scaled, train_mask)
        conditioned_frame = _compute_mode_features(space_name, "conditioned", scaled, train_scaled, train_mask, group_keys=group_keys)

        frames.append(full_frame)
        frames.append(conditioned_frame)
        full_metric_frames[space_name] = full_frame

    out = pd.concat(frames, axis=1)
    contrast_metrics = ["top2_share", "effective_dim", "residual2", "density_log", "fraction_in_envelope"]

    for metric in contrast_metrics:
        cz = full_metric_frames["color_z"]["color_z_full_" + metric].to_numpy(dtype=float)
        color = full_metric_frames["color"]["color_full_" + metric].to_numpy(dtype=float)
        mag = full_metric_frames["mag"]["mag_full_" + metric].to_numpy(dtype=float)
        out["contrast_color_z_color_absdiff_" + metric] = np.abs(cz - color)
        out["contrast_color_z_mag_absdiff_" + metric] = np.abs(cz - mag)

    out.index = raw.index
    return out


FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Train-anchored local manifold codimension and support diagnostics across photometric, color, and redshift spaces.",
    }
]