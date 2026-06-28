import numpy as np
import pandas as pd


N_INITIAL_REDSHIFT_BINS = 30
MIN_BIN_ROWS = 8000
MIN_VALID_BIN_ROWS = 100
MIN_PCA_ROWS = 30
STANDARDIZED_FIT_CLIP = 6.0
LOCAL_SCALE_FLOOR_FRAC = 0.05
EPS = 1e-6
OVERLAP_CENTER = 2.7
OVERLAP_WIDTH = 0.45
COLOR_COLUMNS = ("u", "g", "r", "i", "z")
CUBE_SPECS = (("ugri", (0, 1, 2)), ("griz", (1, 2, 3)))


def _as_numeric_series(frame, column, index, default=0.0):
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").astype(float)
    return pd.Series(default, index=index, dtype=float)


def _scaled_mad(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return 1.4826 * np.median(np.abs(arr - med))


def _robust_center_scale(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0.0, 1.0
    clean = arr[finite]
    center = float(np.median(clean))
    scale = float(_scaled_mad(clean))
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.std(clean))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return center, scale


def _training_mask(raw, aux):
    mask = pd.Series(True, index=raw.index)
    if aux is None or aux.empty:
        if "id" in raw.columns:
            ids = pd.to_numeric(raw["id"], errors="coerce")
            finite_ids = ids[np.isfinite(ids)]
            if len(finite_ids) > 0:
                min_id = float(finite_ids.min())
                contiguous_train_cut = 577346.0
                if min_id <= contiguous_train_cut and float(finite_ids.max()) > contiguous_train_cut:
                    return (ids <= contiguous_train_cut).fillna(False).astype(bool)
        return mask

    for col in ("train_mask", "is_train", "in_train", "fit_mask", "is_fit", "is_training"):
        if col in aux.columns:
            candidate = aux[col].reindex(raw.index)
            if candidate.notna().any():
                return candidate.fillna(False).astype(bool)

    for col in ("split", "dataset", "fold_role", "role"):
        if col in aux.columns:
            text = aux[col].reindex(raw.index).astype(str).str.lower()
            candidate = text.isin(("train", "fit", "training"))
            if candidate.any():
                return candidate

    if "id" in raw.columns:
        ids = pd.to_numeric(raw["id"], errors="coerce")
        finite_ids = ids[np.isfinite(ids)]
        if len(finite_ids) > 0:
            contiguous_train_cut = 577346.0
            if float(finite_ids.min()) <= contiguous_train_cut and float(finite_ids.max()) > contiguous_train_cut:
                return (ids <= contiguous_train_cut).fillna(False).astype(bool)

    return mask


def _build_bins(z_train):
    z_clean = np.asarray(z_train, dtype=float)
    z_clean = z_clean[np.isfinite(z_clean)]
    if z_clean.size == 0:
        return np.array([-np.inf, np.inf], dtype=float), np.array([0.0], dtype=float)

    if z_clean.size < MIN_BIN_ROWS:
        center = float(np.median(z_clean))
        return np.array([-np.inf, np.inf], dtype=float), np.array([center], dtype=float)

    quantiles = np.linspace(0.0, 1.0, N_INITIAL_REDSHIFT_BINS + 1)
    raw_edges = np.quantile(z_clean, quantiles)
    unique_edges = np.unique(raw_edges)
    if unique_edges.size < 2:
        center = float(np.median(z_clean))
        return np.array([-np.inf, np.inf], dtype=float), np.array([center], dtype=float)

    raw_edges = unique_edges
    raw_edges[0] = -np.inf
    raw_edges[-1] = np.inf
    raw_bin = np.searchsorted(raw_edges[1:-1], z_clean, side="right")
    counts = np.bincount(raw_bin, minlength=len(raw_edges) - 1)

    merged_edges = [raw_edges[0]]
    running = 0
    for i, count in enumerate(counts):
        running += int(count)
        is_last = i == len(counts) - 1
        if running >= MIN_BIN_ROWS or is_last:
            merged_edges.append(raw_edges[i + 1])
            running = 0

    if len(merged_edges) < 2:
        merged_edges = [-np.inf, np.inf]
    else:
        merged_edges[0] = -np.inf
        merged_edges[-1] = np.inf

    edges = np.asarray(merged_edges, dtype=float)
    bin_ids = np.searchsorted(edges[1:-1], z_clean, side="right")
    centers = []
    for bin_id in range(len(edges) - 1):
        vals = z_clean[bin_ids == bin_id]
        if vals.size:
            centers.append(float(np.median(vals)))
        else:
            lo = edges[bin_id]
            hi = edges[bin_id + 1]
            if np.isfinite(lo) and np.isfinite(hi):
                centers.append(float((lo + hi) / 2.0))
            elif np.isfinite(lo):
                centers.append(float(lo))
            elif np.isfinite(hi):
                centers.append(float(hi))
            else:
                centers.append(float(np.median(z_clean)))
    return edges, np.asarray(centers, dtype=float)


def _orient_basis(components, previous_basis):
    basis = np.asarray(components, dtype=float).copy()
    if basis.shape != (3, 3):
        return None

    if previous_basis is None:
        if np.dot(basis[0], np.ones(3, dtype=float)) < 0.0:
            basis[0] *= -1.0
        anchor = np.array([1.0, -1.0, 0.0], dtype=float)
        if np.dot(basis[1], anchor) < 0.0:
            basis[1] *= -1.0
        if np.linalg.det(basis) < 0.0:
            basis[2] *= -1.0
        return basis

    aligned = basis.copy()
    for row in range(3):
        if np.dot(aligned[row], previous_basis[row]) < 0.0:
            aligned[row] *= -1.0
    if np.linalg.det(aligned) < 0.0:
        dots = np.abs(np.sum(aligned * previous_basis, axis=1))
        aligned[int(np.argmin(dots))] *= -1.0
    return aligned


def _fit_pca_basis(x_fit, previous_basis):
    x_fit = np.asarray(x_fit, dtype=float)
    if x_fit.shape[0] < MIN_PCA_ROWS or x_fit.shape[1] != 3:
        return None
    if not np.all(np.isfinite(x_fit)):
        return None

    cov = np.cov(x_fit, rowvar=False)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        return None

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order].T
    if not np.all(np.isfinite(eigvals)) or eigvals[0] < EPS:
        return None

    return _orient_basis(eigvecs, previous_basis)


def _coordinate_stats(coords):
    stats = {}
    for key, values in coords.items():
        center, scale = _robust_center_scale(values)
        hat = (np.asarray(values, dtype=float) - center) / (scale + EPS)
        lo, hi = np.nanpercentile(hat[np.isfinite(hat)], [0.5, 99.5]) if np.isfinite(hat).any() else (-10.0, 10.0)
        stats[key] = {
            "center": center,
            "scale": scale,
            "lo": float(lo) if np.isfinite(lo) else -10.0,
            "hi": float(hi) if np.isfinite(hi) else 10.0,
        }

    d = np.asarray(coords["d"], dtype=float)
    d_med = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
    d_scale = float(_scaled_mad(d))
    if not np.isfinite(d_scale) or d_scale < EPS:
        d_scale = float(np.nanstd(d)) if np.isfinite(d).any() else 1.0
    if not np.isfinite(d_scale) or d_scale < EPS:
        d_scale = 1.0
    d_norm = d / (d_med + d_scale + EPS)
    d_hi = np.nanpercentile(d_norm[np.isfinite(d_norm)], 99.5) if np.isfinite(d_norm).any() else 10.0
    stats["d_norm_hi"] = float(d_hi) if np.isfinite(d_hi) else 10.0
    stats["d_med"] = d_med
    stats["d_scale"] = d_scale
    return stats


def _fit_cube_bin_models(colors, z, train_mask, edges, cube_indices):
    train_idx = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    global_values = colors[train_idx][:, cube_indices]
    global_median = np.nanmedian(global_values, axis=0)
    global_mad = np.asarray([_scaled_mad(global_values[:, j]) for j in range(3)], dtype=float)
    global_mad = np.where(np.isfinite(global_mad) & (global_mad > EPS), global_mad, 1.0)

    bin_ids = np.searchsorted(edges[1:-1], z, side="right")
    n_bins = len(edges) - 1
    models = []
    previous_basis = None

    for bin_id in range(n_bins):
        in_bin_train = train_mask & (bin_ids == bin_id)
        idx = np.flatnonzero(in_bin_train)
        if idx.size < MIN_VALID_BIN_ROWS:
            models.append(None)
            continue

        values = colors[idx][:, cube_indices]
        med = np.nanmedian(values, axis=0)
        local_mad = np.asarray([_scaled_mad(values[:, j]) for j in range(3)], dtype=float)
        scale = np.maximum(local_mad, LOCAL_SCALE_FLOOR_FRAC * global_mad)
        scale = np.where(np.isfinite(scale) & (scale > EPS), scale, global_mad)
        if not np.all(np.isfinite(med)) or not np.all(np.isfinite(scale)):
            models.append(None)
            continue

        x = (values - med) / (scale + EPS)
        x_fit = np.clip(x, -STANDARDIZED_FIT_CLIP, STANDARDIZED_FIT_CLIP)
        finite_rows = np.all(np.isfinite(x_fit), axis=1)
        basis = _fit_pca_basis(x_fit[finite_rows], previous_basis)
        if basis is None:
            models.append(None)
            continue

        x_nonclip = x[finite_rows]
        projected = x_nonclip @ basis.T
        coord_map = {
            "t": projected[:, 0],
            "q2": projected[:, 1],
            "q3": projected[:, 2],
            "d": np.sqrt(projected[:, 1] ** 2 + projected[:, 2] ** 2),
        }
        stats = _coordinate_stats(coord_map)
        models.append({"median": med, "scale": scale, "basis": basis, "stats": stats})
        previous_basis = basis

    valid_bins = [i for i, model in enumerate(models) if model is not None]
    if valid_bins:
        return models, valid_bins, False

    med = np.nanmedian(global_values, axis=0)
    scale = np.where(global_mad > EPS, global_mad, 1.0)
    x = (global_values - med) / (scale + EPS)
    basis = _fit_pca_basis(np.clip(x[np.all(np.isfinite(x), axis=1)], -STANDARDIZED_FIT_CLIP, STANDARDIZED_FIT_CLIP), None)
    if basis is None:
        basis = np.eye(3, dtype=float)
    projected = x[np.all(np.isfinite(x), axis=1)] @ basis.T
    coord_map = {
        "t": projected[:, 0],
        "q2": projected[:, 1],
        "q3": projected[:, 2],
        "d": np.sqrt(projected[:, 1] ** 2 + projected[:, 2] ** 2),
    }
    fallback_model = {"median": med, "scale": scale, "basis": basis, "stats": _coordinate_stats(coord_map)}
    return [fallback_model for _ in range(n_bins)], list(range(n_bins)), True


def _nearest_valid_bin(bin_id, valid_bins, centers):
    valid = np.asarray(valid_bins, dtype=int)
    if valid.size == 0:
        return 0
    distances = np.abs(centers[valid] - centers[int(bin_id)])
    return int(valid[int(np.argmin(distances))])


def _transform_cube(colors, z, edges, centers, models, valid_bins):
    bin_ids = np.searchsorted(edges[1:-1], z, side="right")
    n = colors.shape[0]

    out = {
        "t": np.zeros(n, dtype=float),
        "q2": np.zeros(n, dtype=float),
        "q3": np.zeros(n, dtype=float),
        "t_hat": np.zeros(n, dtype=float),
        "q2_hat": np.zeros(n, dtype=float),
        "q3_hat": np.zeros(n, dtype=float),
        "d": np.zeros(n, dtype=float),
        "d_norm": np.zeros(n, dtype=float),
        "log1p_d_norm": np.zeros(n, dtype=float),
        "q2_sign": np.zeros(n, dtype=float),
        "q3_sign": np.zeros(n, dtype=float),
        "q2_ratio": np.zeros(n, dtype=float),
        "q3_ratio": np.zeros(n, dtype=float),
        "d_ratio": np.zeros(n, dtype=float),
        "fallback_indicator": np.zeros(n, dtype=float),
    }

    for raw_bin in range(len(edges) - 1):
        rows = np.flatnonzero(bin_ids == raw_bin)
        if rows.size == 0:
            continue
        model_bin = raw_bin if models[raw_bin] is not None else _nearest_valid_bin(raw_bin, valid_bins, centers)
        model = models[model_bin]
        out["fallback_indicator"][rows] = 1.0 if model_bin != raw_bin else 0.0

        x = (colors[rows] - model["median"]) / (model["scale"] + EPS)
        projected = x @ model["basis"].T
        t = projected[:, 0]
        q2 = projected[:, 1]
        q3 = projected[:, 2]
        d = np.sqrt(q2 ** 2 + q3 ** 2)

        stats = model["stats"]
        t_hat = (t - stats["t"]["center"]) / (stats["t"]["scale"] + EPS)
        q2_hat = (q2 - stats["q2"]["center"]) / (stats["q2"]["scale"] + EPS)
        q3_hat = (q3 - stats["q3"]["center"]) / (stats["q3"]["scale"] + EPS)
        d_norm = d / (stats["d_med"] + stats["d_scale"] + EPS)

        t_hat = np.clip(t_hat, stats["t"]["lo"], stats["t"]["hi"])
        q2_hat = np.clip(q2_hat, stats["q2"]["lo"], stats["q2"]["hi"])
        q3_hat = np.clip(q3_hat, stats["q3"]["lo"], stats["q3"]["hi"])
        d_norm = np.clip(d_norm, 0.0, stats["d_norm_hi"])

        denom = 1.0 + np.abs(t_hat) + d_norm
        out["t"][rows] = t
        out["q2"][rows] = q2
        out["q3"][rows] = q3
        out["t_hat"][rows] = t_hat
        out["q2_hat"][rows] = q2_hat
        out["q3_hat"][rows] = q3_hat
        out["d"][rows] = d
        out["d_norm"][rows] = d_norm
        out["log1p_d_norm"][rows] = np.log1p(np.maximum(d_norm, 0.0))
        out["q2_sign"][rows] = np.sign(q2)
        out["q3_sign"][rows] = np.sign(q3)
        out["q2_ratio"][rows] = q2_hat / denom
        out["q3_ratio"][rows] = q3_hat / denom
        out["d_ratio"][rows] = d_norm / (1.0 + np.abs(t_hat))

    return out


def add_redshift_adaptive_color_tube_residuals(raw, deps, aux):
    index = raw.index
    u = _as_numeric_series(raw, "u", index).to_numpy(dtype=float)
    g = _as_numeric_series(raw, "g", index).to_numpy(dtype=float)
    r = _as_numeric_series(raw, "r", index).to_numpy(dtype=float)
    i = _as_numeric_series(raw, "i", index).to_numpy(dtype=float)
    zmag = _as_numeric_series(raw, "z", index).to_numpy(dtype=float)
    redshift = np.maximum(_as_numeric_series(raw, "redshift", index).to_numpy(dtype=float), 0.0)

    colors = np.column_stack((u - g, g - r, r - i, i - zmag))
    train_mask = _training_mask(raw, aux).reindex(index).fillna(False).to_numpy(dtype=bool)
    if not np.any(train_mask):
        train_mask = np.ones(len(raw), dtype=bool)

    edges, centers = _build_bins(redshift[train_mask])
    z_bin = np.searchsorted(edges[1:-1], redshift, side="right")
    overlap_gate = np.exp(-0.5 * ((redshift - OVERLAP_CENTER) / OVERLAP_WIDTH) ** 2)

    features = {
        "redshift_bin_id": z_bin.astype(float),
        "redshift_bin_center": centers[np.clip(z_bin, 0, len(centers) - 1)],
        "redshift_overlap_gate": overlap_gate,
        "color_ug": colors[:, 0],
        "color_gr": colors[:, 1],
        "color_ri": colors[:, 2],
        "color_iz": colors[:, 3],
    }

    for cube_name, cube_indices in CUBE_SPECS:
        cube_values = colors[:, cube_indices]
        models, valid_bins, global_fallback = _fit_cube_bin_models(colors, redshift, train_mask, edges, cube_indices)
        transformed = _transform_cube(cube_values, redshift, edges, centers, models, valid_bins)

        for name, values in transformed.items():
            features[f"{cube_name}_{name}"] = values

        features[f"{cube_name}_gated_d_norm"] = transformed["d_norm"] * overlap_gate
        features[f"{cube_name}_gated_log1p_d_norm"] = transformed["log1p_d_norm"] * overlap_gate
        features[f"{cube_name}_gated_abs_q2_hat"] = np.abs(transformed["q2_hat"]) * overlap_gate
        features[f"{cube_name}_gated_abs_q3_hat"] = np.abs(transformed["q3_hat"]) * overlap_gate
        features[f"{cube_name}_global_fallback"] = np.full(len(raw), float(global_fallback), dtype=float)

    return pd.DataFrame(features, index=index)


FEATURE_GROUPS = [
    {
        "name": "redshift_adaptive_color_tube_residuals",
        "fn": add_redshift_adaptive_color_tube_residuals,
        "depends_on": [],
        "description": "Redshift-local PCA color-tube coordinates and residual distances for adjacent optical color manifolds.",
    }
]