import numpy as np
import pandas as pd

MAG_BANDS = ("u", "g", "r", "i", "z")
REDSHIFT_COLUMN = "redshift"
N_PCA_COMPONENTS = 3
DESIRED_BIN_COUNT = 8
MIN_BIN_SUPPORT_FLOOR = 24
MAX_BIN_SUPPORT_CEILING = 250
MAG_MIN = -30.0
MAG_MAX = 60.0
EPS = 1e-12


def _to_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _compute_shape_frame(frame):
    band_vectors = []
    for band in MAG_BANDS:
        mag = _to_numeric(frame[band])
        mag = mag.where((mag >= MAG_MIN) & (mag <= MAG_MAX))
        finite = mag.dropna()
        if len(finite):
            fill_value = float(finite.median())
        else:
            fill_value = 0.0
        mag = mag.fillna(fill_value).to_numpy(dtype=float)
        band_vectors.append(mag)

    mags = np.column_stack(band_vectors)
    flux = np.power(10.0, -0.4 * mags)
    flux_sum = np.sum(flux, axis=1, keepdims=True)
    return flux / np.maximum(flux_sum, EPS)


def _extract_redshift(frame):
    red = _to_numeric(frame[REDSHIFT_COLUMN])
    finite = red.dropna()
    if len(finite):
        fill_value = float(finite.median())
    else:
        fill_value = 0.0
    return red.fillna(fill_value).to_numpy(dtype=float)


def _compute_bin_edges(redshift_values):
    z = np.asarray(redshift_values, dtype=float)
    if z.size == 0 or not np.isfinite(z).any():
        return np.array([0.0, 1.0], dtype=float)

    series = pd.Series(z)

    for nb in (DESIRED_BIN_COUNT, max(4, DESIRED_BIN_COUNT - 2), 4, 3, 2):
        try:
            _, edges = pd.qcut(series, q=nb, labels=False, retbins=True, duplicates="drop")
        except Exception:
            continue
        edges = np.asarray(edges, dtype=float)
        if len(edges) < 3:
            continue
        if not np.all(np.isfinite(edges)):
            continue
        edges = np.unique(edges)
        if len(edges) < 3 or np.any(np.diff(edges) <= 0):
            continue
        return edges

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        return np.array([0.0, 1.0], dtype=float)

    if z_min == z_max:
        span = 1.0 if z_min == 0.0 else 0.01 * (abs(z_min) + 1.0)
        return np.array([z_min - span, z_min + span], dtype=float)

    return np.array([z_min, (2.0 * z_min + z_max) / 3.0, (z_min + 2.0 * z_max) / 3.0, z_max], dtype=float)


def _assign_bins(redshift_values, edges):
    z = np.asarray(redshift_values, dtype=float)
    if z.size == 0 or len(edges) < 2:
        return np.zeros(z.size, dtype=np.int32)

    clipped = np.clip(z, edges[0], edges[-1])
    codes = pd.cut(clipped, bins=edges, labels=False, include_lowest=True)
    return codes.fillna(0).astype(np.int32).to_numpy()


def _fit_pca_model(block):
    block = np.asarray(block, dtype=float)
    n_rows, n_features = block.shape
    if n_rows < 2 or n_features == 0:
        mean = block.mean(axis=0) if n_rows else np.zeros(n_features, dtype=float)
        return {"mean": mean, "components": np.zeros((0, n_features), dtype=float)}

    mean = block.mean(axis=0)
    centered = block - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(N_PCA_COMPONENTS, vt.shape[0], vt.shape[1]) if vt.size else 0
        components = vt[:k] if k > 0 else np.zeros((0, n_features), dtype=float)
    except Exception:
        components = np.zeros((0, n_features), dtype=float)

    return {"mean": mean, "components": components}


def add_redshift_partitioned_sed_pca_residuals(raw, deps, aux):
    p_raw = _compute_shape_frame(raw)
    z_raw = _extract_redshift(raw)

    z_fit = z_raw
    p_fit = p_raw

    aux_is_usable = (
        aux is not None
        and hasattr(aux, "__len__")
        and len(aux) > 0
        and all(col in aux.columns for col in MAG_BANDS)
        and REDSHIFT_COLUMN in aux.columns
    )
    if aux_is_usable:
        try:
            aux_shape = _compute_shape_frame(aux)
            aux_z = _extract_redshift(aux)
            z_lo = np.nanmin(z_raw)
            z_hi = np.nanmax(z_raw)
            if np.isfinite(z_lo) and np.isfinite(z_hi):
                aux_z = np.clip(aux_z, z_lo, z_hi)
            p_fit = np.vstack((p_fit, aux_shape))
            z_fit = np.concatenate((z_fit, aux_z))
        except Exception:
            p_fit = p_raw
            z_fit = z_raw

    edges = _compute_bin_edges(z_fit)
    n_fit = len(z_fit)
    n_bins = max(1, len(edges) - 1)

    z_raw_clipped = np.clip(z_raw, edges[0], edges[-1])
    raw_bin = _assign_bins(z_raw_clipped, edges)
    fit_bin = _assign_bins(z_fit, edges)

    bin_sizes = np.bincount(fit_bin, minlength=n_bins)
    min_support = max(MIN_BIN_SUPPORT_FLOOR, min(MAX_BIN_SUPPORT_CEILING, int(np.sqrt(max(1, n_fit)))))
    supported = bin_sizes >= min_support
    if not np.any(supported):
        supported[:] = True

    global_model = _fit_pca_model(p_fit)

    bin_models = {}
    for b in np.where(supported)[0]:
        mask = fit_bin == b
        if mask.any():
            bin_models[int(b)] = _fit_pca_model(p_fit[mask])

    centers = (edges[:-1] + edges[1:]) / 2.0
    good_bins = np.where(supported)[0]
    fallback_map = np.full(n_bins, -1, dtype=np.int32)
    if good_bins.size:
        for b in range(n_bins):
            if supported[b]:
                fallback_map[b] = b
            else:
                fallback_map[b] = int(good_bins[np.argmin(np.abs(centers[b] - centers[good_bins]))])

    effective_bin = np.array(
        [fallback_map[b] if 0 <= b < n_bins else -1 for b in raw_bin],
        dtype=np.int32
    )
    used_global = effective_bin == -1
    fallback_used = (effective_bin != raw_bin) | used_global

    n_rows = len(raw)
    scores = np.zeros((n_rows, N_PCA_COMPONENTS), dtype=float)
    residual = np.zeros((n_rows, len(MAG_BANDS)), dtype=float)

    for b in np.unique(effective_bin):
        mask = np.where(effective_bin == b)[0]
        if mask.size == 0:
            continue
        model = global_model if b == -1 else bin_models.get(int(b), global_model)
        mean = model["mean"]
        components = model["components"]

        block = p_raw[mask]
        if components.size:
            centered = block - mean
            proj = centered @ components.T
            k = proj.shape[1]
            scores[mask, :k] = proj
            reconstructed = proj @ components + mean
        else:
            reconstructed = np.broadcast_to(mean, block.shape)

        residual[mask] = block - reconstructed

    residual_l2 = np.sqrt(np.einsum("ij,ij->i", residual, residual))
    shape_l2 = np.sqrt(np.sum(p_raw * p_raw, axis=1))
    residual_rel = residual_l2 / np.maximum(shape_l2, EPS)
    residual_energy = np.sum(residual * residual, axis=1) / np.maximum(np.sum(p_raw * p_raw, axis=1), EPS)

    bin_support = np.array(
        [float(bin_sizes[b]) if 0 <= b < n_bins else float(n_fit) for b in effective_bin],
        dtype=float
    )

    return pd.DataFrame(
        {
            "sed_c1": scores[:, 0].astype(np.float32),
            "sed_c2": scores[:, 1].astype(np.float32),
            "sed_c3": scores[:, 2].astype(np.float32),
            "sed_residual_norm": residual_l2.astype(np.float32),
            "sed_residual_norm_rel": residual_rel.astype(np.float32),
            "sed_residual_energy": residual_energy.astype(np.float32),
            "sed_residual_u": residual[:, 0].astype(np.float32),
            "sed_residual_g": residual[:, 1].astype(np.float32),
            "sed_residual_r": residual[:, 2].astype(np.float32),
            "sed_residual_i": residual[:, 3].astype(np.float32),
            "sed_residual_z": residual[:, 4].astype(np.float32),
            "sed_bin_id": raw_bin.astype(np.int32),
            "sed_bin_effective": effective_bin.astype(np.int32),
            "sed_bin_support": bin_support.astype(np.float32),
            "sed_bin_fallback": fallback_used.astype(np.uint8),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "redshift_partitioned_sed_pca_residuals",
        "fn": add_redshift_partitioned_sed_pca_residuals,
        "depends_on": [],
        "description": "Compute redshift-stratified PCA residual features on normalized ugri z flux-shape manifolds, with fallback to nearest-supported strata or global PCA.",
    }
]