import numpy as np
import pandas as pd


Z_BIN_EDGES = (
    -0.02, 0.05, 0.15, 0.30, 0.50, 0.80, 1.20,
    1.80, 2.40, 3.00, 3.80, 4.70, 5.60, 7.05,
)
I_BIN_START = 10.0
I_BIN_STOP = 28.0
I_BIN_WIDTH = 0.5


def _mad(values, center=None):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if center is None:
        center = np.nanmedian(arr)
    return float(np.nanmedian(np.abs(arr - center)))


def _orient_vector(vec):
    out = np.asarray(vec, dtype=np.float64).copy()
    if out.size == 0:
        return out
    idx = int(np.argmax(np.abs(out)))
    if out[idx] < 0:
        out *= -1.0
    return out


def _fit_pca_basis(x_std):
    if x_std.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    centered = x_std - np.nanmean(x_std, axis=0)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        v1 = _orient_vector(vh[0])
        v2 = _orient_vector(vh[1]) if vh.shape[0] > 1 else np.array([0.0, 1.0, 0.0])
    except np.linalg.LinAlgError:
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        v1 = _orient_vector(vecs[:, order[0]])
        v2 = _orient_vector(vecs[:, order[1]])

    v2 = v2 - np.dot(v2, v1) * v1
    norm = np.linalg.norm(v2)
    if norm <= 1e-12:
        fallback = np.zeros_like(v1)
        fallback[int(np.argmin(np.abs(v1)))] = 1.0
        v2 = fallback - np.dot(fallback, v1) * v1
        norm = np.linalg.norm(v2)
    return v1, _orient_vector(v2 / max(norm, 1e-12))


def _fit_tube_model(colors, redshift, imag, indices):
    sample = colors[indices]
    med = np.nanmedian(sample, axis=0)
    mad = np.nanmedian(np.abs(sample - med), axis=0)
    scale = np.maximum(1.4826 * mad, 0.03)

    x_std = (sample - med) / scale
    v1_initial, _ = _fit_pca_basis(x_std)
    projected_initial = np.outer(np.dot(x_std, v1_initial), v1_initial)
    dist_initial = np.sqrt(np.sum((x_std - projected_initial) ** 2, axis=1))

    cutoff = np.nanpercentile(dist_initial, 85.0)
    core_mask = dist_initial <= cutoff
    if np.count_nonzero(core_mask) < 20:
        core_mask = np.ones(indices.size, dtype=bool)

    core = x_std[core_mask]
    v1, v2 = _fit_pca_basis(core)

    projected_core = np.outer(np.dot(core, v1), v1)
    residual_core = core - projected_core
    dist_core = np.sqrt(np.sum(residual_core ** 2, axis=1))
    spread = max(1.4826 * _mad(dist_core), 0.05)

    core_indices = indices[core_mask]
    local_floor = _estimate_local_floor(
        core,
        residual_core,
        redshift[core_indices],
        imag[core_indices],
        spread,
    )

    return {
        "median": med,
        "scale": scale,
        "v1": v1,
        "v2": v2,
        "spread": spread,
        "floor": local_floor,
        "fill_abs": float(np.nanmedian(dist_core)),
        "fill_signed": 0.0,
    }


def _estimate_local_floor(core, residual_core, redshift_core, imag_core, spread):
    n = core.shape[0]
    if n < 8:
        return 0.20 * spread

    coord = np.column_stack((np.dot(core, _safe_first_axis(core)), redshift_core, imag_core))
    coord_scale = np.maximum(np.nanmedian(np.abs(coord - np.nanmedian(coord, axis=0)), axis=0), 1e-6)
    coord = coord / coord_scale

    max_rows = min(n, 700)
    if n > max_rows:
        take = np.linspace(0, n - 1, max_rows).astype(np.int64)
    else:
        take = np.arange(n, dtype=np.int64)

    jitters = []
    for idx in take:
        delta = coord - coord[idx]
        dist2 = np.sum(delta * delta, axis=1)
        k = min(8, n)
        near = np.argpartition(dist2, k - 1)[:k]
        near = near[near != idx]
        if near.size == 0:
            continue
        if near.size > 7:
            near = near[np.argsort(dist2[near])[:7]]
        med_res = np.nanmedian(residual_core[near], axis=0)
        jitters.append(np.linalg.norm(residual_core[idx] - med_res))

    if not jitters:
        floor = 0.20 * spread
    else:
        floor = float(np.nanmedian(jitters) / np.sqrt(2.0))

    return float(np.clip(floor, 0.10 * spread, 0.70 * spread))


def _safe_first_axis(core):
    if core.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0])
    v1, _ = _fit_pca_basis(core)
    return v1


def _apply_tube_model(colors, model):
    x_std = (colors - model["median"]) / model["scale"]
    projected = np.outer(np.dot(x_std, model["v1"]), model["v1"])
    residual = x_std - projected
    distance = np.sqrt(np.sum(residual ** 2, axis=1))
    abs_score = np.maximum(distance - model["floor"], 0.0) / (model["spread"] + 1e-6)
    signed_score = np.sign(np.dot(residual, model["v2"])) * abs_score
    return abs_score, signed_score


def _resolve_sample_indices(cell_indices, z_bin, i_bin, cell_map, nz, ni):
    own = cell_indices.get((z_bin, i_bin))
    if own is not None and own.size >= 200:
        return own

    best = own if own is not None else np.empty(0, dtype=np.int64)
    for ir in range(0, 4):
        i_lo = max(0, i_bin - ir)
        i_hi = min(ni - 1, i_bin + ir)
        for zr in range(0, 2):
            z_lo = max(0, z_bin - zr)
            z_hi = min(nz - 1, z_bin + zr)
            parts = []
            total = 0
            for zz in range(z_lo, z_hi + 1):
                for ii in range(i_lo, i_hi + 1):
                    arr = cell_map.get((zz, ii))
                    if arr is not None and arr.size:
                        parts.append(arr)
                        total += arr.size
            if total >= 200:
                return np.concatenate(parts) if len(parts) > 1 else parts[0]
            if total > best.size:
                best = np.concatenate(parts) if len(parts) > 1 else (parts[0] if parts else best)

    if best.size >= 80:
        return best
    return None


def _nearest_fitted_model(models, z_bin, i_bin, global_model):
    if not models:
        return global_model
    best_key = None
    best_dist = None
    for key in models:
        dist = abs(key[0] - z_bin) + abs(key[1] - i_bin)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_key = key
    return models.get(best_key, global_model)


def _compute_space_scores(colors, redshift, imag, z_ids, i_ids, nz, ni):
    n = colors.shape[0]
    global_indices = np.arange(n, dtype=np.int64)
    global_model = _fit_tube_model(colors, redshift, imag, global_indices)

    cell_map = {}
    for zz in range(nz):
        z_mask = z_ids == zz
        if not np.any(z_mask):
            continue
        for ii in range(ni):
            mask = z_mask & (i_ids == ii)
            if np.any(mask):
                cell_map[(zz, ii)] = np.flatnonzero(mask).astype(np.int64)

    models = {}
    pending = []
    for zz in range(nz):
        for ii in range(ni):
            own = cell_map.get((zz, ii))
            if own is None or own.size == 0:
                continue
            sample_idx = _resolve_sample_indices(cell_map, zz, ii, cell_map, nz, ni)
            if sample_idx is None:
                pending.append((zz, ii))
            else:
                models[(zz, ii)] = _fit_tube_model(colors, redshift, imag, sample_idx)

    abs_out = np.empty(n, dtype=np.float64)
    signed_out = np.empty(n, dtype=np.float64)

    for zz in range(nz):
        for ii in range(ni):
            row_idx = cell_map.get((zz, ii))
            if row_idx is None or row_idx.size == 0:
                continue
            model = models.get((zz, ii))
            if model is None:
                model = _nearest_fitted_model(models, zz, ii, global_model)
            abs_vals, signed_vals = _apply_tube_model(colors[row_idx], model)
            abs_vals = np.nan_to_num(abs_vals, nan=model["fill_abs"], posinf=12.0, neginf=0.0)
            signed_vals = np.nan_to_num(signed_vals, nan=model["fill_signed"], posinf=12.0, neginf=-12.0)
            abs_out[row_idx] = np.clip(abs_vals, 0.0, 12.0)
            signed_out[row_idx] = np.clip(signed_vals, -12.0, 12.0)

    return abs_out, signed_out


def add_error_deconvolved_locus_tube_residuals(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=np.float64)
    imag = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=np.float64)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=np.float64)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=np.float64)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=np.float64)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=np.float64)

    c1 = u - g
    c2 = g - r
    c3 = r - imag
    c4 = imag - z

    finite_fill = np.nanmedian(np.column_stack((c1, c2, c3, c4)), axis=0)
    c1 = np.nan_to_num(c1, nan=finite_fill[0], posinf=finite_fill[0], neginf=finite_fill[0])
    c2 = np.nan_to_num(c2, nan=finite_fill[1], posinf=finite_fill[1], neginf=finite_fill[1])
    c3 = np.nan_to_num(c3, nan=finite_fill[2], posinf=finite_fill[2], neginf=finite_fill[2])
    c4 = np.nan_to_num(c4, nan=finite_fill[3], posinf=finite_fill[3], neginf=finite_fill[3])

    redshift = np.nan_to_num(redshift, nan=np.nanmedian(redshift), posinf=Z_BIN_EDGES[-1], neginf=Z_BIN_EDGES[0])
    imag = np.nan_to_num(imag, nan=np.nanmedian(imag), posinf=I_BIN_STOP, neginf=I_BIN_START)

    z_edges = np.asarray(Z_BIN_EDGES, dtype=np.float64)
    i_edges = np.arange(I_BIN_START, I_BIN_STOP + I_BIN_WIDTH * 0.5, I_BIN_WIDTH, dtype=np.float64)

    z_clipped = np.clip(redshift, z_edges[0], z_edges[-1])
    i_clipped = np.clip(imag, i_edges[0], i_edges[-1])

    z_ids = np.searchsorted(z_edges, z_clipped, side="right") - 1
    i_ids = np.searchsorted(i_edges, i_clipped, side="right") - 1
    z_ids = np.clip(z_ids, 0, len(z_edges) - 2)
    i_ids = np.clip(i_ids, 0, len(i_edges) - 2)

    colors_a = np.column_stack((c1, c2, c3))
    colors_b = np.column_stack((c2, c3, c4))

    a_abs, a_signed = _compute_space_scores(colors_a, redshift, imag, z_ids, i_ids, len(z_edges) - 1, len(i_edges) - 1)
    b_abs, b_signed = _compute_space_scores(colors_b, redshift, imag, z_ids, i_ids, len(z_edges) - 1, len(i_edges) - 1)

    mean_abs = 0.5 * (a_abs + b_abs)

    return pd.DataFrame(
        {
            "A_abs": a_abs,
            "A_signed": a_signed,
            "B_abs": b_abs,
            "B_signed": b_signed,
            "mean_abs_z_24_30": mean_abs * ((redshift >= 2.4) & (redshift < 3.0)),
            "mean_abs_z_ge_35": mean_abs * (redshift >= 3.5),
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "error_deconvolved_locus_tube_residuals",
        "fn": add_error_deconvolved_locus_tube_residuals,
        "depends_on": [],
        "description": "Robust redshift- and brightness-conditioned color-locus tube residuals with deconvolved local broadening scores.",
    }
]