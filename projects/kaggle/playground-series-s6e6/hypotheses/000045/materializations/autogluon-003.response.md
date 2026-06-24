import numpy as np
import pandas as pd

_RHO = 0.05
_MAX_COND = 1.0e8
_MAD_FLOOR = 1.0e-6
_MIN_BIN_FLOOR = 1500
_FRAC_MIN_BIN = 0.001
_PROJ_DIMS = 3
_PAIR_INDICES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)
_RESID_NAMES = (
    "pairwise_color_res_u_g",
    "pairwise_color_res_u_r",
    "pairwise_color_res_u_i",
    "pairwise_color_res_u_z",
    "pairwise_color_res_g_r",
    "pairwise_color_res_g_i",
    "pairwise_color_res_g_z",
    "pairwise_color_res_r_i",
    "pairwise_color_res_r_z",
    "pairwise_color_res_i_z",
)
_PROJ_NAMES = (
    "pairwise_color_pc1",
    "pairwise_color_pc2",
    "pairwise_color_pc3",
)
_MAHAL_NAME = "pairwise_color_mahalanobis"


def _pairwise_color_matrix(raw):
    mags = raw[["u", "g", "r", "i", "z"]].to_numpy(dtype=float)
    return np.column_stack([mags[:, i] - mags[:, j] for i, j in _PAIR_INDICES])


def _build_bin_edges(redshift, n_quantiles):
    red = np.asarray(redshift, dtype=float)
    red = red.copy()
    finite_mask = np.isfinite(red)
    if not np.all(finite_mask):
        replacement = np.nanmedian(red[finite_mask]) if np.any(finite_mask) else 0.0
        red[~finite_mask] = replacement

    if red.size == 0:
        return np.array([0.0, 1.0], dtype=float)

    quantiles = np.linspace(0.0, 1.0, int(n_quantiles) + 1)
    edges = np.quantile(red, quantiles).astype(float)

    lo = float(np.nanmin(edges))
    hi = float(np.nanmax(edges))
    if not np.isfinite(lo) or not np.isfinite(hi):
        finite_red = red[np.isfinite(red)]
        if finite_red.size == 0:
            return np.array([0.0, 1.0], dtype=float)
        lo = float(np.min(finite_red))
        hi = float(np.max(finite_red))

    if lo == hi:
        eps = 1.0 if lo == 0.0 else abs(lo) * 1.0e-9 + 1.0e-12
        edges = np.array([lo - eps, hi + eps], dtype=float)

    edges = np.unique(np.clip(edges, lo, hi))
    if edges.size < 2:
        return np.array([lo - 1.0e-9, hi + 1.0e-9], dtype=float)

    m_min = max(_MIN_BIN_FLOOR, int(np.ceil(_FRAC_MIN_BIN * red.size)))
    edges = edges.tolist()
    assignments = np.clip(np.searchsorted(edges, red, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(assignments, minlength=len(edges) - 1).astype(int).tolist()

    while len(counts) > 1:
        sparse_bins = np.where(np.array(counts) < m_min)[0]
        if sparse_bins.size == 0:
            break

        i = int(sparse_bins[0])

        left = i - 1
        while left >= 0 and counts[left] == 0:
            left -= 1

        right = i + 1
        while right < len(counts) and counts[right] == 0:
            right += 1

        if left < 0 and right >= len(counts):
            break

        if left < 0:
            counts[i + 1] += counts[i]
            del counts[i]
            del edges[i + 1]
        elif right >= len(counts):
            counts[i - 1] += counts[i]
            del counts[i]
            del edges[i + 1]
        else:
            dist_left = i - left
            dist_right = right - i
            if dist_left <= dist_right:
                counts[i - 1] += counts[i]
                del counts[i]
                del edges[i + 1]
            else:
                counts[i + 1] += counts[i]
                del counts[i]
                del edges[i + 1]

    return np.asarray(edges, dtype=float)


def _compute_bin_parameters(values):
    x = np.asarray(values, dtype=float)
    n, d = x.shape
    if n == 0:
        return None

    mu = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - mu), axis=0)
    mad = np.where((~np.isfinite(mad)) | (mad < _MAD_FLOOR), _MAD_FLOOR, mad)

    centered = x - mu
    if n <= 1:
        cov = np.zeros((d, d), dtype=float)
    else:
        cov = (centered.T @ centered) / float(n - 1)

    diag_cov = np.diag(np.diag(cov))
    cov = (1.0 - _RHO) * cov + _RHO * diag_cov
    ridge = 1.0e-6 * (np.trace(cov) / float(d))
    cov = cov + np.eye(d) * ridge

    if not np.all(np.isfinite(cov)):
        return None

    try:
        condition = np.linalg.cond(cov)
    except np.linalg.LinAlgError:
        return None
    if (not np.isfinite(condition)) or (condition > _MAX_COND):
        return None

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1.0e-30)
    eigvecs = eigvecs[:, order]

    for j in range(min(_PROJ_DIMS, d)):
        anchor = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[anchor, j] < 0.0:
            eigvecs[:, j] = -eigvecs[:, j]

    inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return {
        "mu": mu,
        "mad": mad,
        "inv_cov": inv_cov,
        "inv_sqrt": inv_sqrt,
        "eig_top": eigvecs[:, :_PROJ_DIMS],
    }


def _fallback_parameters(values):
    x = np.asarray(values, dtype=float)
    mu = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - mu), axis=0)
    mad = np.where((~np.isfinite(mad)) | (mad < _MAD_FLOOR), _MAD_FLOOR, mad)

    d = x.shape[1]
    diag = mad * mad
    inv_cov = np.diag(1.0 / np.maximum(diag, _MAD_FLOOR * _MAD_FLOOR))
    inv_sqrt = np.diag(1.0 / mad)
    eig_top = np.eye(d, _PROJ_DIMS)

    return {
        "mu": mu,
        "mad": mad,
        "inv_cov": inv_cov,
        "inv_sqrt": inv_sqrt,
        "eig_top": eig_top,
    }


def _apply_transform(x, params):
    centered = x - params["mu"]
    residual = centered / params["mad"]
    mahal = np.sqrt(np.einsum("ij,ij->i", centered @ params["inv_cov"], centered))
    whitened = centered @ params["inv_sqrt"]
    proj = whitened @ params["eig_top"]
    return residual, mahal, proj


def add_pairwise_color_lattice_residuals(raw, deps, aux):
    color_vec = _pairwise_color_matrix(raw)
    red = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)

    red_clean = red.copy()
    finite_red = np.isfinite(red_clean)
    if not finite_red.all():
        fill = float(np.nanmedian(red_clean[finite_red])) if np.any(finite_red) else 0.0
        red_clean[~finite_red] = fill

    edges = _build_bin_edges(red_clean, n_quantiles=32)
    if edges.size < 2:
        edges = np.array([float(np.nanmin(red_clean)), float(np.nanmax(red_clean))], dtype=float)

    if edges.size == 2 and edges[0] == edges[1]:
        eps = 1.0e-9 if edges[0] == 0.0 else abs(edges[0]) * 1.0e-9 + 1.0e-12
        edges = np.array([edges[0] - eps, edges[1] + eps], dtype=float)

    bin_idx = np.searchsorted(edges, red_clean, side="right") - 1
    max_bin = max(len(edges) - 2, 0)
    if max_bin > 0:
        bin_idx = np.clip(bin_idx, 0, max_bin)
    else:
        bin_idx = np.zeros(red_clean.shape[0], dtype=int)

    global_params = _compute_bin_parameters(color_vec)
    if global_params is None:
        global_params = _fallback_parameters(color_vec)
    global_resid, global_mahal, global_proj = _apply_transform(color_vec, global_params)

    n_bins = max(len(edges) - 1, 0)
    bin_params = {}
    if n_bins > 0:
        for b in range(n_bins):
            mask = bin_idx == b
            if np.any(mask):
                params = _compute_bin_parameters(color_vec[mask])
                if params is not None:
                    bin_params[b] = params

    valid_bins = np.array(sorted(bin_params.keys()), dtype=int)
    if n_bins > 0:
        valid_mask_by_bin = np.zeros(n_bins, dtype=bool)
        valid_mask_by_bin[valid_bins] = True
        use_fallback = ~valid_mask_by_bin[bin_idx]
    else:
        use_fallback = np.ones(red_clean.shape[0], dtype=bool)

    use_bin = bin_idx.copy()
    if np.any(use_fallback):
        if valid_bins.size > 0:
            centers = 0.5 * (edges[:-1] + edges[1:])
            valid_centers = centers[valid_bins]
            red_invalid = red_clean[use_fallback]
            dists = np.abs(red_invalid[:, None] - valid_centers[None, :])
            nearest = np.argmin(dists, axis=1)
            use_bin[use_fallback] = valid_bins[nearest]
        else:
            use_bin[use_fallback] = -1

    residual = global_resid.copy()
    mahal = global_mahal.copy()
    proj = global_proj.copy()

    for b in valid_bins:
        idx = np.flatnonzero(use_bin == b)
        if idx.size == 0:
            continue
        params = bin_params[int(b)]
        r, m, p = _apply_transform(color_vec[idx], params)
        residual[idx] = r
        mahal[idx] = m
        proj[idx] = p

    out = {
        _RESID_NAMES[0]: residual[:, 0],
        _RESID_NAMES[1]: residual[:, 1],
        _RESID_NAMES[2]: residual[:, 2],
        _RESID_NAMES[3]: residual[:, 3],
        _RESID_NAMES[4]: residual[:, 4],
        _RESID_NAMES[5]: residual[:, 5],
        _RESID_NAMES[6]: residual[:, 6],
        _RESID_NAMES[7]: residual[:, 7],
        _RESID_NAMES[8]: residual[:, 8],
        _RESID_NAMES[9]: residual[:, 9],
        _PROJ_NAMES[0]: proj[:, 0],
        _PROJ_NAMES[1]: proj[:, 1],
        _PROJ_NAMES[2]: proj[:, 2],
        _MAHAL_NAME: mahal,
    }

    return pd.DataFrame(out, index=raw.index)


FEATURE_GROUPS = [
    {
        "name": "pairwise_color_lattice_residuals",
        "fn": add_pairwise_color_lattice_residuals,
        "depends_on": [],
        "description": "Create redshift-binned, robust pairwise ugriz residual features with Mahalanobis and whitened principal-direction projections.",
    }
]