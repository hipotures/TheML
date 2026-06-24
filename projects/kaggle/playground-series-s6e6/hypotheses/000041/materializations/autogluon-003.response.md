import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKEARN_GMM = True
except Exception:
    GaussianMixture = None
    _HAS_SKEARN_GMM = False

_CLASS_NAMES = ("GALAXY", "STAR", "QSO")
_Z_BINS = (0.0, 0.8, 2.5, 4.0, 7.0)
_I_BIN_STEP = 0.2
_EPS = 1e-12
_BACKOFF_SUPPORT = 200.0
_LOCAL_1_COMP_SUPPORT = 500.0
_LOCAL_2_COMP_SUPPORT = 2000.0
_LOCAL_8_COMP_SUPPORT = 5000.0
_PRIOR_SMOOTH = 0.3 / 3.0
_BLEND_WITH_GLOBAL = 0.3


def _triangular_bin_weights(values, centers):
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        idx_left = np.zeros_like(values, dtype=np.int64)
        idx_right = np.zeros_like(values, dtype=np.int64)
        w_left = np.ones_like(values, dtype=np.float64)
        w_right = np.zeros_like(values, dtype=np.float64)
        return idx_left, idx_right, w_left, w_right

    clipped = np.clip(values, centers[0], centers[-1])
    idx_left = np.searchsorted(centers, clipped, side="right") - 1
    idx_left = np.clip(idx_left, 0, centers.size - 1)
    idx_right = np.clip(idx_left + 1, 0, centers.size - 1)

    w_left = np.ones_like(values, dtype=np.float64)
    w_right = np.zeros_like(values, dtype=np.float64)
    mask = idx_right > idx_left
    if np.any(mask):
        left = idx_left[mask]
        right = idx_right[mask]
        span = centers[right] - centers[left]
        span = np.where(span <= 0, 1.0, span)
        frac = (clipped[mask] - centers[left]) / span
        frac = np.clip(frac, 0.0, 1.0)
        w_left[mask] = 1.0 - frac
        w_right[mask] = frac
    return idx_left, idx_right, w_left, w_right


def _build_i_centers(i_values):
    finite_i = np.asarray(i_values, dtype=np.float64)
    finite_i = finite_i[np.isfinite(finite_i)]
    if finite_i.size == 0:
        return (0.0,)
    i_min = float(np.nanmin(finite_i))
    i_max = float(np.nanmax(finite_i))
    step = float(_I_BIN_STEP)
    count = int(np.floor((i_max - i_min) / step))
    centers = [i_min + step * i for i in range(max(1, count + 1))]
    if centers[-1] < i_max:
        centers.append(centers[-1] + step)
    return tuple(centers)


def _components_for_support(support):
    if support >= _LOCAL_8_COMP_SUPPORT:
        return 8
    if support >= _LOCAL_2_COMP_SUPPORT:
        return 4
    if support >= _LOCAL_1_COMP_SUPPORT:
        return 2
    return 1


def _mad_diag_squared(values):
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    mad = np.maximum(mad, 1e-6)
    return mad * mad


def _pack_mixture_model(means, covariances, weights, data):
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    if means.ndim == 1:
        means = means[None, :]
        covariances = covariances[None, :, :]
        weights = weights[None]

    dim = means.shape[1]
    n_comp = means.shape[0]
    weights = np.clip(weights, _EPS, 1.0)
    weights_sum = np.sum(weights)
    weights = weights / max(weights_sum, _EPS)

    mad_diag = _mad_diag_squared(data)
    inv_covs = []
    log_coeffs = np.zeros(n_comp, dtype=np.float64)

    for k in range(n_comp):
        cov = 0.5 * (covariances[k] + covariances[k].T)
        cov = cov + np.diag(mad_diag)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            min_eig = np.min(eigvals)
        except Exception:
            min_eig = -1.0
        if not np.isfinite(min_eig) or min_eig <= 1e-12:
            cov = cov + np.eye(dim) * 1e-6

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            cov = cov + np.eye(dim) * 1e-4
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(logdet):
                cov = np.eye(dim) * np.maximum(np.max(np.diag(cov)), 1e-4)
                sign, logdet = np.linalg.slogdet(cov)

        inv_covs.append(np.linalg.pinv(cov))
        log_coeffs[k] = np.log(weights[k]) - 0.5 * (logdet + dim * np.log(2.0 * np.pi))

    return {
        "means": means,
        "inv_covs": inv_covs,
        "log_coeffs": log_coeffs,
        "n_components": n_comp,
    }


def _weighted_mean_cov(values, sample_weights):
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64).reshape(-1, 1)
    if x.shape[0] == 0:
        d = x.shape[1] if x.ndim == 2 else 1
        return np.zeros(d, dtype=np.float64), np.eye(d, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        w = np.ones((x.shape[0], 1), dtype=np.float64)
        w_sum = float(x.shape[0])
    mean = (x * w).sum(axis=0) / w_sum
    centered = x - mean
    cov = (centered * w).T @ centered / w_sum
    cov = 0.5 * (cov + cov.T)
    if cov.ndim == 1:
        cov = np.atleast_2d(cov)
    return mean, cov


def _fit_single_gaussian(values, sample_weights, support):
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64)
    if x.shape[0] == 0:
        return None
    mean, cov = _weighted_mean_cov(x, w)
    cov = cov + np.diag(np.maximum(np.diag(cov), 1e-12))
    return _pack_mixture_model(mean[None, :], cov[None, :, :], np.array([1.0], dtype=np.float64), x)


def _fit_mixture_model(values, sample_weights, support):
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64).astype(np.float64)
    if x.shape[0] == 0:
        return None

    n_comp = _components_for_support(float(support))
    n_comp = int(min(n_comp, x.shape[0]))
    if n_comp <= 1 or not _HAS_SKEARN_GMM:
        return _fit_single_gaussian(x, w, support)

    try:
        gm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            n_init=1,
            max_iter=80,
            random_state=0,
            reg_covar=1e-8,
        )
        try:
            gm.fit(x, sample_weight=w)
        except TypeError:
            gm.fit(x)
        gm_comp = int(gm.means_.shape[0])
        if gm_comp <= 1:
            return _fit_single_gaussian(x, w, support)
        return _pack_mixture_model(gm.means_, gm.covariances_, gm.weights_, x)
    except Exception:
        return _fit_single_gaussian(x, w, support)


def _score_mixture_density(model, values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] == 0 or model is None:
        return np.zeros((0,), dtype=np.float64)

    means = model["means"]
    inv_covs = model["inv_covs"]
    log_coeffs = model["log_coeffs"]
    n_comp = int(model["n_components"])

    log_terms = np.empty((n_comp, x.shape[0]), dtype=np.float64)
    for c in range(n_comp):
        diff = x - means[c]
        mahal = np.einsum("ij,ij->i", diff @ inv_covs[c], diff)
        log_terms[c] = log_coeffs[c] - 0.5 * mahal

    max_log = np.max(log_terms, axis=0)
    finite = np.isfinite(max_log)
    density = np.zeros(x.shape[0], dtype=np.float64)
    if np.any(finite):
        stable = np.exp(log_terms[:, finite] - max_log[None, finite])
        row_sum = np.sum(stable, axis=0)
        log_prob = max_log[finite] + np.log(np.maximum(row_sum, _EPS))
        density[finite] = np.exp(np.clip(log_prob, -745.0, 745.0))
    return density


def _kmeans_like_labels(values, n_clusters=3):
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    if n < n_clusters:
        labels = np.arange(n, dtype=np.int8) % max(1, n_clusters)
        return labels

    centers_idx = np.linspace(0, n - 1, n_clusters).astype(np.int64)
    centers = x[centers_idx].copy()
    labels = np.zeros(n, dtype=np.int8)

    for _ in range(12):
        diff2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(diff2, axis=1).astype(np.int8)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        new_centers = np.empty_like(centers)
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                new_centers[c] = np.mean(x[mask], axis=0)
            else:
                new_centers[c] = centers[c]
        if np.allclose(new_centers, centers, rtol=1e-7, atol=1e-9):
            centers = new_centers
            break
        centers = new_centers

    return labels


def _pseudo_class_labels(values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.int8)

    labels = None
    centers = None

    if _HAS_SKEARN_GMM:
        try:
            gm = GaussianMixture(
                n_components=3,
                covariance_type="full",
                n_init=1,
                max_iter=80,
                random_state=0,
                reg_covar=1e-8,
            )
            gm.fit(x)
            labels = gm.predict(x).astype(np.int8)
            centers = gm.means_.astype(np.float64)
        except Exception:
            labels = None

    if labels is None:
        labels = _kmeans_like_labels(x, n_clusters=3)
        centers = np.zeros((3, x.shape[1]), dtype=np.float64)
        for k in range(3):
            mask = labels == k
            if np.any(mask):
                centers[k] = np.mean(x[mask], axis=0)
            else:
                centers[k] = np.mean(x, axis=0)

    order = np.argsort(centers[:, 0])
    remap = {int(old): int(new) for new, old in enumerate(order)}
    mapped = np.empty_like(labels)
    for old, new in remap.items():
        mapped[labels == old] = new
    return mapped


def add_xdqso_inspired_flux_density_scores(raw, deps, aux):
    required = ("u", "g", "r", "i", "z", "redshift")
    x_series = []
    for col in required:
        if col not in raw.columns:
            raise KeyError(f"Missing required column: {col}")
        x_series.append(raw[col].to_numpy(dtype=np.float64, copy=False))

    u, g, r, i, zmag, redshift = x_series
    i = np.asarray(i, dtype=np.float64)
    x = np.column_stack([
        -0.4 * (u - i),
        -0.4 * (g - i),
        -0.4 * (r - i),
        -0.4 * (zmag - i),
    ])

    for d in range(x.shape[1]):
        col = x[:, d]
        lo = float(np.nanpercentile(col, 0.5))
        hi = float(np.nanpercentile(col, 99.5))
        if hi > lo:
            x[:, d] = np.clip(col, lo, hi)

    n = raw.shape[0]
    i_centers = _build_i_centers(i)
    z_centers = (
        (_Z_BINS[0] + _Z_BINS[1]) * 0.5,
        (_Z_BINS[1] + _Z_BINS[2]) * 0.5,
        (_Z_BINS[2] + _Z_BINS[3]) * 0.5,
        (_Z_BINS[3] + _Z_BINS[4]) * 0.5,
    )
    n_i_bins = len(i_centers)
    n_z_bins = len(z_centers)

    i_left, i_right, iw_left, iw_right = _triangular_bin_weights(i, i_centers)
    z_left, z_right, wz_left, wz_right = _triangular_bin_weights(redshift, z_centers)

    combos_i = np.concatenate([
        i_left,
        i_left,
        i_right,
        i_right,
    ]).astype(np.int64)
    combos_z = np.concatenate([
        z_left,
        z_right,
        z_left,
        z_right,
    ]).astype(np.int64)
    combos_w = np.concatenate([
        iw_left * wz_left,
        iw_left * wz_right,
        iw_right * wz_left,
        iw_right * wz_right,
    ]).astype(np.float64)

    combo_rows = np.repeat(np.arange(n, dtype=np.int64), 4)
    keep = combos_w > 0
    combos_i = combos_i[keep]
    combos_z = combos_z[keep]
    combos_w = combos_w[keep]
    combo_rows = combo_rows[keep]

    if combos_w.size == 0:
        # Degenerate fallback.
        p = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    else:
        labels = _pseudo_class_labels(x)
        n_classes = len(_CLASS_NAMES)

        local_models = [dict() for _ in range(n_classes)]
        z_models = [dict() for _ in range(n_classes)]
        local_support = np.zeros((n_classes, n_i_bins, n_z_bins), dtype=np.float64)
        z_support = np.zeros((n_classes, n_z_bins), dtype=np.float64)
        class_counts = np.zeros(n_classes, dtype=np.float64)
        global_models = [None for _ in range(n_classes)]

        for c in range(n_classes):
            class_mask = labels == c
            class_rows = np.flatnonzero(class_mask)
            class_counts[c] = float(class_rows.size)

            if class_rows.size > 0:
                global_models[c] = _fit_mixture_model(x[class_rows], np.ones(class_rows.size, dtype=np.float64), float(class_rows.size))

            class_combo_mask = class_mask[combo_rows]
            if not np.any(class_combo_mask):
                continue

            rows_c = combo_rows[class_combo_mask]
            i_c = combos_i[class_combo_mask]
            z_c = combos_z[class_combo_mask]
            w_c = combos_w[class_combo_mask]

            np.add.at(local_support[c], (i_c, z_c), w_c)
            np.add.at(z_support[c], z_c, w_c)

            cell_keys = i_c * n_z_bins + z_c
            order_cell = np.argsort(cell_keys)
            cell_keys = cell_keys[order_cell]
            rows_cell = rows_c[order_cell]
            w_cell = w_c[order_cell]

            uniq_keys, starts = np.unique(cell_keys, return_index=True)
            for ui, key in enumerate(uniq_keys):
                s = starts[ui]
                e = starts[ui + 1] if ui + 1 < starts.size else cell_keys.size
                support = local_support[c].ravel()[int(key)]
                if support < _LOCAL_1_COMP_SUPPORT:
                    continue
                i_idx = int(key // n_z_bins)
                z_idx = int(key - i_idx * n_z_bins)
                m_rows = rows_cell[s:e]
                m_w = w_cell[s:e]
                local_models[c][(i_idx, z_idx)] = _fit_mixture_model(x[m_rows], m_w, support)

            for z_idx in range(n_z_bins):
                support_z = z_support[c, z_idx]
                if support_z < _BACKOFF_SUPPORT:
                    continue
                z_mask = z_c == z_idx
                if not np.any(z_mask):
                    continue
                z_rows = rows_c[z_mask]
                z_w = w_c[z_mask]
                z_models[c][z_idx] = _fit_mixture_model(x[z_rows], z_w, support_z)

        global_density = np.zeros((n, len(_CLASS_NAMES)), dtype=np.float64)
        for c in range(len(_CLASS_NAMES)):
            if global_models[c] is not None:
                global_density[:, c] = _score_mixture_density(global_models[c], x)

        cell_total = local_support.sum(axis=0, keepdims=False)
        z_total = z_support.sum(axis=0, keepdims=False)

        cell_prior = np.full((len(_CLASS_NAMES), n_i_bins, n_z_bins), _PRIOR_SMOOTH, dtype=np.float64)
        z_prior = np.full((len(_CLASS_NAMES), n_z_bins), _PRIOR_SMOOTH, dtype=np.float64)

        for c in range(len(_CLASS_NAMES)):
            denom = cell_total
            mask = denom > 0
            if np.any(mask):
                cell_prior[c][mask] = 0.7 * (local_support[c][mask] / denom[mask]) + _PRIOR_SMOOTH
            denom_z = z_total
            mask_z = denom_z > 0
            if np.any(mask_z):
                z_prior[c][mask_z] = 0.7 * (z_support[c][mask_z] / denom_z[mask_z]) + _PRIOR_SMOOTH

        scores = np.zeros((n, len(_CLASS_NAMES)), dtype=np.float64)
        row_support = np.zeros((n, len(_CLASS_NAMES)), dtype=np.float64)

        cell_keys = combos_i * n_z_bins + combos_z
        order_all = np.argsort(cell_keys)
        rows_all = combo_rows[order_all]
        w_all = combos_w[order_all]
        i_all = combos_i[order_all]
        z_all = combos_z[order_all]
        keys_all = cell_keys[order_all]

        uniq_all, starts_all = np.unique(keys_all, return_index=True)
        for ui, key in enumerate(uniq_all):
            s = starts_all[ui]
            e = starts_all[ui + 1] if ui + 1 < starts_all.size else keys_all.size
            seg_rows = rows_all[s:e]
            seg_w = w_all[s:e]
            if seg_rows.size == 0:
                continue
            i_idx = int(i_all[s])
            z_idx = int(z_all[s])

            x_seg = x[seg_rows]
            for c in range(len(_CLASS_NAMES)):
                model = local_models[c].get((i_idx, z_idx))
                if model is not None:
                    dens = _score_mixture_density(model, x_seg)
                    prior = float(cell_prior[c, i_idx, z_idx])
                    support = float(local_support[c, i_idx, z_idx])
                    scores[seg_rows, c] += seg_w * prior * dens
                    row_support[seg_rows, c] += seg_w * support
                    continue

                z_model = z_models[c].get(z_idx)
                if z_model is not None:
                    dens = _score_mixture_density(z_model, x_seg)
                    prior = float(z_prior[c, z_idx])
                    support = float(z_support[c, z_idx])
                    scores[seg_rows, c] += seg_w * prior * dens
                    row_support[seg_rows, c] += seg_w * support

        scores = np.where(np.isfinite(scores), scores, 0.0)
        for c in range(len(_CLASS_NAMES)):
            low_support = row_support[:, c] < _BACKOFF_SUPPORT
            if np.any(low_support):
                blend = global_density[:, c]
                blend = np.where(np.isfinite(blend), blend, 0.0)
                scores[low_support, c] = (1.0 - _BLEND_WITH_GLOBAL) * scores[low_support, c] + _BLEND_WITH_GLOBAL * blend[low_support]

        row_sum = np.sum(scores, axis=1)
        p = np.full((n, len(_CLASS_NAMES)), 1.0 / len(_CLASS_NAMES), dtype=np.float64)
        valid = (row_sum > 0) & np.isfinite(row_sum)
        if np.any(valid):
            p[valid] = scores[valid] / np.maximum(row_sum[valid, None], _EPS)
        p = np.where(np.isfinite(p), p, 1.0 / len(_CLASS_NAMES))

    p_galaxy = p[:, 0]
    p_star = p[:, 1]
    p_qso = p[:, 2]
    sorted_probs = np.sort(p, axis=1)
    top_two_gap = sorted_probs[:, 2] - sorted_probs[:, 1]
    top_prob = sorted_probs[:, 2]
    safe = np.clip(p, 1e-15, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)

    gxs = np.log(safe[:, 0]) - np.log(safe[:, 1])
    gxq = np.log(safe[:, 0]) - np.log(safe[:, 2])
    sxq = np.log(safe[:, 1]) - np.log(safe[:, 2])

    return pd.DataFrame(
        {
            "p_GALAXY": p_galaxy,
            "p_STAR": p_star,
            "p_QSO": p_qso,
            "margin_GALAXY_STAR": gxs,
            "margin_GALAXY_QSO": gxq,
            "margin_STAR_QSO": sxq,
            "top_class_probability": top_prob,
            "confidence_gap": top_two_gap,
            "posterior_entropy": entropy,
        },
        index=raw.index,
    )


FEATURE_GROUPS = [
    {
        "name": "xdqso_inspired_flux_density_scores",
        "fn": add_xdqso_inspired_flux_density_scores,
        "depends_on": [],
        "description": "Generates adaptive local generative score features from color-dynamics with class-conditional mixture likelihood posteriors and uncertainty-sensitive margins.",
    },
]