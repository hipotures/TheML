import numpy as np
import pandas as pd

_FEATURE_NAME = "xdqso_inspired_flux_density_scores"
_CLASS_NAMES = ("GALAXY", "STAR", "QSO")
_Z_BIN_EDGES = (0.0, 0.8, 2.5, 4.0, 7.0)
_Z_BIN_CENTERS = (0.4, 1.65, 3.25, 5.5)
_I_BIN_STEP = 0.2
_MIN_MODEL_ROWS = 2
_LOCAL_MIN_SUPPORT = 500
_PARENT_MIN_SUPPORT = 200
_BACKOFF_LIKELIHOOD_WEIGHT = 0.3
_PRIOR_SMOOTH = 0.3
_UNIFORM_PRIOR = (0.3333333333333333, 0.3333333333333333, 0.3333333333333333)
_MAD_FLOOR = 1e-6
_EIGENVALUE_FLOOR = 1e-12
_LOG_TWO_PI = 1.8378770664093453


def _gaussian_mixture_class():
    cached = getattr(_gaussian_mixture_class, "_cached", "UNSET")
    if cached != "UNSET":
        return cached
    try:
        from sklearn.mixture import GaussianMixture

        gm_cls = GaussianMixture
    except Exception:
        gm_cls = None
    _gaussian_mixture_class._cached = gm_cls
    return gm_cls


def _uniform_output(index):
    n = len(index)
    uniform = np.full(n, 1.0 / 3.0, dtype=float)
    return pd.DataFrame(
        {
            "p_GALAXY": uniform,
            "p_STAR": uniform,
            "p_QSO": uniform,
            "margin_GALAXY_STAR": np.zeros(n, dtype=float),
            "margin_GALAXY_QSO": np.zeros(n, dtype=float),
            "margin_STAR_QSO": np.zeros(n, dtype=float),
            "top_class_probability": uniform,
            "confidence_gap": np.zeros(n, dtype=float),
            "posterior_entropy": np.full(n, np.log(3.0), dtype=float),
        },
        index=index,
    )


def _build_i_centers(i_min, i_max):
    if not np.isfinite(i_min) or not np.isfinite(i_max):
        return tuple()
    if i_max < i_min:
        return tuple()
    span = float(i_max - i_min)
    if span < _I_BIN_STEP:
        return (float(i_min),)
    count = int(np.ceil(span / _I_BIN_STEP)) + 1
    centers = i_min + _I_BIN_STEP * np.arange(count, dtype=float)
    centers = centers[centers <= i_max + 1e-12]
    if centers.size == 0:
        centers = np.array((i_min,), dtype=float)
    return tuple(float(v) for v in centers)


def _triangular_bin_weights(values, centers):
    vals = np.asarray(values, dtype=float)
    centers_arr = np.asarray(centers, dtype=float)
    n = vals.shape[0]
    idx_low = np.zeros(n, dtype=np.int32)
    idx_high = np.zeros(n, dtype=np.int32)
    w_low = np.ones(n, dtype=float)
    w_high = np.zeros(n, dtype=float)

    if centers_arr.size == 0:
        return idx_low, idx_high, w_low, w_high

    m = centers_arr.size
    for i in range(n):
        v = vals[i]
        if not np.isfinite(v):
            idx_low[i] = 0
            idx_high[i] = 0
            w_low[i] = 1.0
            w_high[i] = 0.0
            continue

        if v <= centers_arr[0]:
            idx_low[i] = 0
            idx_high[i] = 0
            w_low[i] = 1.0
            w_high[i] = 0.0
            continue

        if v >= centers_arr[-1]:
            idx_low[i] = m - 1
            idx_high[i] = m - 1
            w_low[i] = 1.0
            w_high[i] = 0.0
            continue

        high = int(np.searchsorted(centers_arr, v, side="right"))
        low = high - 1
        low = max(0, min(low, m - 1))
        high = max(0, min(high, m - 1))
        if low == high:
            idx_low[i] = low
            idx_high[i] = high
            w_low[i] = 1.0
            w_high[i] = 0.0
            continue

        span = centers_arr[high] - centers_arr[low]
        if not np.isfinite(span) or span <= 0.0:
            idx_low[i] = low
            idx_high[i] = high
            w_low[i] = 1.0
            w_high[i] = 0.0
            continue

        w_hi = (v - centers_arr[low]) / span
        w_lo = 1.0 - w_hi
        if not np.isfinite(w_lo) or not np.isfinite(w_hi):
            w_lo = 1.0
            w_hi = 0.0
        if w_lo < 0.0:
            w_lo = 0.0
            w_hi = 1.0
        elif w_hi < 0.0:
            w_hi = 0.0
            w_lo = 1.0

        idx_low[i] = low
        idx_high[i] = high
        w_low[i] = w_lo
        w_high[i] = w_hi

    return idx_low, idx_high, w_low, w_high


def _component_count_for_support(support):
    if support >= 5000:
        return 8
    if support >= 2000:
        return 4
    if support >= 500:
        return 2
    return 1


def _assign_component_groups_to_classes(means):
    n_comp = means.shape[0]
    if n_comp <= 1:
        return (
            np.array((0,), dtype=np.int32),
            np.array((0,), dtype=np.int32),
            np.array((0,), dtype=np.int32),
        )
    if n_comp == 2:
        return (
            np.array((0,), dtype=np.int32),
            np.array((0,), dtype=np.int32),
            np.array((1,), dtype=np.int32),
        )

    order = np.argsort(means[:, 0].astype(float))
    buckets = np.array_split(order, 3)
    return (
        buckets[0].astype(np.int32),
        buckets[1].astype(np.int32),
        buckets[2].astype(np.int32),
    )


def _regularize_covariance_matrix(covariance, mad2):
    c = np.asarray(covariance, dtype=float)
    c = (c + c.T) * 0.5
    c = c + np.diag(mad2.astype(float))
    for _ in range(4):
        try:
            min_eig = np.min(np.linalg.eigvalsh(c))
            if not np.isfinite(min_eig):
                min_eig = 0.0
            if min_eig <= _EIGENVALUE_FLOOR:
                c = c + np.eye(c.shape[0], dtype=float) * (
                    (_EIGENVALUE_FLOOR - min_eig) + 1e-6
                )
            sign, logdet = np.linalg.slogdet(c)
            if np.isfinite(logdet) and sign > 0.0:
                inv = np.linalg.inv(c)
                return inv, float(logdet), c
            c = c + np.eye(c.shape[0], dtype=float) * 1e-6
        except Exception:
            c = c + np.eye(c.shape[0], dtype=float) * 1e-3

    c = np.eye(c.shape[0], dtype=float) * max(1e-3, np.max(np.abs(c)))
    inv = np.linalg.pinv(c)
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0.0 or not np.isfinite(logdet):
        logdet = float(np.log(np.finfo(float).tiny))
    return inv, float(logdet), c


def _fit_class_model(x_block, support):
    GaussianMixture = _gaussian_mixture_class()
    if GaussianMixture is None:
        return None
    x_block = np.asarray(x_block, dtype=float)
    n = x_block.shape[0]
    if n < _MIN_MODEL_ROWS:
        return None

    x_block = np.nan_to_num(x_block, nan=0.0, posinf=0.0, neginf=0.0)
    base_components = int(_component_count_for_support(support))
    attempts = []
    for c in (base_components, max(1, base_components // 2), 1):
        c = int(min(c, n))
        if c < 1:
            continue
        if c not in attempts:
            attempts.append(c)
    attempts = tuple(attempts)

    gmm = None
    for n_comp in attempts:
        if n_comp < 1:
            continue
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                random_state=0,
                max_iter=200,
                n_init=1,
                reg_covar=1e-6,
                init_params="kmeans",
            )
            gmm.fit(x_block)
            if np.isfinite(gmm.lower_bound_) and np.isfinite(gmm.means_).all():
                break
        except Exception:
            gmm = None
    if gmm is None:
        return None

    means = np.asarray(gmm.means_, dtype=float)
    weights = np.asarray(gmm.weights_, dtype=float)
    covariances = np.asarray(gmm.covariances_, dtype=float)

    if means.shape[0] == 0:
        return None

    mad = np.nanmedian(np.abs(x_block - np.nanmedian(x_block, axis=0)), axis=0)
    mad = np.where(np.isfinite(mad), mad, 0.0)
    mad2 = np.maximum(mad * mad, _MAD_FLOOR)

    inv_covs = []
    log_dets = []
    covs_reg = []
    for c in range(covariances.shape[0]):
        inv, logdet, cov_reg = _regularize_covariance_matrix(covariances[c], mad2)
        if not np.all(np.isfinite(inv)) or not np.isfinite(logdet):
            return None
        inv_covs.append(inv.astype(float))
        log_dets.append(float(logdet))
        covs_reg.append(cov_reg.astype(float))
    inv_covs = np.array(inv_covs, dtype=float)
    log_dets = np.array(log_dets, dtype=float)

    comp_groups = _assign_component_groups_to_classes(means)
    class_prior = np.zeros(3, dtype=float)
    for ci, idx in enumerate(comp_groups):
        if idx.size == 0:
            continue
        class_prior[ci] = float(np.nansum(weights[idx]))
    prior_sum = float(np.sum(class_prior))
    if prior_sum > 0.0 and np.isfinite(prior_sum):
        class_prior = class_prior / prior_sum
    else:
        class_prior = np.array(_UNIFORM_PRIOR, dtype=float)

    return {
        "means": means,
        "weights": weights,
        "inv_covs": inv_covs,
        "log_dets": log_dets,
        "covs_reg": covs_reg,
        "comp_groups": comp_groups,
        "class_prior": class_prior,
        "support": float(support),
    }


def _logsumexp_rows(log_matrix):
    maxv = np.max(log_matrix, axis=1, keepdims=True)
    shifted = log_matrix - maxv
    shifted = np.where(np.isfinite(shifted), shifted, -np.inf)
    exp_sum = np.exp(shifted)
    return maxv[:, 0] + np.log(np.sum(exp_sum, axis=1))


def _score_model(model, x_block):
    if model is None:
        return np.zeros((x_block.shape[0], 3), dtype=float)

    x_block = np.asarray(x_block, dtype=float)
    if x_block.size == 0:
        return np.zeros((0, 3), dtype=float)

    means = model["means"]
    weights = model["weights"]
    inv_covs = model["inv_covs"]
    log_dets = model["log_dets"]
    comp_groups = model["comp_groups"]

    n = x_block.shape[0]
    n_comp = means.shape[0]
    d = means.shape[1]

    log_comp = np.empty((n, n_comp), dtype=float)
    for i in range(n_comp):
        diff = x_block - means[i]
        quad = np.einsum("ij,ij->i", np.dot(diff, inv_covs[i]), diff)
        log_comp[:, i] = np.log(np.maximum(weights[i], 1e-300)) - 0.5 * (
            float(d) * _LOG_TWO_PI + log_dets[i] + quad
        )

    out = np.zeros((n, 3), dtype=float)
    for ci in range(3):
        idx = comp_groups[ci]
        if idx.size == 0:
            continue
        comp_log = log_comp[:, idx]
        out[:, ci] = np.exp(_logsumexp_rows(comp_log))

    return np.where(np.isfinite(out), out, 0.0)


def _accumulate_cell_contributions(
    cell_ids,
    cell_weights,
    row_index,
    x_arr,
    local_models,
    parent_models,
    global_model,
    n_i,
    accum_like,
    accum_prior,
    accum_support,
):
    mask = cell_weights > 0.0
    if not np.any(mask):
        return

    rows = row_index[mask]
    cids = np.asarray(cell_ids, dtype=np.int64)[mask]
    w = cell_weights[mask]

    order = np.argsort(cids, kind="mergesort")
    rows = rows[order]
    cids = cids[order]
    w = w[order]

    boundaries = np.nonzero(np.diff(cids) != 0)[0]
    starts = np.concatenate(
        (np.array((0,), dtype=np.int64), boundaries + 1, np.array((len(cids),), dtype=np.int64))
    )

    for s, e in zip(starts[:-1], starts[1:]):
        cid = int(cids[s])
        idx_rows = rows[s:e]
        ws = w[s:e]
        if idx_rows.size == 0:
            continue

        model = local_models.get(cid)
        if model is None:
            z_idx = int(cid // n_i)
            parent = parent_models[z_idx]
            if parent is not None and parent["support"] >= _PARENT_MIN_SUPPORT:
                model = parent
            else:
                model = global_model
        if model is None:
            continue

        block_scores = _score_model(model, x_arr[idx_rows])
        accum_like[idx_rows] += ws[:, None] * block_scores
        accum_prior[idx_rows] += ws[:, None] * model["class_prior"]
        accum_support[idx_rows] += ws * model["support"]


def add_xdqso_inspired_flux_density_scores(raw, deps, aux):
    index = raw.index
    required = ("u", "g", "r", "i", "z", "redshift")
    if raw is None or len(raw) == 0:
        return _uniform_output(index)

    for col in required:
        if col not in raw.columns:
            return _uniform_output(index)

    u = np.asarray(raw["u"].to_numpy(dtype=float), dtype=float)
    g = np.asarray(raw["g"].to_numpy(dtype=float), dtype=float)
    r = np.asarray(raw["r"].to_numpy(dtype=float), dtype=float)
    i_mag = np.asarray(raw["i"].to_numpy(dtype=float), dtype=float)
    z_mag = np.asarray(raw["z"].to_numpy(dtype=float), dtype=float)
    redshift = np.asarray(raw["redshift"].to_numpy(dtype=float), dtype=float)

    x = np.column_stack(
        (
            -0.4 * (u - i_mag),
            -0.4 * (g - i_mag),
            -0.4 * (r - i_mag),
            -0.4 * (z_mag - i_mag),
        )
    )
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    for j in range(x.shape[1]):
        q0, q1 = np.nanquantile(x[:, j], [0.005, 0.995])
        if np.isfinite(q0) and np.isfinite(q1) and q1 > q0:
            x[:, j] = np.clip(x[:, j], q0, q1)

    n = x.shape[0]
    i_min = float(np.nanmin(i_mag))
    i_max = float(np.nanmax(i_mag))
    i_centers = _build_i_centers(i_min, i_max)
    if len(i_centers) == 0:
        return _uniform_output(index)

    i_centers_arr = np.asarray(i_centers, dtype=float)
    i_hard = np.rint((i_mag - float(i_centers_arr[0])) / _I_BIN_STEP).astype(np.int32)
    i_hard = np.clip(i_hard, 0, len(i_centers_arr) - 1)

    z_edges = np.asarray(_Z_BIN_EDGES, dtype=float)
    z_hard = np.searchsorted(z_edges[1:-1], redshift, side="right").astype(np.int32)
    z_hard = np.clip(z_hard, 0, 3)

    n_i = len(i_centers_arr)
    hard_cell = z_hard * n_i + i_hard
    total_cells = n_i * 4

    cell_counts = np.bincount(hard_cell, minlength=total_cells)
    local_models = {}
    for cell_id in range(total_cells):
        n_cell = int(cell_counts[cell_id])
        if n_cell < _LOCAL_MIN_SUPPORT:
            continue
        rows_cell = np.nonzero(hard_cell == cell_id)[0]
        if rows_cell.size < _MIN_MODEL_ROWS:
            continue
        model = _fit_class_model(x[rows_cell], n_cell)
        if model is not None:
            local_models[cell_id] = model

    parent_models = [None, None, None, None]
    for z_idx in range(4):
        rows_z = np.nonzero(z_hard == z_idx)[0]
        if rows_z.size < _MIN_MODEL_ROWS:
            continue
        model = _fit_class_model(x[rows_z], rows_z.size)
        if model is not None:
            parent_models[z_idx] = model

    global_model = _fit_class_model(x, n)

    i0, i1, wi0, wi1 = _triangular_bin_weights(i_mag, i_centers)
    z0, z1, wz0, wz1 = _triangular_bin_weights(redshift, _Z_BIN_CENTERS)

    c00 = z0 * n_i + i0
    w00 = wi0 * wz0
    c01 = z0 * n_i + i1
    w01 = wi1 * wz0
    c10 = z1 * n_i + i0
    w10 = wi0 * wz1
    c11 = z1 * n_i + i1
    w11 = wi1 * wz1

    accum_like = np.zeros((n, 3), dtype=float)
    accum_prior = np.zeros((n, 3), dtype=float)
    accum_support = np.zeros(n, dtype=float)
    rows = np.arange(n, dtype=np.int64)

    _accumulate_cell_contributions(
        c00,
        w00,
        rows,
        x,
        local_models,
        parent_models,
        global_model,
        n_i,
        accum_like,
        accum_prior,
        accum_support,
    )
    _accumulate_cell_contributions(
        c01,
        w01,
        rows,
        x,
        local_models,
        parent_models,
        global_model,
        n_i,
        accum_like,
        accum_prior,
        accum_support,
    )
    _accumulate_cell_contributions(
        c10,
        w10,
        rows,
        x,
        local_models,
        parent_models,
        global_model,
        n_i,
        accum_like,
        accum_prior,
        accum_support,
    )
    _accumulate_cell_contributions(
        c11,
        w11,
        rows,
        x,
        local_models,
        parent_models,
        global_model,
        n_i,
        accum_like,
        accum_prior,
        accum_support,
    )

    uniform_prior = np.array(_UNIFORM_PRIOR, dtype=float)
    prior_row = np.array(accum_prior, copy=True)
    prior_sum = np.sum(prior_row, axis=1)
    valid_prior = prior_sum > 0.0
    prior_row[valid_prior] = prior_row[valid_prior] / prior_sum[valid_prior, None]
    prior_row[~valid_prior] = uniform_prior
    prior_row = np.nan_to_num(
        prior_row, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0
    )
    prior_row = prior_row / np.sum(prior_row, axis=1, keepdims=True)

    pi_local = (1.0 - _PRIOR_SMOOTH) * prior_row + _PRIOR_SMOOTH * uniform_prior

    if global_model is not None:
        global_like = _score_model(global_model, x)
    else:
        global_like = np.tile(uniform_prior, (n, 1))

    low_support = accum_support < _PARENT_MIN_SUPPORT
    final_like = np.array(accum_like, copy=True)
    if np.any(low_support):
        final_like[low_support] = (
            1.0 - _BACKOFF_LIKELIHOOD_WEIGHT
        ) * accum_like[low_support] + _BACKOFF_LIKELIHOOD_WEIGHT * global_like[low_support]

    ptilde = final_like * pi_local
    ptilde = np.nan_to_num(ptilde, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sum(ptilde, axis=1)

    probs = np.tile(uniform_prior, (n, 1))
    valid = np.isfinite(denom) & (denom > 0.0)
    if np.any(valid):
        probs[valid] = ptilde[valid] / denom[valid, None]

    probs = np.nan_to_num(
        probs, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0
    )
    row_sum = np.sum(probs, axis=1)
    valid_row = row_sum > 0.0
    probs[valid_row] = probs[valid_row] / row_sum[valid_row, None]
    probs[~valid_row] = uniform_prior

    eps = 1e-15
    p_gal = probs[:, 0]
    p_star = probs[:, 1]
    p_qso = probs[:, 2]

    margin_gs = np.log((p_gal + eps) / (p_star + eps))
    margin_gq = np.log((p_gal + eps) / (p_qso + eps))
    margin_sq = np.log((p_star + eps) / (p_qso + eps))

    sorted_p = np.sort(probs, axis=1)
    top_prob = sorted_p[:, 2]
    gap = sorted_p[:, 2] - sorted_p[:, 1]
    entropy = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)

    return pd.DataFrame(
        {
            "p_GALAXY": p_gal,
            "p_STAR": p_star,
            "p_QSO": p_qso,
            "margin_GALAXY_STAR": margin_gs,
            "margin_GALAXY_QSO": margin_gq,
            "margin_STAR_QSO": margin_sq,
            "top_class_probability": top_prob,
            "confidence_gap": gap,
            "posterior_entropy": entropy,
        },
        index=index,
    )


FEATURE_GROUPS = [
    {
        "name": _FEATURE_NAME,
        "fn": add_xdqso_inspired_flux_density_scores,
        "depends_on": [],
        "description": "Adaptive local flux-density score features with overlap-aware GMM likelihood blending into class-like probabilistic margins.",
    }
]