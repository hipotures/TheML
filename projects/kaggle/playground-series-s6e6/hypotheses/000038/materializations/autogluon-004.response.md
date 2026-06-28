import numpy as np
import pandas as pd

N_KNOTS = 180
TRAIN_ID_MAX = 577346
INITIAL_BANDWIDTH = 0.06
BANDWIDTH_GROWTH = 1.5
MAX_BANDWIDTH = 0.8
MIN_EFFECTIVE_N = 1200.0
COV_SHRINK = 0.10
EIGEN_FLOOR = 1e-5
LOW_Z_TAPER = 0.01
TAIL_DAMP_SCALE = 0.05
EPS = 1e-8
COLOR_COLUMNS = ("u", "g", "r", "i", "z")


def _weighted_quantile(values, quantiles):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if values.size == 0:
        return np.zeros_like(quantiles)
    return np.quantile(values, quantiles)


def _huber_center(x, weights, delta=1.5, n_iter=5):
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    sw = np.sum(w) + EPS
    center = np.sum(x * w[:, None], axis=0) / sw
    for _ in range(n_iter):
        residual = x - center
        scale = 1.4826 * np.sum(w[:, None] * np.abs(residual), axis=0) / sw + EPS
        adj = np.minimum(1.0, delta * scale / (np.abs(residual) + EPS))
        rw = w[:, None] * adj
        center = np.sum(x * rw, axis=0) / (np.sum(rw, axis=0) + EPS)
    return center


def _regularize_cov(cov):
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    if not np.isfinite(trace) or trace <= 0.0:
        cov = np.eye(4, dtype=np.float64) * 0.01
    else:
        cov = (1.0 - COV_SHRINK) * cov + COV_SHRINK * np.eye(4, dtype=np.float64) * trace / 4.0
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    vals = np.maximum(vals, EIGEN_FLOOR)
    return (vecs * vals) @ vecs.T


def _weighted_cov_huber(x, weights, center):
    w = np.asarray(weights, dtype=np.float64)
    residual = np.asarray(x, dtype=np.float64) - center
    sw = np.sum(w) + EPS
    scale = 1.4826 * np.sum(w[:, None] * np.abs(residual), axis=0) / sw + EPS
    clipped = np.clip(residual, -3.0 * scale, 3.0 * scale)
    cov = (clipped * w[:, None]).T @ clipped / sw
    return _regularize_cov(cov)


def _smooth_series(y, weights):
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = y.shape[0]
    radius = 5
    out = np.empty_like(y)
    for j in range(y.shape[1]):
        col = y[:, j]
        smooth = np.empty(n, dtype=np.float64)
        for k in range(n):
            lo = max(0, k - radius)
            hi = min(n, k + radius + 1)
            idx = np.arange(lo, hi, dtype=np.float64) - float(k)
            kernel = np.exp(-0.5 * (idx / 2.5) ** 2)
            ww = kernel * np.maximum(weights[lo:hi], 1.0)
            smooth[k] = np.sum(col[lo:hi] * ww) / (np.sum(ww) + EPS)
        out[:, j] = smooth
    return out


def _interp_matrix_array(x, knots, matrices):
    x = np.asarray(x, dtype=np.float64)
    pos = np.searchsorted(knots, x, side="right") - 1
    pos = np.clip(pos, 0, len(knots) - 2)
    denom = knots[pos + 1] - knots[pos]
    frac = np.where(np.abs(denom) > EPS, (x - knots[pos]) / denom, 0.0)
    return (1.0 - frac)[:, None, None] * matrices[pos] + frac[:, None, None] * matrices[pos + 1]


def _interp_array(x, knots, values):
    x = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(x, knots, values)
    cols = [np.interp(x, knots, values[:, j]) for j in range(values.shape[1])]
    return np.vstack(cols).T


def _mad_scale(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 1.0
    if weights is None:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
    else:
        w = np.asarray(weights, dtype=np.float64)
        order = np.argsort(values)
        sv = values[order]
        sw = w[order]
        cw = np.cumsum(sw)
        half = 0.5 * (cw[-1] + EPS)
        med = sv[min(np.searchsorted(cw, half), len(sv) - 1)]
        dev = np.abs(values - med)
        order = np.argsort(dev)
        sd = dev[order]
        sw = w[order]
        cw = np.cumsum(sw)
        mad = sd[min(np.searchsorted(cw, half), len(sd) - 1)]
    return max(1.4826 * float(mad), 1e-4)


def _make_colors(frame):
    vals = frame.loc[:, COLOR_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    return np.column_stack(
        (
            vals[:, 0] - vals[:, 1],
            vals[:, 1] - vals[:, 2],
            vals[:, 2] - vals[:, 3],
            vals[:, 3] - vals[:, 4],
        )
    )


def add_photoz_trajectory_geometry(raw, deps, aux):
    index = raw.index
    z_all = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    colors_all = _make_colors(raw)

    if "id" in raw.columns:
        train_mask = raw["id"].to_numpy(copy=False) <= TRAIN_ID_MAX
        if int(np.sum(train_mask)) < max(5000, N_KNOTS * 20):
            train_mask = np.ones(len(raw), dtype=bool)
    else:
        train_mask = np.ones(len(raw), dtype=bool)

    z_train = z_all[train_mask]
    colors_train = colors_all[train_mask]

    q_lo, q_hi = np.quantile(z_train, (0.005, 0.995))
    quantiles = np.linspace(0.005, 0.995, N_KNOTS)
    knots = _weighted_quantile(z_train, quantiles)
    knots = np.maximum.accumulate(knots)
    for k in range(1, len(knots)):
        if knots[k] <= knots[k - 1]:
            knots[k] = knots[k - 1] + 1e-6
    knots[0] = min(knots[0], q_lo)
    knots[-1] = max(knots[-1], q_hi)

    means = np.zeros((N_KNOTS, 4), dtype=np.float64)
    covs = np.zeros((N_KNOTS, 4, 4), dtype=np.float64)
    eff_n = np.zeros(N_KNOTS, dtype=np.float64)

    global_center = np.median(colors_train, axis=0)
    global_cov = _regularize_cov(np.cov((colors_train - global_center).T))

    for k, knot in enumerate(knots):
        h = INITIAL_BANDWIDTH
        dist = z_train - knot
        weights = np.exp(-0.5 * (dist / h) ** 2)
        effective = (np.sum(weights) ** 2) / (np.sum(weights * weights) + EPS)
        while effective < MIN_EFFECTIVE_N and h < MAX_BANDWIDTH:
            h = min(MAX_BANDWIDTH, h * BANDWIDTH_GROWTH)
            weights = np.exp(-0.5 * (dist / h) ** 2)
            effective = (np.sum(weights) ** 2) / (np.sum(weights * weights) + EPS)

        keep = weights > 1e-6
        if int(np.sum(keep)) < 20:
            means[k] = global_center
            covs[k] = global_cov
            eff_n[k] = max(effective, 1.0)
            continue

        local_x = colors_train[keep]
        local_w = weights[keep]
        center = _huber_center(local_x, local_w)
        means[k] = center
        covs[k] = _weighted_cov_huber(local_x, local_w, center)
        eff_n[k] = max(effective, 1.0)

    smooth_means = _smooth_series(means, eff_n)
    velocity = np.gradient(smooth_means, knots, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, knots, axis=0, edge_order=2)

    z_clip_all = np.clip(z_all, knots[0], knots[-1])
    mu_all = _interp_array(z_clip_all, knots, smooth_means)
    v_all = _interp_array(z_clip_all, knots, velocity)
    a_all = _interp_array(z_clip_all, knots, acceleration)
    cov_all = _interp_matrix_array(z_clip_all, knots, covs)

    residual = colors_all - mu_all
    y = np.zeros_like(residual)
    t = np.zeros_like(residual)
    aw = np.zeros_like(residual)

    for n in range(len(raw)):
        cov = _regularize_cov(cov_all[n])
        chol = np.linalg.cholesky(cov)
        y[n] = np.linalg.solve(chol, residual[n])
        t[n] = np.linalg.solve(chol, v_all[n])
        aw[n] = np.linalg.solve(chol, a_all[n])

    t_norm = np.linalg.norm(t, axis=1) + EPS
    tangent_unit = t / t_norm[:, None]
    tangent = np.sum(y * tangent_unit, axis=1)
    distance = np.sum(y * y, axis=1)
    orthogonal = np.maximum(0.0, distance - tangent * tangent)

    aw_norm = np.linalg.norm(aw, axis=1) + EPS
    curvature = np.sum(y * aw, axis=1) / aw_norm

    outside = np.maximum(0.0, knots[0] - z_all) + np.maximum(0.0, z_all - knots[-1])
    low_z_factor = np.where(z_clip_all < LOW_Z_TAPER, z_clip_all / LOW_Z_TAPER, 1.0)
    tail_factor = np.exp(-outside / TAIL_DAMP_SCALE)
    signed_factor = np.clip(low_z_factor * tail_factor, 0.0, 1.0)
    tangent = tangent * signed_factor
    curvature = curvature * signed_factor

    train_positions = np.flatnonzero(train_mask)
    y_train = y[train_mask]
    t_train = t[train_mask]
    aw_train = aw[train_mask]
    t_train_unit = t_train / (np.linalg.norm(t_train, axis=1)[:, None] + EPS)
    tangent_train = np.sum(y_train * t_train_unit, axis=1)
    distance_train = np.sum(y_train * y_train, axis=1)
    orthogonal_train = np.maximum(0.0, distance_train - tangent_train * tangent_train)
    curvature_train = np.sum(y_train * aw_train, axis=1) / (np.linalg.norm(aw_train, axis=1) + EPS)

    sigma_t = np.zeros(N_KNOTS, dtype=np.float64)
    sigma_o = np.zeros(N_KNOTS, dtype=np.float64)
    sigma_c = np.zeros(N_KNOTS, dtype=np.float64)

    for k, knot in enumerate(knots):
        h = INITIAL_BANDWIDTH
        dist = z_train - knot
        weights = np.exp(-0.5 * (dist / h) ** 2)
        effective = (np.sum(weights) ** 2) / (np.sum(weights * weights) + EPS)
        while effective < MIN_EFFECTIVE_N and h < MAX_BANDWIDTH:
            h = min(MAX_BANDWIDTH, h * BANDWIDTH_GROWTH)
            weights = np.exp(-0.5 * (dist / h) ** 2)
            effective = (np.sum(weights) ** 2) / (np.sum(weights * weights) + EPS)
        keep = weights > 1e-6
        if int(np.sum(keep)) < 20:
            sigma_t[k] = _mad_scale(tangent_train)
            sigma_o[k] = _mad_scale(orthogonal_train)
            sigma_c[k] = _mad_scale(curvature_train)
        else:
            local_w = weights[keep]
            sigma_t[k] = _mad_scale(tangent_train[keep], local_w)
            sigma_o[k] = _mad_scale(orthogonal_train[keep], local_w)
            sigma_c[k] = _mad_scale(curvature_train[keep], local_w)

    sigma_t = _smooth_series(sigma_t[:, None], eff_n)[:, 0]
    sigma_o = _smooth_series(sigma_o[:, None], eff_n)[:, 0]
    sigma_c = _smooth_series(sigma_c[:, None], eff_n)[:, 0]

    st_all = np.maximum(_interp_array(z_clip_all, knots, sigma_t), 1e-4)
    so_all = np.maximum(_interp_array(z_clip_all, knots, sigma_o), 1e-4)
    sc_all = np.maximum(_interp_array(z_clip_all, knots, sigma_c), 1e-4)

    features = pd.DataFrame(
        {
            "tangent_residual": tangent,
            "orthogonal_sq_distance": orthogonal,
            "whitened_sq_distance": distance,
            "curvature_residual": curvature,
            "tangent_residual_norm": tangent / st_all,
            "orthogonal_sq_distance_norm": orthogonal / so_all,
            "curvature_residual_norm": curvature / sc_all,
        },
        index=index,
    )

    for col in features.columns:
        lo, hi = np.quantile(features.iloc[train_positions][col].to_numpy(dtype=np.float64), (0.001, 0.999))
        features[col] = np.clip(features[col].to_numpy(dtype=np.float64), lo, hi)

    return features


FEATURE_GROUPS = [
    {
        "name": "photoz_trajectory_geometry",
        "fn": add_photoz_trajectory_geometry,
        "depends_on": [],
        "description": "Regularized color-redshift trajectory residual geometry features from the empirical photometric manifold.",
    }
]