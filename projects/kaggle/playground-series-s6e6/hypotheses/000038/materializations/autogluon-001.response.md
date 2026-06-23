import numpy as np
import pandas as pd

_COLOR_BANDS = ("u", "g", "r", "i", "z")
_REQUIRED_COLUMNS = ("redshift", "u", "g", "r", "i", "z")

_KNOT_COUNT = 160
_NODE_LOW_QUANTILE = 0.005
_NODE_HIGH_QUANTILE = 0.995
_FIT_LOW_CLIP = 0.001
_FIT_HIGH_CLIP = 0.999

_MIN_LOCAL_POINTS = 180
_INITIAL_HALF_WIDTH = 24
_COV_REG = 1e-6
_EPS = 1e-12
_TAIL_LOW_Z = 0.01
_TAIL_DERIV_SCALE = 0.25
_GROUP_NAME = "photoz_trajectory_geometry"


def _coerce_numeric(frame, columns):
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
    return out


def _build_fit_block(raw, aux):
    raw_block = _coerce_numeric(raw, _REQUIRED_COLUMNS)
    if raw_block.empty or raw_block.isna().all().all():
        return pd.DataFrame(columns=list(_REQUIRED_COLUMNS)), {}

    frames = [raw_block]
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_cols = [c for c in _REQUIRED_COLUMNS if c in aux.columns]
        if len(aux_cols) == len(_REQUIRED_COLUMNS):
            aux_block = _coerce_numeric(aux, _REQUIRED_COLUMNS)
            if not aux_block.empty:
                frames.append(aux_block)

    fit = pd.concat(frames, axis=0, ignore_index=True)
    fit = fit.replace([np.inf, -np.inf], np.nan)
    fit = fit.loc[:, [c for c in _REQUIRED_COLUMNS if c in fit.columns]]
    fit = fit.dropna(subset=_REQUIRED_COLUMNS)

    if fit.empty:
        return pd.DataFrame(columns=list(_REQUIRED_COLUMNS)), {}

    fit = fit.copy()
    fit["redshift"] = fit["redshift"].clip(lower=0.0)

    bounds = {}
    for col in _COLOR_BANDS:
        lo = fit[col].quantile(_FIT_LOW_CLIP)
        hi = fit[col].quantile(_FIT_HIGH_CLIP)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            fit[col] = fit[col].clip(lower=lo, upper=hi)
            bounds[col] = (float(lo), float(hi))

    return fit, bounds


def _apply_bounds(frame, bounds):
    clipped = frame.copy()
    for col, (lo, hi) in bounds.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].clip(lower=lo, upper=hi)
    return clipped


def _color_matrix(df):
    u = df["u"].to_numpy(dtype=float)
    g = df["g"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)
    i = df["i"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)
    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z
    return np.column_stack((c1, c2, c3, c4))


def _regularize_covariance(cov):
    cov = 0.5 * (cov + cov.T)
    if not np.all(np.isfinite(cov)):
        return np.eye(4) * _COV_REG
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except Exception:
        return np.eye(4) * _COV_REG
    min_eig = np.min(eigvals) if eigvals.size else 0.0
    if not np.isfinite(min_eig) or min_eig <= 0.0:
        cov = cov + (-min_eig + _COV_REG) * np.eye(4)
    return cov


def _weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(weights)
    values = values[valid]
    weights = weights[valid]

    if values.shape[0] < 2:
        return np.full(4, np.nan, dtype=float), np.full((4, 4), np.nan, dtype=float)

    weights = np.maximum(weights, 0.0)
    sw = float(weights.sum())
    if sw <= 0.0:
        weights = np.ones(values.shape[0], dtype=float)
        sw = float(values.shape[0])

    weights = weights / sw
    mu = np.dot(weights, values)
    centered = values - mu

    mad = 1.4826 * np.nanmedian(np.abs(centered), axis=0)
    mad = np.where(mad < 1.0, 1.0, mad)
    centered = np.clip(centered, -5.0 * mad, 5.0 * mad)

    cov = np.dot(centered.T, centered * weights[:, None])
    cov = _regularize_covariance(cov)
    return mu, cov


def _inverse_sqrt_covariance(cov):
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, _COV_REG)
    inv_sqrt = np.diag(eigvals ** -0.5)
    return eigvecs @ inv_sqrt @ eigvecs.T


def _build_profile(redshift, colors):
    z = np.asarray(redshift, dtype=float)
    c = np.asarray(colors, dtype=float)

    good = np.isfinite(z) & np.isfinite(c).all(axis=1)
    z = z[good]
    c = c[good]

    n = z.shape[0]
    if n < 12:
        return None

    order = np.argsort(z)
    z = z[order]
    c = c[order]

    lo = float(np.min(z))
    hi = float(np.max(z))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None

    if np.isclose(hi, lo):
        nodes = np.array([lo, lo + 1.0], dtype=float)
    else:
        q = np.linspace(_NODE_LOW_QUANTILE, _NODE_HIGH_QUANTILE, _KNOT_COUNT)
        nodes = np.quantile(z, q)
        nodes = np.asarray(np.unique(nodes), dtype=float)
        nodes = nodes[np.isfinite(nodes)]
        if nodes.size < 2:
            nodes = np.array([lo, hi], dtype=float)
        if nodes.size == 2 and np.isclose(nodes[0], nodes[1]):
            nodes[1] = nodes[0] + 1.0

    global_mu = np.nanmedian(c, axis=0)
    global_cov = _regularize_covariance(np.cov(c, rowvar=False))
    if not np.isfinite(global_cov).all():
        _, global_cov = _weighted_stats(c, np.ones(n))

    nodes_count = nodes.shape[0]
    mu_nodes = np.empty((nodes_count, 4), dtype=float)
    cov_nodes = np.empty((nodes_count, 4, 4), dtype=float)

    for idx, knot in enumerate(nodes):
        half = _INITIAL_HALF_WIDTH
        pos = int(np.searchsorted(z, knot, side="left"))
        left = max(0, pos - half)
        right = min(n, pos + half)

        while (right - left) < _MIN_LOCAL_POINTS and (left > 0 or right < n):
            half = int(np.ceil(half * 1.5))
            left = max(0, pos - half)
            right = min(n, pos + half)
            if half >= n:
                break

        if right <= left or (right - left) < 2:
            mu_nodes[idx] = global_mu
            cov_nodes[idx] = global_cov
            continue

        idx_slice = np.arange(left, right)
        z_local = z[idx_slice]
        c_local = c[idx_slice]

        dist = np.abs(z_local - knot)
        dmax = float(dist.max())
        if dmax > _EPS:
            w = np.power(np.clip(1.0 - dist / dmax, 0.0, 1.0), 1.5)
        else:
            w = np.ones_like(dist, dtype=float)

        mu_k, cov_k = _weighted_stats(c_local, w)
        if not (np.isfinite(mu_k).all() and np.isfinite(cov_k).all()):
            mu_k, cov_k = global_mu, global_cov

        mu_nodes[idx] = mu_k
        cov_nodes[idx] = _regularize_covariance(cov_k)

    if nodes_count >= 3:
        d1 = np.gradient(mu_nodes, nodes, axis=0, edge_order=2)
        d2 = np.gradient(d1, nodes, axis=0, edge_order=2)
    else:
        d1 = np.zeros_like(mu_nodes)
        d2 = np.zeros_like(mu_nodes)

    return {
        "nodes": nodes,
        "mu": mu_nodes,
        "mu_deriv": d1,
        "mu_curv": d2,
        "cov": cov_nodes,
    }


def add_photoz_trajectory_geometry(raw, deps, aux):
    idx = raw.index
    feature_names = (
        "photoz_trajectory_resid_c1",
        "photoz_trajectory_resid_c2",
        "photoz_trajectory_resid_c3",
        "photoz_trajectory_resid_c4",
        "photoz_trajectory_d_tan",
        "photoz_trajectory_d_orth",
        "photoz_trajectory_d_tan_norm",
        "photoz_trajectory_d_orth_norm",
        "photoz_trajectory_m_curv",
        "photoz_trajectory_local_speed",
        "photoz_trajectory_local_curv",
    )
    n_rows = len(idx)
    out = {name: pd.Series(np.nan, index=idx, dtype=float) for name in feature_names}

    raw_block, bounds = _build_fit_block(raw, aux)
    if raw_block.empty:
        return pd.DataFrame(out)

    profile = _build_profile(raw_block["redshift"].to_numpy(dtype=float), _color_matrix(raw_block))
    if profile is None:
        return pd.DataFrame(out)

    nodes = profile["nodes"]
    mu_nodes = profile["mu"]
    deriv_nodes = profile["mu_deriv"]
    curv_nodes = profile["mu_curv"]
    cov_nodes = profile["cov"]

    base = _coerce_numeric(raw, _REQUIRED_COLUMNS)
    if len(base.columns) != len(_REQUIRED_COLUMNS):
        return pd.DataFrame(out)

    base = _apply_bounds(base, bounds)
    base["redshift"] = base["redshift"].clip(lower=0.0)

    z_raw = base["redshift"].to_numpy(dtype=float)
    colors = _color_matrix(base)

    valid = np.isfinite(z_raw) & np.isfinite(colors).all(axis=1)
    if not np.any(valid):
        return pd.DataFrame(out)

    z_clip = np.where(np.isfinite(z_raw), np.maximum(z_raw, 0.0), nodes[0])
    z_use = np.clip(z_clip, nodes[0], nodes[-1])

    loc = np.searchsorted(nodes, z_use, side="right") - 1
    loc = np.clip(loc, 0, len(nodes) - 2)

    zl = nodes[loc]
    zr = nodes[loc + 1]
    span = zr - zl
    span[span <= _EPS] = 1.0
    frac = (z_use - zl) / span

    tail_scale = np.ones(n_rows, dtype=float)
    tail_hi = float(np.quantile(nodes, 0.985))
    tail_scale[z_raw < _TAIL_LOW_Z] = _TAIL_DERIV_SCALE
    tail_scale[z_raw > tail_hi] = _TAIL_DERIV_SCALE

    resid = np.empty((n_rows, 4), dtype=float)
    d_tan = np.full(n_rows, np.nan, dtype=float)
    d_orth = np.full(n_rows, np.nan, dtype=float)
    m_curv = np.full(n_rows, np.nan, dtype=float)
    local_speed = np.full(n_rows, np.nan, dtype=float)
    local_curv = np.full(n_rows, np.nan, dtype=float)
    d_tan_norm = np.full(n_rows, np.nan, dtype=float)
    d_orth_norm = np.full(n_rows, np.nan, dtype=float)
    resid[:] = np.nan

    for i in np.where(valid)[0]:
        k = int(loc[i])
        a = float(frac[i])

        one_minus = 1.0 - a
        mu_i = one_minus * mu_nodes[k] + a * mu_nodes[k + 1]
        v_i = one_minus * deriv_nodes[k] + a * deriv_nodes[k + 1]
        c_i = one_minus * curv_nodes[k] + a * curv_nodes[k + 1]
        cov_i = one_minus * cov_nodes[k] + a * cov_nodes[k + 1]
        r = colors[i] - mu_i

        v_i = v_i * tail_scale[i]
        c_i = c_i * tail_scale[i]

        cov_i = _regularize_covariance(cov_i)
        if not np.isfinite(cov_i).all() or not np.isfinite(v_i).all() or not np.isfinite(c_i).all():
            continue

        local_speed[i] = float(np.linalg.norm(v_i))
        local_curv[i] = float(np.linalg.norm(c_i))

        try:
            L = np.linalg.cholesky(cov_i)
            y = np.linalg.solve(L, r)
            tv = np.linalg.solve(L, v_i)
        except np.linalg.LinAlgError:
            inv_sqrt = _inverse_sqrt_covariance(cov_i)
            y = inv_sqrt.dot(r)
            tv = inv_sqrt.dot(v_i)
        except Exception:
            continue

        y_norm2 = float(np.dot(y, y))
        tv_norm = float(np.linalg.norm(tv))

        if tv_norm > _EPS:
            u_hat = tv / tv_norm
            dt = float(np.dot(y, u_hat))
            orth = float(max(y_norm2 - (dt * dt), 0.0))
        else:
            dt = 0.0
            orth = y_norm2

        cv = float(np.dot(r, c_i))
        cnorm = float(np.linalg.norm(c_i))
        mc = cv / (cnorm + _EPS) if cnorm > _EPS else 0.0

        residual = r
        resid[i] = residual
        d_tan[i] = dt
        d_orth[i] = orth
        m_curv[i] = mc
        d_tan_norm[i] = dt / 1.0
        d_orth_norm[i] = orth / (np.sqrt(3.0) + _EPS)

    out["photoz_trajectory_resid_c1"] = resid[:, 0]
    out["photoz_trajectory_resid_c2"] = resid[:, 1]
    out["photoz_trajectory_resid_c3"] = resid[:, 2]
    out["photoz_trajectory_resid_c4"] = resid[:, 3]
    out["photoz_trajectory_d_tan"] = pd.Series(d_tan, index=idx)
    out["photoz_trajectory_d_orth"] = pd.Series(d_orth, index=idx)
    out["photoz_trajectory_d_tan_norm"] = pd.Series(d_tan_norm, index=idx)
    out["photoz_trajectory_d_orth_norm"] = pd.Series(d_orth_norm, index=idx)
    out["photoz_trajectory_m_curv"] = pd.Series(m_curv, index=idx)
    out["photoz_trajectory_local_speed"] = pd.Series(local_speed, index=idx)
    out["photoz_trajectory_local_curv"] = pd.Series(local_curv, index=idx)

    return pd.DataFrame(out, index=idx)


FEATURE_GROUPS = [
    {
        "name": "photoz_trajectory_geometry",
        "fn": add_photoz_trajectory_geometry,
        "depends_on": [],
        "description": "Builds local color-redshift trajectory geometry features from smooth redshift-conditioned color means, covariances, and derivative-based tangential/orthogonal residual structure.",
    }
]