import numpy as np
import pandas as pd

EPS = 1e-8
WINDOW_WIDTH = 0.8
WINDOW_STEP = 0.4
QSO_LOW_Z = 2.2
QSO_HIGH_Z = 3.5
MIN_GMM_SAMPLES = 32
MAX_GMM_COMPONENTS = 12
COMPONENT_SCALE = 160
NOISE_FLOOR = 1e-6
PRIOR_ALPHA = 1.0
LOG_FEATURE_CLIP = 12.0
CLASS_KEYS = ("STAR", "GALAXY", "QSO_LOW", "QSO_MID", "QSO_HIGH")


def _to_float(series):
    if series is None:
        return np.array([], dtype=np.float64)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _find_column(frame, candidates):
    lower_map = {str(col).lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        key = str(candidate).lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _normalize_class_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip().lower()
        if "star" in text or text in {"s", "stellar"}:
            return "STAR"
        if "gal" in text:
            return "GALAXY"
        if "qso" in text or "quasar" in text:
            return "QSO"
        return None
    try:
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        if float(numeric).is_integer():
            integer = int(round(numeric))
            if integer == 0:
                return "STAR"
            if integer == 1:
                return "GALAXY"
            if integer == 2:
                return "QSO"
    except Exception:
        return None
    return None


def _qso_branch_from_redshift(redshift):
    z = np.asarray(redshift, dtype=np.float64)
    branch = np.full(z.shape[0], 1, dtype=np.int8)
    finite = np.isfinite(z)
    branch[:] = 1
    branch[(finite) & (z < QSO_LOW_Z)] = 0
    branch[(finite) & (z > QSO_HIGH_Z)] = 2
    return branch


def _build_windows(values):
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.array([0.0], dtype=np.float64)
    vmin = float(np.nanmin(values[finite]))
    vmax = float(np.nanmax(values[finite]))
    half = WINDOW_WIDTH / 2.0
    first = np.floor((vmin - half) / WINDOW_STEP) * WINDOW_STEP + half
    last = vmax + half
    return np.arange(first, last + WINDOW_STEP + 1e-9, WINDOW_STEP, dtype=np.float64)


def _assign_windows(values, centers):
    if centers.size == 0:
        return np.zeros(values.shape[0], dtype=np.int64)
    idx = np.floor((values - centers[0]) / WINDOW_STEP + 0.5).astype(np.int64)
    idx = np.where(np.isfinite(values), idx, 0)
    return np.clip(idx, 0, centers.size - 1)


def _nearest_window_map(available, n_windows):
    avail = np.asarray(sorted(set(int(v) for v in available)), dtype=np.int64)
    if avail.size == 0:
        return np.full(n_windows, -1, dtype=np.int64)
    out = np.empty(n_windows, dtype=np.int64)
    for w in range(n_windows):
        pos = np.searchsorted(avail, w)
        left = avail[pos - 1] if pos > 0 else None
        right = avail[pos] if pos < avail.size else None
        if left is None:
            out[w] = right
        elif right is None:
            out[w] = left
        elif (w - left) <= (right - w):
            out[w] = left
        else:
            out[w] = right
    return out


def _robust_variance(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.size == 0:
        return np.full(4, 1e-3, dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    var = (1.4826 * mad) ** 2
    var[~np.isfinite(var)] = NOISE_FLOOR
    var[var < NOISE_FLOOR] = NOISE_FLOOR
    return var


def _build_noise_by_window(X, window_ids, n_windows):
    base = _robust_variance(X)
    noise = np.repeat(base[None, :], n_windows, axis=0)
    finite = np.isfinite(X).all(axis=1)
    for w in range(n_windows):
        idx = np.where((window_ids == w) & finite)[0]
        if idx.size >= 8:
            noise[w] = _robust_variance(X[idx])
    return noise


def _fit_mixture(features):
    x = np.asarray(features, dtype=np.float64)
    if x.shape[0] == 0:
        return None
    n_features = x.shape[1]
    mean = np.nanmean(x, axis=0)
    var = np.nanvar(x, axis=0)
    var[~np.isfinite(var)] = NOISE_FLOOR
    var[var < NOISE_FLOOR] = NOISE_FLOOR
    if x.shape[0] < MIN_GMM_SAMPLES:
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean[None, :],
            "vars": var[None, :],
        }
    n_components = int(x.shape[0] // COMPONENT_SCALE)
    if n_components < 1:
        n_components = 1
    if n_components > MAX_GMM_COMPONENTS:
        n_components = MAX_GMM_COMPONENTS
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        n_components = 1

    if n_components == 1:
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean[None, :],
            "vars": var[None, :],
        }

    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=NOISE_FLOOR,
            random_state=0,
            max_iter=120,
            n_init=1,
        )
        gmm.fit(x)
        weights = np.asarray(gmm.weights_, dtype=np.float64)
        means = np.asarray(gmm.means_, dtype=np.float64)
        vars_ = np.asarray(gmm.covariances_, dtype=np.float64)
        vars_[~np.isfinite(vars_)] = NOISE_FLOOR
        vars_[vars_ < NOISE_FLOOR] = NOISE_FLOOR
        return {"weights": weights, "means": means, "vars": vars_}
    except Exception:
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean[None, :],
            "vars": var[None, :],
        }


def _logsumexp_rows(matrix):
    if matrix.size == 0:
        return np.full(matrix.shape[1], -np.inf, dtype=np.float64)
    m = np.nanmax(matrix, axis=0)
    finite = np.isfinite(m)
    out = np.full(matrix.shape[1], -np.inf, dtype=np.float64)
    if np.any(finite):
        sub = matrix[:, finite]
        mm = m[finite]
        out[finite] = mm + np.log(np.sum(np.exp(sub - mm), axis=0))
    return out


def _mixture_log_likelihood(x, model, noise):
    if model is None:
        return np.full(x.shape[0], -np.inf, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    vars_ = np.asarray(model["vars"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if means.size == 0 or x.shape[1] != means.shape[1]:
        return np.full(x.shape[0], -np.inf, dtype=np.float64)

    v = vars_ + np.asarray(noise, dtype=np.float64)[None, :]
    v[v < NOISE_FLOOR] = NOISE_FLOOR
    diff = x[:, None, :] - means[None, :, :]
    quad = np.sum((diff * diff) / v[None, :, :], axis=2)
    logdet = np.sum(np.log(v), axis=1)
    logw = np.log(np.clip(weights, EPS, 1.0))
    norm = -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + logdet[None, :] + quad)
    comp = logw[None, :] + norm
    return _logsumexp_rows(comp.T)


def _smooth_and_normalize_counts(counts, alpha):
    counts = np.asarray(counts, dtype=np.float64)
    n_windows, n_cols = counts.shape
    sm = np.array(counts, copy=True)
    for c in range(n_cols):
        col = sm[:, c]
        nz = np.where(col > 0)[0]
        if nz.size == 0:
            continue
        fb = _nearest_window_map(nz, n_windows)
        for w in range(n_windows):
            if col[w] <= 0:
                col[w] = col[fb[w]]
        sm[:, c] = col

    row_sum = sm.sum(axis=1)
    out = np.empty_like(sm, dtype=np.float64)
    has = row_sum > 0
    out[has] = (sm[has] + alpha) / (row_sum[has, None] + alpha * n_cols)
    out[~has] = 1.0 / float(n_cols)
    return out


def _build_aux_reference(aux, centers):
    if aux is None or len(aux) == 0:
        return None
    class_col = _find_column(
        aux,
        ["class", "target", "label", "y", "object_class", "classlabel", "class_label"],
    )
    if class_col is None:
        return None

    cols = {
        "u": _find_column(aux, ["u"]),
        "g": _find_column(aux, ["g"]),
        "r": _find_column(aux, ["r"]),
        "i": _find_column(aux, ["i"]),
        "z": _find_column(aux, ["z"]),
        "redshift": _find_column(aux, ["redshift", "zspec", "red_shift"]),
    }
    if any(v is None for v in cols.values()):
        return None

    u = _to_float(aux[cols["u"]])
    g = _to_float(aux[cols["g"]])
    r = _to_float(aux[cols["r"]])
    i = _to_float(aux[cols["i"]])
    z = _to_float(aux[cols["z"]])
    red = _to_float(aux[cols["redshift"]])

    x0 = -0.4 * (u - i)
    x1 = -0.4 * (g - i)
    x2 = -0.4 * (r - i)
    x3 = -0.4 * (z - i)
    X = np.column_stack([x0, x1, x2, x3])

    raw_labels = aux[class_col].apply(_normalize_class_label).to_numpy(dtype=object)
    model_class = np.full(raw_labels.shape[0], "UNKNOWN", dtype=object)
    model_class[raw_labels == "STAR"] = "STAR"
    model_class[raw_labels == "GALAXY"] = "GALAXY"
    model_class[raw_labels == "QSO"] = "QSO"

    branches = _qso_branch_from_redshift(red)
    window_ids = _assign_windows(i, centers)

    finite = np.isfinite(X).all(axis=1) & np.isfinite(i)
    keep = finite & (model_class != "UNKNOWN")
    if not np.any(keep):
        return None

    ref = pd.DataFrame(
        {
            "x0": x0[keep],
            "x1": x1[keep],
            "x2": x2[keep],
            "x3": x3[keep],
            "window_id": window_ids[keep],
            "class": model_class[keep],
            "qso_branch": branches[keep],
        },
        index=np.arange(np.count_nonzero(keep)),
    )
    return ref


def _compute_priors(raw_windows, raw_redshift, aux_reference, n_windows):
    if aux_reference is not None and len(aux_reference) > 0:
        class_counts = (
            aux_reference.groupby(["window_id", "class"]).size().unstack(fill_value=0)
        )
        class_counts = class_counts.reindex(
            range(n_windows), fill_value=0.0
        ).fillna(0.0)
        star_counts = (
            class_counts["STAR"].to_numpy(dtype=np.float64)
            if "STAR" in class_counts
            else np.zeros(n_windows, dtype=np.float64)
        )
        gal_counts = (
            class_counts["GALAXY"].to_numpy(dtype=np.float64)
            if "GALAXY" in class_counts
            else np.zeros(n_windows, dtype=np.float64)
        )
        qso_counts = (
            class_counts["QSO"].to_numpy(dtype=np.float64)
            if "QSO" in class_counts
            else np.zeros(n_windows, dtype=np.float64)
        )
        class_prior = _smooth_and_normalize_counts(
            np.column_stack([star_counts, gal_counts, qso_counts]), PRIOR_ALPHA
        )

        qso_only = aux_reference[aux_reference["class"] == "QSO"]
        branch_counts = qso_only.groupby(["window_id", "qso_branch"]).size().unstack(
            fill_value=0
        )
        branch_counts = branch_counts.reindex(range(n_windows), fill_value=0.0).fillna(0.0)
        b0 = (
            branch_counts[0].to_numpy(dtype=np.float64)
            if 0 in branch_counts.columns
            else np.zeros(n_windows, dtype=np.float64)
        )
        b1 = (
            branch_counts[1].to_numpy(dtype=np.float64)
            if 1 in branch_counts.columns
            else np.zeros(n_windows, dtype=np.float64)
        )
        b2 = (
            branch_counts[2].to_numpy(dtype=np.float64)
            if 2 in branch_counts.columns
            else np.zeros(n_windows, dtype=np.float64)
        )
        branch_prior = _smooth_and_normalize_counts(
            np.column_stack([b0, b1, b2]), PRIOR_ALPHA
        )
        return class_prior, branch_prior

    class_prior = np.tile(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64), (n_windows, 1))
    branches = _qso_branch_from_redshift(raw_redshift)
    branch_counts = np.zeros((n_windows, 3), dtype=np.float64)
    for b in (0, 1, 2):
        idx = np.where(branches == b)[0]
        if idx.size > 0:
            counts = np.bincount(raw_windows[idx], minlength=n_windows).astype(np.float64)
            branch_counts[:, b] = counts
    branch_prior = _smooth_and_normalize_counts(branch_counts, PRIOR_ALPHA)
    return class_prior, branch_prior


def _fit_models(raw_X, raw_windows, raw_redshift, aux_reference, n_windows):
    models = {key: {} for key in CLASS_KEYS}
    branch = _qso_branch_from_redshift(raw_redshift)
    finite = np.isfinite(raw_X).all(axis=1)

    if aux_reference is not None and len(aux_reference) > 0:
        for cls in ("STAR", "GALAXY"):
            subset = aux_reference[aux_reference["class"] == cls]
            for w, window_df in subset.groupby("window_id"):
                x = window_df[["x0", "x1", "x2", "x3"]].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x).all(axis=1)]
                if x.size == 0:
                    continue
                models[cls][int(w)] = _fit_mixture(x)

        qso_ref = aux_reference[aux_reference["class"] == "QSO"]
        qso_map = {0: "QSO_LOW", 1: "QSO_MID", 2: "QSO_HIGH"}
        for branch_val, key in qso_map.items():
            bsub = qso_ref[qso_ref["qso_branch"] == branch_val]
            for w, window_df in bsub.groupby("window_id"):
                x = window_df[["x0", "x1", "x2", "x3"]].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x).all(axis=1)]
                if x.size == 0:
                    continue
                models[key][int(w)] = _fit_mixture(x)
        return models

    for w in range(n_windows):
        w_idx = (raw_windows == w) & finite
        if np.any(w_idx):
            xw = raw_X[w_idx]
            model_w = _fit_mixture(xw)
            if model_w is not None:
                models["STAR"][w] = model_w
                models["GALAXY"][w] = model_w
        for branch_val, key in ((0, "QSO_LOW"), (1, "QSO_MID"), (2, "QSO_HIGH")):
            b_idx = w_idx & (branch == branch_val)
            if np.any(b_idx):
                xb = raw_X[b_idx]
                model_b = _fit_mixture(xb)
                if model_b is not None:
                    models[key][w] = model_b
    return models


def _compute_likelihoods(raw_X, valid_mask, windows, noise_by_window, models, n_windows):
    like = {key: np.full(raw_X.shape[0], -np.inf, dtype=np.float64) for key in CLASS_KEYS}
    for key in CLASS_KEYS:
        by_window = models.get(key, {})
        if by_window:
            fallback = _nearest_window_map(by_window.keys(), n_windows)
        else:
            fallback = np.full(n_windows, -1, dtype=np.int64)

        for w in range(n_windows):
            idx = (windows == w) & valid_mask
            if not np.any(idx):
                continue
            model = by_window.get(int(w))
            if model is None:
                fb = fallback[w]
                if fb < 0:
                    continue
                model = by_window.get(int(fb))
                if model is None:
                    continue
            like[key][idx] = _mixture_log_likelihood(raw_X[idx], model, noise_by_window[w])
    return like


def add_redshift_branch_deconvolved_flux_posteriors(raw, deps, aux):
    raw_index = raw.index
    n = len(raw)

    u = _to_float(raw["u"])
    g = _to_float(raw["g"])
    r = _to_float(raw["r"])
    i = _to_float(raw["i"])
    z = _to_float(raw["z"])
    redshift = _to_float(raw["redshift"])

    rel_u_i = -0.4 * (u - i)
    rel_g_i = -0.4 * (g - i)
    rel_r_i = -0.4 * (r - i)
    rel_z_i = -0.4 * (z - i)
    color_ug = u - g
    color_gr = g - r
    color_ri = r - i
    color_iz = i - z

    X = np.column_stack([rel_u_i, rel_g_i, rel_r_i, rel_z_i])
    i_for_windows = [i[np.isfinite(i)]]
    if aux is not None and len(aux) > 0:
        aux_i_col = _find_column(aux, ["i"])
        if aux_i_col is not None:
            aux_i = _to_float(aux[aux_i_col])
            if aux_i.size > 0:
                i_for_windows.append(aux_i[np.isfinite(aux_i)])

    if len(i_for_windows) == 0:
        all_i = np.array([0.0], dtype=np.float64)
    else:
        all_i = np.concatenate(i_for_windows, axis=0)
    centers = _build_windows(all_i)
    windows = _assign_windows(i, centers)
    n_windows = centers.shape[0]

    noise_by_window = _build_noise_by_window(X, windows, n_windows)
    aux_reference = _build_aux_reference(aux, centers)

    class_prior, branch_prior = _compute_priors(windows, redshift, aux_reference, n_windows)
    models = _fit_models(X, windows, redshift, aux_reference, n_windows)

    valid = np.isfinite(X).all(axis=1)
    like = _compute_likelihoods(X, valid, windows, noise_by_window, models, n_windows)

    prior_rows = class_prior[windows]
    branch_rows = branch_prior[windows]
    prior_star = prior_rows[:, 0]
    prior_galaxy = prior_rows[:, 1]
    prior_qso = prior_rows[:, 2]
    prior_qso_low = prior_qso * branch_rows[:, 0]
    prior_qso_mid = prior_qso * branch_rows[:, 1]
    prior_qso_high = prior_qso * branch_rows[:, 2]

    base_star = prior_star.copy()
    base_gal = prior_galaxy.copy()
    base_low = prior_qso_low.copy()
    base_mid = prior_qso_mid.copy()
    base_high = prior_qso_high.copy()

    base_sum = base_star + base_gal + base_low + base_mid + base_high
    base_nonzero = base_sum > 0
    base_star[base_nonzero] /= base_sum[base_nonzero]
    base_gal[base_nonzero] /= base_sum[base_nonzero]
    base_low[base_nonzero] /= base_sum[base_nonzero]
    base_mid[base_nonzero] /= base_sum[base_nonzero]
    base_high[base_nonzero] /= base_sum[base_nonzero]
    if np.any(~base_nonzero):
        base_star[~base_nonzero] = 1.0 / 3.0
        base_gal[~base_nonzero] = 1.0 / 3.0
        base_low[~base_nonzero] = 1.0 / 9.0
        base_mid[~base_nonzero] = 1.0 / 9.0
        base_high[~base_nonzero] = 1.0 / 9.0

    log_star = np.full(n, -np.inf, dtype=np.float64)
    log_gal = np.full(n, -np.inf, dtype=np.float64)
    log_low = np.full(n, -np.inf, dtype=np.float64)
    log_mid = np.full(n, -np.inf, dtype=np.float64)
    log_high = np.full(n, -np.inf, dtype=np.float64)

    log_star[valid] = np.log(np.clip(prior_star[valid], EPS, 1.0 - EPS)) + like["STAR"][valid]
    log_gal[valid] = np.log(np.clip(prior_galaxy[valid], EPS, 1.0 - EPS)) + like["GALAXY"][valid]
    log_low[valid] = np.log(np.clip(prior_qso_low[valid], EPS, 1.0 - EPS)) + like["QSO_LOW"][valid]
    log_mid[valid] = np.log(np.clip(prior_qso_mid[valid], EPS, 1.0 - EPS)) + like["QSO_MID"][valid]
    log_high[valid] = np.log(np.clip(prior_qso_high[valid], EPS, 1.0 - EPS)) + like["QSO_HIGH"][valid]

    stack = np.vstack([log_star, log_gal, log_low, log_mid, log_high])
    finite_stack = np.isfinite(stack).any(axis=0)

    posterior = np.vstack([base_star, base_gal, base_low, base_mid, base_high]).astype(np.float64)
    if np.any(finite_stack):
        denom = _logsumexp_rows(stack[:, finite_stack])
        posterior[:, finite_stack] = np.exp(stack[:, finite_stack] - denom)

    p_star = np.clip(posterior[0], EPS, 1.0 - EPS)
    p_galaxy = np.clip(posterior[1], EPS, 1.0 - EPS)
    p_qso_low = np.clip(posterior[2], EPS, 1.0 - EPS)
    p_qso_mid = np.clip(posterior[3], EPS, 1.0 - EPS)
    p_qso_high = np.clip(posterior[4], EPS, 1.0 - EPS)

    p_qso = p_qso_low + p_qso_mid + p_qso_high
    p_qso = np.clip(p_qso, EPS, 1.0 - EPS)
    logit_star = np.log((p_star + EPS) / (1.0 - p_star + EPS))
    logit_gal = np.log((p_galaxy + EPS) / (1.0 - p_galaxy + EPS))
    logit_qso = np.log((p_qso + EPS) / (1.0 - p_qso + EPS))

    margin_low_vs_star = np.log((p_qso_low + EPS) / (p_star + EPS))
    margin_mid_vs_star = np.log((p_qso_mid + EPS) / (p_star + EPS))
    margin_high_vs_star = np.log((p_qso_high + EPS) / (p_star + EPS))
    margin_low_vs_galaxy = np.log((p_qso_low + EPS) / (p_galaxy + EPS))
    margin_mid_vs_galaxy = np.log((p_qso_mid + EPS) / (p_galaxy + EPS))
    margin_high_vs_galaxy = np.log((p_qso_high + EPS) / (p_galaxy + EPS))

    qso_branch = np.column_stack([p_qso_low, p_qso_mid, p_qso_high])
    qso_sum = np.sum(qso_branch, axis=1)
    qso_norm = np.divide(
        qso_branch,
        qso_sum[:, None],
        out=np.zeros_like(qso_branch),
        where=qso_sum[:, None] > EPS,
    )
    qso_branch_concentration = np.where(
        qso_sum > EPS, np.max(qso_norm, axis=1), 0.0
    )
    qso_branch_entropy = -np.sum(
        np.where(
            qso_sum[:, None] > EPS,
            qso_norm * np.log(np.clip(qso_norm, EPS, 1.0)),
            0.0,
        ),
        axis=1,
    )
    qso_branch_active_count = np.sum(
        qso_norm > (0.5 * np.max(qso_norm, axis=1, keepdims=True)),
        axis=1,
    )

    out = pd.DataFrame(index=raw_index)
    out["redshift_branch_rel_flux_u_over_i"] = rel_u_i
    out["redshift_branch_rel_flux_g_over_i"] = rel_g_i
    out["redshift_branch_rel_flux_r_over_i"] = rel_r_i
    out["redshift_branch_rel_flux_z_over_i"] = rel_z_i
    out["redshift_branch_color_u_g"] = color_ug
    out["redshift_branch_color_g_r"] = color_gr
    out["redshift_branch_color_r_i"] = color_ri
    out["redshift_branch_color_i_z"] = color_iz
    out["redshift_branch_i_window_id"] = windows.astype(np.int16)
    out["redshift_branch_p_star"] = p_star
    out["redshift_branch_p_galaxy"] = p_galaxy
    out["redshift_branch_p_qso"] = p_qso
    out["redshift_branch_p_qso_low"] = p_qso_low
    out["redshift_branch_p_qso_mid"] = p_qso_mid
    out["redshift_branch_p_qso_high"] = p_qso_high
    out["redshift_branch_logit_star"] = logit_star
    out["redshift_branch_logit_galaxy"] = logit_gal
    out["redshift_branch_logit_qso"] = logit_qso
    out["redshift_branch_qso_low_vs_star_margin"] = margin_low_vs_star
    out["redshift_branch_qso_mid_vs_star_margin"] = margin_mid_vs_star
    out["redshift_branch_qso_high_vs_star_margin"] = margin_high_vs_star
    out["redshift_branch_qso_low_vs_galaxy_margin"] = margin_low_vs_galaxy
    out["redshift_branch_qso_mid_vs_galaxy_margin"] = margin_mid_vs_galaxy
    out["redshift_branch_qso_high_vs_galaxy_margin"] = margin_high_vs_galaxy
    out["redshift_branch_qso_branch_concentration"] = qso_branch_concentration
    out["redshift_branch_qso_branch_entropy"] = qso_branch_entropy
    out["redshift_branch_qso_branch_count"] = qso_branch_active_count.astype(np.float64)

    log_cols = [
        "redshift_branch_logit_star",
        "redshift_branch_logit_galaxy",
        "redshift_branch_logit_qso",
        "redshift_branch_qso_low_vs_star_margin",
        "redshift_branch_qso_mid_vs_star_margin",
        "redshift_branch_qso_high_vs_star_margin",
        "redshift_branch_qso_low_vs_galaxy_margin",
        "redshift_branch_qso_mid_vs_galaxy_margin",
        "redshift_branch_qso_high_vs_galaxy_margin",
    ]
    for c in log_cols:
        out[c] = np.clip(out[c], -LOG_FEATURE_CLIP, LOG_FEATURE_CLIP)

    return out


FEATURE_GROUPS = [
    {
        "name": "redshift_branch_deconvolved_flux_posteriors",
        "fn": add_redshift_branch_deconvolved_flux_posteriors,
        "depends_on": [],
        "description": "Build deconvolved i-band-windowed relative-flux Gaussian-mixture posteriors for STAR, GALAXY, and QSO branches with uncertainty-adjusted likelihoods and branch-ambiguity features.",
    },
]