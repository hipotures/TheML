from __future__ import annotations

import numpy as np
import pandas as pd

_CLASS_NAMES = ("GALAXY", "STAR", "QSO")
_I_BIN_START = 17.7
_I_BIN_STOP = 22.5
_I_BIN_STEP = 0.1
_NUM_I_BINS = 49
_Z_CUT_1 = 0.8
_Z_CUT_2 = 2.5
_Z_CUT_3 = 4.0
_EPS = 1e-12
_MIN_VAR = 1e-8
_RANDOM_STATE = 7


def _to_float_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)


def _canonicalize_class(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    token = str(value).strip().upper()
    if token == "":
        return None
    if token.startswith("GAL"):
        return "GALAXY"
    if token.startswith("STAR"):
        return "STAR"
    if token.startswith("QSO") or token in ("Q", "QUASAR", "AGN"):
        return "QSO"
    return None


def _logsumexp_rows(matrix):
    maxv = np.max(matrix, axis=1, keepdims=True)
    finite = np.isfinite(maxv)
    max_safe = np.where(finite, maxv, 0.0)
    sum_shifted = np.sum(np.exp(matrix - max_safe), axis=1)
    out = np.log(np.where(sum_shifted > 0.0, sum_shifted, 1.0)) + np.squeeze(max_safe, axis=1)
    return np.where(finite[:, 0], out, -np.inf)


def _compute_ratio_features(u, g, r_, z, i):
    u = np.asarray(u, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    r_ = np.asarray(r_, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    i = np.asarray(i, dtype=np.float64)

    valid = np.isfinite(u) & np.isfinite(g) & np.isfinite(r_) & np.isfinite(z) & np.isfinite(i)
    valid &= (i > 0.0)
    if valid.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64), valid

    fi = np.full(i.shape[0], 1.0, dtype=np.float64)
    fi[valid] = 10.0 ** (-0.4 * i[valid])

    fu = np.full(i.shape[0], 0.0, dtype=np.float64)
    fg = np.full(i.shape[0], 0.0, dtype=np.float64)
    fr = np.full(i.shape[0], 0.0, dtype=np.float64)
    fz = np.full(i.shape[0], 0.0, dtype=np.float64)
    good = valid
    fu[good] = 10.0 ** (-0.4 * u[good])
    fg[good] = 10.0 ** (-0.4 * g[good])
    fr[good] = 10.0 ** (-0.4 * r_[good])
    fz[good] = 10.0 ** (-0.4 * z[good])

    ratios = np.column_stack((fu, fg, fr, fz)) / fi[:, None]
    valid &= np.isfinite(ratios).all(axis=1)
    return ratios, valid


def _assign_i_bin_indices(i_vals):
    i_vals = np.asarray(i_vals, dtype=np.float64)
    out = np.full(i_vals.shape[0], -1, dtype=np.int32)
    in_range = np.isfinite(i_vals) & (i_vals >= _I_BIN_START) & (i_vals <= _I_BIN_STOP)
    out[in_range] = np.floor((i_vals[in_range] - _I_BIN_START + 1e-12) / _I_BIN_STEP).astype(np.int32)
    out = np.clip(out, -1, _NUM_I_BINS - 1)
    return out


def _assign_z_slab_indices(z_vals):
    z_vals = np.asarray(z_vals, dtype=np.float64)
    out = np.full(z_vals.shape[0], -1, dtype=np.int8)
    finite = np.isfinite(z_vals)
    out[finite & (z_vals < _Z_CUT_1)] = 0
    out[finite & (z_vals >= _Z_CUT_1) & (z_vals < _Z_CUT_2)] = 1
    out[finite & (z_vals >= _Z_CUT_2) & (z_vals < _Z_CUT_3)] = 2
    out[finite & (z_vals >= _Z_CUT_3)] = 3
    return out


def _robust_diagonal_variance(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == 0:
        return np.full(values.shape[1], _MIN_VAR, dtype=np.float64)
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0)
    var = (1.4826 * mad) ** 2
    return np.maximum(var, _MIN_VAR)


def _compute_proxy_by_bin(ratios, i_bins, n_bins):
    ratios = np.asarray(ratios, dtype=np.float64)
    i_bins = np.asarray(i_bins, dtype=np.int32)

    proxy = np.full((n_bins, ratios.shape[1]), _MIN_VAR, dtype=np.float64)
    good = np.isfinite(ratios).all(axis=1)
    if np.any(good):
        proxy[:] = _robust_diagonal_variance(ratios[good])

    for bin_id in range(n_bins):
        mask = good & (i_bins == bin_id)
        if np.any(mask):
            proxy[bin_id] = _robust_diagonal_variance(ratios[mask])

    return proxy


def _get_gaussian_mixture():
    try:
        from sklearn.mixture import GaussianMixture

        return GaussianMixture
    except Exception:
        return None


def _fit_gmm_model(X):
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] == 0:
        return None

    if X.shape[0] == 1:
        mean = X[0:1]
        var = np.maximum(np.var(X, axis=0, ddof=0)[None, :], _MIN_VAR)
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean,
            "diag_cov": var,
            "log_weights": np.array([0.0], dtype=np.float64),
        }

    GaussianMixture = _get_gaussian_mixture()
    if GaussianMixture is None:
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.maximum(np.var(X, axis=0, ddof=0, keepdims=True), _MIN_VAR)
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean,
            "diag_cov": var,
            "log_weights": np.array([0.0], dtype=np.float64),
        }

    n_components = int(np.sqrt(X.shape[0]))
    n_components = max(1, min(20, n_components))
    n_components = min(n_components, X.shape[0])

    try:
        gm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            reg_covar=1e-7,
            random_state=_RANDOM_STATE,
            max_iter=250,
            n_init=1,
        )
        gm.fit(X)
        diag_cov = np.maximum(gm.covariances_, _MIN_VAR)
        return {
            "weights": gm.weights_.astype(np.float64),
            "means": gm.means_.astype(np.float64),
            "diag_cov": diag_cov.astype(np.float64),
            "log_weights": np.log(np.maximum(gm.weights_, _EPS).astype(np.float64)),
        }
    except Exception:
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.maximum(np.var(X, axis=0, ddof=0, keepdims=True), _MIN_VAR)
        return {
            "weights": np.array([1.0], dtype=np.float64),
            "means": mean,
            "diag_cov": var,
            "log_weights": np.array([0.0], dtype=np.float64),
        }


def _log_pdf_diag_gmm(X, model, proxy_diag):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    means = model["means"]
    log_w = model["log_weights"]
    cov = np.maximum(model["diag_cov"] + np.asarray(proxy_diag, dtype=np.float64)[None, :], _MIN_VAR)

    diff = X[:, None, :] - means[None, :, :]
    inv = 1.0 / cov
    quad = np.sum((diff * diff) * inv, axis=2)
    log_det = np.sum(np.log(cov), axis=1)
    dims = float(X.shape[1])
    comp = log_w[None, :] - 0.5 * (quad + log_det[None, :] + dims * np.log(2.0 * np.pi))
    return _logsumexp_rows(comp)


def _score_one_class(ratios, i_bins, z_bins, class_info, proxy_by_bin, global_proxy):
    n = ratios.shape[0]
    scores = np.full(n, -np.inf, dtype=np.float64)
    if class_info is None:
        return scores

    cell_models = class_info.get("cell_models", {})
    global_model = class_info.get("global_model")
    unresolved = np.ones(n, dtype=bool)

    for offset in (0, -1, 1):
        if not unresolved.any():
            break
        candidate_bins = i_bins + offset
        for z_bin in range(4):
            slab_mask = z_bins == z_bin
            if not np.any(slab_mask & unresolved):
                continue
            for bin_id in range(proxy_by_bin.shape[0]):
                model = cell_models.get((bin_id, z_bin))
                if model is None:
                    continue
                mask = unresolved & slab_mask & (candidate_bins == bin_id)
                if not np.any(mask):
                    continue
                row_pdf = _log_pdf_diag_gmm(ratios[mask], model, proxy_by_bin[bin_id])
                scores[mask] = np.log(model["prior"] + _EPS) + row_pdf
                unresolved[mask] = False

    if global_model is not None and np.any(unresolved):
        idx = np.where(unresolved)[0]
        if idx.size:
            global_ll = _log_pdf_diag_gmm(ratios[idx], global_model, global_proxy)
            prior = float(class_info.get("global_prior", 0.0))
            scores[idx] = np.log(prior + _EPS) + global_ll

    return scores


def _normalize_loglik_to_probs(log_likelihood):
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    n_rows, n_classes = log_likelihood.shape
    probs = np.full((n_rows, n_classes), 1.0 / float(n_classes), dtype=np.float64)

    finite_rows = np.isfinite(np.max(log_likelihood, axis=1))
    if not np.any(finite_rows):
        return probs

    idx = np.where(finite_rows)[0]
    row_ll = log_likelihood[idx]
    row_max = np.max(row_ll, axis=1, keepdims=True)
    exp_like = np.exp(row_ll - row_max)
    denom = np.sum(exp_like, axis=1, keepdims=True)
    good = denom[:, 0] > 0.0

    if np.any(good):
        sel = idx[good]
        probs[sel] = exp_like[good] / np.where(denom[good] <= 0.0, 1.0, denom[good])

    return probs


def _build_xdqso_bundle(aux):
    required = ("u", "g", "r", "i", "z", "redshift", "class")
    if not isinstance(aux, pd.DataFrame):
        return None
    if any(c not in aux.columns for c in required):
        return None
    if aux.empty:
        return None

    u = _to_float_array(aux["u"])
    g = _to_float_array(aux["g"])
    r_ = _to_float_array(aux["r"])
    i = _to_float_array(aux["i"])
    z = _to_float_array(aux["z"])
    rs = aux["redshift"]
    z = _to_float_array(rs) if not isinstance(rs, pd.Series) else _to_float_array(rs)

    class_series = aux["class"]
    label_list = [_canonicalize_class(v) for v in class_series]
    labels = np.asarray(label_list, dtype=object)

    ratios, valid_ratio = _compute_ratio_features(u, g, r_, z, i)
    label_mask = np.array([x in _CLASS_NAMES for x in labels], dtype=bool)
    valid = valid_ratio & label_mask & np.isfinite(z)

    if not np.any(valid):
        return None

    u_v = u[valid]
    g_v = g[valid]
    r_v = r_[valid]
    i_v = i[valid]
    z_v = z[valid]
    labels_v = labels[valid]
    ratios_v = ratios[valid]

    i_bins = _assign_i_bin_indices(i_v)
    z_bins = _assign_z_slab_indices(z_v)
    proxy_by_bin = _compute_proxy_by_bin(ratios_v, i_bins, _NUM_I_BINS)
    global_proxy = np.median(proxy_by_bin, axis=0)

    cell_total = {}
    for bi, zi in zip(i_bins, z_bins):
        if bi >= 0 and zi >= 0:
            key = (int(bi), int(zi))
            cell_total[key] = cell_total.get(key, 0) + 1

    class_models = {}
    total_n = len(labels_v)

    for cname in _CLASS_NAMES:
        class_mask = labels_v == cname
        class_n = int(np.sum(class_mask))
        if class_n == 0:
            continue

        gmm_global = _fit_gmm_model(ratios_v[class_mask])
        if gmm_global is None:
            continue

        cell_models = {}
        for bin_id in range(_NUM_I_BINS):
            for slab_id in range(4):
                mask = class_mask & (i_bins == bin_id) & (z_bins == slab_id)
                if not np.any(mask):
                    continue
                model = _fit_gmm_model(ratios_v[mask])
                if model is None:
                    continue
                prior = np.sum(mask)
                den = float(cell_total.get((bin_id, slab_id), max(1, prior)))
                model["prior"] = float(prior) / float(den)
                cell_models[(bin_id, slab_id)] = model

        class_models[cname] = {
            "global_model": gmm_global,
            "global_prior": float(class_n) / float(total_n),
            "cell_models": cell_models,
        }

    if not class_models:
        return None

    return {
        "n_bins": _NUM_I_BINS,
        "proxy_by_bin": proxy_by_bin,
        "global_proxy": global_proxy,
        "classes": class_models,
    }


def _build_feature_frame(prob_matrix, index):
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    p_galaxy = prob_matrix[:, 0]
    p_star = prob_matrix[:, 1]
    p_qso = prob_matrix[:, 2]

    p_clipped = np.clip(prob_matrix, _EPS, 1.0)
    log_p = np.log(p_clipped)

    logodds_qso_minus_star = log_p[:, 2] - log_p[:, 1]
    logodds_qso_minus_galaxy = log_p[:, 2] - log_p[:, 0]
    logodds_star_minus_galaxy = log_p[:, 1] - log_p[:, 0]

    order = np.sort(prob_matrix, axis=1)
    top_class_confidence = order[:, 2] - order[:, 1]
    posterior_entropy = -np.sum(prob_matrix * np.log(p_clipped), axis=1)

    return pd.DataFrame(
        {
            "p_GALAXY": p_galaxy,
            "p_STAR": p_star,
            "p_QSO": p_qso,
            "logodds_qso_minus_star": logodds_qso_minus_star,
            "logodds_qso_minus_galaxy": logodds_qso_minus_galaxy,
            "logodds_star_minus_galaxy": logodds_star_minus_galaxy,
            "top_class_confidence": top_class_confidence,
            "posterior_entropy": posterior_entropy,
        },
        index=index,
    )


def _default_features(index):
    n = len(index)
    uniform = 1.0 / float(len(_CLASS_NAMES))
    return _build_feature_frame(np.full((n, len(_CLASS_NAMES)), uniform, dtype=np.float64), index)


def add_xdqso_inspired_flux_density_scores(raw, deps, aux):
    required = ("u", "g", "r", "z", "i", "redshift")
    if any(c not in raw.columns for c in required):
        return _default_features(raw.index)

    try:
        u = _to_float_array(raw["u"])
        g = _to_float_array(raw["g"])
        r_ = _to_float_array(raw["r"])
        z = _to_float_array(raw["z"])
        i = _to_float_array(raw["i"])
        redshift = _to_float_array(raw["redshift"])
    except Exception:
        return _default_features(raw.index)

    ratios, valid = _compute_ratio_features(u, g, r_, z, i)
    if ratios.shape[0] == 0:
        return _default_features(raw.index)

    bundle = _build_xdqso_bundle(aux)
    if bundle is None:
        return _default_features(raw.index)

    i_bins = _assign_i_bin_indices(i)
    z_bins = _assign_z_slab_indices(redshift)

    log_likelihood = np.full((raw.shape[0], len(_CLASS_NAMES)), -np.inf, dtype=np.float64)
    valid_idx = np.where(valid)[0]
    if valid_idx.size > 0:
        v_ratios = ratios[valid_idx]
        v_i_bins = i_bins[valid_idx]
        v_z_bins = z_bins[valid_idx]
        for ci, cname in enumerate(_CLASS_NAMES):
            class_info = bundle["classes"].get(cname)
            if class_info is None:
                continue
            ll = _score_one_class(
                v_ratios,
                v_i_bins,
                v_z_bins,
                class_info,
                bundle["proxy_by_bin"],
                bundle["global_proxy"],
            )
            log_likelihood[valid_idx, ci] = ll

    probs = _normalize_loglik_to_probs(log_likelihood)
    return _build_feature_frame(probs, raw.index)


FEATURE_GROUPS = [
    {
        "name": "xdqso_inspired_flux_density_scores",
        "fn": add_xdqso_inspired_flux_density_scores,
        "depends_on": [],
        "description": "Builds XDQSO-like redshift-binned i-band-normalized relative-flux likelihood scores from auxiliary labeled data and outputs class posteriors and discriminative odds features.",
    }
]