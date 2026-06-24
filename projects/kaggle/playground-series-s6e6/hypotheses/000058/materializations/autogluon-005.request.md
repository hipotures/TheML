Fix one failed hypothesis materialization.

Return a corrected version of the same feature-group module. Do not invent a
new hypothesis or change the feature family. Keep the fix narrow: address the
observed failure while preserving the intended preprocessing behavior.

You are repairing Python preprocessing code for one stored hypothesis. The
existing file is a feature-group module imported by a fixed runtime wrapper.
Generate only the corrected module.

# Data Overview

-> test.csv has 247435 rows and 11 columns.
Here is some information about the columns:
id (int64) has range: 577347.00 - 824781.00, 0 nan values
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.96 - 79.17, 0 nan values
u (float64) has range: 13.90 - 27.84, 0 nan values
g (float64) has range: 13.37 - 27.17, 0 nan values
r (float64) has range: 10.39 - 25.29, 0 nan values
i (float64) has range: 10.03 - 24.57, 0 nan values
z (float64) has range: 10.63 - 25.70, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F'], 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud'], 0 nan values

-> train.csv has 577347 rows and 12 columns.
Here is some information about the columns:
id (int64) has range: 0.00 - 577346.00, 0 nan values
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.97 - 79.16, 0 nan values
u (float64) has range: -0.14 - 28.25, 0 nan values
g (float64) has range: 13.54 - 27.62, 0 nan values
r (float64) has range: 12.58 - 25.25, 0 nan values
i (float64) has range: 11.96 - 27.91, 0 nan values
z (float64) has range: 11.68 - 26.83, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F'], 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud'], 0 nan values

# Task

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy

# Failed Materialization

- Mode: autogluon
- Hypothesis ID: 000058
- Source file: autogluon-002.py
- Failed node: 
- Run: 

# Execution Error

```text
Invalid materialization for hypothesis 000058 (autogluon); response=hypotheses/000058/materializations/autogluon-002.response.md: '(' was never closed (<unknown>, line 106)
```

# Repair Notes

- The generated module is syntactically invalid. Fix the exact syntax error first, then ensure the module still obeys the feature-group contract.

# Previous Code

```python
import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
except Exception:
    GaussianMixture = None

GROUP_NAME = "redshift_branch_deconvolved_flux_posteriors"
GROUP_PREFIX = "rbdgpp"

K_LOGFLUX = -0.9210340371976183
WINDOW_STEP = 0.4
WINDOW_WIDTH = 0.8
WINDOW_HALF = 0.4
N_MIN = 240
SPARSE_THRESHOLD = 150
MAD_MIN_LOCAL = 30
SMOOTH_ALPHA = 2.0
BACKOFF_SCALE = 80.0
GMM_RANDOM_SEED = 17
K_BIC_MIN = 2
K_BIC_MAX = 8
K_BIC_SPARSE_MAX = 4
EPS = 1e-8
LOGIT_CLIP = 12.0
NOISE_FLOOR_STD = 0.05
MIN_DENSITY = 1e-300
PI_CONST = 3.141592653589793

def _to_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

def _logsumexp(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return -np.inf
    max_val = np.max(values)
    if not np.isfinite(max_val):
        return -np.inf
    return float(max_val + np.log(np.sum(np.exp(values - max_val))))

def _safe_logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.clip(np.log(p / (1.0 - p)), -LOGIT_CLIP, LOGIT_CLIP)

def _mad(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return None
    arr = np.where(np.isfinite(arr), arr, np.nan)
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0)
    if np.any(~np.isfinite(mad)):
        return None
    return 1.482579685 * mad

def _noise_matrix_from_mad_color(mad_color):
    j = np.array(
        [
            [K_LOGFLUX, K_LOGFLUX, K_LOGFLUX, 0.0],
            [0.0, K_LOGFLUX, K_LOGFLUX, 0.0],
            [0.0, 0.0, K_LOGFLUX, 0.0],
            [0.0, 0.0, 0.0, -K_LOGFLUX],
        ],
        dtype=float,
    )
    floor_var = NOISE_FLOOR_STD ** 2
    if mad_color is None:
        mad_color = np.array([NOISE_FLOOR_STD] * 4, dtype=float)
    mad_color = np.maximum(np.asarray(mad_color, dtype=float), NOISE_FLOOR_STD)
    var = np.square(mad_color)
    v = j @ np.diag(var) @ j.T
    v = 0.5 * (v + v.T)
    v = v + np.eye(4) * 1e-12
    diag = np.diag(v)
    if np.any(~np.isfinite(diag)):
        diag = np.where(np.isfinite(diag), diag, floor_var)
    v[np.arange(4), np.arange(4)] = np.maximum(diag, floor_var)
    return v

def _fit_single_component(X):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return {"weights": np.array([1.0]), "means": np.zeros(4), "covariances": np.eye(4)}
    n_samples, d = X.shape
    mean = np.nanmean(X, axis=0)
    if not np.all(np.isfinite(mean)):
        mean = np.zeros(d, dtype=float)
    if n_samples < 2:
        cov = np.eye(d)
    else:
        cov = np.nan_to_num(np.cov(X, rowvar=False, bias=True), nan=0.0)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        if cov.ndim != 2 or cov.shape != (d, d):
            if cov.ndim == 1 and cov.size == d:
                cov = np.diag(np.maximum(np.abs(cov), 1.0))
            else:
                cov = np.diag(np.maximum(np.abs(np.nanmean(np.diag(np.atleast_2d(cov)))), 1.0)
    if np.all(np.abs(cov) < 1e-12):
        cov = np.eye(d)
    return {
        "weights": np.array([1.0], dtype=float),
        "means": mean.astype(float),
        "covariances": np.asarray(cov, dtype=float),
    }

def _fit_gmm_bic(X, max_components, random_state):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return _fit_single_component(X)
    n_samples, _ = X.shape
    if GaussianMixture is None or n_samples < K_BIC_MIN:
        return _fit_single_component(X)
    max_k = min(max_components, n_samples - 1)
    if max_k < K_BIC_MIN:
        return _fit_single_component(X)
    best = None
    best_bic = np.inf
    for k in range(K_BIC_MIN, max_k + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                reg_covar=1e-6,
                max_iter=250,
                n_init=1,
            )
            gm.fit(X)
            bic = gm.bic(X)
            if np.isfinite(bic) and bic < best_bic:
                best_bic = bic
                best = gm
        except Exception:
            continue
    if best is None:
        return _fit_single_component(X)
    return {
        "weights": best.weights_.astype(float),
        "means": best.means_.astype(float),
        "covariances": best.covariances_.astype(float),
    }

def _prepare_model(model, noise):
    if model is None:
        return None
    means = np.asarray(model["means"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    covs = np.asarray(model["covariances"], dtype=float)

    if means.ndim == 1:
        means = means[None, :]
    if covs.ndim == 2:
        covs = covs[None, :, :]
    if covs.shape[0] != means.shape[0]:
        if covs.shape[0] == 1:
            covs = np.repeat(covs, means.shape[0], axis=0)
        else:
            covs = covs[: means.shape[0]]
    if weights.ndim == 0 or weights.size == 0:
        weights = np.ones(means.shape[0], dtype=float)
    if weights.shape[0] != means.shape[0]:
        weights = np.ones(means.shape[0], dtype=float)
    if noise is None:
        noise = np.zeros((means.shape[1], means.shape[1]), dtype=float)

    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    dim = means.shape[1]
    invs = []
    log_norm = []
    for j in range(means.shape[0]):
        cov = np.asarray(covs[j], dtype=float)
        if cov.ndim != 2 or cov.shape != (dim, dim):
            cov = np.eye(dim)
        cov = 0.5 * (cov + cov.T)
        cov = cov + noise + np.eye(dim) * 1e-12
        jitter = 1e-10
        for _ in range(8):
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if not np.isfinite(sign) or sign <= 0.0 or not np.isfinite(logdet):
                    raise np.linalg.LinAlgError
                inv = np.linalg.inv(cov)
                break
            except Exception:
                cov = cov + np.eye(dim) * jitter
                jitter *= 10.0
        else:
            cov = np.eye(dim) * 1.0
            inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
        invs.append(inv.astype(float))
        log_norm.append(0.5 * (dim * np.log(2.0 * PI_CONST) + logdet))
    return {
        "weights": weights.astype(float),
        "means": means.astype(float),
        "inv": invs,
        "log_norm": np.asarray(log_norm, dtype=float),
        "k": means.shape[0],
    }

def _mixture_logpdf(prepared, x):
    if prepared is None or prepared["k"] == 0:
        return -300.0
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return -300.0
    logs = []
    for k in range(prepared["k"]):
        diff = x - prepared["means"][k]
        quad = float(np.dot(diff, np.dot(prepared["inv"][k], diff)))
        if not np.isfinite(quad):
            quad = 1e6
        logs.append(np.log(prepared["weights"][k]) - prepared["log_norm"][k] - 0.5 * quad)
    return _logsumexp(logs)

def _mixture_pdf(prepared, x):
    lp = _mixture_logpdf(prepared, x)
    if not np.isfinite(lp):
        return MIN_DENSITY
    p = np.exp(lp)
    if not np.isfinite(p) or p <= 0.0:
        return MIN_DENSITY
    return float(max(p, MIN_DENSITY))

def _evaluate_entry(entry, x_row):
    lam = float(entry["lam"])
    local_prep = entry["local"]
    global_prep = entry["global"]
    if lam <= 0.0:
        return _mixture_pdf(global_prep, x_row)
    local_pdf = _mixture_pdf(local_prep, x_row) if local_prep is not None else MIN_DENSITY
    global_pdf = _mixture_pdf(global_prep, x_row) if global_prep is not None else MIN_DENSITY
    return max(lam * local_pdf + (1.0 - lam) * global_pdf, MIN_DENSITY)

def _build_window_left_edges(i_values, step):
    vals = np.asarray(i_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0.0], dtype=float)
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    start = float(np.floor(min_val / step) * step)
    end = float(np.floor(max_val / step) * step)
    if end < start:
        end = start
    count = int(np.round((end - start) / step)) + 1
    return start + np.arange(count, dtype=float) * step

def _window_memberships(i_values, left_edges, valid_mask):
    i = np.asarray(i_values, dtype=float)
    members = []
    for left in left_edges:
        members.append(np.flatnonzero(valid_mask & (i >= left) & (i < left + WINDOW_WIDTH)))
    return members

def _triangle_window_candidates(i_values, start_left, step, n_windows, valid_mask):
    i = np.asarray(i_values, dtype=float)
    left = np.floor((i - start_left) / step).astype(int)
    right = left + 1
    center_left = start_left + (left + 1) * step
    center_right = center_left + step
    left_w = 1.0 - np.abs(i - center_left) / step
    right_w = 1.0 - np.abs(i - center_right) / step
    left = np.where((left >= 0) & (left < n_windows), left, -1)
    right = np.where((right >= 0) & (right < n_windows), right, -1)
    left_w = np.where((left >= 0) & valid_mask, np.clip(left_w, 0.0, 1.0), 0.0)
    right_w = np.where((right >= 0) & valid_mask, np.clip(right_w, 0.0, 1.0), 0.0)
    return left.astype(int), right.astype(int), left_w.astype(float), right_w.astype(float)

def _expand_window_indices(window_members, row_filter, window_idx, target_n):
    n_windows = len(window_members)
    base = window_members[window_idx]
    if row_filter is not None:
        base = base[row_filter[base]]
    if base.size >= target_n:
        return base
    selected = base.copy()
    radius = 1
    while selected.size < target_n and (window_idx - radius >= 0 or window_idx + radius < n_windows):
        add_blocks = []
        left = window_idx - radius
        right = window_idx + radius
        if left >= 0:
            block = window_members[left]
            if row_filter is not None:
                block = block[row_filter[block]]
            if block.size:
                add_blocks.append(block)
        if right < n_windows:
            block = window_members[right]
            if row_filter is not None:
                block = block[row_filter[block]]
            if block.size:
                add_blocks.append(block)
        if not add_blocks:
            radius += 1
            continue
        selected = np.unique(np.concatenate((selected, np.concatenate(add_blocks))))
        radius += 1
        if radius > n_windows:
            break
    return selected

def _noise_for_stratum(local_indices, class_window_indices, class_global_indices, color_matrix):
    if local_indices.size >= MAD_MIN_LOCAL:
        mad_val = _mad(color_matrix[local_indices])
    elif class_window_indices.size >= MAD_MIN_LOCAL:
        mad_val = _mad(color_matrix[class_window_indices])
    elif class_global_indices.size >= MAD_MIN_LOCAL:
        mad_val = _mad(color_matrix[class_global_indices])
    else:
        mad_val = None
    return _noise_matrix_from_mad_color(mad_val)

def _infer_classes(redshift, spectral_type):
    z = np.asarray(redshift, dtype=float)
    spectral = pd.Series(spectral_type).astype("string").fillna("")
    star_like = np.isfinite(z) & (z < 0.02)
    star_like |= np.isfinite(z) & (z < 0.08) & spectral.isin(["M", "G/K", "A/F", "O/B"])
    qso_like = np.isfinite(z) & (z >= 2.2)

    cls = np.where(star_like, 0, 1)
    cls = np.where(qso_like, 2, cls)

    branch = np.full(z.shape, -1, dtype=np.int8)
    branch[np.isfinite(z) & (z < 2.2)] = 0
    branch[np.isfinite(z) & (z >= 2.2) & (z <= 3.5)] = 1
    branch[np.isfinite(z) & (z > 3.5)] = 2
    return cls.astype(np.int8), branch.astype(np.int8)

def add_redshift_branch_deconvolved_flux_posteriors(raw, deps, aux):
    n = len(raw)
    idx = raw.index
    required = ("u", "g", "r", "i", "z", "redshift")
    for c in required:
        if c not in raw.columns:
            raise ValueError(f"Missing required column: {c}")

    u = _to_float_array(raw["u"])
    g = _to_float_array(raw["g"])
    r = _to_float_array(raw["r"])
    i = _to_float_array(raw["i"])
    z = _to_float_array(raw["z"])
    red = _to_float_array(raw["redshift"])

    color = np.column_stack((u - g, g - r, r - i, i - z))
    x = np.column_stack(
        (
            K_LOGFLUX * (u - i),
            K_LOGFLUX * (g - i),
            K_LOGFLUX * (r - i),
            K_LOGFLUX * (z - i),
        )
    )

    finite = np.isfinite(x).all(axis=1)
    if not np.any(finite):
        base_prob = np.array([1.0 / 3.0] * 3)
        p_star = np.full(n, base_prob[0], dtype=float)
        p_gal = np.full(n, base_prob[1], dtype=float)
        p_qso_low = np.full(n, base_prob[2] / 3.0, dtype=float)
        p_qso_mid = np.full(n, base_prob[2] / 3.0, dtype=float)
        p_qso_high = np.full(n, base_prob[2] / 3.0, dtype=float)
        branch_conc = np.full(n, 1.0 / 3.0, dtype=float)
        branch_ent = np.full(n, np.log(3.0), dtype=float)
        dom_low = np.zeros(n, dtype=np.int8)
        dom_mid = np.zeros(n, dtype=np.int8)
        dom_high = np.zeros(n, dtype=np.int8)
    else:
        spectral = raw.get("spectral_type", pd.Series("", index=idx))
        class_idx, branch_idx = _infer_classes(red, spectral)

        finite_idx = np.flatnonzero(finite)
        x_f = x[finite]
        c_f = color[finite]
        class_f = class_idx[finite]
        branch_f = branch_idx[finite]

        left_edges = _build_window_left_edges(i[finite], WINDOW_STEP)
        n_windows = len(left_edges)
        window_members = _window_memberships(i, left_edges, finite)

        class_window_rows = {0: [], 1: [], 2: []}
        qso_window_rows_by_branch = {0: [], 1: [], 2: []}
        qso_window_rows_all = []
        for w, rows in enumerate(window_members):
            cw = class_f[rows - np.where(finite)[0][0] if False else np.zeros(0, dtype=int)]

        for w, rows in enumerate(window_members):
            if rows.size == 0:
                class_window_rows[0].append(np.array([], dtype=int))
                class_window_rows[1].append(np.array([], dtype=int))
                class_window_rows[2].append(np.array([], dtype=int))
                qso_window_rows_by_branch[0].append(np.array([], dtype=int))
                qso_window_rows_by_branch[1].append(np.array([], dtype=int))
                qso_window_rows_by_branch[2].append(np.array([], dtype=int))
                qso_window_rows_all.append(np.array([], dtype=int))
                continue
            c_rows = class_idx[rows]
            b_rows = branch_idx[rows]
            class_window_rows[0].append(rows[c_rows == 0])
            class_window_rows[1].append(rows[c_rows == 1])
            class_window_rows[2].append(rows[c_rows == 2])
            qso_window_rows_all.append(rows[c_rows == 2])
            qso_window_rows_by_branch[0].append(rows[(c_rows == 2) & (b_rows == 0)])
            qso_window_rows_by_branch[1].append(rows[(c_rows == 2) & (b_rows == 1)])
            qso_window_rows_by_branch[2].append(rows[(c_rows == 2) & (b_rows == 2)])

        global_rows_by_class = {
            0: np.flatnonzero(finite & (class_idx == 0)),
            1: np.flatnonzero(finite & (class_idx == 1)),
            2: np.flatnonzero(finite & (class_idx == 2)),
        }

        total_finite = float(finite.sum())
        class_count = np.array([len(global_rows_by_class[0]), len(global_rows_by_class[1]), len(global_rows_by_class[2])], dtype=float)
        if total_finite > 0:
            class_prior_global = (class_count + SMOOTH_ALPHA) / (total_finite + 3.0 * SMOOTH_ALPHA)
        else:
            class_prior_global = np.array([1.0 / 3.0] * 3, dtype=float)

        overall_model = _fit_gmm_bic(x[finite], K_BIC_SPARSE_MAX, GMM_RANDOM_SEED + 123)
        overall_noise = _noise_matrix_from_mad_color(_mad(color[finite]))
        overall_prepared = _prepare_model(overall_model, overall_noise)

        # class-window priors
        class_prior_window = np.zeros((n_windows, 3), dtype=float)
        for w in range(n_windows):
            rows_w = window_members[w]
            nw = float(len(rows_w))
            if nw == 0:
                class_prior_window[w] = class_prior_global
            else:
                for c in (0, 1, 2):
                    nc = float(len(class_window_rows[c][w]))
                    class_prior_window[w, c] = (nc + SMOOTH_ALPHA) / (nw + 3.0 * SMOOTH_ALPHA)

        qso_global_rows = global_rows_by_class[2]
        global_qso_count = float(len(qso_global_rows))
        global_qso_branch_counts = np.array(
            [
                np.sum(branch_idx[qso_global_rows] == 0) if qso_global_rows.size else 0,
                np.sum(branch_idx[qso_global_rows] == 1) if qso_global_rows.size else 0,
                np.sum(branch_idx[qso_global_rows] == 2) if qso_global_rows.size else 0,
            ],
            dtype=float,
        )
        if global_qso_count > 0:
            global_qso_branch_prior = (global_qso_branch_counts + SMOOTH_ALPHA) / (global_qso_count + 3.0 * SMOOTH_ALPHA)
            global_qso_branch_prior = global_qso_branch_prior / np.sum(global_qso_branch_prior)
        else:
            global_qso_branch_prior = np.array([1.0 / 3.0] * 3, dtype=float)

        qso_branch_prior_window = np.zeros((n_windows, 3), dtype=float)
        for w in range(n_windows):
            cnt_b = np.array(
                [
                    float(len(qso_window_rows_by_branch[0][w])),
                    float(len(qso_window_rows_by_branch[1][w])),
                    float(len(qso_window_rows_by_branch[2][w])),
                ]
            )
            n_q = float(len(qso_window_rows_all[w]))
            if n_q == 0:
                qso_branch_prior_window[w] = global_qso_branch_prior
            else:
                local_prior = (cnt_b + SMOOTH_ALPHA) / (n_q + 3.0 * SMOOTH_ALPHA)
                for b in (0, 1, 2):
                    lam_b = cnt_b[b] / (cnt_b[b] + BACKOFF_SCALE) if cnt_b[b] > 0 else 0.0
                    qso_branch_prior_window[w, b] = lam_b * local_prior[b] + (1.0 - lam_b) * global_qso_branch_prior[b]

        # global models
        global_prepared = {}
        for c in (0, 1, 2):
            rows_c = global_rows_by_class[c]
            if rows_c.size > 0:
                kmax = K_BIC_MAX if rows_c.size >= SPARSE_THRESHOLD else K_BIC_SPARSE_MAX
                model_c = _fit_gmm_bic(x[rows_c], kmax, GMM_RANDOM_SEED + 17 * (c + 1))
                noise_c = _noise_matrix_from_mad_color(_mad(color[rows_c]) if rows_c.size >= MAD_MIN_LOCAL else None)
            else:
                model_c = overall_model
                noise_c = _noise_matrix_from_mad_color(_mad(color[finite]))
            global_prepared[(c, -1)] = _prepare_model(model_c, noise_c)

        # branch-specific global QSO models
        qso_rows = global_rows_by_class[2]
        qso_all_prepared = global_prepared[(2, -1)]
        if qso_rows.size > 0:
            qso_color_rows = color[qso_rows]
            qso_global_noise = _noise_matrix_from_mad_color(_mad(qso_color_rows))
            qso_branch_models = {}
            for b in (0, 1, 2):
                idx_b = qso_rows[branch_idx[qso_rows] == b]
                if idx_b.size > 0:
                    model_b = _fit_gmm_bic(x[idx_b], K_BIC_SPARSE_MAX, GMM_RANDOM_SEED + 100 + b)
                    noise_b = _noise_matrix_from_mad_color(_mad(color[idx_b]) if idx_b.size >= MAD_MIN_LOCAL else _mad(color[qso_rows]))
                    qso_branch_models[b] = _prepare_model(model_b, noise_b)
                else:
                    qso_branch_models[b] = qso_all_prepared
            for b in (0, 1, 2):
                global_prepared[(2, b)] = qso_branch_models[b]
        else:
            for b in (0, 1, 2):
                global_prepared[(2, b)] = qso_all_prepared

        model_entries = {}

        for c in (0, 1):
            for w in range(n_windows):
                row_filter = (class_idx == c)
                local_idx = _expand_window_indices(window_members, row_filter, w, N_MIN)
                if local_idx.size == 0:
                    local_prepared = None
                    lam = 0.0
                else:
                    lam = 1.0 if local_idx.size >= SPARSE_THRESHOLD else local_idx.size / (local_idx.size + BACKOFF_SCALE)
                    kmax = K_BIC_MAX if local_idx.size >= SPARSE_THRESHOLD else K_BIC_SPARSE_MAX
                    model_local = _fit_gmm_bic(x[local_idx], kmax, GMM_RANDOM_SEED + 200 + 5 * c + w)
                    noise_local = _noise_for_stratum(
                        local_idx,
                        class_window_rows[c][w],
                        global_rows_by_class[c],
                        color,
                    )
                    local_prepared = _prepare_model(model_local, noise_local)
                model_entries[(c, -1, w)] = {
                    "local": local_prepared,
                    "global": global_prepared[(c, -1)],
                    "lam": lam,
                }

        for b in (0, 1, 2):
            for w in range(n_windows):
                row_filter = (class_idx == 2) & (branch_idx == b)
                local_idx = _expand_window_indices(window_members, row_filter, w, N_MIN)
                if local_idx.size == 0:
                    local_prepared = None
                    lam = 0.0
                else:
                    lam = 1.0 if local_idx.size >= SPARSE_THRESHOLD else local_idx.size / (local_idx.size + BACKOFF_SCALE)
                    kmax = K_BIC_MAX if local_idx.size >= SPARSE_THRESHOLD else K_BIC_SPARSE_MAX
                    model_local = _fit_gmm_bic(x[local_idx], kmax, GMM_RANDOM_SEED + 350 + 7 * b + w)
                    noise_local = _noise_for_stratum(
                        local_idx,
                        class_window_rows[2][w],
                        global_rows_by_class[2],
                        color,
                    )
                    local_prepared = _prepare_model(model_local, noise_local)
                model_entries[(2, b, w)] = {
                    "local": local_prepared,
                    "global": global_prepared[(2, b)],
                    "lam": lam,
                }

        left_idx, right_idx, left_w, right_w = _triangle_window_candidates(
            i,
            float(left_edges[0]) if n_windows > 0 else 0.0,
            WINDOW_STEP,
            n_windows,
            finite,
        )

        p_star = np.zeros(n, dtype=float)
        p_galaxy = np.zeros(n, dtype=float)
        p_qso_low = np.zeros(n, dtype=float)
        p_qso_mid = np.zeros(n, dtype=float)
        p_qso_high = np.zeros(n, dtype=float)

        for r_idx in np.flatnonzero(finite):
            x_row = x[r_idx]
            local_mass_star = 0.0
            local_mass_gal = 0.0
            mass_low = 0.0
            mass_mid = 0.0
            mass_high = 0.0

            cands = []
            if left_w[r_idx] > 0.0 and left_idx[r_idx] >= 0:
                cands.append((left_idx[r_idx], left_w[r_idx]))
            if right_w[r_idx] > 0.0 and right_idx[r_idx] >= 0 and right_idx[r_idx] != left_idx[r_idx]:
                cands.append((right_idx[r_idx], right_w[r_idx]))

            if not cands:
                p = class_prior_global
                p_s = p[0] * _evaluate_entry(
                    {"local": None, "global": global_prepared[(0, -1)], "lam": 0.0},
                    x_row,
                )
                p_g = p[1] * _evaluate_entry(
                    {"local": None, "global": global_prepared[(1, -1)], "lam": 0.0},
                    x_row,
                )
                p_q = p[2]
                b0 = global_qso_branch_prior[0] * _evaluate_entry(
                    {"local": None, "global": global_prepared[(2, 0)], "lam": 0.0},
                    x_row,
                )
                b1 = global_qso_branch_prior[1] * _evaluate_entry(
                    {"local": None, "global": global_prepared[(2, 1)], "lam": 0.0},
                    x_row,
                )
                b2 = global_qso_branch_prior[2] * _evaluate_entry(
                    {"local": None, "global": global_prepared[(2, 2)], "lam": 0.0},
                    x_row,
                )
                total = p_s + p_g + p_q * (b0 + b1 + b2)
                if total <= 0.0:
                    p_star[r_idx] = class_prior_global[0]
                    p_galaxy[r_idx] = class_prior_global[1]
                    p_qso_low[r_idx] = class_prior_global[2] * global_qso_branch_prior[0]
                    p_qso_mid[r_idx] = class_prior_global[2] * global_qso_branch_prior[1]
                    p_qso_high[r_idx] = class_prior_global[2] * global_qso_branch_prior[2]
                else:
                    p_star[r_idx] = p_s / total
                    p_galaxy[r_idx] = p_g / total
                    scale = p_q / total
                    p_qso_low[r_idx] = scale * b0
                    p_qso_mid[r_idx] = scale * b1
                    p_qso_high[r_idx] = scale * b2
                continue

            for w, wgt in cands:
                prior_w = class_prior_window[w]
                prior_q = prior_w[2]

                p_s_entry = model_entries[(0, -1, w)]
                p_g_entry = model_entries[(1, -1, w)]
                p_q0_entry = model_entries[(2, 0, w)]
                p_q1_entry = model_entries[(2, 1, w)]
                p_q2_entry = model_entries[(2, 2, w)]

                local_mass_star += wgt * prior_w[0] * _evaluate_entry(p_s_entry, x_row)
                local_mass_gal += wgt * prior_w[1] * _evaluate_entry(p_g_entry, x_row)

                mass_low += wgt * prior_q * qso_branch_prior_window[w, 0] * _evaluate_entry(p_q0_entry, x_row)
                mass_mid += wgt * prior_q * qso_branch_prior_window[w, 1] * _evaluate_entry(p_q1_entry, x_row)
                mass_high += wgt * prior_q * qso_branch_prior_window[w, 2] * _evaluate_entry(p_q2_entry, x_row)

            total_mass = local_mass_star + local_mass_gal + mass_low + mass_mid + mass_high
            if not np.isfinite(total_mass) or total_mass <= 0.0:
                p_star[r_idx] = class_prior_global[0]
                p_galaxy[r_idx] = class_prior_global[1]
                p_qso_low[r_idx] = class_prior_global[2] * global_qso_branch_prior[0]
                p_qso_mid[r_idx] = class_prior_global[2] * global_qso_branch_prior[1]
                p_qso_high[r_idx] = class_prior_global[2] * global_qso_branch_prior[2]
            else:
                p_star[r_idx] = local_mass_star / total_mass
                p_galaxy[r_idx] = local_mass_gal / total_mass
                p_qso_low[r_idx] = mass_low / total_mass
                p_qso_mid[r_idx] = mass_mid / total_mass
                p_qso_high[r_idx] = mass_high / total_mass

        branch_mass = np.column_stack((p_qso_low, p_qso_mid, p_qso_high))
        sum_qso = branch_mass.sum(axis=1)
        branch_norm = np.zeros_like(branch_mass)
        nz = sum_qso > 0.0
        branch_norm[nz] = branch_mass[nz] / sum_qso[nz, None]
        branch_norm[~nz] = 1.0 / 3.0
        branch_concentration = np.max(branch_norm, axis=1)
        branch_entropy = -np.sum(branch_norm * np.log(np.clip(branch_norm, EPS, 1.0)), axis=1)
        max_branch = np.max(branch_mass, axis=1)
        branch_dominant = (branch_mass > (0.5 * max_branch[:, None])).astype(np.int8)

        p_qso_all = p_qso_low + p_qso_mid + p_qso_high

        logit_star = _safe_logit(np.clip(p_star, EPS, 1.0 - EPS))
        logit_galaxy = _safe_logit(np.clip(p_galaxy, EPS, 1.0 - EPS))
        logit_qso_all = _safe_logit(np.clip(p_qso_all, EPS, 1.0 - EPS))
        logit_qso_low = _safe_logit(np.clip(p_qso_low, EPS, 1.0 - EPS))
        logit_qso_mid = _safe_logit(np.clip(p_qso_mid, EPS, 1.0 - EPS))
        logit_qso_high = _safe_logit(np.clip(p_qso_high, EPS, 1.0 - EPS))

        margin_qso_vs_star = np.clip(logit_qso_all - logit_star, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_vs_galaxy = np.clip(logit_qso_all - logit_galaxy, -LOGIT_CLIP, LOGIT_CLIP)
        margin_star_vs_galaxy = np.clip(logit_star - logit_galaxy, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_low_vs_star = np.clip(logit_qso_low - logit_star, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_low_vs_galaxy = np.clip(logit_qso_low - logit_galaxy, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_mid_vs_star = np.clip(logit_qso_mid - logit_star, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_mid_vs_galaxy = np.clip(logit_qso_mid - logit_galaxy, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_high_vs_star = np.clip(logit_qso_high - logit_star, -LOGIT_CLIP, LOGIT_CLIP)
        margin_qso_high_vs_galaxy = np.clip(logit_qso_high - logit_galaxy, -LOGIT_CLIP, LOGIT_CLIP)

    return pd.DataFrame(
        {
            f"{GROUP_PREFIX}_p_star": p_star,
            f"{GROUP_PREFIX}_p_galaxy": p_galaxy,
            f"{GROUP_PREFIX}_p_qso_all": p_qso_all,
            f"{GROUP_PREFIX}_p_qso_low": p_qso_low,
            f"{GROUP_PREFIX}_p_qso_mid": p_qso_mid,
            f"{GROUP_PREFIX}_p_qso_high": p_qso_high,
            f"{GROUP_PREFIX}_logit_star": _safe_logit(np.clip(p_star, EPS, 1.0 - EPS)),
            f"{GROUP_PREFIX}_logit_galaxy": _safe_logit(np.clip(p_galaxy, EPS, 1.0 - EPS)),
            f"{GROUP_PREFIX}_logit_qso_all": logit_qso_all,
            f"{GROUP_PREFIX}_logit_qso_low": logit_qso_low,
            f"{GROUP_PREFIX}_logit_qso_mid": logit_qso_mid,
            f"{GROUP_PREFIX}_logit_qso_high": logit_qso_high,
            f"{GROUP_PREFIX}_margin_qso_vs_star": margin_qso_vs_star,
            f"{GROUP_PREFIX}_margin_qso_vs_galaxy": margin_qso_vs_galaxy,
            f"{GROUP_PREFIX}_margin_star_vs_galaxy": margin_star_vs_galaxy,
            f"{GROUP_PREFIX}_margin_qso_low_vs_star": margin_qso_low_vs_star,
            f"{GROUP_PREFIX}_margin_qso_low_vs_galaxy": margin_qso_low_vs_galaxy,
            f"{GROUP_PREFIX}_margin_qso_mid_vs_star": margin_qso_mid_vs_star,
            f"{GROUP_PREFIX}_margin_qso_mid_vs_galaxy": margin_qso_mid_vs_galaxy,
            f"{GROUP_PREFIX}_margin_qso_high_vs_star": margin_qso_high_vs_star,
            f"{GROUP_PREFIX}_margin_qso_high_vs_galaxy": margin_qso_high_vs_galaxy,
            f"{GROUP_PREFIX}_qso_branch_concentration": branch_concentration,
            f"{GROUP_PREFIX}_qso_branch_entropy": branch_entropy,
            f"{GROUP_PREFIX}_qso_low_dominant": branch_dominant[:, 0],
            f"{GROUP_PREFIX}_qso_mid_dominant": branch_dominant[:, 1],
            f"{GROUP_PREFIX}_qso_high_dominant": branch_dominant[:, 2],
        },
        index=idx,
    )

FEATURE_GROUPS = [
    {
        "name": GROUP_NAME,
        "fn": add_redshift_branch_deconvolved_flux_posteriors,
        "depends_on": [],
        "description": "Builds adaptive overlapping i-band-window Gaussian-mixture deconvolved posteriors over star/galaxy/QSO branches and outputs calibrated posterior logits and ambiguity diagnostics.",
    }
]
```

# Group Code Contract

Return only Python code. Do not use markdown fences.

Define semantic feature-group functions and `FEATURE_GROUPS`.

Generate only the feature-group module: Python definitions for feature-group
preprocessing. A separate fixed runtime wrapper imports this module and is
responsible for logging, timing, dependency ordering, output-column renaming,
final DataFrame assembly, and all non-preprocessing work.

Each feature function must use this signature:

```python
def add_group_name(raw, deps, aux):
    ...
    return new_features
```

Rules:
- `raw` is the raw/base train+test covariate frame without target labels. It includes ID columns.
- `deps` is a dict of dependency outputs by logical group name. Use it only when this group declares dependencies.
- `aux` is an auxiliary DataFrame when available, otherwise empty.
- Return a pandas DataFrame containing only new local feature columns with `index=raw.index`.
- Preserve row count, row order, and index exactly.
- Do not return raw/input columns.
- Do not mutate `raw`, `deps`, or `aux` in place.
- Use clear local feature names. The executor will rename returned columns after the function finishes.
- Outputs may be numeric, boolean, categorical, or string scalar columns. Do not return nested lists, dicts, tuples, or sets.
- You may compute covariate-only train+test statistics from `raw`; do not use target labels, validation labels, model outputs, or leaderboard feedback.
- Do not read project data files, write files, train models, create `main()`, concatenate final blocks, or implement orchestration.
- Do not implement timing decorators or logging wrappers. The group executor logs every group call and duration.
- Top-level code may contain only imports, function definitions, literal constants, and `FEATURE_GROUPS`.
- Do not call functions in top-level assignments. For example, do not write `EDGES = np.array(...)`, `CUTS = pd.IntervalIndex(...)`, or any other assignment whose right-hand side calls a function or constructor.
- If a constant needs conversion to a NumPy/Pandas object, store it as a literal tuple/list at module level and convert it inside the feature function.

Register groups like this:

```python
FEATURE_GROUPS = [
    {
        "name": "group_name",
        "fn": add_group_name,
        "depends_on": [],
        "description": "One sentence describing this feature group.",
    }
]
```

Mode-specific boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not read project data files.
- The fixed wrapper handles all non-preprocessing work.