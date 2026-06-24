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
- Hypothesis ID: 000038
- Source file: autogluon-002.py
- Failed node: 20260624T061902-b51a857b-346
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 974, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 317, in add_photoz_trajectory_geometry
    cov = _huber_covariance(colors_train, w, mu)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 119, in _huber_covariance
    md2 = np.einsum("ij,ij->i", xc @ inv, xc)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/.venv/lib/python3.12/site-packages/numpy/_core/einsumfunc.py", line 1429, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 653, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 180 seconds
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 1098, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 974, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 317, in add_photoz_trajectory_geometry
    cov = _huber_covariance(colors_train, w, mu)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 119, in _huber_covariance
    md2 = np.einsum("ij,ij->i", xc @ inv, xc)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/.venv/lib/python3.12/site-packages/numpy/_core/einsumfunc.py", line 1429, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061902-b51a857b-346/02-code.py", line 653, in _raise_timeout
    raise TimeoutError(f"AutoGluon preprocess exceeded {seconds} seconds")
TimeoutError: AutoGluon preprocess exceeded 180 seconds

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=photoz_trajectory_geometry
TheML feature group: failed name=photoz_trajectory_geometry elapsed_s=179.953 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_TRAJ_KNOTS = 180
_TRAJ_MIN_EFFECTIVE = 1500.0
_TRAJ_BW_INIT = 0.08
_TRAJ_BW_MAX = 1.0
_TRAJ_EPS = 1e-12
_TRAJ_SHRINK = 0.85
_TRAJ_TRAIN_ID_MAX = 577346
_TRAJ_COV_ENTRIES = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3),
    (3, 3),
)

def _weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return 0.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    half = cumulative[-1] * 0.5
    return float(values[np.searchsorted(cumulative, half)])

def _effective_sample_size(weights):
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = np.sum(weights)
    if total <= _TRAJ_EPS:
        return 0.0
    return float((total * total) / (np.dot(weights, weights) + _TRAJ_EPS))

def _huber_location_1d(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return 0.0

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = np.maximum(weights[valid], 0.0)
    if values.size == 0:
        return 0.0

    location = _weighted_median(values, weights)
    for _ in range(40):
        resid = values - location
        scale = 1.4826 * _weighted_median(np.abs(resid), weights) + _TRAJ_EPS
        c = 1.345 * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            robust_w = np.minimum(1.0, c / (np.abs(resid) + _TRAJ_EPS))
        robust_w = np.where(np.isfinite(robust_w), robust_w, 0.0)
        next_w = weights * robust_w
        denominator = np.sum(next_w)
        if denominator <= _TRAJ_EPS:
            break
        next_loc = float(np.dot(next_w, values) / denominator)
        if np.abs(next_loc - location) < 1e-8:
            location = next_loc
            break
        location = next_loc
    return float(location)

def _huber_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if values.size == 0:
        return np.zeros(values.shape[1] if values.ndim > 1 else 1, dtype=float)
    if values.ndim == 1:
        return np.array([_huber_location_1d(values, weights)], dtype=float)
    out = np.zeros(values.shape[1], dtype=float)
    for j in range(values.shape[1]):
        out[j] = _huber_location_1d(values[:, j], weights)
    return out

def _huber_covariance(values, weights, center):
    x = np.asarray(values, dtype=float)
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if x.size == 0:
        p = x.shape[1] if x.ndim > 1 else 1
        return np.eye(p, dtype=float) * _TRAJ_EPS
    valid = np.all(np.isfinite(x), axis=1) & np.isfinite(w) & (w > 0)
    x = x[valid]
    w = w[valid]
    if x.size == 0:
        p = x.shape[1] if x.ndim > 1 else 1
        return np.eye(p, dtype=float) * _TRAJ_EPS
    center = np.asarray(center, dtype=float)
    xc = x - center[np.newaxis, :]
    weights = w.copy()
    p = xc.shape[1]
    cov = np.eye(p, dtype=float)
    for _ in range(8):
        wsqrt = np.sqrt(np.maximum(weights, 0.0))
        denom = np.sum(weights) + _TRAJ_EPS
        cov = (xc * wsqrt[:, None]).T @ (xc * wsqrt[:, None]) / denom
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(p, dtype=float) * _TRAJ_EPS
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)

        md2 = np.einsum("ij,ij->i", xc @ inv, xc)
        valid_md = np.isfinite(md2) & (md2 > 0)
        if not np.any(valid_md):
            break
        med = float(np.median(md2[valid_md]))
        if med <= _TRAJ_EPS:
            break
        c = 4.685 * np.sqrt(med)
        c2 = c * c + _TRAJ_EPS
        robust_w = np.where(md2 <= c2, 1.0, c2 / (md2 + _TRAJ_EPS))
        weights = w * np.maximum(robust_w, 0.0)

    return cov

def _shrink_covariance(cov):
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[0]
    tr = np.trace(cov)
    if not np.isfinite(tr) or tr <= 0:
        tr = d
    target = (tr / d) if d > 0 else 1.0
    shr = _TRAJ_SHRINK * cov + (1.0 - _TRAJ_SHRINK) * np.eye(d, dtype=float) * target
    vals, vecs = np.linalg.eigh(0.5 * (shr + shr.T))
    vals = np.maximum(vals, 1e-6)
    return (vecs * vals[np.newaxis, :]) @ vecs.T

def _safe_cholesky(matrix):
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (4, 4):
        return None
    mat = 0.5 * (mat + mat.T)
    jitter = 1e-9
    for _ in range(10):
        try:
            return np.linalg.cholesky(mat + jitter * np.eye(4))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return None

def _fit_curve_model(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return ("constant", 0.0)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ux, inv = np.unique(x, return_inverse=True)
    if ux.size != x.size:
        y_sum = np.bincount(inv, weights=y)
        cnt = np.bincount(inv)
        y = y_sum / np.maximum(cnt, 1.0)
        x = ux

    for i in range(1, x.size):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-12

    if x.size == 1:
        return ("constant", float(y[0]))
    if x.size == 2:
        return ("constant", float(np.mean(y)))

    if x.size < 4:
        d1 = np.gradient(y, x, edge_order=2)
        d2 = np.gradient(d1, x, edge_order=2)
        return ("fallback", (x, y, d1, d2))

    try:
        from scipy.interpolate import UnivariateSpline

        spline = UnivariateSpline(x, y, k=3, s=max(1.0, 0.2 * len(x)))
        return ("scipy", (spline, spline.derivative(1), spline.derivative(2)))
    except Exception:
        d1 = np.gradient(y, x, edge_order=2)
        d2 = np.gradient(d1, x, edge_order=2)
        return ("fallback", (x, y, d1, d2))

def _eval_curve_model(model, x, deriv=0):
    xq = np.asarray(x, dtype=float)
    kind = model[0]
    if kind == "constant":
        return np.full_like(xq, float(model[1]), dtype=float)
    if kind == "scipy":
        spline, d1, d2 = model[1]
        if deriv == 0:
            return np.asarray(spline(xq), dtype=float)
        if deriv == 1:
            return np.asarray(d1(xq), dtype=float)
        return np.asarray(d2(xq), dtype=float)
    xk, yk, d1k, d2k = model[1]
    if deriv == 0:
        return np.interp(xq, xk, yk)
    if deriv == 1:
        return np.interp(xq, xk, d1k)
    return np.interp(xq, xk, d2k)

def _infer_train_mask(raw, aux):
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        for key in ("is_train", "is_train_row", "is_training", "train", "train_mask"):
            if key in aux.columns:
                vals = aux[key]
                if vals.dtype == bool:
                    return np.asarray(vals.to_numpy(dtype=bool), dtype=bool)
                as_num = pd.to_numeric(vals, errors="coerce")
                if as_num.notna().all():
                    if set(np.unique(np.floor(as_num.to_numpy()))).issubset({0.0, 1.0}):
                        return np.asarray(as_num.to_numpy() > 0.5, dtype=bool)

    if "id" not in raw.columns:
        return np.ones(len(raw), dtype=bool)

    ids = pd.to_numeric(raw["id"], errors="coerce")
    if not ids.notna().all():
        return np.ones(len(raw), dtype=bool)
    id_vals = ids.to_numpy(dtype=float)
    if np.min(id_vals) == 0 and np.max(id_vals) >= 824781 and len(raw) == 824782:
        return id_vals <= _TRAJ_TRAIN_ID_MAX
    return np.ones(len(raw), dtype=bool)

def add_photoz_trajectory_geometry(raw, deps, aux):
    z = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z_band = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float)

    colors = np.column_stack((u - g, g - r, r - i, i - z_band))
    n_rows = len(raw)

    train_mask = _infer_train_mask(raw, aux)
    if len(train_mask) != n_rows:
        train_mask = np.ones(n_rows, dtype=bool)
    if not np.any(train_mask):
        train_mask = np.ones(n_rows, dtype=bool)

    train_valid = np.isfinite(z) & np.all(np.isfinite(colors), axis=1) & train_mask
    z_train = z[train_valid]
    colors_train = colors[train_valid]

    if z_train.size == 0:
        train_mask = np.ones(n_rows, dtype=bool)
        train_valid = np.isfinite(z) & np.all(np.isfinite(colors), axis=1)
        z_train = z[train_valid]
        colors_train = colors[train_valid]

    if z_train.size == 0:
        z_train = z
        colors_train = colors

    z_train = np.asarray(z_train, dtype=float)
    colors_train = np.asarray(colors_train, dtype=float)

    z_min = float(np.nanmin(z_train))
    z_max = float(np.nanmax(z_train))
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))

    global_mu = _huber_mean(colors_train, np.ones(len(z_train), dtype=float))
    global_cov = _huber_covariance(colors_train, np.ones(len(z_train), dtype=float), global_mu)
    global_cov = _shrink_covariance(global_cov)

    qlevels = np.linspace(0.005, 0.995, _TRAJ_KNOTS)
    knots = np.quantile(z_train, qlevels)
    knots = np.clip(knots, z_min, z_max)
    for k in range(1, knots.size):
        if knots[k] <= knots[k - 1]:
            knots[k] = knots[k - 1] + 1e-12 * (k + 1)

    mu_knots = np.zeros((_TRAJ_KNOTS, 4), dtype=float)
    cov_knots = np.zeros((_TRAJ_KNOTS, 4, 4), dtype=float)

    for i_k, knot in enumerate(knots):
        h = _TRAJ_BW_INIT
        w = np.exp(-0.5 * np.square((z_train - knot) / h))
        n_eff = _effective_sample_size(w)
        while n_eff < _TRAJ_MIN_EFFECTIVE and h < (_TRAJ_BW_MAX - _TRAJ_EPS):
            h = min(h * 2.0, _TRAJ_BW_MAX)
            w = np.exp(-0.5 * np.square((z_train - knot) / h))
            n_eff = _effective_sample_size(w)

        if n_eff < _TRAJ_MIN_EFFECTIVE:
            mu = global_mu
            cov = global_cov
        else:
            mu = _huber_mean(colors_train, w)
            cov = _huber_covariance(colors_train, w, mu)
            cov = _shrink_covariance(cov)

        mu_knots[i_k, :] = mu
        cov_knots[i_k, :, :] = cov

    mu_models = [
        _fit_curve_model(knots, mu_knots[:, dim])
        for dim in range(4)
    ]
    cov_entries_knots = np.zeros((_TRAJ_KNOTS, len(_TRAJ_COV_ENTRIES)), dtype=float)
    for e_idx, (a_idx, b_idx) in enumerate(_TRAJ_COV_ENTRIES):
        cov_entries_knots[:, e_idx] = cov_knots[:, a_idx, b_idx]
    cov_models = [
        _fit_curve_model(knots, cov_entries_knots[:, e_idx])
        for e_idx in range(len(_TRAJ_COV_ENTRIES))
    ]

    dmu_knots = np.zeros_like(mu_knots)
    ddmu_knots = np.zeros_like(mu_knots)
    for dim, model in enumerate(mu_models):
        dmu_knots[:, dim] = _eval_curve_model(model, knots, deriv=1)
        ddmu_knots[:, dim] = _eval_curve_model(model, knots, deriv=2)

    sigma_t_knots = np.ones(_TRAJ_KNOTS, dtype=float)
    sigma_e_knots = np.ones(_TRAJ_KNOTS, dtype=float) * np.sqrt(3.0)
    sigma_m_knots = np.ones(_TRAJ_KNOTS, dtype=float)

    for i_k in range(_TRAJ_KNOTS):
        cov_k = cov_knots[i_k]
        v_k = dmu_knots[i_k]
        a_k = ddmu_knots[i_k]
        try:
            inv_cov_k = np.linalg.pinv(cov_k + np.eye(4) * 1e-9)
            vv = float(np.dot(v_k, inv_cov_k @ v_k))
            nv = float(np.dot(v_k, v_k))
            if nv > _TRAJ_EPS:
                sigma_t_knots[i_k] = np.sqrt(max(vv / nv, _TRAJ_EPS))
            aa = float(np.dot(a_k, a_k))
            if aa > _TRAJ_EPS:
                sigma_m_knots[i_k] = np.sqrt(max(np.dot(a_k, inv_cov_k @ a_k) / aa, _TRAJ_EPS))
        except Exception:
            pass

    sigma_t_knots = np.maximum(sigma_t_knots, 1e-6)
    sigma_e_knots = np.maximum(sigma_e_knots, 1e-6)
    sigma_m_knots = np.maximum(sigma_m_knots, 1e-6)

    sigma_t_model = _fit_curve_model(knots, sigma_t_knots)
    sigma_e_model = _fit_curve_model(knots, sigma_e_knots)
    sigma_m_model = _fit_curve_model(knots, sigma_m_knots)

    z_eval = np.clip(z, z_min, z_max)
    mu_eval = np.column_stack([_eval_curve_model(m, z_eval, deriv=0) for m in mu_models])
    v_eval = np.column_stack([_eval_curve_model(m, z_eval, deriv=1) for m in mu_models])
    a_eval = np.column_stack([_eval_curve_model(m, z_eval, deriv=2) for m in mu_models])
    cov_eval = np.column_stack([_eval_curve_model(m, z_eval, deriv=0) for m in cov_models])

    sigma_t_eval = _eval_curve_model(sigma_t_model, z_eval, deriv=0)
    sigma_e_eval = _eval_curve_model(sigma_e_model, z_eval, deriv=0)
    sigma_m_eval = _eval_curve_model(sigma_m_model, z_eval, deriv=0)

    sigma_t_eval = np.maximum(sigma_t_eval, 1e-6)
    sigma_e_eval = np.maximum(sigma_e_eval, 1e-6)
    sigma_m_eval = np.maximum(sigma_m_eval, 1e-6)

    outside_mask = (z < z_min) | (z > z_max)
    alpha = np.ones(n_rows, dtype=float)
    if np.any(outside_mask):
        edge = np.where(z < z_min, z_min, z_max)
        alpha[outside_mask] = np.minimum(1.0, np.abs(z[outside_mask] - edge[outside_mask]) / 0.03)

    a_eval = a_eval.copy()
    if np.any(outside_mask):
        a_eval[outside_mask, :] = 0.0

    T = np.zeros(n_rows, dtype=float)
    E = np.zeros(n_rows, dtype=float)
    M = np.zeros(n_rows, dtype=float)

    for i in range(n_rows):
        c00, c01, c02, c03, c11, c12, c13, c22, c23, c33 = cov_eval[i]
        cov_i = np.array(
            [
                [c00, c01, c02, c03],
                [c01, c11, c12, c13],
                [c02, c12, c22, c23],
                [c03, c13, c23, c33],
            ],
            dtype=float,
        )
        L = _safe_cholesky(cov_i)
        if L is None:
            continue

        mu_i = mu_eval[i]
        v_i = v_eval[i]
        a_i = a_eval[i]
        diff = colors[i] - mu_i

        try:
            y = np.linalg.solve(L, diff)
        except Exception:
            continue

        try:
            t = np.linalg.solve(L, v_i)
        except Exception:
            t = np.zeros(4, dtype=float)

        t_norm = np.linalg.norm(t)
        if t_norm > _TRAJ_EPS:
            u_t = t / t_norm
            T[i] = float(np.dot(y, u_t))
        else:
            T[i] = 0.0

        E[i] = float(np.dot(y, y) - T[i] * T[i])

        a_norm = float(np.linalg.norm(a_i))
        if a_norm > _TRAJ_EPS:
            try:
                cov_inv_a = np.linalg.solve(L.T, np.linalg.solve(L, a_i))
                M[i] = float(np.dot(diff, cov_inv_a) / (a_norm + 1e-6))
            except Exception:
                M[i] = 0.0
        else:
            M[i] = 0.0

        if outside_mask[i]:
            T[i] *= alpha[i]
            M[i] *= alpha[i]

    lowz_scale = np.where(z < 0.01, np.clip(z, 0.0, 0.01) / 0.01, 1.0)
    T *= lowz_scale
    M *= lowz_scale
    E *= lowz_scale

    T_scaled = T / sigma_t_eval
    E_scaled = E / sigma_e_eval
    M_scaled = M / sigma_m_eval

    T_scaled *= lowz_scale
    E_scaled *= lowz_scale
    M_scaled *= lowz_scale
    if np.any(outside_mask):
        T_scaled[outside_mask] *= alpha[outside_mask]
        M_scaled[outside_mask] *= alpha[outside_mask]

    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    T_scaled = np.nan_to_num(T_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    E_scaled = np.nan_to_num(E_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    M_scaled = np.nan_to_num(M_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(
        {
            "photoz_trajectory_geometry_T": T,
            "photoz_trajectory_geometry_E": E,
            "photoz_trajectory_geometry_M": M,
            "photoz_trajectory_geometry_T_scaled": T_scaled,
            "photoz_trajectory_geometry_E_scaled": E_scaled,
            "photoz_trajectory_geometry_M_scaled": M_scaled,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "photoz_trajectory_geometry",
        "fn": add_photoz_trajectory_geometry,
        "depends_on": [],
        "description": "Computes redshift-dependent manifold-trajectory color diagnostics from adaptive weighted Huber tracks in u-g, g-r, r-i, i-z space.",
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