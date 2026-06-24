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
- Source file: autogluon-007.py
- Failed node: 20260624T063804-1a43aa05-367
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063804-1a43aa05-367/02-code.py", line 1156, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063804-1a43aa05-367/02-code.py", line 424, in add_redshift_branch_deconvolved_flux_posteriors
    class_window_prior = (window_class_counts.astype(np.float64) + ALPHA_PRIOR) / np.where(denom == 0.0, 1.0, denom)
                         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (45,3) (45,) 
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063804-1a43aa05-367/02-code.py", line 1280, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063804-1a43aa05-367/02-code.py", line 1156, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T063804-1a43aa05-367/02-code.py", line 424, in add_redshift_branch_deconvolved_flux_posteriors
    class_window_prior = (window_class_counts.astype(np.float64) + ALPHA_PRIOR) / np.where(denom == 0.0, 1.0, denom)
                         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (45,3) (45,)

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_branch_deconvolved_flux_posteriors
TheML feature group: failed name=redshift_branch_deconvolved_flux_posteriors elapsed_s=1.960 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

FEATURE_GROUP_NAME = "redshift_branch_deconvolved_flux_posteriors"
SEED = 2026
N_CLASSES = 3
CLASS_STAR = 0
CLASS_GALAXY = 1
CLASS_QSO = 2
BRANCH_LOW = 0
BRANCH_MID = 1
BRANCH_HIGH = 2
WINDOW_STEP = 0.4
WINDOW_WIDTH = 0.8
WINDOW_HALF = 0.4
ALPHA_PRIOR = 2.0
BACKOFF_BRANCH = 80.0
N_MIN = 240
N_SPARSE = 150
N_MAD_MIN = 30
MAX_GMM_COMPONENTS = 8
NOISE_FLOOR_SIGMA = 0.05
EPS = 1e-12
LOG2PI = 1.8378770664093453
KAPPA = -0.9210340371976183
JACOBIAN_TO_X = (
    (KAPPA, KAPPA, KAPPA, 0.0),
    (0.0, KAPPA, KAPPA, 0.0),
    (0.0, 0.0, KAPPA, 0.0),
    (0.0, 0.0, 0.0, -KAPPA),
)

def _to_numeric_array(series):
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.float64)
    fill = np.nanmedian(arr[mask])
    arr[~np.isfinite(arr)] = fill
    return arr

def _build_windows(i_min, i_max):
    if not (np.isfinite(i_min) and np.isfinite(i_max)):
        i_min = 0.0
        i_max = 0.0
    if i_max - i_min < WINDOW_WIDTH:
        starts = (i_min,)
    else:
        count = int(np.floor((i_max - i_min) / WINDOW_STEP)) + 1
        starts = tuple(i_min + i * WINDOW_STEP for i in range(count))
    centers = tuple(start + WINDOW_HALF for start in starts)
    return starts, centers

def _assign_windows(i_values, starts, centers):
    w_count = len(starts)
    if w_count == 0:
        return [], []

    centers_arr = np.asarray(centers, dtype=np.float64)
    c0 = centers_arr[0]
    row_members = [[] for _ in range(w_count)]
    row_weights = [[] for _ in range(w_count)]

    for row_idx, i_val in enumerate(i_values):
        base = int(np.floor((i_val - c0) / WINDOW_STEP))
        for w in (base - 1, base, base + 1):
            if 0 <= w < w_count:
                wt = 1.0 - abs(i_val - centers_arr[w]) / WINDOW_HALF
                if wt > 0.0:
                    row_members[w].append(row_idx)
                    row_weights[w].append(wt)

    return row_members, row_weights

def _build_expanded_rows(base_rows, total_rows, target_count):
    all_rows = np.arange(total_rows, dtype=np.int64)
    w_count = len(base_rows)
    if w_count == 0:
        return []

    expanded = []
    for w in range(w_count):
        expanded_rows = np.array([], dtype=np.int64)
        radius = 0
        while radius <= w_count:
            lo = max(0, w - radius)
            hi = min(w_count - 1, w + radius)
            chunks = []
            for k in range(lo, hi + 1):
                rows_k = base_rows[k]
                if rows_k.size:
                    chunks.append(rows_k)
            if chunks:
                merged = np.concatenate(chunks)
                if merged.size >= target_count or (lo == 0 and hi == w_count - 1):
                    expanded_rows = np.unique(merged)
                    break
            radius += 1
        if expanded_rows.size < target_count:
            expanded_rows = all_rows
        expanded.append(expanded_rows)
    return expanded

def _mad(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0) * 1.4826
    mad = np.asarray(mad, dtype=np.float64)
    if mad.size == 0 or np.any(~np.isfinite(mad)):
        return None
    return mad

def _noise_matrix_from_mad(mad):
    if mad is None:
        return None
    mad = np.where(mad < 0.0, 0.0, mad)
    mad = np.maximum(mad, 0.0)
    j = np.asarray(JACOBIAN_TO_X, dtype=np.float64)
    cov_color = np.diag(mad * mad)
    cov_x = j @ cov_color @ j.T
    cov_x = (cov_x + cov_x.T) * 0.5
    np.fill_diagonal(cov_x, np.maximum(np.diag(cov_x), NOISE_FLOOR_SIGMA * NOISE_FLOOR_SIGMA))
    if not np.all(np.isfinite(cov_x)):
        return None
    return cov_x

def _noise_matrix_floor():
    return np.eye(4, dtype=np.float64) * (NOISE_FLOOR_SIGMA * NOISE_FLOOR_SIGMA)

def _noise_for_rows(primary_rows, fallback_rows, all_rows, colors):
    if primary_rows is not None and primary_rows.size >= N_MAD_MIN:
        noise = _noise_matrix_from_mad(_mad(colors[primary_rows]))
        if noise is not None:
            return noise
    if fallback_rows is not None and fallback_rows.size >= N_MAD_MIN:
        noise = _noise_matrix_from_mad(_mad(colors[fallback_rows]))
        if noise is not None:
            return noise
    if all_rows is not None and all_rows.size >= N_MAD_MIN:
        noise = _noise_matrix_from_mad(_mad(colors[all_rows]))
        if noise is not None:
            return noise
    return _noise_matrix_floor()

def _single_component_model(x_block):
    x = np.asarray(x_block, dtype=np.float64)
    d = x.shape[1]
    if x.size == 0:
        mean = np.zeros(d, dtype=np.float64)
        cov = np.eye(d, dtype=np.float64)
    elif x.shape[0] == 1:
        mean = x[0]
        cov = np.eye(d, dtype=np.float64)
    else:
        mean = np.mean(x, axis=0)
        cov = np.cov(x, rowvar=False, bias=True)
        if np.ndim(cov) == 0:
            cov = np.eye(d, dtype=np.float64) * float(cov)
    cov = np.asarray(cov, dtype=np.float64)
    cov = (cov + cov.T) * 0.5
    cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
    cov = np.nan_to_num(cov, nan=1e-6, posinf=1e-6, neginf=1e-6)
    return {
        "weights": np.array([1.0], dtype=np.float64),
        "means": mean[None, :],
        "covariances": cov[None, :, :],
    }

def _fit_gmm_block(x_block, seed, max_components=MAX_GMM_COMPONENTS):
    x = np.asarray(x_block, dtype=np.float64)
    n, d = x.shape
    if n <= 1:
        return _single_component_model(x)

    max_k = min(MAX_GMM_COMPONENTS, n, max_components)
    if max_k < 2:
        return _single_component_model(x)

    best_model = None
    best_bic = np.inf
    for k in range(2, max_k + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=seed + k,
                n_init=2,
                max_iter=250,
            )
            gm.fit(x)
            bic = gm.bic(x)
            if bic < best_bic:
                best_bic = bic
                best_model = gm
        except Exception:
            continue

    if best_model is None:
        return _single_component_model(x)

    return {
        "weights": best_model.weights_.astype(np.float64).copy(),
        "means": best_model.means_.astype(np.float64).copy(),
        "covariances": best_model.covariances_.astype(np.float64).copy(),
    }

def _log_multivariate_normal(x, mean, cov):
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    d = x.shape[1]

    jitter = 1e-8
    chol = None
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(cov)
            break
        except np.linalg.LinAlgError:
            cov = cov + np.eye(d, dtype=np.float64) * jitter
            jitter *= 10.0

    if chol is None:
        cov = np.eye(d, dtype=np.float64)
        chol = np.linalg.cholesky(cov)

    diff = x - mean
    y = np.linalg.solve(chol, diff.T)
    mahal = np.sum(y * y, axis=0)
    logdet = 2.0 * np.sum(np.log(np.clip(np.diag(chol), EPS, None)))
    return -0.5 * (d * LOG2PI + logdet + mahal)

def _log_gaussian_mixture(x, model, noise):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] == 0:
        return np.array([], dtype=np.float64)
    if model is None:
        return np.full(x.shape[0], -np.inf, dtype=np.float64)

    weights = np.asarray(model["weights"], dtype=np.float64)
    means = np.asarray(model["means"], dtype=np.float64)
    covs = np.asarray(model["covariances"], dtype=np.float64)

    component_logs = []
    for k in range(weights.shape[0]):
        w = weights[k]
        if w <= 0:
            continue
        comp_cov = covs[k] + noise
        component_logs.append(np.log(np.clip(w, EPS, 1.0)) + _log_multivariate_normal(x, means[k], comp_cov))

    if not component_logs:
        return np.full(x.shape[0], -np.inf, dtype=np.float64)
    return logsumexp(np.vstack(component_logs).T, axis=1)

def _blended_loglik(x, local_model, local_noise, local_mix, global_model, global_noise):
    if x.shape[0] == 0:
        return np.array([], dtype=np.float64)

    ll_global = _log_gaussian_mixture(x, global_model, global_noise)
    if local_model is None or local_mix <= 0.0:
        return ll_global
    if local_mix >= 0.999999:
        return _log_gaussian_mixture(x, local_model, local_noise)

    ll_local = _log_gaussian_mixture(x, local_model, local_noise)
    a = np.log(np.clip(local_mix, EPS, 1.0)) + ll_local
    b = np.log(np.clip(1.0 - local_mix, EPS, 1.0)) + ll_global
    return logsumexp(np.column_stack((a, b)), axis=1)

def _infer_class_labels(redshift, colors):
    z = np.asarray(redshift, dtype=np.float64)
    n = z.shape[0]
    if n == 0:
        return np.array([], dtype=np.int8)

    feats = np.column_stack((z, colors))
    med = np.nanmedian(feats, axis=0)
    mad = np.nanmedian(np.abs(feats - med), axis=0)
    mad = np.where(mad > 1e-6, mad, 1.0)
    feats = (feats - med) / mad

    try:
        km = MiniBatchKMeans(
            n_clusters=3,
            random_state=SEED,
            batch_size=min(4096, max(256, n)),
            n_init=4,
        )
        cluster = km.fit_predict(feats)

        med_by_cluster = np.full(3, np.inf, dtype=np.float64)
        for c in range(3):
            idx = np.flatnonzero(cluster == c)
            if idx.size:
                med_by_cluster[c] = float(np.nanmedian(z[idx]))

        if not np.all(np.isfinite(med_by_cluster)):
            raise ValueError("Invalid clustering medians")

        order = np.argsort(med_by_cluster)
        cls = np.empty(n, dtype=np.int8)
        cls[cluster == order[0]] = CLASS_STAR
        cls[cluster == order[1]] = CLASS_GALAXY
        cls[cluster == order[2]] = CLASS_QSO
        return cls
    except Exception:
        q1 = float(np.quantile(z, 0.33))
        q2 = float(np.quantile(z, 0.90))
        cls = np.empty(n, dtype=np.int8)
        cls[z <= q1] = CLASS_STAR
        cls[(z > q1) & (z <= q2)] = CLASS_GALAXY
        cls[z > q2] = CLASS_QSO
        return cls

def _assign_qso_branches(redshift, class_ids):
    z = np.asarray(redshift, dtype=np.float64)
    branches = np.full(z.shape[0], -1, dtype=np.int8)
    is_qso = class_ids == CLASS_QSO
    branches[is_qso & (z < 2.2)] = BRANCH_LOW
    branches[is_qso & (z >= 2.2) & (z <= 3.5)] = BRANCH_MID
    branches[is_qso & (z > 3.5)] = BRANCH_HIGH
    return branches

def add_redshift_branch_deconvolved_flux_posteriors(raw, deps, aux):
    if raw is None:
        return pd.DataFrame()
    if raw.shape[0] == 0:
        return pd.DataFrame(index=raw.index)

    i_vals = _to_numeric_array(raw["i"])
    u = _to_numeric_array(raw["u"])
    g = _to_numeric_array(raw["g"])
    r = _to_numeric_array(raw["r"])
    z_mag = _to_numeric_array(raw["z"])
    redshift = _to_numeric_array(raw["redshift"])

    colors = np.column_stack((u - g, g - r, r - i_vals, i_vals - z_mag))
    x = np.column_stack(
        (
            KAPPA * (u - i_vals),
            KAPPA * (g - i_vals),
            KAPPA * (r - i_vals),
            KAPPA * (z_mag - i_vals),
        )
    )

    n_rows = raw.shape[0]
    class_ids = _infer_class_labels(redshift, colors)
    branches = _assign_qso_branches(redshift, class_ids)

    class_counts = np.array(
        [
            np.sum(class_ids == CLASS_STAR),
            np.sum(class_ids == CLASS_GALAXY),
            np.sum(class_ids == CLASS_QSO),
        ],
        dtype=np.float64,
    )
    global_class_prior = (class_counts + ALPHA_PRIOR) / (n_rows + ALPHA_PRIOR * N_CLASSES)

    i_min = float(np.nanmin(i_vals))
    i_max = float(np.nanmax(i_vals))
    starts, centers = _build_windows(i_min, i_max)

    base_rows, base_weights = _assign_windows(i_vals, starts, centers)
    base_rows = [np.asarray(r, dtype=np.int64) for r in base_rows]
    base_weights = [np.asarray(w, dtype=np.float64) for w in base_weights]
    num_windows = len(base_rows)

    expanded_rows = _build_expanded_rows(base_rows, n_rows, N_MIN)

    class_window_rows = [[np.array([], dtype=np.int64) for _ in range(num_windows)] for _ in range(N_CLASSES)]
    qso_branch_window_rows = [[np.array([], dtype=np.int64) for _ in range(num_windows)] for _ in range(3)]
    window_class_counts = np.zeros((num_windows, N_CLASSES), dtype=np.int64)
    window_branch_counts = np.zeros((num_windows, 3), dtype=np.int64)

    for w, rows in enumerate(base_rows):
        if rows.size == 0:
            continue
        cids = class_ids[rows]
        window_class_counts[w] = np.bincount(cids, minlength=N_CLASSES)

        for c in range(N_CLASSES):
            class_window_rows[c][w] = rows[cids == c]

        q_rows = rows[cids == CLASS_QSO]
        if q_rows.size:
            bvals = branches[q_rows]
            for b in range(3):
                qso_branch_window_rows[b][w] = q_rows[bvals == b]
            for b in range(3):
                window_branch_counts[w, b] = np.sum(bvals == b)

    class_window_total = window_class_counts.sum(axis=1)
    denom = class_window_total.astype(np.float64) + ALPHA_PRIOR * N_CLASSES
    class_window_prior = (window_class_counts.astype(np.float64) + ALPHA_PRIOR) / np.where(denom == 0.0, 1.0, denom)
    class_window_prior[class_window_total == 0] = global_class_prior

    global_qso_branch_counts = np.array(
        [
            np.sum((class_ids == CLASS_QSO) & (branches == BRANCH_LOW)),
            np.sum((class_ids == CLASS_QSO) & (branches == BRANCH_MID)),
            np.sum((class_ids == CLASS_QSO) & (branches == BRANCH_HIGH)),
        ],
        dtype=np.float64,
    )
    global_qso_count = float(np.sum(global_qso_branch_counts))
    if global_qso_count > 0.0:
        global_branch_prior = (global_qso_branch_counts + ALPHA_PRIOR) / (global_qso_count + 3.0 * ALPHA_PRIOR)
    else:
        global_branch_prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    qso_window_total = window_branch_counts.sum(axis=1)
    branch_local = np.zeros((num_windows, 3), dtype=np.float64)
    branch_prior = np.zeros((num_windows, 3), dtype=np.float64)
    for w in range(num_windows):
        total = qso_window_total[w]
        if total > 0.0:
            branch_local[w] = (window_branch_counts[w].astype(np.float64) + ALPHA_PRIOR) / (total + 3.0 * ALPHA_PRIOR)
        else:
            branch_local[w] = global_branch_prior
        lam = window_branch_counts[w].astype(np.float64) / (window_branch_counts[w].astype(np.float64) + BACKOFF_BRANCH)
        branch_prior[w] = lam * branch_local[w] + (1.0 - lam) * global_branch_prior

    global_class_rows = [
        np.flatnonzero(class_ids == CLASS_STAR),
        np.flatnonzero(class_ids == CLASS_GALAXY),
        np.flatnonzero(class_ids == CLASS_QSO),
    ]

    global_class_models = []
    for c in range(N_CLASSES):
        idx = global_class_rows[c]
        model = _fit_gmm_block(x[idx], SEED + c * 11 + 17, max_components=MAX_GMM_COMPONENTS)
        noise = _noise_for_rows(
            idx if idx.size >= N_MAD_MIN else np.array([], dtype=np.int64),
            idx,
            idx,
            colors,
        )
        global_class_models.append((model, noise))

    qso_rows = global_class_rows[CLASS_QSO]
    global_qso_branch_models = []
    for b in range(3):
        if qso_rows.size:
            bidx = qso_rows[branches[qso_rows] == b]
        else:
            bidx = np.array([], dtype=np.int64)
        model = (
            _fit_gmm_block(x[bidx], SEED + 100 + b * 13 + 3, max_components=MAX_GMM_COMPONENTS)
            if bidx.size >= 2
            else global_class_models[CLASS_QSO][0]
        )
        noise = _noise_for_rows(
            bidx if bidx.size >= N_MAD_MIN else np.array([], dtype=np.int64),
            bidx,
            qso_rows,
            colors,
        )
        global_qso_branch_models.append((model, noise))

    class_window_models = [[None for _ in range(num_windows)] for _ in range(N_CLASSES)]
    qso_window_models = [[None for _ in range(num_windows)] for _ in range(3)]

    for c in range(N_CLASSES):
        g_model, g_noise = global_class_models[c]
        fallback_rows = global_class_rows[c]
        for w in range(num_windows):
            rows = expanded_rows[w]
            c_rows = rows[class_ids[rows] == c]
            if c_rows.size >= 2:
                if c_rows.size >= N_SPARSE:
                    kmax = MAX_GMM_COMPONENTS
                else:
                    kmax = max(2, min(MAX_GMM_COMPONENTS, c_rows.size // 30 + 2))
                local_model = _fit_gmm_block(x[c_rows], SEED + c * 53 + w, max_components=kmax)
                mix = min(1.0, c_rows.size / float(N_SPARSE))
            else:
                local_model = None
                mix = 0.0
            local_noise = _noise_for_rows(c_rows, class_window_rows[c][w], fallback_rows, colors)
            class_window_models[c][w] = {
                "local_model": local_model,
                "local_noise": local_noise,
                "local_mix": mix,
            }

    for b in range(3):
        for w in range(num_windows):
            rows = expanded_rows[w]
            b_rows = rows[(class_ids[rows] == CLASS_QSO) & (branches[rows] == b)]
            if b_rows.size >= 2:
                if b_rows.size >= N_SPARSE:
                    kmax = MAX_GMM_COMPONENTS
                else:
                    kmax = max(2, min(MAX_GMM_COMPONENTS, b_rows.size // 30 + 2))
                local_model = _fit_gmm_block(x[b_rows], SEED + 700 + b * 29 + w, max_components=kmax)
                mix = min(1.0, b_rows.size / float(N_SPARSE))
            else:
                local_model = None
                mix = 0.0
            local_noise = _noise_for_rows(
                b_rows,
                qso_branch_window_rows[b][w],
                qso_rows,
                colors,
            )
            qso_window_models[b][w] = {
                "local_model": local_model,
                "local_noise": local_noise,
                "local_mix": mix,
            }

    log_class_mass = np.full((n_rows, N_CLASSES), -np.inf, dtype=np.float64)
    log_qso_branch_mass = np.full((n_rows, 3), -np.inf, dtype=np.float64)

    for w in range(num_windows):
        rows = base_rows[w]
        if rows.size == 0:
            continue
        x_sub = x[rows]
        w_log = np.log(np.clip(base_weights[w], EPS, 1.0))

        for c in range(N_CLASSES):
            cls_info = class_window_models[c][w]
            g_model, g_noise = global_class_models[c]
            cls_ll = _blended_loglik(
                x_sub,
                cls_info["local_model"],
                cls_info["local_noise"],
                cls_info["local_mix"],
                g_model,
                g_noise,
            )
            term = np.log(np.clip(class_window_prior[w, c], EPS, 1.0)) + cls_ll + w_log
            log_class_mass[rows, c] = np.logaddexp(log_class_mass[rows, c], term)

            if c == CLASS_QSO:
                q_prior = np.log(np.clip(class_window_prior[w, c], EPS, 1.0))
                for b in range(3):
                    q_info = qso_window_models[b][w]
                    g_b_model, g_b_noise = global_qso_branch_models[b]
                    q_ll = _blended_loglik(
                        x_sub,
                        q_info["local_model"],
                        q_info["local_noise"],
                        q_info["local_mix"],
                        g_b_model,
                        g_b_noise,
                    )
                    q_term = (
                        q_prior
                        + np.log(np.clip(branch_prior[w, b], EPS, 1.0))
                        + q_ll
                        + w_log
                    )
                    log_qso_branch_mass[rows, b] = np.logaddexp(log_qso_branch_mass[rows, b], q_term)

    total_log = logsumexp(log_class_mass, axis=1)
    finite_total = np.isfinite(total_log)

    class_probs = np.zeros((n_rows, N_CLASSES), dtype=np.float64)
    if np.any(finite_total):
        class_probs[finite_total] = np.exp(log_class_mass[finite_total] - total_log[finite_total][:, None])
    if np.any(~finite_total):
        class_probs[~finite_total] = 1.0 / N_CLASSES

    qso_raw_mass = np.zeros((n_rows, 3), dtype=np.float64)
    if np.any(finite_total):
        qso_raw_mass[finite_total] = np.exp(log_qso_branch_mass[finite_total] - total_log[finite_total][:, None])

    qso_branch_mass = qso_raw_mass.sum(axis=1)
    qso_branch_norm = np.zeros((n_rows, 3), dtype=np.float64)
    nz = qso_branch_mass > 0.0
    qso_branch_norm[nz] = qso_raw_mass[nz] / qso_branch_mass[nz][:, None]

    qso_concentration = np.zeros(n_rows, dtype=np.float64)
    if np.any(nz):
        qso_concentration[nz] = np.max(qso_branch_norm[nz], axis=1)

    qso_entropy = np.zeros(n_rows, dtype=np.float64)
    if np.any(nz):
        qso_entropy[nz] = -np.sum(
            qso_branch_norm[nz] * np.log(np.clip(qso_branch_norm[nz], EPS, 1.0)),
            axis=1,
        )

    p_star = class_probs[:, CLASS_STAR]
    p_galaxy = class_probs[:, CLASS_GALAXY]
    p_qso = class_probs[:, CLASS_QSO]
    p_q_low = qso_raw_mass[:, BRANCH_LOW]
    p_q_mid = qso_raw_mass[:, BRANCH_MID]
    p_q_high = qso_raw_mass[:, BRANCH_HIGH]

    q_peak = np.maximum.reduce([p_q_low, p_q_mid, p_q_high])
    threshold = 0.5 * q_peak
    q_dominant_low = (p_q_low > threshold).astype(np.int8)
    q_dominant_mid = (p_q_mid > threshold).astype(np.int8)
    q_dominant_high = (p_q_high > threshold).astype(np.int8)

    logit_cols = {
        "logit_star": np.log((p_star + EPS) / (1.0 - p_star + EPS)),
        "logit_galaxy": np.log((p_galaxy + EPS) / (1.0 - p_galaxy + EPS)),
        "logit_qso": np.log((p_qso + EPS) / (1.0 - p_qso + EPS)),
        "margin_star_vs_galaxy": np.log((p_star + EPS) / (p_galaxy + EPS)),
        "margin_star_vs_qso": np.log((p_star + EPS) / (p_qso + EPS)),
        "margin_galaxy_vs_qso": np.log((p_galaxy + EPS) / (p_qso + EPS)),
        "qso_low_vs_star": np.log((p_q_low + EPS) / (p_star + EPS)),
        "qso_mid_vs_star": np.log((p_q_mid + EPS) / (p_star + EPS)),
        "qso_high_vs_star": np.log((p_q_high + EPS) / (p_star + EPS)),
        "qso_low_vs_galaxy": np.log((p_q_low + EPS) / (p_galaxy + EPS)),
        "qso_mid_vs_galaxy": np.log((p_q_mid + EPS) / (p_galaxy + EPS)),
        "qso_high_vs_galaxy": np.log((p_q_high + EPS) / (p_galaxy + EPS)),
    }

    features = pd.DataFrame(
        {
            "posterior_star": p_star,
            "posterior_galaxy": p_galaxy,
            "posterior_qso": p_qso,
            "posterior_qso_low": p_q_low,
            "posterior_qso_mid": p_q_mid,
            "posterior_qso_high": p_q_high,
            "qso_concentration": qso_concentration,
            "qso_entropy": qso_entropy,
            "qso_low_dominant": q_dominant_low,
            "qso_mid_dominant": q_dominant_mid,
            "qso_high_dominant": q_dominant_high,
            **logit_cols,
        },
        index=raw.index,
    )

    feature_log_cols = list(logit_cols.keys())
    features[feature_log_cols] = features[feature_log_cols].clip(-12.0, 12.0)
    return features

FEATURE_GROUPS = [
    {
        "name": FEATURE_GROUP_NAME,
        "fn": add_redshift_branch_deconvolved_flux_posteriors,
        "depends_on": [],
        "description": "Generate adaptive windowed deconvolved relative-flux posteriors, branch-aware class likelihoods, and calibrated log margin features.",
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