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

# External Data Description for star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

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
- Hypothesis ID: 000046
- Source file: autogluon-003.py
- Failed node: 20260623T165254-95af45e8-221
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 725, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 230, in add_local_manifold_codimension
    f1 = _compute_local_manifold_features(color_with_z, "local_manifold_codimension_colors_with_redshift", raw.index)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 154, in _compute_local_manifold_features
    coeff = np.einsum("mij,mj->mi", basis2, x_rel)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/einsumfunc.py", line 1429, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (4096,5,2)->(4096,5,2) (4096,5)->(4096,newaxis,5) 
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 849, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 725, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 230, in add_local_manifold_codimension
    f1 = _compute_local_manifold_features(color_with_z, "local_manifold_codimension_colors_with_redshift", raw.index)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T165254-95af45e8-221/02-code.py", line 154, in _compute_local_manifold_features
    coeff = np.einsum("mij,mj->mi", basis2, x_rel)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/einsumfunc.py", line 1429, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (4096,5,2)->(4096,5,2) (4096,5)->(4096,newaxis,5)

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=local_manifold_codimension
TheML feature group: failed name=local_manifold_codimension elapsed_s=36.404 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_NEIGHBOR_K = 64
_EPS = 1e-12
_MAD_SCALE = 1.4826
_CHUNK_SIZE = 4096
_SINGULAR_RATIO_THRESHOLD = 1e12
_LOW_Q = 0.05
_HIGH_Q = 0.95

def _safe_divide(numerator, denominator):
    return np.asarray(numerator, dtype=np.float64) / np.maximum(np.asarray(denominator, dtype=np.float64), _EPS)

def _robust_scale(values):
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    iqr = np.percentile(values, (25.0, 75.0), axis=0)
    scale = np.where(mad > _EPS, _MAD_SCALE * mad, (iqr[1] - iqr[0]) / 1.349)
    scale = np.where((np.isfinite(scale)) & (scale > _EPS), scale, 1.0)
    return (values - median) / scale

def _fallback_from_global_eigs(eigenvalues):
    evals = np.maximum(np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1], _EPS)
    eval_sum = float(np.sum(evals))
    if eval_sum <= _EPS:
        eval_sum = _EPS

    return {
        "r2": float(_safe_divide(evals[0] + evals[1], eval_sum)),
        "residual2": 0.0,
        "ratio12": float(_safe_divide(evals[0], evals[1])),
        "ratio23": float(_safe_divide(evals[1], evals[2])),
        "ratio4sum": float(_safe_divide(evals[3], eval_sum)),
    }

def _compute_local_manifold_features(X_raw, prefix, index):
    X = np.asarray(X_raw, dtype=np.float64, order="C")
    n_rows, d = X.shape

    col_r2 = f"{prefix}_r2"
    col_resid = f"{prefix}_residual2"
    col_ratio12 = f"{prefix}_ratio_lambda1_lambda2"
    col_ratio23 = f"{prefix}_ratio_lambda2_lambda3"
    col_ratio4 = f"{prefix}_ratio_lambda4_sum"
    col_knn_k = f"{prefix}_knn_kdist"
    col_knn_med = f"{prefix}_knn_median_radius"
    col_knn_density = f"{prefix}_knn_density"
    col_box_support = f"{prefix}_inbox_support"
    col_fallback = f"{prefix}_fallback_flag"

    if n_rows <= 0:
        return pd.DataFrame(
            {
                col_r2: np.array([], dtype=np.float32),
                col_resid: np.array([], dtype=np.float32),
                col_ratio12: np.array([], dtype=np.float32),
                col_ratio23: np.array([], dtype=np.float32),
                col_ratio4: np.array([], dtype=np.float32),
                col_knn_k: np.array([], dtype=np.float32),
                col_knn_med: np.array([], dtype=np.float32),
                col_knn_density: np.array([], dtype=np.float32),
                col_box_support: np.array([], dtype=np.float32),
                col_fallback: np.array([], dtype=np.int8),
            },
            index=index,
        )

    if n_rows == 1 or d < 2:
        zeros = np.zeros(n_rows, dtype=np.float32)
        ones = np.ones(n_rows, dtype=np.int8)
        return pd.DataFrame(
            {
                col_r2: zeros,
                col_resid: zeros,
                col_ratio12: zeros,
                col_ratio23: zeros,
                col_ratio4: zeros,
                col_knn_k: zeros,
                col_knn_med: zeros,
                col_knn_density: zeros,
                col_box_support: zeros,
                col_fallback: ones,
            },
            index=index,
        )

    X = _robust_scale(X)
    k = min(_NEIGHBOR_K, n_rows - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1 : k + 1].astype(np.float32, copy=False)
    indices = indices[:, 1 : k + 1].astype(np.int64, copy=False)

    global_cov = np.cov(X, rowvar=False, bias=False)
    global_fallback = _fallback_from_global_eigs(np.linalg.eigvalsh(global_cov))
    global_knn_k = float(np.median(distances[:, -1]))
    global_knn_med = float(np.median(np.median(distances, axis=1)))
    global_density = float(np.median(1.0 / (distances[:, -1] + _EPS)))
    global_low = np.quantile(X, _LOW_Q, axis=0)
    global_high = np.quantile(X, _HIGH_Q, axis=0)
    global_box_support = float(np.mean(np.mean((X >= global_low) & (X <= global_high), axis=1)))

    r2 = np.full(n_rows, global_fallback["r2"], dtype=np.float32)
    residual2 = np.full(n_rows, global_fallback["residual2"], dtype=np.float32)
    ratio12 = np.full(n_rows, global_fallback["ratio12"], dtype=np.float32)
    ratio23 = np.full(n_rows, global_fallback["ratio23"], dtype=np.float32)
    ratio4sum = np.full(n_rows, global_fallback["ratio4sum"], dtype=np.float32)
    knn_kdist = np.empty(n_rows, dtype=np.float32)
    knn_med = np.empty(n_rows, dtype=np.float32)
    knn_density = np.empty(n_rows, dtype=np.float32)
    inbox_support = np.empty(n_rows, dtype=np.float32)
    fallback_flag = np.ones(n_rows, dtype=np.int8)

    for start in range(0, n_rows, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, n_rows)
        rows = slice(start, end)
        local_indices = indices[rows]
        local_dist = distances[rows]
        x_center = X[rows]
        neigh = X[local_indices]

        centroid = np.mean(neigh, axis=1)
        centered = neigh - centroid[:, None, :]
        denom = float(max(k - 1, 1))
        cov = np.einsum("mki,mkj->mij", centered, centered, optimize=True) / denom
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[:, ::-1]
        eigvecs = eigvecs[:, :, ::-1]

        eigsum = np.sum(eigvals, axis=1)
        stable = np.isfinite(eigsum) & (eigsum > _EPS)
        stable &= np.all(np.isfinite(eigvals), axis=1)
        stable &= eigvals[:, 0] > _EPS
        stable &= eigvals[:, 1] > _EPS
        stable &= eigvals[:, 2] > _EPS
        if d >= 4:
            stable &= eigvals[:, 3] > _EPS
        stable &= (eigvals[:, 0] / (eigvals[:, -1] + _EPS)) < _SINGULAR_RATIO_THRESHOLD

        local_r2 = _safe_divide(eigvals[:, 0] + eigvals[:, 1], np.maximum(eigsum, _EPS))
        basis2 = eigvecs[:, :, :2]
        x_rel = x_center - centroid
        coeff = np.einsum("mij,mj->mi", basis2, x_rel)
        recon = np.einsum("mji,mi->mj", basis2, coeff)
        resid = x_rel - recon
        orth_sum = np.sum(eigvals[:, 2:4], axis=1)
        local_resid2 = np.sum(resid * resid, axis=1) / np.maximum(orth_sum, _EPS)
        local_ratio12 = _safe_divide(eigvals[:, 0], np.maximum(eigvals[:, 1], _EPS))
        local_ratio23 = _safe_divide(eigvals[:, 1], np.maximum(eigvals[:, 2], _EPS))
        local_ratio4sum = _safe_divide(eigvals[:, 3], np.maximum(eigsum, _EPS))

        q_low = np.quantile(neigh, _LOW_Q, axis=1)
        q_high = np.quantile(neigh, _HIGH_Q, axis=1)
        local_box = np.mean((x_center >= q_low) & (x_center <= q_high), axis=1)

        knn_kdist[start:end] = local_dist[:, -1]
        knn_med[start:end] = np.median(local_dist, axis=1)
        knn_density[start:end] = 1.0 / (local_dist[:, -1] + _EPS)
        inbox_support[start:end] = local_box

        idx = np.arange(start, end, dtype=np.int64)
        stable_idx = np.flatnonzero(stable)
        if stable_idx.size:
            g = idx[stable_idx]
            fallback_flag[g] = 0
            r2[g] = local_r2[stable_idx].astype(np.float32)
            residual2[g] = local_resid2[stable_idx].astype(np.float32)
            ratio12[g] = local_ratio12[stable_idx].astype(np.float32)
            ratio23[g] = local_ratio23[stable_idx].astype(np.float32)
            ratio4sum[g] = local_ratio4sum[stable_idx].astype(np.float32)

    fallback_global = np.zeros(n_rows, dtype=np.float32)
    fallback_global[:] = global_knn_k
    knn_kdist[~np.isfinite(knn_kdist)] = global_knn_k
    fallback_global_med = np.zeros(n_rows, dtype=np.float32)
    fallback_global_med[:] = global_knn_med
    fallback_density = np.zeros(n_rows, dtype=np.float32)
    fallback_density[:] = global_density
    fallback_support = np.zeros(n_rows, dtype=np.float32)
    fallback_support[:] = global_box_support

    knn_kdist = np.where(np.isfinite(knn_kdist), knn_kdist, fallback_global)
    knn_med = np.where(np.isfinite(knn_med), knn_med, fallback_global_med)
    knn_density = np.where(np.isfinite(knn_density), knn_density, fallback_density)
    inbox_support = np.where(np.isfinite(inbox_support), inbox_support, fallback_support)

    return pd.DataFrame(
        {
            col_r2: r2,
            col_resid: residual2,
            col_ratio12: ratio12,
            col_ratio23: ratio23,
            col_ratio4: ratio4sum,
            col_knn_k: knn_kdist,
            col_knn_med: knn_med,
            col_knn_density: knn_density,
            col_box_support: inbox_support,
            col_fallback: fallback_flag,
        },
        index=index,
    )

def add_local_manifold_codimension(raw, deps, aux):
    _ = deps
    _ = aux

    u = raw["u"].to_numpy(dtype=np.float64, copy=False)
    g = raw["g"].to_numpy(dtype=np.float64, copy=False)
    r = raw["r"].to_numpy(dtype=np.float64, copy=False)
    i = raw["i"].to_numpy(dtype=np.float64, copy=False)
    z = raw["z"].to_numpy(dtype=np.float64, copy=False)
    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)

    color_with_z = np.column_stack((u - g, g - r, r - i, i - z, redshift))
    color_only = np.column_stack((u - g, g - r, r - i, i - z))
    mags = raw[["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float64, copy=False)

    f1 = _compute_local_manifold_features(color_with_z, "local_manifold_codimension_colors_with_redshift", raw.index)
    f2 = _compute_local_manifold_features(color_only, "local_manifold_codimension_colors", raw.index)
    f3 = _compute_local_manifold_features(mags, "local_manifold_codimension_magnitudes", raw.index)

    return pd.concat((f1, f2, f3), axis=1)

FEATURE_GROUPS = [
    {
        "name": "local_manifold_codimension",
        "fn": add_local_manifold_codimension,
        "depends_on": [],
        "description": "Computes neighborhood manifold-codimension, residual-extrapolation, anisotropy, support, and density geometry features from kNN local covariance structure.",
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