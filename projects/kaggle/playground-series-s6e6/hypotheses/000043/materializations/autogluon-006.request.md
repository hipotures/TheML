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
- Hypothesis ID: 000043
- Source file: autogluon-005.py
- Failed node: 20260624T112448-cf1f1242-389
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T112448-cf1f1242-389/02-code.py", line 713, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T112448-cf1f1242-389/02-code.py", line 166, in add_redshift_partitioned_sed_pca_residuals
    local_bin_ids = local_bin.to_numpy(dtype=np.float64)
                    ^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T112448-cf1f1242-389/02-code.py", line 837, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T112448-cf1f1242-389/02-code.py", line 713, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T112448-cf1f1242-389/02-code.py", line 166, in add_redshift_partitioned_sed_pca_residuals
    local_bin_ids = local_bin.to_numpy(dtype=np.float64)
                    ^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_partitioned_sed_pca_residuals
TheML feature group: failed name=redshift_partitioned_sed_pca_residuals elapsed_s=0.358 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

EPSILON = 1e-12
SVD_TOLERANCE = 1e-15
RESIDUAL_BAND_CAP = 0.2
MIN_BIN_FRACTION = 0.004
MIN_BIN_ABSOLUTE = 4000
BANDS = ("u", "g", "r", "i", "z")
REDSHIFT_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

def _build_redshift_edges(redshift_values):
    redshift = np.asarray(redshift_values, dtype=np.float64)
    quantiles = np.asarray(REDSHIFT_QUANTILES, dtype=np.float64)
    edges = np.quantile(redshift, quantiles)
    edges[0] = float(np.min(redshift))
    edges[-1] = float(np.max(redshift))

    dedup = []
    for value in edges:
        if not dedup or value > dedup[-1] + EPSILON:
            dedup.append(float(value))

    if len(dedup) < 2:
        center = float(redshift[0] if redshift.size else 0.0)
        pad = max(1e-6, abs(center) * 1e-6 + 1e-4)
        return np.array([center - pad, center + pad], dtype=np.float64)

    return np.asarray(dedup, dtype=np.float64)

def _merge_bin_edges(edges, counts, min_size):
    edges = np.asarray(edges, dtype=np.float64).copy()
    counts = np.asarray(counts, dtype=np.int64).copy()

    if edges.size < 2 or counts.size == 0:
        return edges, counts

    while counts.size > 1 and counts.min() < min_size:
        idx = int(np.argmin(counts))
        if counts[idx] >= min_size:
            break

        if idx == 0:
            counts[idx] += counts[idx + 1]
            counts = np.delete(counts, idx + 1)
            edges = np.delete(edges, idx + 1)
        elif idx == counts.size - 1:
            counts[idx - 1] += counts[idx]
            counts = np.delete(counts, idx)
            edges = np.delete(edges, idx)
        else:
            if counts[idx - 1] <= counts[idx + 1]:
                counts[idx - 1] += counts[idx]
                counts = np.delete(counts, idx)
                edges = np.delete(edges, idx)
            else:
                counts[idx] += counts[idx + 1]
                counts = np.delete(counts, idx + 1)
                edges = np.delete(edges, idx + 1)

    return edges, counts

def _fit_standardized_pca_residual(p_block):
    block = np.asarray(p_block, dtype=np.float64)
    mu = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    std = np.maximum(std, EPSILON)
    standardized = (block - mu) / std
    standardized = standardized - standardized.mean(axis=0, keepdims=True)

    if standardized.shape[0] <= 1:
        loadings = np.zeros((block.shape[1], 0), dtype=np.float64)
        residual_std = standardized
        return mu, std, loadings, residual_std

    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    k = int(np.sum(singular_values > SVD_TOLERANCE))
    if k == 0:
        loadings = np.zeros((block.shape[1], 0), dtype=np.float64)
        residual_std = standardized
        return mu, std, loadings, residual_std

    k = min(3, k)
    loadings = vt[:k].T
    residual_std = standardized - (standardized @ loadings) @ loadings.T
    return mu, std, loadings, residual_std

def _rank_pct(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    if arr.size == 1:
        return np.array([0.5], dtype=np.float64)
    return pd.Series(arr).rank(method="average", pct=True).to_numpy(dtype=np.float64)

def add_redshift_partitioned_sed_pca_residuals(raw, deps, aux):
    del deps, aux

    index = raw.index
    redshift = raw["redshift"].to_numpy(dtype=np.float64)
    mags = raw.loc[:, BANDS].to_numpy(dtype=np.float64)
    n_rows = redshift.shape[0]

    if n_rows == 0:
        return pd.DataFrame(index=index)

    flux = np.power(10.0, -0.4 * mags)
    shape = flux / (flux.sum(axis=1, keepdims=True) + EPSILON)

    # Global fallback PCA on all rows.
    g_mu, g_std, g_loadings, g_res_std = _fit_standardized_pca_residual(shape)
    g_res = np.clip(g_res_std * g_std, -RESIDUAL_BAND_CAP, RESIDUAL_BAND_CAP)
    g_norm = np.linalg.norm(g_res, axis=1)
    g_rel = g_norm / (np.linalg.norm(shape, axis=1) + EPSILON)
    g_frac = (g_res * g_res).sum(axis=1) / ((shape * shape).sum(axis=1) + EPSILON)
    g_orth = (g_res_std * g_res_std).sum(axis=1)

    g_dir = np.zeros((n_rows, 3), dtype=np.float64)
    if g_loadings.shape[1] > 0:
        g_scores = g_res_std @ g_loadings
        g_n = min(3, g_scores.shape[1])
        g_dir[:, :g_n] = g_scores[:, :g_n]

    g_rank = _rank_pct(g_norm)

    # Start from global values, then replace by local-bin values when available.
    residual = g_res.copy()
    residual_norm = g_norm.copy()
    residual_rel = g_rel.copy()
    residual_energy = g_frac.copy()
    residual_orth = g_orth.copy()
    residual_dir = g_dir.copy()
    residual_bin_rank = g_rank.copy()

    # Build redshift bins from quantiles and merge until minimum support constraint.
    edges = _build_redshift_edges(redshift)
    initial_bin = pd.cut(redshift, edges, labels=False, include_lowest=True, right=True, duplicates="drop")
    initial_bin_series = pd.Series(initial_bin, index=index)
    if initial_bin_series.notna().any():
        valid_bins = initial_bin_series.dropna().astype(np.int64).to_numpy()
        init_counts = np.bincount(valid_bins, minlength=int(valid_bins.max()) + 1)
    else:
        init_counts = np.array([], dtype=np.int64)

    init_bin_count = max(1, init_counts.size)
    min_bin_size = max(MIN_BIN_ABSOLUTE, int(np.floor(MIN_BIN_FRACTION * float(n_rows) / init_bin_count)))
    merged_edges, merged_counts = _merge_bin_edges(edges, init_counts, min_bin_size)

    local_possible = not (
        merged_counts.size == 0 or (merged_counts.size == 1 and merged_counts[0] < min_bin_size)
    )

    if local_possible:
        clipped_redshift = np.clip(redshift, merged_edges[0], merged_edges[-1])
        local_bin = pd.cut(
            clipped_redshift,
            merged_edges,
            labels=False,
            include_lowest=True,
            right=True,
            duplicates="drop",
        )
        local_bin_ids = local_bin.to_numpy(dtype=np.float64)
        local_mask = ~np.isnan(local_bin_ids)
        assigned = np.full(n_rows, -1, dtype=np.int64)
        assigned[local_mask] = local_bin_ids[local_mask].astype(np.int64)

        if (~local_mask).any():
            midpoints = 0.5 * (merged_edges[:-1] + merged_edges[1:])
            nearest = np.argmin(np.abs(clipped_redshift[:, None] - midpoints[None, :]), axis=1)
            assigned[~local_mask] = nearest[~local_mask]
            local_mask[~local_mask] = True

        for b in range(merged_counts.size):
            row_idx = np.flatnonzero(assigned == b)
            if row_idx.size == 0:
                continue

            block = shape[row_idx]
            b_mu, b_std, b_loadings, b_res_std = _fit_standardized_pca_residual(block)
            b_res = np.clip(b_res_std * b_std, -RESIDUAL_BAND_CAP, RESIDUAL_BAND_CAP)

            b_norm = np.linalg.norm(b_res, axis=1)
            b_rel = b_norm / (np.linalg.norm(block, axis=1) + EPSILON)
            b_frac = (b_res * b_res).sum(axis=1) / ((block * block).sum(axis=1) + EPSILON)
            b_orth = (b_res_std * b_res_std).sum(axis=1)

            b_dir = np.zeros((row_idx.size, 3), dtype=np.float64)
            if b_loadings.shape[1] > 0:
                b_scores = b_res_std @ b_loadings
                b_n = min(3, b_scores.shape[1])
                b_dir[:, :b_n] = b_scores[:, :b_n]

            b_rank = _rank_pct(b_norm)

            residual[row_idx] = b_res
            residual_norm[row_idx] = b_norm
            residual_rel[row_idx] = b_rel
            residual_energy[row_idx] = b_frac
            residual_orth[row_idx] = b_orth
            residual_dir[row_idx] = b_dir
            residual_bin_rank[row_idx] = b_rank

    features = {
        "sed_pca_residual_norm_l2": residual_norm,
        "sed_pca_residual_relative_norm": residual_rel,
        "sed_pca_residual_energy_fraction": residual_energy,
        "sed_pca_residual_outside_pc_energy": residual_orth,
        "sed_pca_residual_dir_pc1": residual_dir[:, 0],
        "sed_pca_residual_dir_pc2": residual_dir[:, 1],
        "sed_pca_residual_dir_pc3": residual_dir[:, 2],
        "sed_pca_residual_band_u": residual[:, 0],
        "sed_pca_residual_band_g": residual[:, 1],
        "sed_pca_residual_band_r": residual[:, 2],
        "sed_pca_residual_band_i": residual[:, 3],
        "sed_pca_residual_band_z": residual[:, 4],
        "sed_pca_recon_error_bin_quantile": residual_bin_rank,
    }

    return pd.DataFrame(features, index=index)

FEATURE_GROUPS = [
    {
        "name": "redshift_partitioned_sed_pca_residuals",
        "fn": add_redshift_partitioned_sed_pca_residuals,
        "depends_on": [],
        "description": "Build redshift-partitioned PCA residual spectral-manifold features from normalized ugriz shape vectors.",
    },
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