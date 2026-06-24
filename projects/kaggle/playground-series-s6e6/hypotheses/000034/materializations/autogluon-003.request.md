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
- Hypothesis ID: 000034
- Source file: autogluon-002.py
- Failed node: 20260624T061613-a62f2ae7-342
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061613-a62f2ae7-342/02-code.py", line 656, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061613-a62f2ae7-342/02-code.py", line 75, in add_blackbody_continuum_distance
    arg = PLANCK_COEFF / (lam_prime[:, None, :] * chunk[None, :, :])
                                                  ~~~~~^^^^^^^^^^^^
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061613-a62f2ae7-342/02-code.py", line 780, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061613-a62f2ae7-342/02-code.py", line 656, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061613-a62f2ae7-342/02-code.py", line 75, in add_blackbody_continuum_distance
    arg = PLANCK_COEFF / (lam_prime[:, None, :] * chunk[None, :, :])
                                                  ~~~~~^^^^^^^^^^^^
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=blackbody_continuum_distance
TheML feature group: failed name=blackbody_continuum_distance elapsed_s=0.096 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BAND_COLUMNS = ("u", "g", "r", "i", "z")
WAVELENGTHS_ANGSTROM = (3551.0, 4686.0, 6165.0, 7481.0, 8931.0)
TEMP_COUNT = 180
MIN_TEMP = 1500.0
MAX_TEMP = 50000.0
PLANCK_COEFF = 14387.7
MAX_PLANCK_ARG = 700.0
MAG_FLOOR = 30.0
FLUX_SUM_GUARD = 1e-30
TEMP_CHUNK = 8

def _fit_planck_shape(log_t_values, shape_matrix, lam_prime, lam5_inv):
    t = np.power(10.0, log_t_values)
    arg = PLANCK_COEFF / (lam_prime * t[:, None])
    arg = np.clip(arg, a_min=None, a_max=MAX_PLANCK_ARG)
    blackbody = lam5_inv / np.expm1(arg)

    row_sum = blackbody.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    template = blackbody / row_sum

    numer = np.einsum("bi,bi->b", shape_matrix, template)
    denom = np.einsum("bi,bi->b", template, template)
    denom = np.where(denom == 0.0, 1.0, denom)

    scale = numer / denom
    residual = shape_matrix - template * scale[:, None]
    ssr = np.einsum("bi,bi->b", residual, residual)

    return template, scale, ssr

def add_blackbody_continuum_distance(raw, deps, aux):
    _ = deps, aux

    raw_mags = raw.loc[:, BAND_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    raw_mags = np.where(np.isfinite(raw_mags), raw_mags, MAG_FLOOR)
    flux = np.power(10.0, -0.4 * raw_mags)

    flux_sum = np.sum(flux, axis=1)
    shape = np.empty_like(flux)
    finite_mask = flux_sum > FLUX_SUM_GUARD
    shape[finite_mask] = flux[finite_mask] / flux_sum[finite_mask][:, None]
    if np.any(~finite_mask):
        shape[~finite_mask] = 0.2

    redshift = raw["redshift"].to_numpy(dtype=np.float64, copy=False)
    z_eff = np.where((~np.isfinite(redshift)) | (redshift <= 0.0), 0.0, redshift)

    wavelengths = np.asarray(WAVELENGTHS_ANGSTROM, dtype=np.float64)
    lam_prime = wavelengths[None, :] / (1.0 + z_eff[:, None])
    lam5_inv = np.power(lam_prime, -5.0)

    shape_sq = np.einsum("bi,bi->b", shape, shape)

    log_t_grid = np.linspace(np.log10(MIN_TEMP), np.log10(MAX_TEMP), TEMP_COUNT, dtype=np.float64)
    temp_grid = np.power(10.0, log_t_grid)

    n = raw.shape[0]
    best_ssr = np.full(n, np.inf, dtype=np.float64)
    best_idx = np.full(n, -1, dtype=np.int64)
    second_ssr = np.full(n, np.inf, dtype=np.float64)

    for start in range(0, TEMP_COUNT, TEMP_CHUNK):
        stop = min(TEMP_COUNT, start + TEMP_CHUNK)
        chunk = temp_grid[start:stop]
        arg = PLANCK_COEFF / (lam_prime[:, None, :] * chunk[None, :, :])
        arg = np.clip(arg, a_min=None, a_max=MAX_PLANCK_ARG)

        raw_template = lam5_inv[:, None, :] / np.expm1(arg)
        temp_sum = raw_template.sum(axis=2, keepdims=True)
        temp_sum = np.where(temp_sum == 0.0, 1.0, temp_sum)
        templates = raw_template / temp_sum

        template_sq = np.einsum("bci,bci->bc", templates, templates)
        template_sq = np.where(template_sq == 0.0, 1.0, template_sq)
        numer = np.einsum("bi,bci->bc", shape, templates)
        scale = numer / template_sq

        ssr = shape_sq[:, None] - 2.0 * scale * numer + (scale * scale) * template_sq
        ssr = np.where(np.isfinite(ssr), ssr, np.inf)

        local_min = ssr.min(axis=1)
        local_arg = np.argmin(ssr, axis=1).astype(np.int64)
        local_idx = start + local_arg
        if chunk.shape[0] > 1:
            local_second = np.partition(ssr, 1, axis=1)[:, 1]
        else:
            local_second = np.full(n, np.inf, dtype=np.float64)

        prev_best = best_ssr
        prev_second = second_ssr

        take_new = local_min < prev_best

        best_ssr = np.where(take_new, local_min, prev_best)
        best_idx = np.where(take_new, local_idx, best_idx)
        second_ssr = np.where(
            take_new,
            np.minimum(prev_best, local_second),
            np.minimum(prev_second, local_min),
        )

    if np.any(best_idx < 0):
        best_idx = np.where(best_idx < 0, 0, best_idx)
        best_ssr = np.where(best_ssr == np.inf, 0.0, best_ssr)

    best_log_t = log_t_grid[best_idx]
    dlog = log_t_grid[1] - log_t_grid[0]

    prev_log_t = best_log_t.copy()
    next_log_t = best_log_t.copy()
    interior = (best_idx > 0) & (best_idx < (TEMP_COUNT - 1))

    if np.any(interior):
        prev_log_t[interior] = log_t_grid[best_idx[interior] - 1]
        next_log_t[interior] = log_t_grid[best_idx[interior] + 1]

    _, _, prev_ssr = _fit_planck_shape(prev_log_t, shape, lam_prime, lam5_inv)
    _, _, next_ssr = _fit_planck_shape(next_log_t, shape, lam_prime, lam5_inv)

    curvature = np.full(n, np.nan, dtype=np.float64)
    curvature_valid = interior & np.isfinite(best_ssr) & np.isfinite(prev_ssr) & np.isfinite(next_ssr)
    curvature_num = prev_ssr - 2.0 * best_ssr + next_ssr
    curvature[curvature_valid] = np.where(
        np.abs(curvature_num[curvature_valid]) > 1e-18,
        curvature_num[curvature_valid] / (dlog * dlog),
        np.nan,
    )

    denom = prev_ssr - 2.0 * best_ssr + next_ssr
    refined_log_t = best_log_t.copy()
    refine_valid = interior & np.isfinite(denom) & (np.abs(denom) > 1e-18)
    refined_log_t[refine_valid] = np.clip(
        best_log_t[refine_valid] + 0.5 * (prev_ssr[refine_valid] - next_ssr[refine_valid]) / denom[refine_valid] * dlog,
        log_t_grid[0],
        log_t_grid[-1],
    )

    _, refined_scale, refined_ssr = _fit_planck_shape(refined_log_t, shape, lam_prime, lam5_inv)
    template_refined, _, _ = _fit_planck_shape(refined_log_t, shape, lam_prime, lam5_inv)
    residuals = shape - template_refined * refined_scale[:, None]

    features = {
        "bbcd_log10_T_star": refined_log_t,
        "bbcd_ssr_min": refined_ssr,
        "bbcd_ssr_rms": np.sqrt(np.maximum(refined_ssr, 0.0)),
        "bbcd_ssr_gap_to_second_best": second_ssr - refined_ssr,
        "bbcd_ssr_curvature_logT": curvature,
        "bbcd_residual_u": residuals[:, 0],
        "bbcd_residual_g": residuals[:, 1],
        "bbcd_residual_r": residuals[:, 2],
        "bbcd_residual_i": residuals[:, 3],
        "bbcd_residual_z": residuals[:, 4],
    }

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "blackbody_continuum_distance",
        "fn": add_blackbody_continuum_distance,
        "depends_on": [],
        "description": "Fits a redshift-adjusted blackbody manifold in ugriz shape space and outputs fit distance diagnostics and residual shape features.",
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