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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

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
- Hypothesis ID: 000045
- Source file: autogluon-001.py
- Failed node: 20260623T151146-8fcb30de-193
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 711, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 163, in add_pairwise_color_lattice_residuals
    raw_bin_ids, num_bins = _compute_bin_assignments(redshift_values)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 112, in _compute_bin_assignments
    bins = labels.fillna(0).astype(int).to_numpy()
           ^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 835, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 711, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 163, in add_pairwise_color_lattice_residuals
    raw_bin_ids, num_bins = _compute_bin_assignments(redshift_values)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151146-8fcb30de-193/02-code.py", line 112, in _compute_bin_assignments
    bins = labels.fillna(0).astype(int).to_numpy()
           ^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=pairwise_color_lattice_residuals
TheML feature group: failed name=pairwise_color_lattice_residuals elapsed_s=0.114 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

PAIRWISE_COLOR_DEFS = (
    ("color_u_g", "u", "g"),
    ("color_u_r", "u", "r"),
    ("color_u_i", "u", "i"),
    ("color_u_z", "u", "z"),
    ("color_g_r", "g", "r"),
    ("color_g_i", "g", "i"),
    ("color_g_z", "g", "z"),
    ("color_r_i", "r", "i"),
    ("color_r_z", "r", "z"),
    ("color_i_z", "i", "z"),
)

REDSHIFT_BIN_COUNT = 32
MIN_BIN_ROWS = 120
STD_RESID_CLIP = 8.0
COV_REGULARIZATION = 1e-6
TOP_PCA_COMPONENTS = 3
COV_EIGEN_EPS = 1e-12

def _safe_float_array(raw, column):
    if column not in raw.columns:
        return np.zeros(len(raw), dtype=float)
    return pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)

def _build_pairwise_colors(raw):
    data = {}
    for feature_name, left, right in PAIRWISE_COLOR_DEFS:
        left_values = _safe_float_array(raw, left)
        right_values = _safe_float_array(raw, right)
        data[feature_name] = left_values - right_values
    return pd.DataFrame(data, index=raw.index)

def _bin_statistics(color_block):
    block = np.asarray(color_block, dtype=float)
    if block.size == 0:
        return None

    center = np.nanmedian(block, axis=0)
    mad = np.nanmedian(np.abs(block - center), axis=0)
    mad = np.where(np.isfinite(mad) & (mad > 0.0), mad, 1.0)

    standardized = (block - center) / mad
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    standardized = np.clip(standardized, -STD_RESID_CLIP, STD_RESID_CLIP)

    n_rows, n_features = block.shape
    if n_rows <= 1:
        covariance = np.eye(n_features, dtype=float)
    else:
        covariance = np.cov(standardized, rowvar=False, bias=True)
        if covariance.shape != (n_features, n_features):
            covariance = np.eye(n_features, dtype=float)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

    covariance = (covariance + covariance.T) * 0.5
    covariance = covariance + COV_REGULARIZATION * np.eye(covariance.shape[0])

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        n_features = block.shape[1]
        eigenvalues = np.ones(n_features, dtype=float)
        eigenvectors = np.eye(n_features, dtype=float)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=float)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=float)
    eigenvalues = np.where(np.isfinite(eigenvalues) & (eigenvalues > 0.0), eigenvalues, COV_EIGEN_EPS)

    try:
        inv_covariance = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inv_covariance = np.linalg.pinv(covariance)

    return center, mad, inv_covariance, eigenvectors, eigenvalues

def _compute_bin_assignments(redshift_values):
    n = len(redshift_values)
    redshift_for_bins = np.asarray(redshift_values, dtype=float)

    finite = redshift_for_bins[np.isfinite(redshift_for_bins)]
    if finite.size == 0:
        return np.zeros(n, dtype=int), 1

    redshift_for_bins = np.where(redshift_for_bins < 0.0, 0.0, redshift_for_bins)
    finite = redshift_for_bins[np.isfinite(redshift_for_bins)]

    if np.unique(finite).size <= 1:
        z0 = float(np.nanmin(finite))
        edges = np.array([z0, z0 + 1.0], dtype=float)
    else:
        requested_bins = int(min(REDSHIFT_BIN_COUNT, finite.size))
        probs = np.linspace(0.0, 1.0, requested_bins + 1)
        edges = np.quantile(finite, probs)
        edges = np.unique(edges)
        if edges.size < 2:
            z0 = float(np.nanmin(finite))
            edges = np.array([z0, z0 + 1.0], dtype=float)

    labels = pd.cut(redshift_for_bins, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    bins = labels.fillna(0).astype(int).to_numpy()
    num_bins = int(bins.max()) + 1
    if num_bins < 1:
        num_bins = 1
        bins[:] = 0
    return bins, num_bins

def _resolve_bins(bin_ids, num_bins):
    counts = np.bincount(np.asarray(bin_ids, dtype=np.int64), minlength=num_bins).astype(np.int64)
    resolved = np.arange(num_bins, dtype=np.int64)

    if MIN_BIN_ROWS <= 0:
        return resolved, counts

    good_bins = np.flatnonzero(counts >= MIN_BIN_ROWS)
    if good_bins.size == 0:
        resolved[:] = -1
        return resolved, counts

    populated = np.flatnonzero(counts > 0)
    for b in range(num_bins):
        if counts[b] >= MIN_BIN_ROWS:
            continue

        replacement = -1
        for step in range(1, num_bins + 1):
            left = b - step
            if left >= 0 and counts[left] >= MIN_BIN_ROWS:
                replacement = left
                break
            right = b + step
            if right < num_bins and counts[right] >= MIN_BIN_ROWS:
                replacement = right
                break

        if replacement == -1 and populated.size > 0:
            replacement = int(populated[np.argmin(np.abs(populated - b))])

        resolved[b] = replacement

    return resolved, counts

def add_pairwise_color_lattice_residuals(raw, deps, aux):
    color_df = _build_pairwise_colors(raw)
    color_names = [name for name, _, _ in PAIRWISE_COLOR_DEFS]
    colors = np.asarray(color_df.to_numpy(dtype=float))
    n_rows, n_features = colors.shape

    redshift_values = _safe_float_array(raw, "redshift")
    raw_bin_ids, num_bins = _compute_bin_assignments(redshift_values)
    resolved_bins, _ = _resolve_bins(raw_bin_ids, num_bins)

    global_stats = _bin_statistics(colors)

    if num_bins <= 0:
        mapped_bin_ids = np.full(n_rows, -1, dtype=np.int64)
    else:
        mapped_bin_ids = resolved_bins[np.clip(raw_bin_ids, 0, num_bins - 1)]

    stats_by_source = {-1: global_stats}

    for source_bin in np.unique(mapped_bin_ids):
        if source_bin in stats_by_source:
            continue
        source_idx = np.flatnonzero(raw_bin_ids == int(source_bin))
        if source_idx.size == 0:
            stats_by_source[int(source_bin)] = global_stats
            continue
        source_stats = _bin_statistics(colors[source_idx])
        if source_stats is None:
            source_stats = global_stats
        stats_by_source[int(source_bin)] = source_stats

    residual_features = np.zeros((n_rows, n_features), dtype=float)
    mahalanobis = np.zeros(n_rows, dtype=float)
    principal_scores = np.zeros((n_rows, TOP_PCA_COMPONENTS), dtype=float)

    for source_bin in np.unique(mapped_bin_ids):
        row_idx = np.flatnonzero(mapped_bin_ids == source_bin)
        if row_idx.size == 0:
            continue

        center, mad, inv_covariance, eigvecs, eigvals = stats_by_source[int(source_bin)]

        block = colors[row_idx]
        block_std = (block - center) / mad
        block_std = np.nan_to_num(block_std, nan=0.0, posinf=0.0, neginf=0.0)
        block_std = np.clip(block_std, -STD_RESID_CLIP, STD_RESID_CLIP)

        residual_features[row_idx, :] = block_std

        rotated = block_std @ inv_covariance
        mahalanobis[row_idx] = np.sqrt(np.einsum("ij,ij->i", rotated, block_std))

        n_components = min(TOP_PCA_COMPONENTS, n_features)
        proj = block_std @ eigvecs[:, :n_components]
        proj = proj / np.sqrt(eigvals[:n_components])
        principal_scores[row_idx, :n_components] = proj

    residual_columns = [f"{name}_zbin_residual" for name in color_names]
    feature_df = pd.DataFrame(residual_features, index=raw.index, columns=residual_columns)
    feature_df["pairwise_mahalanobis"] = mahalanobis
    feature_df["pairwise_pc1"] = principal_scores[:, 0]
    feature_df["pairwise_pc2"] = principal_scores[:, 1]
    feature_df["pairwise_pc3"] = principal_scores[:, 2]

    return feature_df

FEATURE_GROUPS = [
    {
        "name": "pairwise_color_lattice_residuals",
        "fn": add_pairwise_color_lattice_residuals,
        "depends_on": [],
        "description": "Builds redshift-binned pairwise ugriz color residual geometry using robust median/MAD normalization, Mahalanobis distance, and bin-conditioned PCA residual scores.",
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