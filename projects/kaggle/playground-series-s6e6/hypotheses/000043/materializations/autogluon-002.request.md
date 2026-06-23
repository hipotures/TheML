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
- Hypothesis ID: 000043
- Source file: autogluon-001.py
- Failed node: 20260623T151019-cb846318-191
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 726, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 148, in add_redshift_partitioned_sed_pca_residuals
    raw_bin = _assign_bins(z_raw_clipped, edges)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 93, in _assign_bins
    return codes.fillna(0).astype(np.int32).to_numpy()
           ^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 850, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 726, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 148, in add_redshift_partitioned_sed_pca_residuals
    raw_bin = _assign_bins(z_raw_clipped, edges)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260623T151019-cb846318-191/02-code.py", line 93, in _assign_bins
    return codes.fillna(0).astype(np.int32).to_numpy()
           ^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'fillna'. Did you mean: 'fill'?

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_partitioned_sed_pca_residuals
TheML feature group: failed name=redshift_partitioned_sed_pca_residuals elapsed_s=0.208 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

MAG_BANDS = ("u", "g", "r", "i", "z")
REDSHIFT_COLUMN = "redshift"
N_PCA_COMPONENTS = 3
DESIRED_BIN_COUNT = 8
MIN_BIN_SUPPORT_FLOOR = 24
MAX_BIN_SUPPORT_CEILING = 250
MAG_MIN = -30.0
MAG_MAX = 60.0
EPS = 1e-12

def _to_numeric(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)

def _compute_shape_frame(frame):
    band_vectors = []
    for band in MAG_BANDS:
        mag = _to_numeric(frame[band])
        mag = mag.where((mag >= MAG_MIN) & (mag <= MAG_MAX))
        finite = mag.dropna()
        if len(finite):
            fill_value = float(finite.median())
        else:
            fill_value = 0.0
        mag = mag.fillna(fill_value).to_numpy(dtype=float)
        band_vectors.append(mag)

    mags = np.column_stack(band_vectors)
    flux = np.power(10.0, -0.4 * mags)
    flux_sum = np.sum(flux, axis=1, keepdims=True)
    return flux / np.maximum(flux_sum, EPS)

def _extract_redshift(frame):
    red = _to_numeric(frame[REDSHIFT_COLUMN])
    finite = red.dropna()
    if len(finite):
        fill_value = float(finite.median())
    else:
        fill_value = 0.0
    return red.fillna(fill_value).to_numpy(dtype=float)

def _compute_bin_edges(redshift_values):
    z = np.asarray(redshift_values, dtype=float)
    if z.size == 0 or not np.isfinite(z).any():
        return np.array([0.0, 1.0], dtype=float)

    series = pd.Series(z)

    for nb in (DESIRED_BIN_COUNT, max(4, DESIRED_BIN_COUNT - 2), 4, 3, 2):
        try:
            _, edges = pd.qcut(series, q=nb, labels=False, retbins=True, duplicates="drop")
        except Exception:
            continue
        edges = np.asarray(edges, dtype=float)
        if len(edges) < 3:
            continue
        if not np.all(np.isfinite(edges)):
            continue
        edges = np.unique(edges)
        if len(edges) < 3 or np.any(np.diff(edges) <= 0):
            continue
        return edges

    z_min = float(np.nanmin(z))
    z_max = float(np.nanmax(z))
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        return np.array([0.0, 1.0], dtype=float)

    if z_min == z_max:
        span = 1.0 if z_min == 0.0 else 0.01 * (abs(z_min) + 1.0)
        return np.array([z_min - span, z_min + span], dtype=float)

    return np.array([z_min, (2.0 * z_min + z_max) / 3.0, (z_min + 2.0 * z_max) / 3.0, z_max], dtype=float)

def _assign_bins(redshift_values, edges):
    z = np.asarray(redshift_values, dtype=float)
    if z.size == 0 or len(edges) < 2:
        return np.zeros(z.size, dtype=np.int32)

    clipped = np.clip(z, edges[0], edges[-1])
    codes = pd.cut(clipped, bins=edges, labels=False, include_lowest=True)
    return codes.fillna(0).astype(np.int32).to_numpy()

def _fit_pca_model(block):
    block = np.asarray(block, dtype=float)
    n_rows, n_features = block.shape
    if n_rows < 2 or n_features == 0:
        mean = block.mean(axis=0) if n_rows else np.zeros(n_features, dtype=float)
        return {"mean": mean, "components": np.zeros((0, n_features), dtype=float)}

    mean = block.mean(axis=0)
    centered = block - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(N_PCA_COMPONENTS, vt.shape[0], vt.shape[1]) if vt.size else 0
        components = vt[:k] if k > 0 else np.zeros((0, n_features), dtype=float)
    except Exception:
        components = np.zeros((0, n_features), dtype=float)

    return {"mean": mean, "components": components}

def add_redshift_partitioned_sed_pca_residuals(raw, deps, aux):
    p_raw = _compute_shape_frame(raw)
    z_raw = _extract_redshift(raw)

    z_fit = z_raw
    p_fit = p_raw

    aux_is_usable = (
        aux is not None
        and hasattr(aux, "__len__")
        and len(aux) > 0
        and all(col in aux.columns for col in MAG_BANDS)
        and REDSHIFT_COLUMN in aux.columns
    )
    if aux_is_usable:
        try:
            aux_shape = _compute_shape_frame(aux)
            aux_z = _extract_redshift(aux)
            z_lo = np.nanmin(z_raw)
            z_hi = np.nanmax(z_raw)
            if np.isfinite(z_lo) and np.isfinite(z_hi):
                aux_z = np.clip(aux_z, z_lo, z_hi)
            p_fit = np.vstack((p_fit, aux_shape))
            z_fit = np.concatenate((z_fit, aux_z))
        except Exception:
            p_fit = p_raw
            z_fit = z_raw

    edges = _compute_bin_edges(z_fit)
    n_fit = len(z_fit)
    n_bins = max(1, len(edges) - 1)

    z_raw_clipped = np.clip(z_raw, edges[0], edges[-1])
    raw_bin = _assign_bins(z_raw_clipped, edges)
    fit_bin = _assign_bins(z_fit, edges)

    bin_sizes = np.bincount(fit_bin, minlength=n_bins)
    min_support = max(MIN_BIN_SUPPORT_FLOOR, min(MAX_BIN_SUPPORT_CEILING, int(np.sqrt(max(1, n_fit)))))
    supported = bin_sizes >= min_support
    if not np.any(supported):
        supported[:] = True

    global_model = _fit_pca_model(p_fit)

    bin_models = {}
    for b in np.where(supported)[0]:
        mask = fit_bin == b
        if mask.any():
            bin_models[int(b)] = _fit_pca_model(p_fit[mask])

    centers = (edges[:-1] + edges[1:]) / 2.0
    good_bins = np.where(supported)[0]
    fallback_map = np.full(n_bins, -1, dtype=np.int32)
    if good_bins.size:
        for b in range(n_bins):
            if supported[b]:
                fallback_map[b] = b
            else:
                fallback_map[b] = int(good_bins[np.argmin(np.abs(centers[b] - centers[good_bins]))])

    effective_bin = np.array(
        [fallback_map[b] if 0 <= b < n_bins else -1 for b in raw_bin],
        dtype=np.int32
    )
    used_global = effective_bin == -1
    fallback_used = (effective_bin != raw_bin) | used_global

    n_rows = len(raw)
    scores = np.zeros((n_rows, N_PCA_COMPONENTS), dtype=float)
    residual = np.zeros((n_rows, len(MAG_BANDS)), dtype=float)

    for b in np.unique(effective_bin):
        mask = np.where(effective_bin == b)[0]
        if mask.size == 0:
            continue
        model = global_model if b == -1 else bin_models.get(int(b), global_model)
        mean = model["mean"]
        components = model["components"]

        block = p_raw[mask]
        if components.size:
            centered = block - mean
            proj = centered @ components.T
            k = proj.shape[1]
            scores[mask, :k] = proj
            reconstructed = proj @ components + mean
        else:
            reconstructed = np.broadcast_to(mean, block.shape)

        residual[mask] = block - reconstructed

    residual_l2 = np.sqrt(np.einsum("ij,ij->i", residual, residual))
    shape_l2 = np.sqrt(np.sum(p_raw * p_raw, axis=1))
    residual_rel = residual_l2 / np.maximum(shape_l2, EPS)
    residual_energy = np.sum(residual * residual, axis=1) / np.maximum(np.sum(p_raw * p_raw, axis=1), EPS)

    bin_support = np.array(
        [float(bin_sizes[b]) if 0 <= b < n_bins else float(n_fit) for b in effective_bin],
        dtype=float
    )

    return pd.DataFrame(
        {
            "sed_c1": scores[:, 0].astype(np.float32),
            "sed_c2": scores[:, 1].astype(np.float32),
            "sed_c3": scores[:, 2].astype(np.float32),
            "sed_residual_norm": residual_l2.astype(np.float32),
            "sed_residual_norm_rel": residual_rel.astype(np.float32),
            "sed_residual_energy": residual_energy.astype(np.float32),
            "sed_residual_u": residual[:, 0].astype(np.float32),
            "sed_residual_g": residual[:, 1].astype(np.float32),
            "sed_residual_r": residual[:, 2].astype(np.float32),
            "sed_residual_i": residual[:, 3].astype(np.float32),
            "sed_residual_z": residual[:, 4].astype(np.float32),
            "sed_bin_id": raw_bin.astype(np.int32),
            "sed_bin_effective": effective_bin.astype(np.int32),
            "sed_bin_support": bin_support.astype(np.float32),
            "sed_bin_fallback": fallback_used.astype(np.uint8),
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "redshift_partitioned_sed_pca_residuals",
        "fn": add_redshift_partitioned_sed_pca_residuals,
        "depends_on": [],
        "description": "Compute redshift-stratified PCA residual features on normalized ugri z flux-shape manifolds, with fallback to nearest-supported strata or global PCA.",
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