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
- Hypothesis ID: 000042
- Source file: autogluon-003.py
- Failed node: 
- Run: 

# Execution Error

```text
Invalid materialization for hypothesis 000042 (autogluon); response=hypotheses/000042/materializations/autogluon-003.response.md: Group materialization must not execute top-level code outside imports, definitions, and registry assignments
```

# Previous Code

```python
import numpy as np
import pandas as pd
from collections import defaultdict

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

Z_BIN_EDGES = (
    -0.01, 0.15, 0.30, 0.50, 0.80, 1.20, 1.80, 2.40, 3.00,
    3.80, 4.70, 5.60, 7.05
)
I_BIN_EDGES = (
    10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
    14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5,
    18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5,
    22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0
)
MAD_SCALE = 1.4826
EPSILON = 1e-6
NEED_ROWS = 180
BACKFILL_MIN_ROWS = 120
BASELINE_FLOOR_SCALE = 0.15
NN_K = 5

def _mad(values, axis=0):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        if axis is None:
            return 0.0
        return np.zeros(arr.shape[0], dtype=float) if axis == 1 else np.zeros(arr.shape[1], dtype=float)
    center = np.nanmedian(arr, axis=axis)
    return np.nanmedian(np.abs(arr - center), axis=axis)

def _unit_vector(vec):
    v = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return v / norm

def _stable_sign(vec):
    v = _unit_vector(vec)
    if v is None:
        return None
    idx = np.argmax(np.abs(v))
    if v[idx] < 0.0:
        v = -v
    return v

def _principal_vectors(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 2:
        return np.array((1.0, 0.0, 0.0), dtype=float), np.array((0.0, 1.0, 0.0), dtype=float)
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    if not np.isfinite(centered).any() or np.allclose(centered, 0.0):
        return np.array((1.0, 0.0, 0.0), dtype=float), np.array((0.0, 1.0, 0.0), dtype=float)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return np.array((1.0, 0.0, 0.0), dtype=float), np.array((0.0, 1.0, 0.0), dtype=float)
    if vh.shape[0] < 2:
        v1 = _stable_sign(vh[0])
        if v1 is None:
            v1 = np.array((1.0, 0.0, 0.0), dtype=float)
        v2 = np.array((0.0, 1.0, 0.0), dtype=float)
        if np.allclose(v1, v2):
            v2 = np.array((0.0, 0.0, 1.0), dtype=float)
        return v1, v2
    v1 = _stable_sign(vh[0])
    v2 = _stable_sign(vh[1])
    if v1 is None:
        v1 = np.array((1.0, 0.0, 0.0), dtype=float)
    if v2 is None:
        v2 = np.array((0.0, 1.0, 0.0), dtype=float)
    # Ensure orthogonality after possible sign flips/numeric noise
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = _stable_sign(v2)
    if v2 is None:
        v2 = np.array((0.0, 1.0, 0.0), dtype=float)
        if np.allclose(v1, v2):
            v2 = np.array((0.0, 0.0, 1.0), dtype=float)
        v2 = _stable_sign(v2)
        if v2 is None:
            v2 = np.array((0.0, 1.0, 0.0), dtype=float)
    return v1, v2

def _knn_cell_floor(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    n = arr.shape[0]
    if cKDTree is None or n < (NN_K + 1):
        return None
    try:
        tree = cKDTree(arr)
        dists, _ = tree.query(arr, k=NN_K + 1)
    except Exception:
        return None
    if dists.ndim != 2 or dists.shape[1] < (NN_K + 1):
        return None
    nn = dists[:, 1:NN_K + 1]
    per_point = np.nanmedian(nn, axis=1)
    return float(np.nanmedian(per_point))

def _fit_tube_model(tube, row_idx):
    idx = np.asarray(row_idx, dtype=np.int64)
    if idx.size == 0:
        return None
    x = np.asarray(tube, dtype=float)[idx]
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] == 0:
        return None
    median = np.nanmedian(x, axis=0)
    mad = _mad(x, axis=0)
    scale = MAD_SCALE * np.asarray(mad, dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)

    whitened = (x - median) / scale
    v1, v2 = _principal_vectors(whitened)

    proj1 = np.dot(whitened, v1)
    residual = whitened - np.outer(proj1, v1)
    d = np.linalg.norm(residual, axis=1)
    spread = MAD_SCALE * np.nanmedian(np.abs(d - np.nanmedian(d))) + EPSILON

    floor = _knn_cell_floor(whitened)
    floor_min = BASELINE_FLOOR_SCALE * spread
    if floor is None or not np.isfinite(floor):
        floor = floor_min
    else:
        floor = max(float(floor), float(floor_min))

    return {
        "m": median.astype(float),
        "q": scale.astype(float),
        "v1": v1.astype(float),
        "v2": v2.astype(float),
        "s": float(spread),
        "n_floor": float(floor),
    }

def _build_global_model(tube):
    model = _fit_tube_model(tube, np.arange(tube.shape[0], dtype=np.int64))
    if model is not None:
        return model
    return {
        "m": np.zeros(3, dtype=float),
        "q": np.ones(3, dtype=float),
        "v1": np.array((1.0, 0.0, 0.0), dtype=float),
        "v2": np.array((0.0, 1.0, 0.0), dtype=float),
        "s": 1.0,
        "n_floor": EPSILON,
    }

def _concat_parts(parts):
    if not parts:
        return np.array([], dtype=np.int64)
    if len(parts) == 1:
        return parts[0].astype(np.int64, copy=False)
    return np.concatenate(parts)

def _collect_rows(cell_rows, z_bin, i_min, i_max, n_i_bins):
    base = z_bin * n_i_bins
    parts = []
    for i_bin in range(i_min, i_max + 1):
        rows = cell_rows.get(base + i_bin)
        if rows is not None and rows.size:
            parts.append(rows)
    return _concat_parts(parts)

def _nearest_populated_cell(z_bin, i_bin, exclude_key, n_i_bins, populated_keys):
    best_key = -1
    best_dist = None
    for key in populated_keys:
        if key == exclude_key:
            continue
        kz = key // n_i_bins
        ki = key - kz * n_i_bins
        dist = abs(kz - z_bin) + abs(ki - i_bin)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_key = key
    if best_key < 0:
        return None
    return best_key

def _resolve_training_rows(z_bin, i_bin, cell_rows, n_z_bins, n_i_bins, populated_keys):
    base_key = z_bin * n_i_bins + i_bin
    rows = _collect_rows(cell_rows, z_bin, i_bin, i_bin, n_i_bins)
    if rows.size >= NEED_ROWS:
        return rows

    i_lo = i_bin
    i_hi = i_bin
    while rows.size < NEED_ROWS and (i_lo > 0 or i_hi < n_i_bins - 1):
        i_lo = max(i_lo - 1, 0)
        i_hi = min(i_hi + 1, n_i_bins - 1)
        rows = _collect_rows(cell_rows, z_bin, i_lo, i_hi, n_i_bins)

    if rows.size < NEED_ROWS:
        for z_candidate in (z_bin - 1, z_bin + 1):
            if 0 <= z_candidate < n_z_bins:
                extra = _collect_rows(cell_rows, z_candidate, i_lo, i_hi, n_i_bins)
                if extra.size:
                    if rows.size:
                        rows = np.concatenate((rows, extra))
                    else:
                        rows = extra

    if rows.size >= BACKFILL_MIN_ROWS:
        return rows

    nearest_key = _nearest_populated_cell(z_bin, i_bin, base_key, n_i_bins, populated_keys)
    if nearest_key is None:
        return np.array([], dtype=np.int64)
    return cell_rows.get(nearest_key, np.array([], dtype=np.int64))

def _tube_scores(tube_values, z_bins_idx, i_bins_idx, n_z_bins, n_i_bins, cell_rows, global_model):
    n = tube_values.shape[0]
    abs_scores = np.full(n, np.nan, dtype=float)
    signed_scores = np.full(n, np.nan, dtype=float)

    populated_keys = tuple(sorted(k for k in cell_rows.keys() if k >= 0))
    model_cache = {}

    for cell_key, row_idx in cell_rows.items():
        if cell_key < 0:
            model = global_model
        else:
            model = model_cache.get(cell_key)
            if model is None:
                z_bin = cell_key // n_i_bins
                i_bin = cell_key - z_bin * n_i_bins
                train_idx = _resolve_training_rows(
                    z_bin, i_bin, cell_rows, n_z_bins, n_i_bins, populated_keys
                )
                if train_idx.size == 0:
                    model = global_model
                else:
                    candidate_model = _fit_tube_model(tube_values, train_idx)
                    model = candidate_model if candidate_model is not None else global_model
                model_cache[cell_key] = model

        block = np.asarray(tube_values, dtype=float)[row_idx]
        finite = np.isfinite(block).all(axis=1)
        if not np.any(finite):
            continue
        rows = row_idx[finite]
        xw = (block[finite] - model["m"]) / model["q"]
        proj = np.dot(xw, model["v1"])
        residual = xw - np.outer(proj, model["v1"])
        d = np.linalg.norm(residual, axis=1)
        deconv = np.maximum(d - model["n_floor"], 0.0) / model["s"]

        abs_scores[rows] = np.clip(deconv, 0.0, 10.0)
        signed_scores[rows] = np.clip(np.sign(np.dot(xw, model["v2"]) * deconv), -10.0, 10.0)

    return abs_scores, signed_scores

def add_error_deconvolved_locus_tube_residuals(raw, deps, aux):
    raw = raw

    u = raw["u"].to_numpy(dtype=float)
    g = raw["g"].to_numpy(dtype=float)
    r = raw["r"].to_numpy(dtype=float)
    i = raw["i"].to_numpy(dtype=float)
    z = raw["z"].to_numpy(dtype=float)
    redshift = raw["redshift"].to_numpy(dtype=float)

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z

    z_edges = np.asarray(Z_BIN_EDGES, dtype=float)
    i_edges = np.asarray(I_BIN_EDGES, dtype=float)
    n_z_bins = z_edges.shape[0] - 1
    n_i_bins = i_edges.shape[0] - 1

    z_idx = np.searchsorted(z_edges, redshift, side="right") - 1
    i_idx = np.searchsorted(i_edges, i, side="right") - 1

    z_valid = np.isfinite(redshift)
    i_valid = np.isfinite(i)
    z_idx = np.where(z_valid, np.clip(z_idx, 0, n_z_bins - 1), -1)
    i_idx = np.where(i_valid, np.clip(i_idx, 0, n_i_bins - 1), -1)

    n_rows = raw.shape[0]
    cell_rows_dict = defaultdict(list)
    for pos in range(n_rows):
        zi = int(z_idx[pos])
        ii = int(i_idx[pos])
        if zi < 0 or ii < 0:
            key = -1
        else:
            key = zi * n_i_bins + ii
        cell_rows_dict[key].append(pos)

    cell_rows = {int(k): np.asarray(v, dtype=np.int64) for k, v in cell_rows_dict.items()}

    A = np.column_stack((c1, c2, c3)).astype(float, copy=False)
    B = np.column_stack((c2, c3, c4)).astype(float, copy=False)

    global_model_a = _build_global_model(A)
    global_model_b = _build_global_model(B)

    A_abs, A_signed = _tube_scores(A, z_idx, i_idx, n_z_bins, n_i_bins, cell_rows, global_model_a)
    B_abs, B_signed = _tube_scores(B, z_idx, i_idx, n_z_bins, n_i_bins, cell_rows, global_model_b)

    abs_mid = np.abs((A_abs + B_abs) * 0.5)
    g_2_4_3_0 = abs_mid * ((redshift >= 2.4) & (redshift < 3.0)).astype(float)
    g_3_5_plus = abs_mid * (redshift >= 3.5).astype(float)

    return pd.DataFrame(
        {
            "A_abs": A_abs,
            "A_signed": A_signed,
            "B_abs": B_abs,
            "B_signed": B_signed,
            "g_2_4_3_0": g_2_4_3_0,
            "g_3_5_plus": g_3_5_plus,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "error_deconvolved_locus_tube_residuals",
        "fn": add_error_deconvolved_locus_tube_residuals,
        "depends_on": [],
        "description": (
            "Build redshift- and i-band-binned color-tube models with adaptive widening, "
            "5-NN deconvolution floors, and signed/unsigned orthogonal tube residual features with contextual redshift gates."
        ),
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