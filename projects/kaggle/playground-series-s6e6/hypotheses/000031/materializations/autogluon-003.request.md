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
- Hypothesis ID: 000031
- Source file: autogluon-002.py
- Failed node: 20260624T061452-7ef9300f-339
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py:327: DeprecationWarning: the `interpolation=` argument to quantile was renamed to `method=`, which has additional options.
Users of the modes 'nearest', 'lower', 'higher', or 'midpoint' are encouraged to review the method they used. (Deprecated NumPy 1.22)
  edges = _build_equal_freq_edges(z)
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py", line 858, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py", line 330, in add_redshift_adaptive_color_tube_residuals
    bin_idx = np.clip(np.searchsorted(edges, z_clipped, side="right") - 1, 0, edges.size - 2).astype(np.int64)
                                             ^^^^^^^^^
NameError: name 'z_clipped' is not defined. Did you mean: 'z_clamped'?
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py", line 982, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py", line 858, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T061452-7ef9300f-339/02-code.py", line 330, in add_redshift_adaptive_color_tube_residuals
    bin_idx = np.clip(np.searchsorted(edges, z_clipped, side="right") - 1, 0, edges.size - 2).astype(np.int64)
                                             ^^^^^^^^^
NameError: name 'z_clipped' is not defined. Did you mean: 'z_clamped'?

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_adaptive_color_tube_residuals
TheML feature group: failed name=redshift_adaptive_color_tube_residuals elapsed_s=0.063 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

NUM_EQUAL_FREQ_BINS = 30
MIN_BIN_ROWS = 5000
MAD_SCALE = 1.4826
SCALE_FLOOR = 1e-6
CLIP_PERCENTILE = 99.5
OVERLAP_GAUSS_CENTER = 2.7
OVERLAP_GAUSS_SIGMA = 0.45
RATIO_DENOM_EPS = 1e-4

def _to_float(values):
    return np.asarray(values, dtype=np.float64)

def _robust_mad(values):
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return med, mad

def _empty_cube_param():
    return {
        "valid": False,
        "center": np.zeros(3, dtype=np.float64),
        "color_scale": np.ones(3, dtype=np.float64),
        "components": np.eye(3, dtype=np.float64),
        "scale_t": SCALE_FLOOR,
        "scale_q2": SCALE_FLOOR,
        "scale_q3": SCALE_FLOOR,
        "scale_d": SCALE_FLOOR,
        "clip_t": 0.0,
        "clip_q2": 0.0,
        "clip_q3": 0.0,
        "clip_d": 0.0,
    }

def _copy_cube_param(p):
    return {
        "valid": bool(p["valid"]),
        "center": np.array(p["center"], copy=True),
        "color_scale": np.array(p["color_scale"], copy=True),
        "components": np.array(p["components"], copy=True),
        "scale_t": float(p["scale_t"]),
        "scale_q2": float(p["scale_q2"]),
        "scale_q3": float(p["scale_q3"]),
        "scale_d": float(p["scale_d"]),
        "clip_t": float(p["clip_t"]),
        "clip_q2": float(p["clip_q2"]),
        "clip_q3": float(p["clip_q3"]),
        "clip_d": float(p["clip_d"]),
    }

def _build_equal_freq_edges(redshift_nonneg):
    z = np.asarray(redshift_nonneg, dtype=np.float64)
    z = np.maximum(z, 0.0)
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax):
        zmin = 0.0
        zmax = 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    probs = np.linspace(0.0, 1.0, NUM_EQUAL_FREQ_BINS + 1)
    quant = np.quantile(z, probs, interpolation="linear")
    edges = [float(quant[0])]
    for v in quant[1:]:
        if float(v) > edges[-1] + 1e-12:
            edges.append(float(v))

    if len(edges) < 2:
        edges = [zmin, zmax]

    edges = np.array(edges, dtype=np.float64)
    return _merge_small_bins(z, edges)

def _merge_small_bins(z, edges):
    edges = np.array(edges, dtype=np.float64)
    if edges.size < 2:
        return np.array([0.0, 1.0], dtype=np.float64)

    while True:
        if edges.size <= 2:
            break

        bin_idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, edges.size - 2)
        counts = np.bincount(bin_idx, minlength=edges.size - 1)
        small_bins = np.flatnonzero(counts < MIN_BIN_ROWS)

        if small_bins.size == 0:
            break

        b = int(small_bins[0])
        if b == 0:
            remove = 1
        elif b == counts.size - 1:
            remove = b
        else:
            left = counts[b - 1]
            right = counts[b + 1]
            remove = b if left <= right else b + 1

        if remove <= 0 or remove >= edges.size - 1:
            break

        edges = np.delete(edges, remove)

    if edges.size < 2:
        edges = np.array([np.nanmin(z), np.nanmax(z)], dtype=np.float64)
        if not np.isfinite(edges[0]) or not np.isfinite(edges[1]) or edges[1] <= edges[0]:
            edges[1] = edges[0] + 1.0

    return edges

def _nearest_index(reference_centers, candidates, idx):
    c = np.asarray(candidates, dtype=np.int64)
    deltas = np.abs(reference_centers[c] - reference_centers[idx])
    return int(c[int(np.argmin(deltas))])

def _fit_bin_cube_params(colors, row_idx):
    p = _empty_cube_param()
    if row_idx.size == 0:
        return p

    x = colors[row_idx]
    if x.size == 0 or x.shape[0] < 3:
        return p

    center, mad = _robust_mad(x)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(mad)):
        return p
    if np.any(mad <= 0.0):
        return p

    color_scale = np.maximum(mad * MAD_SCALE, SCALE_FLOOR)
    x_std = (x - center) / color_scale

    cov = np.cov(x_std, rowvar=False, bias=True)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        return p

    eigvals, eigvecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
        return p

    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order].T
    if not np.all(np.isfinite(components)):
        return p

    proj = x_std @ components.T
    t = proj[:, 0]
    q2 = proj[:, 1]
    q3 = proj[:, 2]
    d = np.sqrt(q2 * q2 + q3 * q3)

    med_t, mad_t = _robust_mad(t)
    med_q2, mad_q2 = _robust_mad(q2)
    med_q3, mad_q3 = _robust_mad(q3)
    med_d, mad_d = _robust_mad(d)

    if not np.isfinite(med_t) or not np.isfinite(med_q2) or not np.isfinite(med_q3) or not np.isfinite(med_d):
        return p
    if mad_t <= 0.0 or mad_q2 <= 0.0 or mad_q3 <= 0.0 or mad_d <= 0.0:
        return p

    scale_t = max(mad_t * MAD_SCALE, SCALE_FLOOR)
    scale_q2 = max(mad_q2 * MAD_SCALE, SCALE_FLOOR)
    scale_q3 = max(mad_q3 * MAD_SCALE, SCALE_FLOOR)
    scale_d = max(mad_d * MAD_SCALE, SCALE_FLOOR)

    t_hat = t / scale_t
    q2_hat = q2 / scale_q2
    q3_hat = q3 / scale_q3
    d_hat = d / scale_d

    clip_t = float(np.nanpercentile(np.abs(t_hat), CLIP_PERCENTILE))
    clip_q2 = float(np.nanpercentile(np.abs(q2_hat), CLIP_PERCENTILE))
    clip_q3 = float(np.nanpercentile(np.abs(q3_hat), CLIP_PERCENTILE))
    clip_d = float(np.nanpercentile(np.abs(d_hat), CLIP_PERCENTILE))

    p["valid"] = True
    p["center"] = center
    p["color_scale"] = color_scale
    p["components"] = components
    p["scale_t"] = scale_t
    p["scale_q2"] = scale_q2
    p["scale_q3"] = scale_q3
    p["scale_d"] = scale_d
    p["clip_t"] = clip_t
    p["clip_q2"] = clip_q2
    p["clip_q3"] = clip_q3
    p["clip_d"] = clip_d
    return p

def _prepare_cube_params(colors, bin_idx, n_bins, bin_centers, merged_from_small):
    base = []
    for b in range(n_bins):
        rows = np.flatnonzero(bin_idx == b)
        base.append(_fit_bin_cube_params(colors, rows))

    # Deterministic sign convention with continuity across adjacent bins.
    ones = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    prev = None
    for b in range(n_bins):
        if not base[b]["valid"]:
            continue
        comps = np.array(base[b]["components"], copy=True)

        if np.dot(comps[0], ones) < 0.0:
            comps = -comps

        if prev is not None:
            for k in range(3):
                if np.dot(comps[k], prev[k]) < 0.0:
                    comps[k] = -comps[k]

        base[b]["components"] = comps
        prev = comps

    valid_indices = [i for i in range(n_bins) if base[i]["valid"] and (not merged_from_small[i])]
    if len(valid_indices) == 0:
        fallback_flags = np.ones(n_bins, dtype=np.int8)
        return [_copy_cube_param(_empty_cube_param()) for _ in range(n_bins)], fallback_flags

    resolved = []
    fallback_flags = np.zeros(n_bins, dtype=np.int8)
    for b in range(n_bins):
        needs_fallback = (not base[b]["valid"]) or bool(merged_from_small[b])
        if not needs_fallback:
            resolved.append(_copy_cube_param(base[b]))
            continue

        nearest = _nearest_index(bin_centers, np.array(valid_indices, dtype=np.int64), b)
        resolved.append(_copy_cube_param(base[nearest]))
        fallback_flags[b] = 1

    return resolved, fallback_flags

def _build_cube_feature_block(prefix, colors, overlap_weight, bin_idx, params):
    n = colors.shape[0]
    t_hat = np.zeros(n, dtype=np.float64)
    q2_hat = np.zeros(n, dtype=np.float64)
    q3_hat = np.zeros(n, dtype=np.float64)
    d_hat = np.zeros(n, dtype=np.float64)

    t_hat_clipped = np.zeros(n, dtype=np.float64)
    q2_hat_clipped = np.zeros(n, dtype=np.float64)
    q3_hat_clipped = np.zeros(n, dtype=np.float64)
    d_hat_clipped = np.zeros(n, dtype=np.float64)

    sign2 = np.zeros(n, dtype=np.float64)
    sign3 = np.zeros(n, dtype=np.float64)

    for b, p in enumerate(params):
        idx = np.flatnonzero(bin_idx == b)
        if idx.size == 0:
            continue

        x = colors[idx]
        x_std = (x - p["center"]) / p["color_scale"]
        proj = x_std @ p["components"].T

        t = proj[:, 0]
        q2 = proj[:, 1]
        q3 = proj[:, 2]
        d = np.sqrt(q2 * q2 + q3 * q3)

        t_u = t / p["scale_t"]
        q2_u = q2 / p["scale_q2"]
        q3_u = q3 / p["scale_q3"]
        d_u = d / p["scale_d"]

        t_hat[idx] = t_u
        q2_hat[idx] = q2_u
        q3_hat[idx] = q3_u
        d_hat[idx] = d_u

        t_hat_clipped[idx] = np.clip(t_u, -p["clip_t"], p["clip_t"])
        q2_hat_clipped[idx] = np.clip(q2_u, -p["clip_q2"], p["clip_q2"])
        q3_hat_clipped[idx] = np.clip(q3_u, -p["clip_q3"], p["clip_q3"])
        d_hat_clipped[idx] = np.clip(d_u, -p["clip_d"], p["clip_d"])

        sign2[idx] = np.sign(q2_u)
        sign3[idx] = np.sign(q3_u)

    ratio_t_over_q2 = t_hat / (np.abs(q2_hat) + RATIO_DENOM_EPS)
    ratio_t_over_q3 = t_hat / (np.abs(q3_hat) + RATIO_DENOM_EPS)
    g_d_hat = overlap_weight * d_hat
    g_absq2_hat = overlap_weight * np.abs(q2_hat)

    return {
        f"{prefix}_t_hat": t_hat,
        f"{prefix}_q2_hat": q2_hat,
        f"{prefix}_q3_hat": q3_hat,
        f"{prefix}_d_hat": d_hat,
        f"{prefix}_t_hat_clipped": t_hat_clipped,
        f"{prefix}_q2_hat_clipped": q2_hat_clipped,
        f"{prefix}_q3_hat_clipped": q3_hat_clipped,
        f"{prefix}_d_hat_clipped": d_hat_clipped,
        f"{prefix}_sign2": sign2,
        f"{prefix}_sign3": sign3,
        f"{prefix}_ratio_t_over_abs_q2": ratio_t_over_q2,
        f"{prefix}_ratio_t_over_abs_q3": ratio_t_over_q3,
        f"{prefix}_g_d_hat": g_d_hat,
        f"{prefix}_g_abs_q2_hat": g_absq2_hat,
    }

def add_redshift_adaptive_color_tube_residuals(raw, deps, aux):
    z = _to_float(raw["redshift"])
    z = np.maximum(z, 0.0)

    edges = _build_equal_freq_edges(z)

    z_clamped = np.clip(z, edges[0], edges[-1])
    bin_idx = np.clip(np.searchsorted(edges, z_clipped, side="right") - 1, 0, edges.size - 2).astype(np.int64)
    n_bins = edges.size - 1

    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    merged_from_small = bin_counts < MIN_BIN_ROWS
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    u = _to_float(raw["u"])
    g = _to_float(raw["g"])
    r = _to_float(raw["r"])
    i = _to_float(raw["i"])
    z_band = _to_float(raw["z"])

    c1 = u - g
    c2 = g - r
    c3 = r - i
    c4 = i - z_band

    ugri = np.column_stack((c1, c2, c3))
    griz = np.column_stack((c2, c3, c4))

    ugri_params, ugri_fallback_by_bin = _prepare_cube_params(ugri, bin_idx, n_bins, bin_centers, merged_from_small)
    griz_params, griz_fallback_by_bin = _prepare_cube_params(griz, bin_idx, n_bins, bin_centers, merged_from_small)

    overlap_weight = np.exp(-0.5 * ((z - OVERLAP_GAUSS_CENTER) / OVERLAP_GAUSS_SIGMA) ** 2)

    ugri_feats = _build_cube_feature_block("ugri", ugri, overlap_weight, bin_idx, ugri_params)
    griz_feats = _build_cube_feature_block("griz", griz, overlap_weight, bin_idx, griz_params)

    fallback_ugri = ugri_fallback_by_bin[bin_idx].astype(np.int8)
    fallback_griz = griz_fallback_by_bin[bin_idx].astype(np.int8)

    features = {}
    features.update(ugri_feats)
    features.update(griz_feats)
    features["ugri_fallback_from_bin"] = fallback_ugri
    features["griz_fallback_from_bin"] = fallback_griz
    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "redshift_adaptive_color_tube_residuals",
        "fn": add_redshift_adaptive_color_tube_residuals,
        "depends_on": [],
        "description": "Builds redshift-adaptive PCA color-tube residual coordinates in Ugri and Griz spaces with continuity-stable orientation and overlap-window emphasized residual features.",
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