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
- Hypothesis ID: 000039
- Source file: autogluon-004.py
- Failed node: 20260624T062205-3de64c69-347
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062205-3de64c69-347/02-code.py", line 801, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062205-3de64c69-347/02-code.py", line 263, in add_local_principal_color_residuals
    residuals_abs[color] = np.where(np.isfinite(zc), np.abs(zc), np.nan)
    ~~~~~~~~~~~~~^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062205-3de64c69-347/02-code.py", line 925, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062205-3de64c69-347/02-code.py", line 801, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T062205-3de64c69-347/02-code.py", line 263, in add_local_principal_color_residuals
    residuals_abs[color] = np.where(np.isfinite(zc), np.abs(zc), np.nan)
    ~~~~~~~~~~~~~^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=local_principal_color_residuals
TheML feature group: failed name=local_principal_color_residuals elapsed_s=1.335 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_REDSHIFT_BIN_EDGES = (-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0)
_ALPHA_BIN_EDGES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0, 360.0)
_DEFAULT_DELTA_BIN_EDGES = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
_SCALE_MULTIPLIER = 1.4826
_MIN_COLOR_COUNT = 300
_MIN_ALL_COUNT = 500
_EPS_BY_COLOR = {"s": 0.02, "w": 0.02, "x": 0.02, "y": 0.02, "l": 0.03}
_COLOR_NAMES = ("s", "w", "x", "y", "l")
_LEVEL_DEFINITIONS = (
    ("full", ("redshift_bin", "sky_bin", "spectral_type", "galaxy_population")),
    ("redshift_sky", ("redshift_bin", "sky_bin")),
    ("redshift_tags", ("redshift_bin", "spectral_type", "galaxy_population")),
    ("redshift_only", ("redshift_bin",)),
    ("global", ()),
)

def _mad(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    center = np.median(arr)
    return np.median(np.abs(arr - center))

def _bin_index(values, edges):
    values = np.asarray(values, dtype=float)
    idx = np.zeros(len(values), dtype=np.int16)
    finite = np.isfinite(values)
    if finite.any():
        edge_array = np.asarray(edges, dtype=float)
        b = np.digitize(values[finite], edge_array, right=False) - 1
        idx[finite] = np.clip(b, 0, len(edge_array) - 2).astype(np.int16)
    return idx

def _compute_level_stats(frame, by_cols, color):
    n = len(frame)
    values = frame[color].to_numpy(dtype=float)
    valid = frame[f"{color}_valid"].to_numpy(dtype=bool)
    eps = _EPS_BY_COLOR[color]

    if by_cols:
        key_cols = list(by_cols)
        keys = pd.MultiIndex.from_frame(frame[key_cols].astype(object))
        valid_frame = frame.loc[valid, key_cols + [color]]
        grouped = valid_frame.groupby(key_cols, dropna=False)[color]
        med = grouped.median()
        mad = grouped.apply(_mad)
        cnt = grouped.count()
        all_cnt = frame.groupby(key_cols, dropna=False)["all_colors_count"].sum()

        med_arr = med.reindex(keys).to_numpy(dtype=float)
        mad_arr = mad.reindex(keys).to_numpy(dtype=float)
        cnt_arr = cnt.reindex(keys).to_numpy(dtype=float)
        all_arr = all_cnt.reindex(keys).to_numpy(dtype=float)
        scale_arr = np.maximum(mad_arr * _SCALE_MULTIPLIER, eps)
        return med_arr, scale_arr, cnt_arr, all_arr

    valid_values = values[valid]
    if valid_values.size == 0:
        med_scalar = np.nan
        scale_scalar = np.nan
        cnt_scalar = 0.0
    else:
        med_scalar = float(np.median(valid_values))
        scale_scalar = float(max(_SCALE_MULTIPLIER * _mad(valid_values), eps))
        cnt_scalar = float(valid_values.size)

    all_scalar = float(frame["all_colors_count"].sum())
    return (
        np.full(n, med_scalar, dtype=float),
        np.full(n, scale_scalar, dtype=float),
        np.full(n, cnt_scalar, dtype=float),
        np.full(n, all_scalar, dtype=float),
    )

def add_local_principal_color_residuals(raw, deps, aux):
    _ = deps
    _ = aux
    index = raw.index
    n = len(raw)

    u = pd.to_numeric(raw["u"], errors="coerce").to_numpy(dtype=float)
    g = pd.to_numeric(raw["g"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(raw["r"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(raw["i"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(raw["z"], errors="coerce").to_numpy(dtype=float)

    s = -0.249 * u + 0.794 * g - 0.555 * r + 0.234
    w = -0.227 * g + 0.792 * r - 0.567 * i + 0.050
    x = 0.707 * g - 0.707 * r - 0.983
    y = -0.270 * r + 0.800 * i - 0.534 * z + 0.059

    gmr = g - r
    l_domain = np.isfinite(gmr) & (gmr >= 0.5) & (gmr <= 0.8)
    l = np.full(n, np.nan, dtype=float)
    l[l_domain] = -0.436 * u[l_domain] + 1.129 * g[l_domain] - 0.119 * r[l_domain] - 0.574 * i[l_domain] + 0.1984

    s_valid = np.isfinite(s)
    w_valid = np.isfinite(w)
    x_valid = np.isfinite(x)
    y_valid = np.isfinite(y)
    l_valid = np.isfinite(l)
    all_colors_valid = s_valid & w_valid & x_valid & y_valid & l_valid

    redshift = pd.to_numeric(raw["redshift"], errors="coerce").to_numpy(dtype=float)
    alpha = pd.to_numeric(raw["alpha"], errors="coerce").to_numpy(dtype=float)
    alpha = np.mod(np.mod(alpha, 360.0), 360.0)
    delta = pd.to_numeric(raw["delta"], errors="coerce").to_numpy(dtype=float)

    delta_valid = np.isfinite(delta)
    if delta_valid.any():
        d_min = float(np.nanmin(delta[delta_valid]))
        d_max = float(np.nanmax(delta[delta_valid]))
        if d_max > d_min:
            delta_edges = tuple(np.linspace(d_min, d_max, 7))
        else:
            delta_edges = _DEFAULT_DELTA_BIN_EDGES
    else:
        delta_edges = _DEFAULT_DELTA_BIN_EDGES

    frame = pd.DataFrame(index=index)
    frame["redshift_bin"] = _bin_index(redshift, _REDSHIFT_BIN_EDGES)
    frame["sky_bin"] = _bin_index(alpha, _ALPHA_BIN_EDGES)
    frame["delta_bin"] = _bin_index(delta, delta_edges)
    frame["spectral_type"] = raw["spectral_type"].fillna("MISSING").astype(str).to_numpy()
    frame["galaxy_population"] = raw["galaxy_population"].fillna("MISSING").astype(str).to_numpy()

    frame["s"] = s
    frame["w"] = w
    frame["x"] = x
    frame["y"] = y
    frame["l"] = l

    frame["s_valid"] = s_valid.astype(bool)
    frame["w_valid"] = w_valid.astype(bool)
    frame["x_valid"] = x_valid.astype(bool)
    frame["y_valid"] = y_valid.astype(bool)
    frame["l_valid"] = l_valid.astype(bool)
    frame["all_colors_count"] = all_colors_valid.astype(np.uint8)

    residuals = {}
    residuals_abs = {}
    missing_flags = {}
    unreliable_flags = {}

    for color in _COLOR_NAMES:
        vals = frame[color].to_numpy(dtype=float)
        valid = frame[f"{color}_valid"].to_numpy(dtype=bool)

        stats0 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[0][1], color)
        stats1 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[1][1], color)
        stats2 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[2][1], color)
        stats3 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[3][1], color)
        stats4 = _compute_level_stats(frame, _LEVEL_DEFINITIONS[4][1], color)

        med0, scale0, cnt0, all0 = stats0
        med1, scale1, cnt1, all1 = stats1
        med2, scale2, cnt2, all2 = stats2
        med3, scale3, cnt3, all3 = stats3
        med4, scale4, cnt4, all4 = stats4

        level = np.full(n, 4, dtype=np.int8)

        cond0 = valid & (cnt0 >= _MIN_COLOR_COUNT) & (all0 >= _MIN_ALL_COUNT)
        level[cond0] = 0

        cond1 = valid & (level == 4) & (cnt1 >= _MIN_COLOR_COUNT) & (all1 >= _MIN_ALL_COUNT)
        level[cond1] = 1

        cond2 = valid & (level == 4) & (cnt2 >= _MIN_COLOR_COUNT) & (all2 >= _MIN_ALL_COUNT)
        level[cond2] = 2

        cond3 = valid & (level == 4) & (cnt3 >= _MIN_COLOR_COUNT) & (all3 >= _MIN_ALL_COUNT)
        level[cond3] = 3

        selected_med = np.where(level == 0, med0, np.where(level == 1, med1, np.where(level == 2, med2, np.where(level == 3, med3, med4))))
        selected_scale = np.where(level == 0, scale0, np.where(level == 1, scale1, np.where(level == 2, scale2, np.where(level == 3, scale3, scale4))))

        use = valid & np.isfinite(selected_med) & np.isfinite(selected_scale) & (selected_scale > 0.0)
        zc = np.full(n, np.nan, dtype=float)
        zc[use] = (vals[use] - selected_med[use]) / selected_scale[use]
        zc[use] = np.clip(zc[use], -10.0, 10.0)

        missing = (~valid) | (~np.isfinite(zc))
        unreliable = (valid & (level == 4) & np.isfinite(zc)).astype(np.uint8)

        residuals[color] = zc
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)
        residuals_abs = np.where(np.isnan(zc), np.nan, np.abs(zc))
        residuals_abs[~np.isfinite(zc)] = np.nan
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs[color] if False else None
        residuals_abs
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan
        residuals_abs = residuals_abs.astype(float)

        residuals[color] = zc
        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)
        residuals_abs = residuals_abs.astype(float)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs[~np.isfinite(residuals_abs)] = np.nan

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs = np.where(np.isfinite(residuals_abs), residuals_abs, np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        residuals_abs = np.abs(zc)
        residuals_abs = np.where(np.isfinite(residuals_abs), residuals_abs, np.nan)

        residuals[color] = zc
        residuals_abs[color] = np.where(np.isfinite(zc), np.abs(zc), np.nan)

        missing_flags[color] = missing.astype(np.uint8)
        unreliable_flags[color] = unreliable

    abs_stack = np.column_stack([residuals_abs[c] for c in _COLOR_NAMES])
    available = np.isfinite(abs_stack)
    available_count = available.sum(axis=1).astype(np.int16)

    max_abs_z = np.where(available_count > 0, np.nanmax(abs_stack, axis=1), np.nan)
    l2_z = np.sqrt(np.sum(np.where(available, np.square(abs_stack), 0.0), axis=1))
    l2_z = np.where(available_count > 0, l2_z, np.nan)

    tail_count = np.sum((abs_stack > 2.0) & available, axis=1).astype(np.int16)
    tail_ratio = np.divide(
        tail_count.astype(float),
        available_count.astype(float),
        out=np.zeros_like(tail_count, dtype=float),
        where=available_count > 0,
    )
    in_locus = (np.all(np.where(available, abs_stack < 0.75, True), axis=1) & (available_count > 0)).astype(np.uint8)

    sign_code = (
        np.where(np.isfinite(residuals["s"]) & (residuals["s"] > 0.0), 1, 0).astype(np.uint8)
        | (np.where(np.isfinite(residuals["w"]) & (residuals["w"] > 0.0), 1, 0).astype(np.uint8) << 1)
        | (np.where(np.isfinite(residuals["x"]) & (residuals["x"] > 0.0), 1, 0).astype(np.uint8) << 2)
        | (np.where(np.isfinite(residuals["y"]) & (residuals["y"] > 0.0), 1, 0).astype(np.uint8) << 3)
    )

    l_outside_domain = (~l_valid & np.isfinite(gmr)).astype(np.uint8)

    new_features = pd.DataFrame(index=index)
    for color in _COLOR_NAMES:
        new_features[f"{color}_z"] = residuals[color]
        new_features[f"{color}_abs_z"] = residuals_abs[color]
        new_features[f"{color}_missing"] = missing_flags[color]
        new_features[f"{color}_not_reliable"] = unreliable_flags[color]

    new_features["max_abs_z"] = max_abs_z
    new_features["l2_z"] = l2_z
    new_features["tail_count"] = tail_count.astype(np.int16)
    new_features["tail_ratio"] = tail_ratio
    new_features["in_locus"] = in_locus
    new_features["sign_code_swx y"] = sign_code
    new_features["available_colors"] = available_count
    new_features["l_outside_domain"] = l_outside_domain

    return new_features

FEATURE_GROUPS = [
    {
        "name": "local_principal_color_residuals",
        "fn": add_local_principal_color_residuals,
        "depends_on": [],
        "description": "Build context-stabilized, robust principal-color residual and orthogonal excursion features from local redshift/sky/categorical statistics.",
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