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
- Hypothesis ID: 000020
- Source file: autogluon-002.py
- Failed node: 20260624T060107-8308e982-326
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T060107-8308e982-326/02-code.py", line 652, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T060107-8308e982-326/02-code.py", line 105, in add_survey_depth_limit_margins
    for lo, hi in __R_INTERVALS:
                  ^^^^^^^^^^^^^
NameError: name '__R_INTERVALS' is not defined. Did you mean: '_R_INTERVALS'?
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T060107-8308e982-326/02-code.py", line 776, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T060107-8308e982-326/02-code.py", line 652, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T060107-8308e982-326/02-code.py", line 105, in add_survey_depth_limit_margins
    for lo, hi in __R_INTERVALS:
                  ^^^^^^^^^^^^^
NameError: name '__R_INTERVALS' is not defined. Did you mean: '_R_INTERVALS'?

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=survey_depth_limit_margins
TheML feature group: failed name=survey_depth_limit_margins elapsed_s=0.059 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_QSO_THRESHOLDS = (
    ("i", (15.0, 19.1, 20.2, 20.4, 21.3, 22.45)),
    ("g", (22.0,)),
    ("r", (21.85,)),
)
_GALAXY_R_THRESHOLDS = (17.77, 19.2, 19.5)
_I_INTERVALS = (
    (15.0, 19.1),
    (15.0, 20.2),
    (20.2, 21.3),
    (21.3, 22.45),
)
_R_INTERVALS = (
    (17.77, 19.2),
    (19.2, 19.5),
)

def _threshold_token(value):
    return str(value).replace(".", "_")

def _as_float_array(values):
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

def _describe_regime(margin_matrix, band_names):
    abs_matrix = np.abs(np.where(np.isnan(margin_matrix), np.inf, margin_matrix))
    min_abs = np.min(abs_matrix, axis=1)
    min_abs = np.where(np.isfinite(min_abs), min_abs, 0.0)

    nearest_idx = np.argmin(abs_matrix, axis=1)
    valid = np.isfinite(min_abs)

    nearest_band = np.array(["none"] * len(margin_matrix), dtype=object)
    nearest_sign = np.zeros(len(margin_matrix), dtype=np.int8)

    if np.any(valid):
        name_lookup = np.array(band_names, dtype=object)
        nearest_band[valid] = name_lookup[nearest_idx[valid]]
        nearest_margin = margin_matrix[np.arange(len(margin_matrix)), nearest_idx]
        nearest_sign[valid] = np.sign(nearest_margin[valid]).astype(np.int8)

    return min_abs, nearest_band, nearest_sign

def add_survey_depth_limit_margins(raw, deps, aux):
    del deps, aux

    i = _as_float_array(raw["i"])
    g = _as_float_array(raw["g"])
    r = _as_float_array(raw["r"])

    i_finite = np.isfinite(i)
    g_finite = np.isfinite(g)
    r_finite = np.isfinite(r)

    features = {}

    for threshold in _GALAXY_R_THRESHOLDS:
        label = _threshold_token(threshold)
        margin = r - float(threshold)
        features[f"margin_r_{label}"] = np.where(r_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[0][1]:
        label = _threshold_token(threshold)
        margin = i - float(threshold)
        features[f"margin_i_{label}"] = np.where(i_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[1][1]:
        label = _threshold_token(threshold)
        margin = g - float(threshold)
        features[f"margin_g_{label}"] = np.where(g_finite, margin, 0.0)

    for threshold in _QSO_THRESHOLDS[2][1]:
        label = _threshold_token(threshold)
        margin = r - float(threshold)
        features[f"margin_r_qso_{label}"] = np.where(r_finite, margin, 0.0)

    n = len(raw)

    for lo, hi in _I_INTERVALS:
        lo_f = float(lo)
        hi_f = float(hi)
        token_lo = _threshold_token(lo_f)
        token_hi = _threshold_token(hi_f)
        in_interval = (i_finite & (i >= lo_f) & (i <= hi_f)).astype(np.int8)

        dist = np.zeros(n, dtype=float)
        below = i_finite & (i < lo_f)
        above = i_finite & (i > hi_f)
        dist[below] = i[below] - lo_f
        dist[above] = i[above] - hi_f

        features[f"in_interval_i_{token_lo}_{token_hi}"] = in_interval
        features[f"dist_to_interval_i_{token_lo}_{token_hi}"] = np.where(i_finite, dist, 0.0)

    for lo, hi in __R_INTERVALS:
        lo_f = float(lo)
        hi_f = float(hi)
        token_lo = _threshold_token(lo_f)
        token_hi = _threshold_token(hi_f)
        in_interval = (r_finite & (r >= lo_f) & (r <= hi_f)).astype(np.int8)

        dist = np.zeros(n, dtype=float)
        below = r_finite & (r < lo_f)
        above = r_finite & (r > hi_f)
        dist[below] = r[below] - lo_f
        dist[above] = r[above] - hi_f

        features[f"in_interval_r_{token_lo}_{token_hi}"] = in_interval
        features[f"dist_to_interval_r_{token_lo}_{token_hi}"] = np.where(r_finite, dist, 0.0)

    qso_margins = []
    qso_names = []
    for band, thresholds in _QSO_THRESHOLDS:
        if band == "i":
            arr = i
            finite = i_finite
            band_prefix = "i"
        elif band == "g":
            arr = g
            finite = g_finite
            band_prefix = "g"
        else:
            arr = r
            finite = r_finite
            band_prefix = "r"

        for threshold in thresholds:
            token = _threshold_token(threshold)
            qso_margins.append(np.where(finite, arr - float(threshold), np.nan))
            qso_names.append(f"{band_prefix}_{token}")

    d_qso, nearest_qso_band, nearest_qso_sign = _describe_regime(np.column_stack(qso_margins), qso_names)
    features["d_qso"] = d_qso
    features["nearest_qso_band"] = nearest_qso_band
    features["nearest_qso_margin_sign"] = nearest_qso_sign

    gal_margins = []
    gal_names = []
    for threshold in _GALAXY_R_THRESHOLDS:
        token = _threshold_token(threshold)
        gal_margins.append(np.where(r_finite, r - float(threshold), np.nan))
        gal_names.append(f"r_{token}")

    d_gal, nearest_gal_band, nearest_gal_sign = _describe_regime(np.column_stack(gal_margins), gal_names)
    features["d_gal"] = d_gal
    features["nearest_gal_band"] = nearest_gal_band
    features["nearest_gal_margin_sign"] = nearest_gal_sign

    features["boundary_gap"] = d_gal - d_qso

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "survey_depth_limit_margins",
        "fn": add_survey_depth_limit_margins,
        "depends_on": [],
        "description": "Builds signed SDSS targeting-boundary margin, interval-membership, and boundary-proximity descriptors in ugriz space for selection-depth geometry.",
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