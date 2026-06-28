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
- Hypothesis ID: 000010
- Source file: autogluon-004.py
- Failed node: 20260628T060529-1ce00ba8-90
- Run: 20260624T200326-681cd81b

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py:66: RuntimeWarning: divide by zero encountered in log
  entropy_terms = np.where(bin_fracs > 0.0, bin_fracs * np.log(bin_fracs), 0.0)
/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py:66: RuntimeWarning: invalid value encountered in multiply
  entropy_terms = np.where(bin_fracs > 0.0, bin_fracs * np.log(bin_fracs), 0.0)
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py", line 623, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py", line 86, in add_rest_frame_filter_landmarks
    bracket_left_idx = np.searchsorted(rest_lambdas, landmark, side="right") - 1
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 1534, in searchsorted
    return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^
ValueError: object too deep for desired array
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py", line 747, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py", line 623, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260624T200326-681cd81b/artifacts/20260628T060529-1ce00ba8-90/02-code.py", line 86, in add_rest_frame_filter_landmarks
    bracket_left_idx = np.searchsorted(rest_lambdas, landmark, side="right") - 1
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 1534, in searchsorted
    return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
           ^^^^^^^^^^^^^^^^^^^^
ValueError: object too deep for desired array

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=rest_frame_filter_landmarks
TheML feature group: failed name=rest_frame_filter_landmarks elapsed_s=0.270 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BAND_NAMES = ("u", "g", "r", "i", "z")
OBS_WAVELENGTHS_ANGSTROM = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
LANDMARKS_ANGSTROM = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
LANDMARK_NAMES = ("lya_limit_912", "lya_1216", "uv_2500", "balmer_3646", "break_4000", "optical_7000")
REST_BIN_NAMES = (
    "below_912",
    "912_1216",
    "1216_2500",
    "2500_3646",
    "3646_4000",
    "4000_7000",
    "above_7000",
)
REST_BIN_EDGES = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)

def add_rest_frame_filter_landmarks(raw, deps, aux):
    z_safe = np.maximum(raw["redshift"].to_numpy(dtype=float), 0.0)
    denom = 1.0 + z_safe

    obs_waves = np.asarray(OBS_WAVELENGTHS_ANGSTROM, dtype=float)
    rest_lambdas = obs_waves.reshape(1, -1) / denom.reshape(-1, 1)
    log_rest_lambdas = np.log(rest_lambdas)

    mags = raw.loc[:, BAND_NAMES].to_numpy(dtype=float)

    span_min = rest_lambdas[:, 0]
    span_max = rest_lambdas[:, -1]
    span_width = span_max - span_min

    features = pd.DataFrame(index=raw.index)

    features["neg_redshift_flag"] = (raw["redshift"].to_numpy(dtype=float) < 0.0).astype(np.int8)
    for idx, band in enumerate(BAND_NAMES):
        features[f"{band}_rest_lambda"] = rest_lambdas[:, idx]

    features["span_min"] = span_min
    features["span_max"] = span_max
    features["span_width"] = span_width
    features["log_span_width"] = np.log1p(span_width)

    for idx in range(len(BAND_NAMES) - 1):
        left = BAND_NAMES[idx]
        right = BAND_NAMES[idx + 1]
        center = np.exp((log_rest_lambdas[:, idx] + log_rest_lambdas[:, idx + 1]) / 2.0)
        features[f"{left}_{right}_interval_rest_center"] = center

    bin_ids = np.searchsorted(np.asarray(REST_BIN_EDGES, dtype=float), rest_lambdas, side="right")
    bin_counts = np.zeros((len(raw), len(REST_BIN_NAMES)), dtype=float)
    row_idx = np.arange(len(raw))
    for band_idx in range(len(BAND_NAMES)):
        bin_counts[row_idx, bin_ids[:, band_idx]] += 1.0

    bin_fracs = bin_counts / float(len(BAND_NAMES))
    for idx, bin_name in enumerate(REST_BIN_NAMES):
        features[f"{bin_name}_count"] = bin_counts[:, idx]
        features[f"{bin_name}_frac"] = bin_fracs[:, idx]

    entropy_terms = np.where(bin_fracs > 0.0, bin_fracs * np.log(bin_fracs), 0.0)
    features["rest_bin_count_entropy"] = -np.sum(entropy_terms, axis=1)

    interval_left_logs = log_rest_lambdas[:, :-1]
    interval_right_logs = log_rest_lambdas[:, 1:]
    interval_center_logs = (interval_left_logs + interval_right_logs) / 2.0

    for landmark_name, landmark in zip(LANDMARK_NAMES, LANDMARKS_ANGSTROM):
        log_landmark = np.log(float(landmark))
        in_span = (span_min <= landmark) & (landmark <= span_max)
        side = np.where(landmark < span_min, -1, np.where(landmark > span_max, 1, 0)).astype(np.int8)
        edge_distance = np.where(landmark < span_min, span_min - landmark, np.where(landmark > span_max, landmark - span_max, 0.0))

        log_ratios = log_rest_lambdas - log_landmark
        abs_log_ratios = np.abs(log_ratios)
        nearest_idx = np.argmin(abs_log_ratios, axis=1)
        nearest_mag = mags[row_idx, nearest_idx]
        nearest_lambda = rest_lambdas[row_idx, nearest_idx]
        signed_log_ratio = log_ratios[row_idx, nearest_idx]

        bracket_left_idx = np.searchsorted(rest_lambdas, landmark, side="right") - 1
        bracket_left_idx = np.clip(bracket_left_idx, 0, len(BAND_NAMES) - 2)
        bracket_right_idx = bracket_left_idx + 1

        bracket_left_mag = mags[row_idx, bracket_left_idx]
        bracket_right_mag = mags[row_idx, bracket_right_idx]
        bracket_left_log_lambda = log_rest_lambdas[row_idx, bracket_left_idx]
        bracket_right_log_lambda = log_rest_lambdas[row_idx, bracket_right_idx]
        bracket_den = bracket_right_log_lambda - bracket_left_log_lambda

        frac_position_inside = (log_landmark - bracket_left_log_lambda) / bracket_den
        interp_mag_inside = bracket_left_mag + frac_position_inside * (bracket_right_mag - bracket_left_mag)
        curvature_inside = interp_mag_inside - ((bracket_left_mag + bracket_right_mag) / 2.0)

        frac_position = np.where(in_span, frac_position_inside, 0.0)
        interp_mag = np.where(in_span, interp_mag_inside, 0.0)
        curvature = np.where(in_span, curvature_inside, 0.0)

        prefix = f"{landmark_name}"
        features[f"{prefix}_in_span"] = in_span.astype(np.int8)
        features[f"{prefix}_side"] = side
        features[f"{prefix}_outside_log_distance"] = np.log1p(edge_distance)
        features[f"{prefix}_nearest_band_idx"] = nearest_idx.astype(np.int8)
        features[f"{prefix}_nearest_band_mag"] = nearest_mag
        features[f"{prefix}_nearest_band_rest_lambda"] = nearest_lambda
        features[f"{prefix}_signed_log_ratio"] = signed_log_ratio
        features[f"{prefix}_abs_log_ratio"] = np.abs(signed_log_ratio)
        features[f"{prefix}_bracket_left_idx"] = bracket_left_idx.astype(np.int8)
        features[f"{prefix}_bracket_right_idx"] = bracket_right_idx.astype(np.int8)
        features[f"{prefix}_bracket_left_mag"] = bracket_left_mag
        features[f"{prefix}_bracket_right_mag"] = bracket_right_mag
        features[f"{prefix}_bracket_color"] = bracket_left_mag - bracket_right_mag
        features[f"{prefix}_fractional_position"] = frac_position
        features[f"{prefix}_interpolated_mag_at_landmark"] = interp_mag
        features[f"{prefix}_curvature_proxy"] = curvature

        for interval_idx in range(len(BAND_NAMES) - 1):
            left = BAND_NAMES[interval_idx]
            right = BAND_NAMES[interval_idx + 1]
            interval_in = (
                (rest_lambdas[:, interval_idx] <= landmark)
                & (landmark <= rest_lambdas[:, interval_idx + 1])
            )
            features[f"{left}_{right}_{prefix}_center_signed_log_distance"] = interval_center_logs[:, interval_idx] - log_landmark
            features[f"{left}_{right}_{prefix}_inside_interval"] = interval_in.astype(np.int8)

    return features

FEATURE_GROUPS = [
    {
        "name": "rest_frame_filter_landmarks",
        "fn": add_rest_frame_filter_landmarks,
        "depends_on": [],
        "description": "Rest-frame passband coverage and spectral landmark alignment features from ugriz photometry and redshift.",
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