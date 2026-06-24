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
- Source file: autogluon-002.py
- Failed node: 20260624T055012-ce35fbdc-316
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py:36: RuntimeWarning: divide by zero encountered in log
  entropy = -np.sum(np.where(has_count, probs * np.log(probs), 0.0), axis=1)
/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py:36: RuntimeWarning: invalid value encountered in multiply
  entropy = -np.sum(np.where(has_count, probs * np.log(probs), 0.0), axis=1)
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py", line 628, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py", line 82, in add_rest_frame_filter_landmarks
    right_wave = np.where(has_right, rest_waves[row_idx, nearest_idx + 1], np.nan)
                                     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: index 5 is out of bounds for axis 1 with size 5
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py", line 752, in <module>
    main()
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py", line 628, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.worktrees/root-hypothesis-revisions/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260624T055012-ce35fbdc-316/02-code.py", line 82, in add_rest_frame_filter_landmarks
    right_wave = np.where(has_right, rest_waves[row_idx, nearest_idx + 1], np.nan)
                                     ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: index 5 is out of bounds for axis 1 with size 5

stdout.log:
AutoGluon materialization: loaded aux file star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=rest_frame_filter_landmarks
TheML feature group: failed name=rest_frame_filter_landmarks elapsed_s=0.426 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BASE_WAVELENGTHS = (3543.0, 4770.0, 6231.0, 7625.0, 9134.0)
REST_BIN_EDGES = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
LANDMARKS = (912.0, 1216.0, 2500.0, 3646.0, 4000.0, 7000.0)
MAG_BANDS = ("u", "g", "r", "i", "z")

def add_rest_frame_filter_landmarks(raw, deps, aux):
    redshift = raw["redshift"].to_numpy(dtype=float)
    z_safe = np.maximum(redshift, 0.0)
    z_neg_flag = redshift < 0.0

    mags = raw[list(MAG_BANDS)].to_numpy(dtype=float)
    obs_waves = np.array(BASE_WAVELENGTHS, dtype=float)
    rest_waves = obs_waves / (1.0 + z_safe[:, None])

    n_rows = raw.shape[0]
    row_idx = np.arange(n_rows)

    span_min = np.min(rest_waves, axis=1)
    span_max = np.max(rest_waves, axis=1)
    span_width = span_max - span_min

    bin_edges = np.array(REST_BIN_EDGES, dtype=float)
    bin_idx = np.digitize(rest_waves, bin_edges, right=False)
    counts = np.zeros((n_rows, bin_edges.size + 1), dtype=np.int16)
    np.add.at(counts, (np.repeat(row_idx, rest_waves.shape[1]), bin_idx.ravel()), 1)

    probs = counts.astype(float) / float(obs_waves.shape[0])
    has_count = counts > 0
    entropy = -np.sum(np.where(has_count, probs * np.log(probs), 0.0), axis=1)
    nonzero_bins = has_count.sum(axis=1).astype(float)
    rest_bin_entropy = np.divide(
        entropy,
        np.log(nonzero_bins),
        out=np.zeros_like(entropy),
        where=nonzero_bins > 1.0,
    )

    features = {
        "z_safe": z_safe,
        "z_neg_flag": z_neg_flag.astype(np.int8),
        "rest_wave_u": rest_waves[:, 0],
        "rest_wave_g": rest_waves[:, 1],
        "rest_wave_r": rest_waves[:, 2],
        "rest_wave_i": rest_waves[:, 3],
        "rest_wave_z": rest_waves[:, 4],
        "span_min": span_min,
        "span_max": span_max,
        "span_width": span_width,
        "n_lt_912": counts[:, 0],
        "n_912_1216": counts[:, 1],
        "n_1216_2500": counts[:, 2],
        "n_2500_3646": counts[:, 3],
        "n_3646_4000": counts[:, 4],
        "n_4000_7000": counts[:, 5],
        "n_gt_7000": counts[:, 6],
        "total_bands": np.full(n_rows, 5, dtype=np.int8),
        "rest_bin_entropy": rest_bin_entropy,
    }

    for landmark in LANDMARKS:
        name = f"{int(landmark)}"
        nearest_idx = np.argmin(np.abs(rest_waves - landmark), axis=1)
        selected_wave = rest_waves[row_idx, nearest_idx]
        selected_mag = mags[row_idx, nearest_idx]
        d = selected_wave - landmark

        in_span = (span_min <= landmark) & (span_max >= landmark)
        below_span = landmark < span_min
        above_span = landmark > span_max

        has_left = nearest_idx > 0
        has_right = nearest_idx < (mags.shape[1] - 1)

        left_wave = np.where(has_left, rest_waves[row_idx, nearest_idx - 1], np.nan)
        right_wave = np.where(has_right, rest_waves[row_idx, nearest_idx + 1], np.nan)
        left_mag = np.where(has_left, mags[row_idx, nearest_idx - 1], np.nan)
        right_mag = np.where(has_right, mags[row_idx, nearest_idx + 1], np.nan)

        interior_straddle = has_left & has_right & (left_wave <= landmark) & (right_wave >= landmark)

        left_delta = np.full(n_rows, np.nan, dtype=float)
        right_delta = np.full(n_rows, np.nan, dtype=float)

        left_delta[interior_straddle] = selected_mag[interior_straddle] - left_mag[interior_straddle]
        right_delta[interior_straddle] = right_mag[interior_straddle] - selected_mag[interior_straddle]

        idx_last = nearest_idx == (mags.shape[1] - 1)
        idx_first = nearest_idx == 0
        left_delta[idx_last] = selected_mag[idx_last] - left_mag[idx_last]
        right_delta[idx_first] = right_mag[idx_first] - selected_mag[idx_first]

        residual = np.zeros(n_rows, dtype=float)

        upper_ok = (landmark > selected_wave) & (nearest_idx < (mags.shape[1] - 1))
        upper_mask = upper_ok & (landmark < right_wave)
        if np.any(upper_mask):
            pred_upper = selected_mag + (right_mag - selected_mag) * (landmark - selected_wave) / (right_wave - selected_wave)
            residual[upper_mask] = selected_mag[upper_mask] - pred_upper[upper_mask]

        lower_ok = (landmark < selected_wave) & (nearest_idx > 0)
        lower_mask = lower_ok & (landmark > left_wave)
        if np.any(lower_mask):
            pred_lower = left_mag + (selected_mag - left_mag) * (landmark - left_wave) / (selected_wave - left_wave)
            residual[lower_mask] = selected_mag[lower_mask] - pred_lower[lower_mask]

        residual = np.where(in_span, residual, 0.0)

        outside_distance = np.where(
            in_span,
            0.0,
            np.log1p(np.abs(landmark - np.where(below_span, span_min, span_max))),
        )
        outside_side = np.where(in_span, 0, np.where(below_span, 0, 1)).astype(np.int8)
        outside_endpoint_band = np.where(in_span, 0, np.where(below_span, 1, 5)).astype(np.int8)

        features[f"landmark_{name}_in_span"] = in_span.astype(np.int8)
        features[f"landmark_{name}_below_span"] = below_span.astype(np.int8)
        features[f"landmark_{name}_above_span"] = above_span.astype(np.int8)
        features[f"landmark_{name}_nearest_band_id"] = nearest_idx + 1
        features[f"landmark_{name}_signed_log_dist"] = np.sign(d) * np.log1p(np.abs(d))
        features[f"landmark_{name}_abs_log_dist"] = np.log1p(np.abs(d))
        features[f"landmark_{name}_nearest_band_mag"] = selected_mag
        features[f"landmark_{name}_left_delta"] = left_delta
        features[f"landmark_{name}_right_delta"] = right_delta
        features[f"landmark_{name}_residual"] = residual
        features[f"landmark_{name}_outside_distance"] = outside_distance
        features[f"landmark_{name}_outside_side"] = outside_side
        features[f"landmark_{name}_outside_endpoint_band"] = outside_endpoint_band

    return pd.DataFrame(features, index=raw.index)

FEATURE_GROUPS = [
    {
        "name": "rest_frame_filter_landmarks",
        "fn": add_rest_frame_filter_landmarks,
        "depends_on": [],
        "description": "Construct rest-frame band-location occupancy and landmark-alignment descriptors from ugriz magnitudes and redshift-aware wavelength placement.",
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