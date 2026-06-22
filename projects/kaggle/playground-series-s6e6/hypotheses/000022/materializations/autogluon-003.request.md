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
- Hypothesis ID: 000022
- Source file: autogluon-002.py
- Failed node: 20260622T221852-cd064c8e-57
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T221852-cd064c8e-57/02-code.py", line 644, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T221852-cd064c8e-57/02-code.py", line 127, in add_aide_sky_cell_local_residuals
    cell_means = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("mean")
                 ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/groupby/generic.py", line 1947, in __getitem__
    raise ValueError(
ValueError: Cannot subset columns with a tuple with more than one element. Use a list instead.
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T221852-cd064c8e-57/02-code.py", line 768, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T221852-cd064c8e-57/02-code.py", line 644, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T221852-cd064c8e-57/02-code.py", line 127, in add_aide_sky_cell_local_residuals
    cell_means = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("mean")
                 ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/groupby/generic.py", line 1947, in __getitem__
    raise ValueError(
ValueError: Cannot subset columns with a tuple with more than one element. Use a list instead.

stdout.log:
AutoGluon materialization: loaded aux file /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv rows=100000 cols=18 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=aide_sky_cell_local_residuals
TheML feature group: failed name=aide_sky_cell_local_residuals elapsed_s=0.093 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

_AIDE_CELLS_PER_DEGREE = 2
_AIDE_RA_BINS = 360 * _AIDE_CELLS_PER_DEGREE
_AIDE_DEC_BINS = 180 * _AIDE_CELLS_PER_DEGREE

_AIDE_BASE_FEATURES = ("u", "g", "r", "i", "z", "redshift_clipped")
_AIDE_COLOR_FEATURES = (
    "u_g",
    "g_r",
    "r_i",
    "i_z",
    "u_i",
    "u_r",
    "g_z",
    "r_z",
    "u_2r_i",
    "g_2i_z",
)

def _compute_sky_cells(alpha_series, delta_series):
    alpha = np.asarray(alpha_series, dtype=np.float64)
    delta = np.asarray(delta_series, dtype=np.float64)

    alpha_wrapped = np.mod(alpha, 360.0)
    ra_bin = np.floor(alpha_wrapped * float(_AIDE_CELLS_PER_DEGREE)).astype(np.int64)
    ra_bin = np.clip(ra_bin, 0, _AIDE_RA_BINS - 1)

    delta_shift = np.clip(delta + 90.0, 0.0, 180.0 - 1e-12)
    dec_bin = np.floor(delta_shift * float(_AIDE_CELLS_PER_DEGREE)).astype(np.int64)
    dec_bin = np.clip(dec_bin, 0, _AIDE_DEC_BINS - 1)

    cell_key = dec_bin * _AIDE_RA_BINS + ra_bin
    return ra_bin, dec_bin, cell_key

def _compute_cell_neighbors(ra_bin, dec_bin):
    cell_counts = np.zeros((_AIDE_DEC_BINS, _AIDE_RA_BINS), dtype=np.int64)
    np.add.at(cell_counts, (dec_bin, ra_bin), 1)

    cell_count = cell_counts[dec_bin, ra_bin]

    # Wrap longitude with neighbors across 0/360 edges.
    wrapped_ra = np.concatenate(
        [cell_counts[:, -1:], cell_counts, cell_counts[:, :1]],
        axis=1
    )
    padded = np.pad(wrapped_ra, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    ra_idx = ra_bin + 2
    dec_idx = dec_bin + 1

    neighbor_count = (
        padded[dec_idx - 1, ra_idx - 1] + padded[dec_idx - 1, ra_idx] + padded[dec_idx - 1, ra_idx + 1] +
        padded[dec_idx, ra_idx - 1] + padded[dec_idx, ra_idx] + padded[dec_idx, ra_idx + 1] +
        padded[dec_idx + 1, ra_idx - 1] + padded[dec_idx + 1, ra_idx] + padded[dec_idx + 1, ra_idx + 1]
    )

    concentration = np.zeros_like(cell_count, dtype=np.float64)
    np.divide(
        cell_count,
        neighbor_count,
        out=concentration,
        where=neighbor_count > 0
    )
    return cell_count, neighbor_count, concentration

def _safe_zscore(residuals, scales):
    z = residuals / scales
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def add_aide_sky_cell_local_residuals(raw, deps, aux):
    alpha = raw["alpha"]
    delta = raw["delta"]

    ra_bin, dec_bin, cell_key = _compute_sky_cells(alpha, delta)
    cell_count, neighbor_count, concentration = _compute_cell_neighbors(ra_bin, dec_bin)

    u = raw["u"].to_numpy(dtype=np.float64)
    g = raw["g"].to_numpy(dtype=np.float64)
    r = raw["r"].to_numpy(dtype=np.float64)
    i = raw["i"].to_numpy(dtype=np.float64)
    z = raw["z"].to_numpy(dtype=np.float64)
    redshift_clipped = np.clip(raw["redshift"].to_numpy(dtype=np.float64), 0.0, 7.0)

    color_table = {
        "u_g": u - g,
        "g_r": g - r,
        "r_i": r - i,
        "i_z": i - z,
        "u_i": u - i,
        "u_r": u - r,
        "g_z": g - z,
        "r_z": r - z,
        "u_2r_i": u - 2.0 * r + i,
        "g_2i_z": g - 2.0 * i + z,
    }

    all_features = pd.DataFrame(
        {
            "cell_key": cell_key,
            "u": u,
            "g": g,
            "r": r,
            "i": i,
            "z": z,
            "redshift_clipped": redshift_clipped,
            "u_g": color_table["u_g"],
            "g_r": color_table["g_r"],
            "r_i": color_table["r_i"],
            "i_z": color_table["i_z"],
            "u_i": color_table["u_i"],
            "u_r": color_table["u_r"],
            "g_z": color_table["g_z"],
            "r_z": color_table["r_z"],
            "u_2r_i": color_table["u_2r_i"],
            "g_2i_z": color_table["g_2i_z"],
        },
        index=raw.index,
    )

    grouped = all_features.groupby("cell_key", sort=False)

    cell_means = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("mean")
    cell_stds = grouped[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES].transform("std").replace(0.0, np.nan)

    residuals = all_features[_AIDE_BASE_FEATURES + _AIDE_COLOR_FEATURES] - cell_means
    zscores = _safe_zscore(residuals, cell_stds)

    color_residuals = residuals[_AIDE_COLOR_FEATURES].to_numpy(dtype=np.float64)
    color_zscores = zscores[_AIDE_COLOR_FEATURES].to_numpy(dtype=np.float64)

    out = pd.DataFrame(index=raw.index)
    out["aide_sky_cell_count"] = cell_count
    out["aide_sky_cell_3x3_count"] = neighbor_count
    out["aide_sky_cell_concentration"] = concentration

    for feat in _AIDE_BASE_FEATURES:
        out[f"aide_sky_cell_residual_{feat}"] = residuals[feat]
        out[f"aide_sky_cell_zscore_{feat}"] = zscores[feat]

    for feat in _AIDE_COLOR_FEATURES:
        out[f"aide_sky_cell_residual_{feat}"] = residuals[feat]
        out[f"aide_sky_cell_zscore_{feat}"] = zscores[feat]

    out["aide_sky_cell_color_residual_l2"] = np.sqrt((color_residuals ** 2).sum(axis=1))
    out["aide_sky_cell_color_residual_abs_l1"] = np.abs(color_residuals).sum(axis=1)
    out["aide_sky_cell_color_residual_mean_signed"] = color_residuals.mean(axis=1)
    out["aide_sky_cell_color_residual_std"] = color_residuals.std(axis=1, ddof=0)
    out["aide_sky_cell_color_max_abs_zscore"] = np.abs(color_zscores).max(axis=1)

    return out

FEATURE_GROUPS = [
    {
        "name": "aide_sky_cell_local_residuals",
        "fn": add_aide_sky_cell_local_residuals,
        "depends_on": [],
        "description": "Build deterministic half-degree sky-cell features with local neighbor density and within-cell residual statistics over magnitudes, clipped redshift, and color-shape features.",
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