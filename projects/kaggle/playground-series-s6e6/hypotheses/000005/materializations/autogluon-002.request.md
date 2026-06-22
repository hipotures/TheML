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
- Hypothesis ID: 000005
- Source file: autogluon-001.py
- Failed node: 20260622T142200-d8d959be-8
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142200-d8d959be-8/02-code.py", line 584, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142200-d8d959be-8/02-code.py", line 69, in add_catalog_template_residuals
    type_ok = type_count.reindex(raw.index).ge(1000)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 5172, in reindex
    return super().reindex(
           ^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5632, in reindex
    return self._reindex_axes(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5655, in _reindex_axes
    new_index, indexer = ax.reindex(
                         ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 4436, in reindex
    raise ValueError("cannot reindex on an axis with duplicate labels")
ValueError: cannot reindex on an axis with duplicate labels
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142200-d8d959be-8/02-code.py", line 708, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142200-d8d959be-8/02-code.py", line 584, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142200-d8d959be-8/02-code.py", line 69, in add_catalog_template_residuals
    type_ok = type_count.reindex(raw.index).ge(1000)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/series.py", line 5172, in reindex
    return super().reindex(
           ^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5632, in reindex
    return self._reindex_axes(
           ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 5655, in _reindex_axes
    new_index, indexer = ax.reindex(
                         ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 4436, in reindex
    raise ValueError("cannot reindex on an axis with duplicate labels")
ValueError: cannot reindex on an axis with duplicate labels

stdout.log:
AutoGluon materialization: starting feature groups
TheML feature group: start name=catalog_template_residuals
TheML feature group: failed name=catalog_template_residuals elapsed_s=2.194 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

def _mad_series(values: pd.Series) -> float:
    s = values.dropna()
    if s.empty:
        return 0.0
    med = s.median()
    return (s - med).abs().median()

def add_catalog_template_residuals(raw, deps, aux):
    # Colors in adjacent-color space
    colors = pd.DataFrame(
        {
            "c1": raw["u"] - raw["g"],
            "c2": raw["g"] - raw["r"],
            "c3": raw["r"] - raw["i"],
            "c4": raw["i"] - raw["z"],
        },
        index=raw.index,
    )
    color_cols = ["c1", "c2", "c3", "c4"]

    spectral_type = raw["spectral_type"].astype(object)
    galaxy_population = raw["galaxy_population"].astype(object)

    pair_key = pd.DataFrame(
        {
            "spectral_type": spectral_type,
            "galaxy_population": galaxy_population,
        },
        index=raw.index,
    )
    pair_data = pd.concat([pair_key, colors], axis=1)

    pair_counts = pair_data.groupby(["spectral_type", "galaxy_population"]).size()
    pair_medians = pair_data.groupby(["spectral_type", "galaxy_population"])[color_cols].median()
    pair_mads = pair_data.groupby(["spectral_type", "galaxy_population"])[color_cols].agg(_mad_series)

    pair_medians = pair_medians.rename(columns={c: f"{c}_median" for c in color_cols})
    pair_mads = pair_mads.rename(columns={c: f"{c}_mad" for c in color_cols})

    type_key = pd.DataFrame({"spectral_type": spectral_type}, index=raw.index)
    type_data = pd.concat([type_key, colors], axis=1)

    type_counts = type_data.groupby("spectral_type").size()
    type_medians = type_data.groupby("spectral_type")[color_cols].median().rename(columns={c: f"{c}_median" for c in color_cols})
    type_mads = type_data.groupby("spectral_type")[color_cols].agg(_mad_series).rename(columns={c: f"{c}_mad" for c in color_cols})

    global_medians = colors.median()
    global_mads = colors.apply(_mad_series)

    idx_pairs = pd.MultiIndex.from_arrays([spectral_type, galaxy_population])

    pair_count = pair_counts.reindex(idx_pairs).fillna(0).astype(int)
    type_count = type_counts.reindex(spectral_type).fillna(0).astype(int)

    pair_med_lookup = pair_medians.reindex(idx_pairs)
    pair_mad_lookup = pair_mads.reindex(idx_pairs)
    type_med_lookup = type_medians.reindex(spectral_type)
    type_mad_lookup = type_mads.reindex(spectral_type)

    pair_ok = pair_count.ge(1000)
    type_ok = type_count.reindex(raw.index).ge(1000)

    chosen_median = pd.DataFrame(
        {
            c: pd.Series(float(global_medians[c]), index=raw.index)
            for c in color_cols
        },
        index=raw.index,
    )
    chosen_mad = pd.DataFrame(
        {
            c: pd.Series(float(global_mads[c]), index=raw.index)
            for c in color_cols
        },
        index=raw.index,
    )

    for c in color_cols:
        chosen_median[c] = pair_med_lookup[f"{c}_median"].where(pair_ok, chosen_median[c])
        chosen_median[c] = type_med_lookup[f"{c}_median"].where((~pair_ok) & type_ok, chosen_median[c])

        chosen_mad[c] = pair_mad_lookup[f"{c}_mad"].where(pair_ok, chosen_mad[c])
        chosen_mad[c] = type_mad_lookup[f"{c}_mad"].where((~pair_ok) & type_ok, chosen_mad[c])

    mad_safe = chosen_mad.clip(lower=0.02)
    residuals = (colors - chosen_median) / mad_safe
    residuals = residuals.clip(lower=-10.0, upper=10.0)

    residuals = residuals.rename(columns={c: f"template_residual_{c}" for c in color_cols})

    abs_residual = residuals.abs()
    out = pd.concat(
        [
            residuals,
            pd.DataFrame(
                {
                    "template_residual_mean_abs": abs_residual.mean(axis=1),
                    "template_residual_max_abs": abs_residual.max(axis=1),
                    "template_residual_rss": (residuals**2).sum(axis=1),
                    "template_residual_count_exceeds_2": (abs_residual > 2.0).sum(axis=1),
                },
                index=raw.index,
            ),
        ],
        axis=1,
    )

    return out

FEATURE_GROUPS = [
    {
        "name": "catalog_template_residuals",
        "fn": add_catalog_template_residuals,
        "depends_on": [],
        "description": "Build template-mismatch residual features by comparing object colors to tag-conditioned color centroids with MAD normalization and fallback priors.",
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