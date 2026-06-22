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
- Hypothesis ID: 000012
- Source file: autogluon-001.py
- Failed node: 20260622T142948-858583ae-15
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: ('faint_blue_wedge_m_margin_1_ug_gr_upper', 'faint_blue_wedge_m_margin_2_ug_gr_lower', 'faint_blue_wedge_m_margin_3_ug_lower', 'faint_blue_wedge_m_margin_4_ug_upper', 'faint_blue_wedge_m_margin_5_gr_lower', 'faint_blue_wedge_m_margin_6_gr_upper', 'faint_blue_wedge_m_margin_7_ri_lower', 'faint_blue_wedge_m_margin_8_ri_upper', 'faint_blue_wedge_m_margin_9_iz_lower', 'faint_blue_wedge_m_margin_10_iz_upper')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142948-858583ae-15/02-code.py", line 614, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142948-858583ae-15/02-code.py", line 108, in add_faint_blue_galaxy_wedge_margins
    color_margin_min = margins[color_cols].min(axis=1)
                       ~~~~~~~^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: ('faint_blue_wedge_m_margin_1_ug_gr_upper', 'faint_blue_wedge_m_margin_2_ug_gr_lower', 'faint_blue_wedge_m_margin_3_ug_lower', 'faint_blue_wedge_m_margin_4_ug_upper', 'faint_blue_wedge_m_margin_5_gr_lower', 'faint_blue_wedge_m_margin_6_gr_upper', 'faint_blue_wedge_m_margin_7_ri_lower', 'faint_blue_wedge_m_margin_8_ri_upper', 'faint_blue_wedge_m_margin_9_iz_lower', 'faint_blue_wedge_m_margin_10_iz_upper')
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: ('faint_blue_wedge_m_margin_1_ug_gr_upper', 'faint_blue_wedge_m_margin_2_ug_gr_lower', 'faint_blue_wedge_m_margin_3_ug_lower', 'faint_blue_wedge_m_margin_4_ug_upper', 'faint_blue_wedge_m_margin_5_gr_lower', 'faint_blue_wedge_m_margin_6_gr_upper', 'faint_blue_wedge_m_margin_7_ri_lower', 'faint_blue_wedge_m_margin_8_ri_upper', 'faint_blue_wedge_m_margin_9_iz_lower', 'faint_blue_wedge_m_margin_10_iz_upper')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142948-858583ae-15/02-code.py", line 738, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142948-858583ae-15/02-code.py", line 614, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T142948-858583ae-15/02-code.py", line 108, in add_faint_blue_galaxy_wedge_margins
    color_margin_min = margins[color_cols].min(axis=1)
                       ~~~~~~~^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: ('faint_blue_wedge_m_margin_1_ug_gr_upper', 'faint_blue_wedge_m_margin_2_ug_gr_lower', 'faint_blue_wedge_m_margin_3_ug_lower', 'faint_blue_wedge_m_margin_4_ug_upper', 'faint_blue_wedge_m_margin_5_gr_lower', 'faint_blue_wedge_m_margin_6_gr_upper', 'faint_blue_wedge_m_margin_7_ri_lower', 'faint_blue_wedge_m_margin_8_ri_upper', 'faint_blue_wedge_m_margin_9_iz_lower', 'faint_blue_wedge_m_margin_10_iz_upper')

stdout.log:
AutoGluon materialization: starting feature groups
TheML feature group: start name=faint_blue_galaxy_wedge_margins
TheML feature group: failed name=faint_blue_galaxy_wedge_margins elapsed_s=0.156 rows=824782 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

def _clean_nonfinite_dataframe(df):
    arr = df.to_numpy(dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return pd.DataFrame(arr, index=df.index, columns=df.columns)

def _clean_nonfinite_series(values, index):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    return pd.Series(arr, index=index)

def add_faint_blue_galaxy_wedge_margins(raw, deps, aux):
    u = pd.to_numeric(raw["u"], errors="coerce")
    g = pd.to_numeric(raw["g"], errors="coerce")
    r = pd.to_numeric(raw["r"], errors="coerce")
    i = pd.to_numeric(raw["i"], errors="coerce")
    z = pd.to_numeric(raw["z"], errors="coerce")

    ug = u - g
    gr = g - r
    ri = r - i
    iz = i - z

    margin_1 = gr - (0.40 + 0.6 * ug)
    margin_2 = (1.7 - 0.1 * ug) - gr
    margin_3 = ug + 0.5
    margin_4 = 3.0 - ug
    margin_5 = gr - 0.0
    margin_6 = 1.8 - gr
    margin_7 = ri + 0.5
    margin_8 = 1.5 - ri
    margin_9 = iz + 1.0
    margin_10 = 1.5 - iz
    margin_11 = u - 18.0
    margin_12 = 24.0 - u
    margin_13 = g - 18.0
    margin_14 = 21.5 - g
    margin_15 = r - 17.8
    margin_16 = 19.5 - r
    margin_17 = i - 16.5
    margin_18 = 20.5 - i
    margin_19 = z - 16.0
    margin_20 = 20.0 - z

    margins = _clean_nonfinite_dataframe(
        pd.DataFrame(
            {
                "faint_blue_wedge_m_margin_1_ug_gr_upper": margin_1,
                "faint_blue_wedge_m_margin_2_ug_gr_lower": margin_2,
                "faint_blue_wedge_m_margin_3_ug_lower": margin_3,
                "faint_blue_wedge_m_margin_4_ug_upper": margin_4,
                "faint_blue_wedge_m_margin_5_gr_lower": margin_5,
                "faint_blue_wedge_m_margin_6_gr_upper": margin_6,
                "faint_blue_wedge_m_margin_7_ri_lower": margin_7,
                "faint_blue_wedge_m_margin_8_ri_upper": margin_8,
                "faint_blue_wedge_m_margin_9_iz_lower": margin_9,
                "faint_blue_wedge_m_margin_10_iz_upper": margin_10,
                "faint_blue_wedge_m_margin_11_u_lower": margin_11,
                "faint_blue_wedge_m_margin_12_u_upper": margin_12,
                "faint_blue_wedge_m_margin_13_g_lower": margin_13,
                "faint_blue_wedge_m_margin_14_g_upper": margin_14,
                "faint_blue_wedge_m_margin_15_r_lower": margin_15,
                "faint_blue_wedge_m_margin_16_r_upper": margin_16,
                "faint_blue_wedge_m_margin_17_i_lower": margin_17,
                "faint_blue_wedge_m_margin_18_i_upper": margin_18,
                "faint_blue_wedge_m_margin_19_z_lower": margin_19,
                "faint_blue_wedge_m_margin_20_z_upper": margin_20,
            },
            index=raw.index,
        )
    )

    color_cols = (
        "faint_blue_wedge_m_margin_1_ug_gr_upper",
        "faint_blue_wedge_m_margin_2_ug_gr_lower",
        "faint_blue_wedge_m_margin_3_ug_lower",
        "faint_blue_wedge_m_margin_4_ug_upper",
        "faint_blue_wedge_m_margin_5_gr_lower",
        "faint_blue_wedge_m_margin_6_gr_upper",
        "faint_blue_wedge_m_margin_7_ri_lower",
        "faint_blue_wedge_m_margin_8_ri_upper",
        "faint_blue_wedge_m_margin_9_iz_lower",
        "faint_blue_wedge_m_margin_10_iz_upper",
    )
    mag_cols = (
        "faint_blue_wedge_m_margin_11_u_lower",
        "faint_blue_wedge_m_margin_12_u_upper",
        "faint_blue_wedge_m_margin_13_g_lower",
        "faint_blue_wedge_m_margin_14_g_upper",
        "faint_blue_wedge_m_margin_15_r_lower",
        "faint_blue_wedge_m_margin_16_r_upper",
        "faint_blue_wedge_m_margin_17_i_lower",
        "faint_blue_wedge_m_margin_18_i_upper",
        "faint_blue_wedge_m_margin_19_z_lower",
        "faint_blue_wedge_m_margin_20_z_upper",
    )

    all_margin_min = margins.min(axis=1)
    color_margin_min = margins[color_cols].min(axis=1)
    mag_margin_min = margins[mag_cols].min(axis=1)
    violation_count = (margins < 0.0).sum(axis=1).astype("int64")

    sampling_input = margins["faint_blue_wedge_m_margin_1_ug_gr_upper"].to_numpy(dtype=np.float64)
    sampling_intensity = np.exp(0.1411 * sampling_input)
    sampling_intensity = np.clip(sampling_intensity, 0.0, 10.0)
    sampling_intensity = _clean_nonfinite_series(sampling_intensity, raw.index)

    return pd.DataFrame(
        {
            "faint_blue_wedge_m_margin_1_ug_gr_upper": margins["faint_blue_wedge_m_margin_1_ug_gr_upper"],
            "faint_blue_wedge_m_margin_2_ug_gr_lower": margins["faint_blue_wedge_m_margin_2_ug_gr_lower"],
            "faint_blue_wedge_m_margin_3_ug_lower": margins["faint_blue_wedge_m_margin_3_ug_lower"],
            "faint_blue_wedge_m_margin_4_ug_upper": margins["faint_blue_wedge_m_margin_4_ug_upper"],
            "faint_blue_wedge_m_margin_5_gr_lower": margins["faint_blue_wedge_m_margin_5_gr_lower"],
            "faint_blue_wedge_m_margin_6_gr_upper": margins["faint_blue_wedge_m_margin_6_gr_upper"],
            "faint_blue_wedge_m_margin_7_ri_lower": margins["faint_blue_wedge_m_margin_7_ri_lower"],
            "faint_blue_wedge_m_margin_8_ri_upper": margins["faint_blue_wedge_m_margin_8_ri_upper"],
            "faint_blue_wedge_m_margin_9_iz_lower": margins["faint_blue_wedge_m_margin_9_iz_lower"],
            "faint_blue_wedge_m_margin_10_iz_upper": margins["faint_blue_wedge_m_margin_10_iz_upper"],
            "faint_blue_wedge_m_margin_11_u_lower": margins["faint_blue_wedge_m_margin_11_u_lower"],
            "faint_blue_wedge_m_margin_12_u_upper": margins["faint_blue_wedge_m_margin_12_u_upper"],
            "faint_blue_wedge_m_margin_13_g_lower": margins["faint_blue_wedge_m_margin_13_g_lower"],
            "faint_blue_wedge_m_margin_14_g_upper": margins["faint_blue_wedge_m_margin_14_g_upper"],
            "faint_blue_wedge_m_margin_15_r_lower": margins["faint_blue_wedge_m_margin_15_r_lower"],
            "faint_blue_wedge_m_margin_16_r_upper": margins["faint_blue_wedge_m_margin_16_r_upper"],
            "faint_blue_wedge_m_margin_17_i_lower": margins["faint_blue_wedge_m_margin_17_i_lower"],
            "faint_blue_wedge_m_margin_18_i_upper": margins["faint_blue_wedge_m_margin_18_i_upper"],
            "faint_blue_wedge_m_margin_19_z_lower": margins["faint_blue_wedge_m_margin_19_z_lower"],
            "faint_blue_wedge_m_margin_20_z_upper": margins["faint_blue_wedge_m_margin_20_z_upper"],
            "faint_blue_wedge_m_margin_min_all": all_margin_min,
            "faint_blue_wedge_m_margin_min_color": color_margin_min,
            "faint_blue_wedge_m_margin_min_magnitude": mag_margin_min,
            "faint_blue_wedge_m_violation_count": violation_count,
            "faint_blue_wedge_sampling_intensity": sampling_intensity,
        },
        index=raw.index,
    )

FEATURE_GROUPS = [
    {
        "name": "faint_blue_galaxy_wedge_margins",
        "fn": add_faint_blue_galaxy_wedge_margins,
        "depends_on": [],
        "description": "Build signed faint-blue wedge margins in color-magnitude space and summarize them with minimum scores, violation counts, and a capped SDSS-style sampling intensity proxy.",
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