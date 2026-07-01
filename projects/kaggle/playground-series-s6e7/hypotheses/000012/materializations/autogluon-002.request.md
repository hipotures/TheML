Fix one failed hypothesis materialization.

Return a corrected version of the same feature-group module. Do not invent a
new hypothesis or change the feature family. Keep the fix narrow: address the
observed failure while preserving the intended preprocessing behavior.

You are repairing Python preprocessing code for one stored hypothesis. The
existing file is a feature-group module imported by a fixed runtime wrapper.
Generate only the corrected module.

# Data Overview

-> test.csv has 295753 rows and 14 columns.
Here is some information about the columns:
id (int64) has range: 690088.00 - 985840.00, 0 nan values
sleep_duration (float64) has range: 3.00 - 10.00, 32571 nan values
heart_rate (float64) has range: 50.00 - 103.70, 3357 nan values
bmi (float64) has range: 16.00 - 34.82, 5956 nan values
calorie_expenditure (float64) has range: 1201.00 - 3574.00, 22652 nan values
step_count (float64) has range: 1002.00 - 14999.00, 5964 nan values
exercise_duration (float64) has range: 0.00 - 98.40, 2958 nan values
water_intake (float64) has range: 0.50 - 4.72, 18633 nan values
diet_type (object) has 3 unique values: ['veg', 'balanced', 'non-veg'], 2958 nan values
stress_level (object) has 3 unique values: ['high', 'medium', 'low'], 35490 nan values
sleep_quality (object) has 3 unique values: ['poor', 'good', 'average'], 24999 nan values
physical_activity_level (object) has 3 unique values: ['active', 'sedentary', 'moderate'], 15695 nan values
smoking_alcohol (object) has 3 unique values: ['occasional', 'yes', 'no'], 12249 nan values
gender (object) has 3 unique values: ['male', 'other', 'female'], 9160 nan values

-> train.csv has 690088 rows and 15 columns.
Here is some information about the columns:
id (int64) has range: 0.00 - 690087.00, 0 nan values
sleep_duration (float64) has range: 3.00 - 10.00, 75999 nan values
heart_rate (float64) has range: 50.00 - 107.70, 7833 nan values
bmi (float64) has range: 16.00 - 34.82, 13898 nan values
calorie_expenditure (float64) has range: 1200.00 - 3580.00, 52853 nan values
step_count (float64) has range: 1002.00 - 14999.00, 13916 nan values
exercise_duration (float64) has range: 0.00 - 99.80, 6901 nan values
water_intake (float64) has range: 0.50 - 4.72, 43477 nan values
diet_type (object) has 3 unique values: ['veg', 'non-veg', 'balanced'], 6901 nan values
stress_level (object) has 3 unique values: ['high', 'low', 'medium'], 82811 nan values
sleep_quality (object) has 3 unique values: ['average', 'poor', 'good'], 58331 nan values
physical_activity_level (object) has 3 unique values: ['sedentary', 'moderate', 'active'], 36621 nan values
smoking_alcohol (object) has 3 unique values: ['yes', 'occasional', 'no'], 28582 nan values
gender (object) has 3 unique values: ['female', 'other', 'male'], 21373 nan values

# External Data Overview for data/student_health_dataset_50k.csv.gz

-> student_health_dataset_50k.csv has 50000 rows and 16 columns.
Here is some information about the columns:
student_id (int64) has range: 1000.00 - 1999.00, 0 nan values
timestamp (object) has 0 nan values
sleep_duration (float64) has range: 3.00 - 10.00, 0 nan values
heart_rate (float64) has range: 50.00 - 120.00, 0 nan values
bmi (float64) has range: 16.00 - 35.00, 0 nan values
calorie_expenditure (float64) has range: 1200.00 - 3901.00, 0 nan values
step_count (int64) has range: 1000.00 - 14999.00, 0 nan values
exercise_duration (float64) has range: 0.00 - 120.00, 0 nan values
water_intake (float64) has range: 0.50 - 4.94, 0 nan values
diet_type (object) has 3 unique values: ['veg', 'non-veg', 'balanced'], 0 nan values
stress_level (object) has 3 unique values: ['high', 'medium', 'low'], 0 nan values
sleep_quality (object) has 3 unique values: ['average', 'poor', 'good'], 0 nan values
physical_activity_level (object) has 3 unique values: ['active', 'sedentary', 'moderate'], 0 nan values
smoking_alcohol (object) has 3 unique values: ['yes', 'occasional', 'no'], 0 nan values
gender (object) has 3 unique values: ['other', 'male', 'female'], 0 nan values
health_condition (object) has 3 unique values: ['at-risk', 'unhealthy', 'fit'], 0 nan values

# External Data Description for data/student_health_dataset_50k.csv.gz

Original source-style student health dataset downloaded from Kaggle dataset
`ziya07/college-student-health-behavior-dataset`. The file is stored as
`data/student_health_dataset_50k.csv.gz`.

Kaggle dataset page description: this is a 50,000-record student health dataset
intended to capture lifestyle, physiological, and psychological characteristics
of college students. The page describes records as time-stamped observations
combining survey-style attributes, wearable-device-style measurements, and
institutional health-record-style signals. The dataset is described as modeled
on trends from large-scale student health studies in China and intended to
represent variation in sleep, stress, physical activity, academic pressure,
mental well-being, and health risk.

It has 50,000 rows and includes the target column `health_condition` with
classes `at-risk`, `unhealthy`, and `fit`. Class counts are 43,718 at-risk,
3,816 unhealthy, and 2,466 fit.

Configured external file overview:

-> student_health_dataset_50k.csv has 50000 rows and 16 columns.
Here is some information about the columns:
student_id (int64) has range: 1000.00 - 1999.00, 0 nan values
timestamp (object) has 0 nan values
sleep_duration (float64) has range: 3.00 - 10.00, 0 nan values
heart_rate (float64) has range: 50.00 - 120.00, 0 nan values
bmi (float64) has range: 16.00 - 35.00, 0 nan values
calorie_expenditure (float64) has range: 1200.00 - 3901.00, 0 nan values
step_count (int64) has range: 1000.00 - 14999.00, 0 nan values
exercise_duration (float64) has range: 0.00 - 120.00, 0 nan values
water_intake (float64) has range: 0.50 - 4.94, 0 nan values
diet_type (object) has 3 unique values: ['veg', 'non-veg', 'balanced'], 0 nan values
stress_level (object) has 3 unique values: ['high', 'medium', 'low'], 0 nan values
sleep_quality (object) has 3 unique values: ['average', 'poor', 'good'], 0 nan values
physical_activity_level (object) has 3 unique values: ['active', 'sedentary', 'moderate'], 0 nan values
smoking_alcohol (object) has 3 unique values: ['yes', 'occasional', 'no'], 0 nan values
gender (object) has 3 unique values: ['other', 'male', 'female'], 0 nan values
health_condition (object) has 3 unique values: ['at-risk', 'unhealthy', 'fit'], 0 nan values

Columns:
- `student_id`
- `timestamp`
- `sleep_duration`
- `heart_rate`
- `bmi`
- `calorie_expenditure`
- `step_count`
- `exercise_duration`
- `water_intake`
- `diet_type`
- `stress_level`
- `sleep_quality`
- `physical_activity_level`
- `smoking_alcohol`
- `gender`
- `health_condition`

Note: the Kaggle dataset page also describes broader feature categories such as
screen time, sitting time, academic pressure, mental health status, and social
relationships. Those fields are present in the separate
`enhanced_student_health_dataset_50k.xls` file listed by Kaggle, but they are
not present in the `student_health_dataset_50k.csv` file configured here for TML.

Overlap with Kaggle train/test: all feature columns except `student_id` and
`timestamp` match Kaggle feature names. The external `student_id` is separate
from Kaggle `id`; do not assume the identifiers can be joined. The `timestamp`
column is external-only and may support source-domain temporal summaries, but it
is not present in Kaggle train/test rows.

Use this dataset only as auxiliary/external data. Do not assume `student_id`
matches Kaggle `id`; treat it as a separate identifier. Avoid target leakage:
`health_condition` can be used for auxiliary supervised statistics only when the
implementation fits those statistics on training folds or otherwise avoids using
validation/test labels.

# Task

## Goal
Predict student health_condition risk class from health and lifestyle features.

## Evaluation
Submissions are scored by balanced accuracy between predicted class labels and the true health_condition values. Submission must contain id and a predicted class label for health_condition for each test row.

## Data description
train.csv has 690088 rows with target column health_condition and 13 feature columns plus id; test.csv has the same features and id without the target; sample_submission.csv shows the required output format. The target has three classes: unhealthy, at-risk, and fit.

# Project Target
- ID column: id
- Target column: health_condition
- Problem type: multiclass
- Evaluation metric: balanced_accuracy

# Failed Materialization

- Mode: autogluon
- Hypothesis ID: 000012
- Source file: autogluon-001.py
- Failed node: 20260701T181931-47b328dc-58
- Run: 20260701T140514-c0d16688

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T181931-47b328dc-58/02-code.py", line 891, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T181931-47b328dc-58/02-code.py", line 45, in add_substance_use_context_buffer
    + sleep_quality.eq("good").astype("int8")
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 6665, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 449, in astype
    return self.apply(
           ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 784, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/dtypes/astype.py", line 237, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/dtypes/astype.py", line 179, in astype_array
    values = values.astype(dtype, copy=copy)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 587, in astype
    raise ValueError("cannot convert NA to integer")
ValueError: cannot convert NA to integer
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T181931-47b328dc-58/02-code.py", line 1054, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T181931-47b328dc-58/02-code.py", line 891, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T181931-47b328dc-58/02-code.py", line 45, in add_substance_use_context_buffer
    + sleep_quality.eq("good").astype("int8")
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/generic.py", line 6665, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 449, in astype
    return self.apply(
           ^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 784, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/dtypes/astype.py", line 237, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/dtypes/astype.py", line 179, in astype_array
    values = values.astype(dtype, copy=copy)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 587, in astype
    raise ValueError("cannot convert NA to integer")
ValueError: cannot convert NA to integer

stdout.log:
TML_RUNTIME|autogluon_options|audit_score_enabled=False|audit_fraction=0.1|feature_importance_enabled=False|fi_subsample_size=0.1|fi_num_shuffle_sets=10|fi_include_confidence_band=True
AutoGluon materialization: loaded aux file student_health_dataset_50k.csv.gz rows=50000 cols=16 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=substance_use_context_buffer
TheML feature group: failed name=substance_use_context_buffer elapsed_s=0.895 rows=985841 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

SUBSTANCE_BURDEN_MAP = {
    "no": 0,
    "occasional": 1,
    "yes": 2,
}

def _column(raw, name):
    if name in raw.columns:
        return raw[name]
    return pd.Series(pd.NA, index=raw.index)

def _normalized_text(raw, name):
    return _column(raw, name).astype("string").str.strip().str.lower()

def add_substance_use_context_buffer(raw, deps, aux):
    index = raw.index

    smoking_alcohol = _normalized_text(raw, "smoking_alcohol")
    diet_type = _normalized_text(raw, "diet_type")
    stress_level = _normalized_text(raw, "stress_level")
    sleep_quality = _normalized_text(raw, "sleep_quality")
    physical_activity_level = _normalized_text(raw, "physical_activity_level")

    sleep_duration = pd.to_numeric(_column(raw, "sleep_duration"), errors="coerce")
    heart_rate = pd.to_numeric(_column(raw, "heart_rate"), errors="coerce")
    bmi = pd.to_numeric(_column(raw, "bmi"), errors="coerce")
    step_count = pd.to_numeric(_column(raw, "step_count"), errors="coerce")
    exercise_duration = pd.to_numeric(_column(raw, "exercise_duration"), errors="coerce")

    substance_burden = smoking_alcohol.map(SUBSTANCE_BURDEN_MAP).fillna(-1).astype("int8")

    protective_context_score = (
        diet_type.isin(("balanced", "veg")).astype("int8")
        + physical_activity_level.isin(("moderate", "active")).astype("int8")
        + sleep_quality.eq("good").astype("int8")
        + stress_level.eq("low").astype("int8")
        + sleep_duration.between(7.0, 9.0, inclusive="both").astype("int8")
        + step_count.ge(8000.0).astype("int8")
        + exercise_duration.ge(30.0).astype("int8")
        + bmi.ge(18.5).mul(bmi.lt(25.0)).astype("int8")
    ).astype("int8")

    adverse_context_score = (
        stress_level.eq("high").astype("int8")
        + sleep_quality.eq("poor").astype("int8")
        + physical_activity_level.eq("sedentary").astype("int8")
        + diet_type.eq("non-veg").astype("int8")
        + (sleep_duration.lt(6.0) | sleep_duration.gt(9.5)).astype("int8")
        + step_count.lt(5000.0).astype("int8")
        + exercise_duration.lt(10.0).astype("int8")
        + (bmi.lt(18.5) | bmi.ge(30.0)).astype("int8")
        + heart_rate.ge(90.0).astype("int8")
    ).astype("int8")

    has_substance_value = substance_burden.ge(0)
    uses_substance = substance_burden.ge(1)
    no_substance = substance_burden.eq(0)

    substance_buffer_gap = pd.Series(np.nan, index=index, dtype="float32")
    substance_buffer_gap.loc[has_substance_value] = (
        adverse_context_score.loc[has_substance_value].astype("float32")
        + substance_burden.loc[has_substance_value].astype("float32")
        - protective_context_score.loc[has_substance_value].astype("float32")
    )

    substance_amplification = pd.Series(np.nan, index=index, dtype="float32")
    substance_amplification.loc[has_substance_value] = (
        substance_burden.loc[has_substance_value].astype("float32")
        * adverse_context_score.loc[has_substance_value].astype("float32")
    )

    substance_buffering = pd.Series(np.nan, index=index, dtype="float32")
    substance_buffering.loc[has_substance_value] = (
        substance_burden.loc[has_substance_value].astype("float32")
        * protective_context_score.loc[has_substance_value].astype("float32")
    )

    context_bucket = pd.Series("mixed_context", index=index, dtype="object")
    context_bucket.loc[~has_substance_value] = "missing_substance"
    context_bucket.loc[
        no_substance & protective_context_score.ge(5) & adverse_context_score.le(1)
    ] = "clean_supported"
    context_bucket.loc[
        no_substance & adverse_context_score.ge(4)
    ] = "non_substance_lifestyle_risk"
    context_bucket.loc[
        uses_substance & protective_context_score.ge(4) & adverse_context_score.le(2)
    ] = "buffered_use"
    context_bucket.loc[
        uses_substance & adverse_context_score.ge(4)
    ] = "compounding_use"

    return pd.DataFrame(
        {
            "substance_burden": substance_burden,
            "protective_context_score": protective_context_score,
            "adverse_context_score": adverse_context_score,
            "substance_buffer_gap": substance_buffer_gap,
            "substance_amplification": substance_amplification,
            "substance_buffering": substance_buffering,
            "context_bucket": context_bucket,
        },
        index=index,
    )

FEATURE_GROUPS = [
    {
        "name": "substance_use_context_buffer",
        "fn": add_substance_use_context_buffer,
        "depends_on": [],
        "description": "Encodes whether smoking and alcohol burden is buffered or amplified by protective and adverse lifestyle context.",
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