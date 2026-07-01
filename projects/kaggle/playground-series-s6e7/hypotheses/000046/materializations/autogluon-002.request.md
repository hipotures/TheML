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
- Hypothesis ID: 000046
- Source file: autogluon-001.py
- Failed node: 20260701T192351-125573be-63
- Run: 20260701T140514-c0d16688

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
KeyError: ('sleep_amount_load', 'movement_load', 'exercise_load', 'hydration_load', 'diet_load', 'stress_load', 'sleep_quality_load', 'stated_activity_load', 'substance_load')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 989, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 157, in add_behavior_physiology_lag_phase
    behavior_load, behavior_coverage = _bounded_mean(
                                       ^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 69, in _bounded_mean
    available = frame[columns].notna().sum(axis=1).astype("float64")
                ~~~~~^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: ('sleep_amount_load', 'movement_load', 'exercise_load', 'hydration_load', 'diet_load', 'stress_load', 'sleep_quality_load', 'stated_activity_load', 'substance_load')
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: ('sleep_amount_load', 'movement_load', 'exercise_load', 'hydration_load', 'diet_load', 'stress_load', 'sleep_quality_load', 'stated_activity_load', 'substance_load')

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 1160, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 989, in main
    transformed = run_feature_groups(
                  ^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/src/tml/features/groups.py", line 59, in run_feature_groups
    block = group["fn"](raw.copy(), {key: value.copy() for key, value in deps.items()}, aux.copy())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 157, in add_behavior_physiology_lag_phase
    behavior_load, behavior_coverage = _bounded_mean(
                                       ^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e7/runs/20260701T140514-c0d16688/artifacts/20260701T192351-125573be-63/02-code.py", line 69, in _bounded_mean
    available = frame[columns].notna().sum(axis=1).astype("float64")
                ~~~~~^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/frame.py", line 4113, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: ('sleep_amount_load', 'movement_load', 'exercise_load', 'hydration_load', 'diet_load', 'stress_load', 'sleep_quality_load', 'stated_activity_load', 'substance_load')

stdout.log:
TML_RUNTIME|autogluon_options|audit_score_enabled=False|audit_fraction=0.1|feature_importance_enabled=False|fi_subsample_size=0.1|fi_num_shuffle_sets=10|fi_include_confidence_band=True|lgbm_gpu_cat_fallback_action=fallback_to_cpu|lgbm_gpu_cat_max_cardinality=512
AutoGluon materialization: loaded aux file student_health_dataset_50k.csv.gz rows=50000 cols=16 passed_to_groups=True
AutoGluon materialization: starting feature groups
TheML feature group: start name=behavior_physiology_lag_phase
TheML feature group: failed name=behavior_physiology_lag_phase elapsed_s=0.942 rows=985841 cols=0
```

# Previous Code

```python
import numpy as np
import pandas as pd

BEHAVIOR_COMPONENT_COLUMNS = (
    "sleep_duration",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
)

PHYSIOLOGY_COMPONENT_COLUMNS = (
    "bmi",
    "heart_rate",
)

DIET_LOAD_MAP = {
    "balanced": 0.0,
    "veg": 0.2,
    "non-veg": 0.45,
}

STRESS_LOAD_MAP = {
    "low": 0.0,
    "medium": 0.5,
    "high": 1.0,
}

SLEEP_QUALITY_LOAD_MAP = {
    "good": 0.0,
    "average": 0.5,
    "poor": 1.0,
}

ACTIVITY_LOAD_MAP = {
    "active": 0.0,
    "moderate": 0.4,
    "sedentary": 1.0,
}

SMOKING_ALCOHOL_LOAD_MAP = {
    "no": 0.0,
    "occasional": 0.55,
    "yes": 1.0,
}

def _numeric_series(raw, column):
    if column in raw.columns:
        return pd.to_numeric(raw[column], errors="coerce")
    return pd.Series(np.nan, index=raw.index, dtype="float64")

def _category_series(raw, column):
    if column in raw.columns:
        return raw[column].astype("string").str.lower().str.strip()
    return pd.Series(pd.NA, index=raw.index, dtype="string")

def _bounded_mean(frame, columns, default_value):
    available = frame[columns].notna().sum(axis=1).astype("float64")
    load = frame[columns].mean(axis=1, skipna=True).fillna(default_value)
    coverage = available / float(len(columns))
    return load.clip(0.0, 1.0), coverage.clip(0.0, 1.0)

def _sleep_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 7.0
    high_mask = values > 9.0
    load.loc[low_mask] = ((7.0 - values.loc[low_mask]) / 1.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 9.0) / 0.5).clip(0.0, 1.0)
    load.loc[values.isna()] = np.nan
    return load

def _steps_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 7500.0
    high_mask = values > 12000.0
    load.loc[low_mask] = ((7500.0 - values.loc[low_mask]) / 2500.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 12000.0) / 3000.0).clip(0.0, 0.25)
    load.loc[values.isna()] = np.nan
    return load

def _exercise_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 30.0
    high_mask = values > 75.0
    load.loc[low_mask] = ((30.0 - values.loc[low_mask]) / 10.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 75.0) / 45.0).clip(0.0, 0.2)
    load.loc[values.isna()] = np.nan
    return load

def _water_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 1.5
    high_mask = values > 3.5
    load.loc[low_mask] = ((1.5 - values.loc[low_mask]) / 1.0).clip(0.0, 1.0)
    load.loc[high_mask] = ((values.loc[high_mask] - 3.5) / 1.5).clip(0.0, 0.35)
    load.loc[values.isna()] = np.nan
    return load

def _bmi_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 18.5
    overweight_mask = (values >= 25.0) & (values < 30.0)
    high_mask = values >= 30.0

    load.loc[low_mask] = ((18.5 - values.loc[low_mask]) / 2.5).clip(0.0, 1.0) * 0.65
    load.loc[overweight_mask] = 0.35 + ((values.loc[overweight_mask] - 25.0) / 5.0).clip(0.0, 1.0) * 0.25
    load.loc[high_mask] = 0.75 + ((values.loc[high_mask] - 30.0) / 5.0).clip(0.0, 1.0) * 0.25
    load.loc[values.isna()] = np.nan
    return load.clip(0.0, 1.0)

def _heart_rate_load(values):
    load = pd.Series(0.0, index=values.index, dtype="float64")
    low_mask = values < 60.0
    mild_high_mask = (values > 80.0) & (values <= 90.0)
    high_mask = values > 90.0

    load.loc[low_mask] = ((60.0 - values.loc[low_mask]) / 10.0).clip(0.0, 1.0) * 0.45
    load.loc[mild_high_mask] = ((values.loc[mild_high_mask] - 80.0) / 10.0).clip(0.0, 1.0) * 0.55
    load.loc[high_mask] = 0.65 + ((values.loc[high_mask] - 90.0) / 30.0).clip(0.0, 1.0) * 0.35
    load.loc[values.isna()] = np.nan
    return load.clip(0.0, 1.0)

def add_behavior_physiology_lag_phase(raw, deps, aux):
    behavior_components = pd.DataFrame(index=raw.index)
    behavior_components["sleep_amount_load"] = _sleep_load(_numeric_series(raw, "sleep_duration"))
    behavior_components["movement_load"] = _steps_load(_numeric_series(raw, "step_count"))
    behavior_components["exercise_load"] = _exercise_load(_numeric_series(raw, "exercise_duration"))
    behavior_components["hydration_load"] = _water_load(_numeric_series(raw, "water_intake"))
    behavior_components["diet_load"] = _category_series(raw, "diet_type").map(DIET_LOAD_MAP).astype("float64")
    behavior_components["stress_load"] = _category_series(raw, "stress_level").map(STRESS_LOAD_MAP).astype("float64")
    behavior_components["sleep_quality_load"] = _category_series(raw, "sleep_quality").map(SLEEP_QUALITY_LOAD_MAP).astype("float64")
    behavior_components["stated_activity_load"] = _category_series(raw, "physical_activity_level").map(ACTIVITY_LOAD_MAP).astype("float64")
    behavior_components["substance_load"] = _category_series(raw, "smoking_alcohol").map(SMOKING_ALCOHOL_LOAD_MAP).astype("float64")

    physiology_components = pd.DataFrame(index=raw.index)
    physiology_components["bmi_strain_load"] = _bmi_load(_numeric_series(raw, "bmi"))
    physiology_components["heart_rate_strain_load"] = _heart_rate_load(_numeric_series(raw, "heart_rate"))

    behavior_load, behavior_coverage = _bounded_mean(
        behavior_components,
        tuple(behavior_components.columns),
        0.5,
    )
    physiology_load, physiology_coverage = _bounded_mean(
        physiology_components,
        tuple(physiology_components.columns),
        0.5,
    )

    lag_gap = physiology_load - behavior_load
    absolute_lag = lag_gap.abs()
    coverage_weighted_lag = lag_gap * np.sqrt(behavior_coverage * physiology_coverage)

    both_low = (behavior_load <= 0.33) & (physiology_load <= 0.33)
    both_high = (behavior_load >= 0.67) & (physiology_load >= 0.67)
    behavior_high_phys_low = (behavior_load >= 0.67) & (physiology_load <= 0.33)
    physiology_high_behavior_low = (physiology_load >= 0.67) & (behavior_load <= 0.33)

    phase_code = pd.Series("transitional_mixed", index=raw.index, dtype="object")
    phase_code.loc[both_low] = "consistent_fit_like"
    phase_code.loc[both_high] = "entrenched_risk"
    phase_code.loc[behavior_high_phys_low] = "latent_behavior_risk"
    phase_code.loc[physiology_high_behavior_low] = "embodied_legacy_risk"

    phase_ordinal = pd.Series(2, index=raw.index, dtype="int8")
    phase_ordinal.loc[both_low] = 0
    phase_ordinal.loc[behavior_high_phys_low] = 1
    phase_ordinal.loc[physiology_high_behavior_low] = 3
    phase_ordinal.loc[both_high] = 4

    new_features = pd.DataFrame(index=raw.index)
    new_features["behavior_load"] = behavior_load.astype("float32")
    new_features["physiology_load"] = physiology_load.astype("float32")
    new_features["behavior_coverage"] = behavior_coverage.astype("float32")
    new_features["physiology_coverage"] = physiology_coverage.astype("float32")
    new_features["lag_gap"] = lag_gap.astype("float32")
    new_features["absolute_lag"] = absolute_lag.astype("float32")
    new_features["coverage_weighted_lag"] = coverage_weighted_lag.astype("float32")
    new_features["phase_code"] = phase_code
    new_features["phase_ordinal"] = phase_ordinal
    new_features["behavior_high_physiology_low"] = behavior_high_phys_low.astype("int8")
    new_features["physiology_high_behavior_low"] = physiology_high_behavior_low.astype("int8")
    new_features["both_domains_low_load"] = both_low.astype("int8")
    new_features["both_domains_high_load"] = both_high.astype("int8")
    return new_features

FEATURE_GROUPS = [
    {
        "name": "behavior_physiology_lag_phase",
        "fn": add_behavior_physiology_lag_phase,
        "depends_on": [],
        "description": "Contrasts near-term behavior load with embodied physiology load to identify lagged health-risk phases.",
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