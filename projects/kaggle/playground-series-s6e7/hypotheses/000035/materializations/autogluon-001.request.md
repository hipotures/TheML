Write preprocessing feature-group code for an AutoGluon wrapper.

Your output must define one semantic feature group for preprocessing.
The fixed wrapper imports this file, runs `FEATURE_GROUPS`, logs group timing,
renames returned columns, assembles the final DataFrame, and handles all
non-preprocessing work.

Do not write the wrapper.

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

# Hypothesis
- title: Lifestyle Risk Cascade
- group_name: lifestyle_risk_cascade_alignment
- family: ordered_domain_interaction
- summary: Represent whether upstream lifestyle stressors, recovery impairment, activity shortfall, and physiological strain form a coherent risk cascade rather than isolated or contained warning signs.
- strategy: Build four [0,1] domain risk scores from raw columns only: exposure_load as the observed-component mean of stress_level mapped low/medium/high to 0/0.5/1, smoking_alcohol mapped no/occasional/yes to 0/0.5/1, diet_type mapped balanced/veg/non-veg to 0/0.25/0.5, and water deficit clipped as (2.0 - water_intake) / 1.5; recovery_impairment as the mean of sleep_quality mapped good/average/poor to 0/0.5/1 and a U-shaped sleep_duration risk max((7 - sleep_duration) / 4, (sleep_duration - 9) / 1, 0) clipped to [0,1]; activity_shortfall as the mean of physical_activity_level mapped active/moderate/sedentary to 0/0.5/1, step deficit clipped as (10000 - step_count) / 9000, exercise deficit clipped as (30 - exercise_duration) / 30, and calorie-output deficit clipped as (1800 - calorie_expenditure) / 600; physiology_strain as the mean of BMI distance from 18.5-25 scaled to the observed 16-35 range and heart-rate strain clipped as max((heart_rate - 70) / 30, 0.5 * (55 - heart_rate) / 10, 0). For each domain, average observed components only; if all components in a domain are missing, set that domain to 0.5. Emit cascade_slope from a least-squares line over ordered scores [exposure_load, recovery_impairment, activity_shortfall, physiology_strain], the three adjacent signed gaps, monotone_propagation_fraction as the share of adjacent pairs where downstream risk is at least upstream risk minus 0.1, propagated_risk as the geometric mean of the four scores with a small 0.05 offset then clipped to [0,1], contained_exposure as max(0, exposure_load - max(recovery_impairment, activity_shortfall, physiology_strain)), and unexplained_physiology as max(0, physiology_strain - mean(exposure_load, recovery_impairment, activity_shortfall)).
- expected_signal: Public-health sources treat sleep, activity, nutrition, tobacco/alcohol, and body status as linked lifestyle risk domains, while student-focused evidence suggests poor lifestyle can weaken activity-related fitness and sleep quality is central to health outcomes; this cascade encoding may separate fit students with isolated manageable issues from at-risk students whose risks propagate across domains and unhealthy students with broad downstream strain. Sources consulted: https://www.cdc.gov/environmental-health-tracking/php/data-research/lifestyle-risk-factors.html, https://www.heart.org/en/healthy-living/healthy-lifestyle/lifes-essential-8, https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1664822/full
- risk: The hand-ordered cascade may be too assumption-driven for a synthetic target, and some component mappings overlap with prior burden, bottleneck, and interaction groups; diet, hydration, calorie, and heart-rate thresholds may be source-specific, and neutral handling of fully missing domains can dilute informative missingness already captured elsewhere.

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

AutoGluon wrapper boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not define `preprocess(df)`.
- Do not read project data files or `./input`.
- The fixed AutoGluon wrapper handles all of those steps outside this generated file.