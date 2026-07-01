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
- title: Sleep Strain Echo
- group_name: sleep_strain_echo
- family: recovery_physiology_interaction
- summary: Represent whether a student's sleep quantity and perceived sleep quality look physiologically restorative or instead coincide with body-mass, pulse, and exertion strain.
- strategy: Create sleep-duration bands using fixed cut points: severe_short < 5, short 5 to < 7, normal 7 to 9, and long > 9; compute sleep_debt = max(0, 7 - sleep_duration), sleep_excess = max(0, sleep_duration - 9), and map sleep_quality to good = 0, average = 0.5, poor = 1, missing = 0.5. Define rest_deficit = min(3, sleep_debt) + 0.5 * min(2, sleep_excess) + sleep_quality_penalty. Learn heart_rate q75 and q90 plus step_count and exercise_duration q25/q75 from training rows only; set pulse_tail to 0 below q75, 0.5 from q75 to q90, and 1 above q90. Define bmi_tail as 0 for 18.5 <= bmi < 25, 0.5 for 16 <= bmi < 18.5 or 25 <= bmi < 30, and 1 for bmi >= 30. Define exertion_high as either step_count or exercise_duration >= its q75 and exertion_low as both <= q25 when observed. Emit coherent derived columns: rest_deficit, sleep_physio_echo = rest_deficit * (pulse_tail + bmi_tail), sleep_exertion_overreach = rest_deficit * pulse_tail * exertion_high, quiet_sleep_deficit = rest_deficit * (1 - pulse_tail) * (1 - bmi_tail), restorative_alignment = 1 when normal sleep, good sleep_quality, bmi_tail = 0, and pulse_tail = 0 else 0, plus a compact phenotype code from sleep band x quality_bad flag x high_physio_strain flag. For missing numeric inputs use training medians only inside these calculations; do not add missingness columns here; if all required numeric context is missing, return neutral zero strain terms except the quality penalty if present.
- expected_signal: This may improve balanced accuracy by separating students with isolated sleep irregularity from those whose sleep deficit has a physiological strain echo, helping distinguish at-risk from unhealthy while preserving a clean fit-like restorative profile; background sources connect 7-9 hour sleep with BMI patterns and sleep deprivation with cardiac autonomic effects: https://socialhealthjournal.ust.edu.ph/wp-content/uploads/2022/02/Associations-between-Sleep-Duration-and-Body-Mass-Index-in-the-US-Adult-Population.pdf and https://www.mdpi.com/2071-1050/13/16/8769.
- risk: The fixed sleep and BMI thresholds may not match this synthetic student population perfectly, and the group overlaps partly with prior recovery and cardiometabolic concepts, so its marginal value depends on whether sleep-linked physiological coupling is not already captured by those broader groups.

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