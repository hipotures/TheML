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
- title: Weakest lifestyle pillar bottleneck
- group_name: lifestyle_pillar_bottleneck
- family: domain_balance_score
- summary: Represent each student by the weakest and most uneven lifestyle-health pillar rather than by overall wellness alone, capturing whether risk is driven by a single severe bottleneck or broad multi-domain deterioration.
- strategy: Source-informed by lifestyle-medicine and cardiovascular-health domain frameworks (https://lifestylemedicine.org/about-lifestyle-medicine/ and https://www.heart.org/en/healthy-living/healthy-lifestyle/lifes-essential-8), create six row-level pillar scores on a 0 to 1 scale: sleep recovery from sleep duration bands and sleep quality; movement from step count, exercise duration, and physical activity level; nutrition-support from diet type and water intake; stress-load from stress level; substance-avoidance from smoking_alcohol; cardiometabolic steadiness from BMI and heart-rate bands. Use deterministic ordinal mappings where protective states score 1, intermediate states score 0.5, adverse states score 0, and numeric bands use broadly accepted healthy ranges with adjacent borderline bands scored 0.5. For each pillar, average available component scores; if all components for a pillar are missing, assign neutral 0.5 so this group does not become a pure missingness signal. Emit a coherent set of bottleneck features: minimum pillar score, weakest pillar id, second-weakest pillar score, count of pillars below 0.34, count above 0.66, spread between best and worst pillar, mean of the two weakest pillars, mean of the two strongest pillars, and strongest-minus-weakest compensation gap.
- expected_signal: Balanced accuracy may improve because fit students should tend to have no severe weak pillar, unhealthy students may show multiple low pillars or a very low cardiometabolic/stress/substance bottleneck, and at-risk students may occupy mixed profiles where one domain fails despite otherwise moderate habits.
- risk: This may overlap with prior guideline-deviation and categorical-polarity groups, and the fixed health bands may be unstable if the synthetic target encodes different thresholds or treats categories such as diet type in a non-domain-realistic way.

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