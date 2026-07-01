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
- title: Auxiliary temporal stability affinity
- group_name: external_temporal_stability_affinity
- family: external_temporal_behavior
- summary: Represent each static train/test row by how closely its observable profile resembles auxiliary-source students with stable versus erratic longitudinal health and lifestyle routines, using the external timestamped repeated observations as a source-only behavioral regularity reference.
- strategy: From data/student_health_dataset_50k.csv.gz only, parse timestamp, sort by student_id and time, and for each student compute source-only temporal instability statistics: numeric IQR divided by the source-column IQR, standard deviation divided by source-column IQR, and median absolute successive change divided by source-column IQR for sleep_duration, heart_rate, bmi, calorie_expenditure, step_count, exercise_duration, and water_intake; for diet_type, stress_level, sleep_quality, physical_activity_level, smoking_alcohol, and gender compute normalized categorical entropy and successive switch rate. Average these components into one per-student instability score after clipping each component to [0, 5], then split source students into low, middle, and high instability tertiles. For each tertile, build a centroid over the shared train/test feature space using source medians for numeric columns and category frequencies for categorical columns, with numeric scaling by source IQR plus a small epsilon. For every train/test row, compute observed-dimension normalized distance to each tertile centroid: squared scaled numeric differences plus categorical mismatch distance equal to 1 minus the centroid frequency of the row category, ignoring missing dimensions and dividing by the number of observed components. Convert distances to similarities with exp(-distance), add the similarity-weighted expected instability score, the high-minus-low stability contrast, and an observed-feature coverage fraction. If a row has no observed shared features, emit neutral tertile similarities, the global source mean instability, zero contrast, and zero coverage. Background used: https://arxiv.org/html/2503.08002v1, https://www.nature.com/articles/s41591-024-03155-8, https://www.c4tbh.org/predicting-stress-in-first-year-college-students-using-sleep-data-from-wearable-devices/.
- expected_signal: The target may reflect not only a one-time health state but also the kind of routine stability that generated it; external longitudinal evidence links changes or irregularity in sleep and related biometric measures with stress and chronic health outcomes, so source-derived stable/unstable routine affinities can expose a risk axis absent from static raw values while avoiding train/test label leakage.
- risk: The auxiliary timestamped source may encode temporal behavior patterns that were weakened or removed in the synthetic train/test distribution, making the affinities noisy; it also partially overlaps with other external support ideas and can add redundant distance-style columns if the source instability tertiles have very similar centroids.

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