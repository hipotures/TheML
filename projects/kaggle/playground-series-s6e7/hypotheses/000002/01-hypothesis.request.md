You are designing feature-group hypothesis.

# Task

## Goal
Predict student health_condition risk class from health and lifestyle features.

## Evaluation
Submissions are scored by balanced accuracy between predicted class labels and the true health_condition values. Submission must contain id and a predicted class label for health_condition for each test row.

## Data description
train.csv has 690088 rows with target column health_condition and 13 feature columns plus id; test.csv has the same features and id without the target; sample_submission.csv shows the required output format. The target has three classes: unhealthy, at-risk, and fit.

# Data Overview

-> sample_submission.csv has 295753 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 690088.00 - 985840.00, 0 nan values
health_condition (object) has 1 unique values: ['at-risk'], 0 nan values

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
health_condition (object) has 3 unique values: ['unhealthy', 'at-risk', 'fit'], 0 nan values
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

# Project Target
- ID column: id
- Target column: health_condition
- Problem type: multiclass
- Evaluation metric: balanced_accuracy
- Submission kind: labels

# Prior Root Groups

Use this only to avoid obvious repetition. Do not imitate the best previous group.
- raw_autogluon_baseline (baseline): Train AutoGluon on the raw project columns without generated feature groups.
- guideline_deviation_burden (domain_threshold_risk_index): Create a compact lifestyle-health burden representation by translating each measured or reported habit into distance from broadly accepted healthy ranges and aggregating the cumulative deviation pattern.

# Web search

Live web search is available and should be used before proposing the hypothesis.
Use the `web_search` tool for both stages:
- Search queries: use `web_search` to find feature engineering ideas,
  preprocessing directions, data representations, validation traps, simple
  baseline patterns, public Kaggle discussions/notebooks, package
  documentation, and domain or method background relevant to this task.
- Page retrieval: use `web_search` again to open/fetch the most relevant result
  pages or direct URLs and read their page content.

Analyze the retrieved information before using it. Do not rely only on search
result titles, snippets, or URLs. Convert externally found techniques into one
generalized, leakage-safe, budget-compatible semantic feature-group idea. Do not
copy public solutions directly. If `web_search` is not useful or not available,
continue normally without inventing sources.

# Rules

- Create exactly one new semantic feature group.
- Do not create a full preprocessing pipeline.
- Do not modify existing groups.
- Do not discuss model families, ensembling, validation folds, training time, hyperparameters, or AutoGluon configuration.
- The hypothesis must later generate exactly one feature-group function.
- The group may contain multiple derived columns only when they belong to one coherent semantic idea.
- A hypothesis must represent one homogeneous group-level change. Multiple implementation details are allowed only when they are necessary parts of that same semantic change.
- Use prior root groups only to avoid obvious repetition.
- must use `depends_on: []`.
- Do not output code.

Return valid JSON only. Do not use markdown fences.

Return exactly this JSON object:

{
  "title": "short descriptive title",
  "group_name": "concise_snake_case_group_name",
  "family": "compact feature-family label",
  "summary": "one comprehensive sentence describing the semantic feature-group idea; do not list formulas, exact feature names, or implementation details here",
  "depends_on": [],
  "strategy": "concrete deterministic feature logic, formulas, bins, statistics, and edge handling",
  "expected_signal": "why this group may improve or clarify the metric",
  "risk": "specific overfitting, leakage, cost, redundancy, or instability risk"
}