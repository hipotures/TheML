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

# Prior Root Groups

Use this only to avoid obvious repetition. Do not imitate the best previous group.
- raw_autogluon_baseline (baseline): Train AutoGluon on the raw project columns without generated feature groups.
- guideline_deviation_burden (domain_threshold_risk_index): Create a compact lifestyle-health burden representation by translating each measured or reported habit into distance from broadly accepted healthy ranges and aggregating the cumulative deviation pattern.
- activity_normalized_cardio_strain (physiological_efficiency_ratios): Represent whether a student's cardiovascular and energy signals look elevated relative to their movement and exercise output, capturing strain per unit of activity rather than absolute lifestyle levels.
- response_completeness_signature (informative_missingness): Represent each student's pattern of unanswered or unavailable health and lifestyle fields as a behavioral and measurement-completeness signal rather than treating missing values only as preprocessing noise.
- perceived_observed_alignment (self_report_objective_discordance): Represent whether each student's subjective categorical health reports agree or conflict with adjacent measured behavior and physiology signals, treating perception-observation misalignment as a distinct health-risk pattern.
- stress_recovery_triad_profiles (behavioral_interaction): Represent each student's rest, pressure, and movement pattern as a recovery-balance state that separates buffered high-demand lifestyles from depleted low-recovery lifestyles.
- external_health_archetype_affinity (external_prototype_similarity): Represent each row by how closely its complete lifestyle and physiology profile resembles labeled health-condition archetypes learned only from the auxiliary student-health source.
- waking_day_behavior_density (24h_time_budget): Represent each student's lifestyle intensity relative to estimated waking time, capturing whether movement, exercise, energy use, and hydration are sparse or concentrated within the non-sleep portion of the day.
- gender_relative_numeric_percentiles (peer_context_normalization): Represent each student's measured health and lifestyle quantities by how unusual they are compared with demographically similar peers, exposing relative physiological and behavioral extremeness that raw absolute values can hide.
- behavioral_cooccurrence_rarity (unsupervised_profile_density): Represent each student by how familiar or unusual their combined lifestyle, physiology, and self-report pattern is relative to the training population, treating atypical behavior constellations as potential health-risk signals.
- activity_adjusted_hydration_reserve (contextual_hydration_balance): Estimate whether each student's fluid intake is sufficient for the water demand implied by their body context and exertion pattern, separating well-supported active lifestyles from exertion-with-deficit and low-demand excess-intake profiles.
- external_source_state_likelihood (external_supervised_density): Represent each student by smoothed health-condition affinity signals implied by matching labeled auxiliary-source lifestyle states, turning source-domain class neighborhoods into compact risk evidence.
- substance_use_context_buffer (behavioral_risk_context): Represent whether substance-use risk is buffered or amplified by the surrounding lifestyle context, separating students whose smoking/alcohol pattern co-occurs with protective habits from those where it clusters with broader health-risk behavior.
- activity_modality_balance (activity_composition): Capture whether a student's movement pattern is primarily structured exercise, incidental daily ambulation, broadly active, or behaviorally inconsistent with their stated activity level.
- fitness_contextualized_cardiometabolic_stage (clinical_threshold_interaction): Create a compact cardiometabolic staging signal that distinguishes benign athletic-like pulse patterns from potentially adverse combinations of body-mass status and elevated resting pulse.
- contextual_heart_rate_strain (contextual_physiology_residual): Represent whether a student's pulse is unusually elevated or suppressed relative to peers with similar observable recovery, activity, and body-context patterns, isolating physiological strain that raw heart rate alone may obscure.
- report_granularity_clipping_signature (measurement_quality): Represent whether each student's recorded health measurements look self-reported, sensor-like, or clipped at source limits, treating measurement granularity and conformance as risk-adjacent context rather than preprocessing noise.
- weight_status_activity_buffer (body_context_buffer): Represent whether non-ideal body-mass status is behaviorally buffered or amplified by observed movement, separating weight-related risk that occurs alongside active habits from weight-related risk paired with sedentary behavior.
- self_report_health_polarity (ordinal_categorical_composite): Represent each student's subjective lifestyle reports as a compact polarity profile that captures whether their stated habits consistently lean protective, adverse, neutral, or internally mixed.
- lifestyle_pillar_bottleneck (domain_balance_score): Represent each student by the weakest and most uneven lifestyle-health pillar rather than by overall wellness alone, capturing whether risk is driven by a single severe bottleneck or broad multi-domain deterioration.
- sleep_strain_echo (recovery_physiology_interaction): Represent whether a student's sleep quantity and perceived sleep quality look physiologically restorative or instead coincide with body-mass, pulse, and exertion strain.
- external_source_support_signature (external_unsupervised_domain_support): Represent each row by how well its observed lifestyle and physiology profile is supported by the original auxiliary source distribution, treating source-domain coverage and density as risk-adjacent context without using auxiliary labels.
- diet_metabolic_alignment (contextual_nutrition_residual): Represent how consistent a student's measured metabolic and self-care profile is with the diet style they report, treating nutrition category as context rather than as an isolated label.
- external_temporal_stability_affinity (external_temporal_behavior): Represent each static train/test row by how closely its observable profile resembles auxiliary-source students with stable versus erratic longitudinal health and lifestyle routines, using the external timestamped repeated observations as a source-on…
- campus_lifestyle_archetype_margins (behavioral_archetype_affinity): Represent each student by deterministic affinities to whole campus-lifestyle archetypes, capturing whether the overall pattern looks consistently healthy, consistently depleted, active-but-overstrained, inactive-but-otherwise-low-risk, or mixed.
- coarse_health_state_class_affinity (supervised_state_likelihood): Represent each student by the smoothed empirical health-condition tendencies of coarse lifestyle and physiology states, turning recurring multi-feature wellness patterns into compact class-affinity evidence.

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