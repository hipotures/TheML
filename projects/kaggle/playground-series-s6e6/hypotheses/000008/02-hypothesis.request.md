# Project Context

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Data Overview

-> sample_submission.csv has 247435 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 577347.00 - 824781.00, 0 nan values
class (object) has 1 unique values: ['GALAXY'], 0 nan values

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
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR'], 0 nan values

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy
- Submission kind: labels

# Instruction

Analyze the selected ROOT hypothesis, the project context, and the previous
revisions.

Create the next revision by expanding or improving the hypothesis while
preserving its substantive feature-group idea. You are not writing code. You are
not creating a new feature group.

The revision should make the hypothesis more precise, internally consistent,
implementable, or better justified. Improve the hypothesis text, especially
`strategy`, when there is a concrete useful improvement. Do not force arbitrary
changes.

Use previous revisions only as context. If they contain useful details, use that
context, but do not mechanically merge them. The output must be one coherent
revised hypothesis using exactly the same JSON schema as a newly generated
hypothesis.

# Web Search

Live web search is enabled. If internet verification is useful, use the
`web_search` tool before writing the revised hypothesis. Check at most 3 sources.
Use web search only to verify domain facts, feature-engineering background,
validation risks, or implementation constraints relevant to the selected
hypothesis. Do not use web search to invent a different feature-group idea, copy
a public solution, or broaden the scope. If web search is not useful, continue
without inventing sources.

# Selected Hypothesis

Hypothesis ID: 000008

Previous versions:

- rev 1: Survey Manifold Rarity
  group_name=survey_manifold_rarity
  family=density_rarity
  summary=Estimate how typical or unusual each object is inside the unlabeled survey manifold formed by coarse redshift regime, catalog tags, and broad photometric energy allocation, so the model can use rarity itself as a class cue.
  strategy=For each object, convert ugriz magnitudes to pseudo-fluxes f_b = 10^(-0.4*m_b), compute normalized band shares p_b = f_b / sum(f), then derive three survey-manifold coordinates: blue_balance = (p_u + p_g) - (p_i + p_z), flux_concentration = max(p_u,p_g,p_r,p_i,p_z), and total_brightness = -2.5*log10(sum(f)). Build bins from train.csv only: redshift bins with fixed edges [-inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, inf], plus decile bins for blue_balance, flux_concentration, and total_brightness; if quantile edges collapse because of ties, merge duplicate edges and use the resulting ordered bins. Form a joint cell from (redshift_bin, spectral_type, galaxy_population, blue_balance_bin, flux_concentration_bin, total_brightness_bin). From train.csv only, compute Laplace-smoothed counts with alpha = 1 for the full joint cell, for the tag-regime block (redshift_bin, spectral_type, galaxy_population), and for the photometric block (blue_balance_bin, flux_concentration_bin, total_brightness_bin). Output one coherent rarity group consisting of: joint_log_density = log((count_joint + 1) / (N + K_joint)), rarity_score = -joint_log_density, and interaction_surprise = log((count_tag_regime + 1) * (count_photo_block + 1)) - log((count_joint + 1) * N), where N is the train row count and K_joint is the number of observed-or-possible joint cells under the constructed bins. Apply the same train-derived bin edges and smoothed lookup tables to test rows; values outside the train range fall into the nearest end bin, and any unseen categorical level would map to an UNK bucket before lookup.
  expected_signal=QSO examples often occupy sparse or cross-inconsistent regions of the survey manifold, while many STAR and GALAXY objects cluster in denser loci, so explicit rarity and interaction surprise can improve per-class separation and help balanced accuracy rather than only overall majority fit.
  risk=If the joint grid is too sparse, these features can become noisy and effectively memorize idiosyncratic train cells; they may also be partially redundant with existing color- and redshift-based groups and can drift if the test distribution populates different density regions.

Return valid JSON only. Do not use markdown fences.

Return exactly this JSON object:

{
  "title": "short descriptive title",
  "group_name": "same concise_snake_case_group_name as the previous version",
  "family": "compact feature-family label",
  "summary": "one comprehensive sentence describing the semantic feature-set idea; do not list formulas, exact feature names, or implementation details here",
  "depends_on": [],
  "strategy": "concrete deterministic feature logic, formulas, bins, statistics, and edge handling",
  "expected_signal": "why this group may improve or clarify the metric",
  "risk": "specific overfitting, leakage, cost, redundancy, or instability risk"
}