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

Hypothesis ID: 000025

Previous versions:

- rev 1: AIDE auxiliary reference distance
  group_name=aide_aux_reference_distribution_distance
  family=aide_auxiliary_reference
  summary=Compare each object to the auxiliary star_classification reference distribution using robust per-column z-scores, empirical CDF positions, absolute median deltas, joint L2 distance, and robust Mahalanobis distance.

  strategy=Load the provided auxiliary star_classification table when available and use only predictor columns overlapping alpha, delta, u, g, r, i, z, and redshift. Clean sentinel photometry values at or below -9000 and clip redshift to nonnegative values. For each overlapping column, compute robust reference median, MAD-based scale, empirical CDF rank, robust z-score, and absolute distance from the reference median. Aggregate per-column robust z-scores into a joint L2 distance. On complete auxiliary rows, estimate a robust centered/scaled covariance and use its pseudoinverse to emit Mahalanobis squared distance, Mahalanobis distance, and an inlier-style indicator. If auxiliary data is unavailable, emit neutral deterministic defaults. Do not use target labels or fit a classifier.
  expected_signal=The AIDE top solutions used the auxiliary reference distribution to identify objects that are close to or far from the external survey-like population; this can help separate stars from galaxy/quasar-like outliers in joint photometric and sky-coordinate space.
  risk=Auxiliary data may not match the competition distribution perfectly, and robust covariance features can become unstable if too few complete auxiliary rows are available or if sentinel values are not cleaned.

- rev 2: AIDE auxiliary distribution distance with robust circular sky geometry
  group_name=aide_aux_reference_distribution_distance
  family=aide_auxiliary_reference
  summary=Construct robust outlier-style signals by comparing each test object against an external reference population in overlapping sky, photometric, and redshift predictor space, using stable marginal and joint distance statistics to capture how typical or atypical its profile is.
  strategy=Load the auxiliary reference table and restrict to predictor columns in the overlap set {alpha, delta, u, g, r, i, z, redshift}; reject any other fields from distance computation. Clean numeric predictors deterministically: values <= -9000 in u, g, r, i, z are set to missing, and redshift < 0 is clipped to 0.0001. Convert alpha to radians and create two deterministic positional components sin_alpha = sin(2*pi*alpha/360) and cos_alpha = cos(2*pi*alpha/360); keep delta as a linear feature. For each used column j, compute reference median m_j and scale s_j from MAD; if MAD is zero, fall back to IQR/1.348, and if still zero, replace with the median positive fallback scale across all columns. For each row x, compute per-column robust z value z_j=(x_j-m_j)/s_j with clipping to [-8,8] when finite, empirical CDF rank p_j using 1-indexed rank ordering on complete reference values (p_j=(rank+1)/(n_ref+1) and clipped to [1/(n_ref+2), 1-1/(n_ref+2)]), and median distance a_j=|x_j-m_j|. Emit a per-feature missing flag and maintain missing_count and available_ratio over overlap columns. Aggregate marginal features as mean_abs_z, max_abs_z, mean_rank_shift = mean(|p_j-0.5|), and mean_abs_median_delta = mean(a_j), each computed over available columns only. Build a joint standardized vector w from [sin_alpha, cos_alpha, delta, u, g, r, i, z, redshift] after excluding unavailable coordinates per row; estimate reference covariance on complete rows after centering and clipping z-like standardized values to [-6,6], then regularize with ridge τ = 1e-6 * trace(C)/d if d>=2 or if covariance rank is deficient. Compute Mahalanobis squared distance d2 = w^T * inv(C_reg) * w, Mahalanobis distance d = sqrt(max(d2,0)), and inlier score s = log1p(d2); if row-wise completeness is insufficient for joint computation, use deterministic row-level fallback by shrinking to available components and a missingness-aware mean distance, but keep availability flags. If auxiliary data is missing or effective overlap is below three usable numeric columns, return explicit neutral constants for all distance features plus full missing/unavailable indicators; never use target labels and never train a downstream classifier in this hypothesis.
  expected_signal=This feature group provides a calibrated notion of how likely each object is under a broad external population; class boundaries are often separable by how far objects move from the joint photometric/redshift manifold (e.g., unusual color-redshift combinations or sky-position concentration), so these robustness-stabilized marginal and Mahalanobis-derived distances add strong discriminative structure for balanced multiclass scoring.
  risk=Potential distributional shift between auxiliary and competition data (calibration, selection, or footprint bias) can introduce systematic ordering errors, and aggressive clipping/regularization may obscure subtle class structure; joint distance estimates are sensitive to low-variance features and incomplete rows, so missingness and fallback paths are essential but may add redundancy or noisy proxies.

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