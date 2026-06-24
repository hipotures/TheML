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

Hypothesis ID: 000031

Previous versions:

- rev 1: Redshift-adaptive color-manifold tube residuals
  group_name=redshift_adaptive_color_tube_residuals
  family=color_manifold_geometry
  summary=Model the thin stellar color manifold separately in redshift slices and provide signed, scale-normalized manifold-distance features that quantify whether each source lies on, orthogonally off, or near the ambiguity intersection of the quasar and stellar loci.
  strategy=Compute c1=u-g, c2=g-r, c3=r-i, c4=i-z for every row and define z_plus = max(redshift, 0). Create equal-frequency redshift bins on z_plus (for example 20 bins), then merge adjacent bins until each bin has at least 5,000 rows; use the same bin edges for both train and test so transformation is deterministic. For each final bin, fit robust scaling on [c1,c2,c3] (ugri cube) and [c2,c3,c4] (griz cube) using median subtraction and MAD scaling, then fit PCA (centered on the bin medians) to obtain v1 as the primary locus axis and v2,v3 as orthogonal residual axes for each cube. For each row in a bin, compute t_ugri=v1_ugri·(x_ugri−μ_ugri), r_ugri=(I−v1_ugri v1_ugri^T)(x_ugri−μ_ugri), d_ugri=||r_ugri||2, and s_ugri=sign(v2_ugri·(x_ugri−μ_ugri)); compute the analogous t_griz, r_griz, d_griz, s_griz for the griz cube. Normalize d_ugri and d_griz by bin-specific 1.4826*MAD(d), clip normalized residuals to [0,99.5th percentile] per bin, and output both raw residuals, normalized residuals, signs, and axis-position ratios t/(MAD(t)+1e-6). Add ambiguity-aware gating using g(z)=exp(−(z_plus−2.7)^2/(2*0.35^2)), based on the known quasar–stellar color overlap near z≈2.7; emit g(z)*d_norm_ugri and g(z)*d_norm_griz as extra features to emphasize redshift regions where off-manifold distances are most diagnostic. For rows falling in undefined/empty bins (boundary artifacts), use the nearest non-empty bin by bin centroid redshift and reuse that bin’s PCA statistics and scales.
  expected_signal=This group converts implicit manifold geometry into explicit supervised-friendly signals: stars should have small orthogonal residuals in their local color tubes, while galaxies and quasars more often produce larger standardized departures, with the overlap-gated variants giving extra separation in the known problematic quasar-star regime around z 2.7 where raw colors alone are less decisive.
  risk=If redshift bins are too fine, PCA directions can be noisy or contaminated by class mixing, especially at sparse high-redshift extremes, and residual normalization can become unstable for ultra-sparse or highly noisy color regions unless robust clipping and minimum-bin constraints are enforced.

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