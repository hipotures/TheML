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

Hypothesis ID: 000044

Previous versions:

- rev 1: Redshift-tracked absorption trough residuals in ugriz
  group_name=redshifted_absorption_trough_residuals
  family=absorption_line_geometry
  summary=Model the broadband imprint of age/metallicity-sensitive stellar and galaxy absorption complexes by measuring how much flux is locally suppressed when known absorption-line rest-frame wavelengths are shifted into the observed ugriz bands, so class separation uses spectroscopic-style line-blanketing patterns in addition to global colors.
  strategy=Convert ugriz magnitudes to linear flux proxies f_u..f_z = 10^-0.4*m. Define band effective wavelengths λ = [3551, 4686, 6165, 7481, 8931] Å and their contiguous passband spans (u 3000-4100, g 4100-5500, r 5500-6900, i 6900-8200, z 8200-9200 Å approximations). For each object compute observed line centers λ_obs = λ_rest*(1+z) for a fixed set of absorption-sensitive features: Ca II K (3933.7), Ca II H (3968.5), G-band (4304), Hδ (4102), Hγ (4341), Hβ (4861), Mg b triplet proxy 5175, and Na I D 5893. If λ_obs lies inside any ugriz span, estimate smooth local continuum at λ_obs by log-flux interpolation between the two neighboring filters straddling λ_obs; if at an edge, use a 5-point robust linear fit in log f vs log λ to predict f_cont. Compute signed residual r = (f_cont - f_band)/(f_band + eps), with eps small (e.g. 1e-12), then split into deficit d = max(r,0) and excess e = max(-r,0). Build grouped columns: total_blue_abs_deficit = mean(d for blue lines), total_red_abs_deficit = mean(d for Hβ/Mg b/Na D), absorption_excess_ratio = (sum(e))/ (sum(d)+1e-6), metal_trough_balance = total_blue_abs_deficit - total_red_abs_deficit, and line_family_contrast = CaK+CaH combined deficit vs Hδ+Hγ+G-band deficit. Add visibility masks per line (1 if in-band, 0 else) and per-redshift-regime masks (z<0.4, 0.4-1.2, 1.2-2.0, 2.0-7.0) so regime-limited aggregates are zeroed when no line is visible.
  expected_signal=Quasar continua are comparatively smooth and line-deficit patterns in broad bands are weaker or irregular, while stars and galaxies more often show redshift-dependent absorption-blanketing structure; tracking these signed deviations at physically meaningful rest-frame line positions should help disentangle ambiguous QSO-vs-galaxy and STAR-vs-QSO regions that are often separable only by subtle spectral-shape perturbations.
  risk=Broadband integration strongly dilutes narrow-line effects, so signal can be weak or noisy in faint objects and at high redshift where key lines redshift out of ugriz; approximate band pass boundaries or single-value continuum proxies may inject systematic bias, and the features are partially redundant with other continuum/break geometry groups, increasing overfit risk unless regularized.

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