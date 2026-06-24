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

Hypothesis ID: 000049

Previous versions:

- rev 1: Asinh-band censoring and dropout regime geometry
  group_name=asinh_censoring_regime_geometry
  family=detection_regime_geometry
  summary=Represent each object by whether each ugriz measurement is in a reliable detection regime or in the SDSS low-signal asinh regime, and encode the resulting band-visibility and dropout geometry so class-separating patterns tied to faint-band suppression are made explicit.
  strategy=For each band b in {u,g,r,i,z}, invert the SDSS asinh magnitude m_b using the published relation m = -2.5/ln(10) * [asinh((f/f0)/(2 b_b)) + ln b_b], where b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10, then compute f_b/f0 = 2 b_b sinh(-(m_b * ln(10)/2.5) - ln b_b) and q_b = (f_b/f0)/(2 b_b). Create per-band regime features: detect_10b_b = 1[f_b/f0 > 10 b_b] (equiv m_b < m10b_b), neg_b = 1[f_b/f0 < 0], and delta_b = clip((m_b - m10b_b)/(m0_b - m10b_b), 0, 1), with m10b_b={22.12,22.60,22.29,21.85,20.32} and m0_b={24.63,25.11,24.80,24.36,22.83} for u,g,r,i,z respectively. Add joint descriptors: count of detected bands, count of non-detected redder-than-blue bands, longest contiguous missing-band run in wavelength order, blue-dropout pattern flags (e.g., u-drop, g-drop, u+g-drop while r/i/z detected), detected-band-only color slopes and curvatures from f_b/f0, and mass-center entropy of abs(f_b/f0) after zeroing not-detected bands; clip f_b/f0 and delta_b to training quantile tails to avoid single-object blow-up and keep edge handling stable when very bright or extreme values appear.
  expected_signal=Because asinh magnitudes compress faint flux, raw magnitude differences can hide true non-detections; regime-aware features expose low-SNR structure and dropout signatures (especially blue-band suppression at specific redshifts and low-confidence detections) that are known to be relevant for distinguishing stars from galaxies and quasars, which should improve class separation in the ambiguous boundary regions measured by balanced accuracy.
  risk=Reliance on published softening constants and threshold magnitudes creates calibration sensitivity if the competition data were reduced with different photometric conventions, and regime flags are intrinsically noisy near boundaries, so they can inject instability or duplicate existing flux-asinh or depth-based features unless regularized.

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