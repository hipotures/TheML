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

Hypothesis ID: 000054

Previous versions:

- rev 1: Restframe anchored SED slope and curvature profiles
  group_name=restframe_anchor_sed_shape
  family=restframe_shape
  summary=Transform each object into a redshift-normalized five-point SED traced at fixed rest-frame wavelength anchors and expose explicit UV-to-optical slope and curvature signals so class-specific continuum behavior is compared on a shared physical wavelength basis rather than fixed observed filters.
  strategy=For each row, convert ugriz magnitudes to linear fluxes F_b = 10^{-0.4 m_b}. Use rest-frame effective wavelengths λ'_b = λ_b / (1 + redshift) with λ_u=3551, λ_g=4686, λ_r=6165, λ_i=7481, λ_z=8931 Å (use 1+redshift clipped at a small positive floor, e.g. 0.1, only if redshift < -0.9). Sort the five (log10 λ', log10 F) pairs by wavelength and build a monotonic piecewise-linear continuum in log-space. Evaluate interpolated rest-frame flux values at fixed anchors A=[1500,2200,2900,3650,4500,6200,7600] Å; if an anchor is outside the interpolated range, set the anchor value to missing and emit a valid-mask feature for each anchor plus a count of usable anchors. From available anchors, compute adjacent segment slopes s1=ΔlogF/Δlogλ in 1500–2200, 2200–2900, 2900–3650, 3650–4500, 4500–6200, 6200–7600 Å, and derive curvature deltas d1=s2-s3 (UV bend around metal-line regime), d2=s4-s5 (optical curvature), and a 3650-Å break residual b4000 = logF(3650) - linear prediction from anchors at 2900 and 4500. Add mismatch terms as RMS residuals of (a) global 1-slope power-law fit over all available anchors and (b) a constrained 2-slope piecewise fit with an enforced break at 3650 Å to quantify break sharpness; replace undefined fits with NaN and carry coverage-count masks.
  expected_signal=Stars are expected to show smooth blackbody-like rest-frame slopes with modest curvature, galaxies should exhibit stronger positive/negative curvature near the 3650–4500 Å region from stellar population breaks, and quasars should remain closer to a broken/power-law continuum with weaker broad-band stellar-break structure; anchoring to physical wavelengths removes redshift-induced feature drift and makes these class differences directly comparable across the full z range.
  risk=Interpolation is unstable when a redshift places most anchors outside observed bands, causing many missing values at extremes, and the method inherits redshift noise; if redshift errors or selection artifacts correlate with class, piecewise fit scores could overfit those artifacts, and the extra non-linear features can become correlated with other redshift-based geometry groups.

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