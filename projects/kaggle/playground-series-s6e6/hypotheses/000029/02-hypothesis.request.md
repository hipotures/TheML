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

Hypothesis ID: 000029

Previous versions:

- rev 1: Lyman-break dropout geometry across redshifted bands
  group_name=redshifted_lyman_discontinuity
  family=restframe_discontinuity
  summary=Create features that quantify where and how strongly the broadband SED drops across the rest-frame Lyman-limit and Lyman-alpha edges as they sweep through ugriz bands, giving direct physically interpretable dropout geometry for class separation.
  strategy=Use the fixed SDSS effective wavelengths [u,g,r,i,z] = [3551, 4686, 6165, 7481, 8931] Å and each object's redshift z to evaluate two breaks B in {912 Å, 1216 Å}. Convert ugriz magnitudes to linear flux proxies f_b = 10^{-0.4 m_b}. For each break B, compute λ_rest = λ/(1+z) for all five bands and let idx = first index where λ_rest[idx] >= B. If idx == 0 or idx == 5, mark edge_available_B = 0 and emit neutral defaults (jump_B = 0, local_B = 0, phase_B = 0); else set edge_available_B = 1, define phase_B = clip((B - λ_rest[idx-1]) / (λ_rest[idx] - λ_rest[idx-1]), 0, 1), jump_B = log10((mean(f[idx:]) + eps)/(mean(f[:idx]) + eps)), and local_B = m[idx-1] - m[idx], with eps = 1e-12. Add a compact positional encoding break_band_B = idx (0..5 or one-hot bins: below u, u-g, g-r, r-i, i-z, above z) and optionally interaction terms jump_B * break_band_B and local_B * edge_available_B to stabilize extreme edges.
  expected_signal=These features should capture physically motivated absorption/continuum-break behavior that is not fully represented by global color slopes, especially for high-redshift quasars and galaxies where the blue-to-red flux jump around Lyα/Lyman-limit is a strong discriminator, while neutralizing to near-zero for low-redshift objects where breaks are outside ugriz coverage.
  risk=Dependence on provided redshift introduces sensitivity to redshift errors and can create unstable boundary assignments when z is noisy; u-band sensitivity and calibration variation can also generate spurious jump magnitudes, and some information may overlap with other break/line-geometry families so gains may be incremental if those are already strong.

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