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

Hypothesis ID: 000048

Previous versions:

- rev 1: Luptitude regime-aware flux re-parameterization
  group_name=luptitude_regime_flux_features
  family=asinh_flux_domain
  summary=Transform SDSS asinh magnitudes into a flux-domain representation that explicitly models the low-signal linear regime and AB calibration idiosyncrasies, then describe each object’s spectral shape and color behavior with regime-aware flux-based descriptors to stabilize class clues near the photometric detection floor.
  strategy=Apply the SDSS u-band AB correction u_ab = u - 0.04 first. Convert each band magnitude m_b in {u_ab,g,r,i,z} to linear flux proxy f_b with the inverse Luptitude relation f_b = 2*b_b*sinh(-0.4*ln(10)*m_b - ln(b_b)), using SDSS softening constants b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10 (f0 normalized to 1). For each band define a reliability regime flag s_b = 1(|f_b| <= 10*b_b), and a soft-clipped flux \tilde f_b = sign(f_b)*max(|f_b|, b_b). Build features from \tilde f_b: flux-log colors for adjacent bands, second-order differences across log10 wavelength order, and a least-squares slope/curvature of log(\tilde f) along {u,g,r,i,z}; compute magnitude-vs-flux color mismatch for each adjacent pair, e.g. delta_c_ug = (u_ab-g) - log(\tilde f_u/\tilde f_g), similarly for other pairs. Add shape summary features on soft-clipped normalized flux allocation (sum abs \tilde f, max share, entropy-like concentration, L2 concentration), and regime descriptors per row: soft_count = sum(s_b), soft_fraction = soft_count/5, negative_flux_count = sum(f_b<0), and a soft_regime_weight = 1 - soft_fraction. Interact slope/curvature features with (1-soft_fraction) and soft_fraction to separate high-SNR and low-SNR regime behavior. For all divisions include epsilon-safe clipping to avoid instability.
  expected_signal=By replacing noisy magnitude-domain colors with physical flux-domain structure near the asinh linearization break, the group can better preserve true SED ordering for faint objects and explicitly flag when color geometry is dominated by measurement floor effects, which should help disentangle quasars, galaxies, and stars in overlapping color-redshift regions and reduce ambiguity-driven boundary errors.
  risk=The transformation is SDSS-specific and tied to published b constants and u-band offset, so performance can degrade if the dataset’s photometry comes from different processing or zero-point conventions; clipping and ratio construction can still amplify noise for extreme values, the feature set may overlap with prior flux/locus groups, and additional nonlinear operations add some compute overhead for very large tables.

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