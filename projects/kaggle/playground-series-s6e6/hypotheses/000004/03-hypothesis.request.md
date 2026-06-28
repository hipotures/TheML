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

Hypothesis ID: 000004

Previous versions:

- rev 1: Distance-Corrected Luminosity Signature
  group_name=distance_corrected_luminosity_signature
  family=luminosity_distance
  summary=Convert apparent broadband brightness into a redshift-adjusted intrinsic-luminosity signature so the model can compare whether an object is physically star-like, galaxy-like, or quasar-like rather than only how bright it looks from Earth.
  strategy=Define z_pos = max(redshift, 0). Compute a deterministic luminosity-distance surrogate in Mpc with H0 = 70.0, c = 299792.458, q0 = -0.55: d_mpc = max(1e-4, (c / H0) * z_pos * (1 + 0.5 * (1 - q0) * z_pos)). Convert each band m in {u,g,r,i,z} into a pseudo-absolute magnitude M = m - 5 * log10(d_mpc) - 25. From these distance-corrected magnitudes create only same-idea summaries: the median pseudo-absolute magnitude across the five bands, the minimum pseudo-absolute magnitude (brightest inferred intrinsic band), the maximum pseudo-absolute magnitude (faintest inferred intrinsic band), and the band-to-band pseudo-absolute range max(M) - min(M). Also add coarse intrinsic-brightness regime flags by binning the median pseudo-absolute magnitude into <= -24, (-24,-20], (-20,-16], (-16,-8], and > -8. For redshift <= 0, the floor d_mpc = 1e-4 keeps the transformation finite and intentionally maps such objects into a nearby-object regime instead of exploding the magnitude scale.
  expected_signal=Balanced accuracy may improve because stars, galaxies, and QSOs can share similar observed colors while differing sharply in implied intrinsic luminosity once apparent brightness is interpreted through redshift, giving the classifier a class-separating signal that is complementary to raw magnitudes and color-only descriptors.
  risk=The distance surrogate is only approximate, especially at high redshift and for noisy near-zero stellar redshifts, so the group can inject modeling error, duplicate some information already present in magnitudes plus redshift, and overfit if the train set contains class-specific redshift calibration artifacts.

- rev 2: Redshift-Calibrated Intrinsic Luminosity Profile
  group_name=distance_corrected_luminosity_signature
  family=luminosity_distance
  summary=Transform the five-band photometry into an approximate intrinsic brightness representation using redshift, then summarize its within-row level and spread so objects are compared by physical luminosity regime and dispersion rather than only apparent magnitudes and positional features.
  strategy=Compute a deterministic luminosity-distance surrogate with constants H0=70.0, c=299792.458, q0=-0.55. Let z_safe = max(redshift, 1e-4) and d_L(Mpc) = (c / H0) * (1 + z_safe) * (z_safe + 0.5 * (1 - q0) * z_safe^2); clip d_L to [1e-4, 1e8] before applying logs so z<=0 and extreme values cannot destabilize transforms. For each band b in {u, g, r, i, z}, compute pseudo-absolute magnitude M_b = mag_b - 5 * log10(d_L) - 25. From the 5-value vector {M_u, M_g, M_r, M_i, M_z}, derive robust summaries and range descriptors: median, min, max, 25th percentile, 75th percentile, IQR = p75 - p25, and full span = max - min. Bin the median pseudo-absolute magnitude into five regimes: <= -27, (-27, -23], (-23, -19], (-19, -14], and > -14. Keep all edge handling deterministic: if redshift is missing (unexpected), treat as 0 before clipping, and if any magnitude column is missing at row-level, that row’s derived feature is dropped/recomputed only from available bands with explicit fixed fallback counts.
  expected_signal=This feature family adds a physically interpretable luminosity axis that can separate classes whose observed colors overlap but whose implied absolute-scale power differs, and the added spread statistics stabilize against single-band noise or outliers while adding nonlinear structure likely helpful for balanced multiclass discrimination.
  risk=The proxy for d_L is a low-order approximation and can mis-estimate distance at high redshift, so the derived magnitudes may add model bias or encode survey artifacts tied to redshift calibration; redundant information with raw magnitudes and redshift may also increase collinearity and overfitting if regularization is weak.

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