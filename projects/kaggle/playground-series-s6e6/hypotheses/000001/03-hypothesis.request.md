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

Hypothesis ID: 000001

Previous versions:

- rev 1: Broadband color-shape features
  group_name=broadband_color_shape
  family=photometric_sed
  summary=Transform the five ugriz magnitudes into color slopes and curvature features that describe each object's broadband spectral energy distribution shape.
  strategy=Create deterministic photometric-shape columns from magnitude differences: color_u_g = u - g, color_g_r = g - r, color_r_i = r - i, color_i_z = i - z, color_u_r = u - r, color_g_i = g - i, color_r_z = r - z, and color_u_z = u - z. Add second-difference curvature terms to capture bends in the spectrum: curve_ugr = (u - g) - (g - r) = u - 2*g + r, curve_gri = (g - r) - (r - i) = g - 2*r + i, and curve_riz = (r - i) - (i - z) = r - 2*i + z. Do not bin, rank, or target-encode anything. Leave raw subtraction values unchanged, including negatives and extreme values; if any source value is non-finite in future data, return null for the affected derived column only.
  expected_signal=GALAXY, STAR, and QSO classes typically separate more clearly in color-color space than in raw magnitudes because these differences remove overall brightness and isolate continuum slope, spectral breaks, and UV excess patterns that align with stellar temperature sequences, galaxy populations, and quasar spectra.
  risk=This group is partly redundant with the existing raw magnitudes and may overlap conceptually with spectral_type and galaxy_population; very noisy magnitudes or redshift-driven bandpass shifts can also make some color differences unstable at class boundaries.

- rev 2: Wavelength-aware ugriz shape diagnostics
  group_name=broadband_color_shape
  family=photometric_sed
  summary=Encode each object’s optical spectral shape with deterministic slope- and curvature-derived descriptors from the ugriz bands so class separation depends on continuum geometry rather than absolute brightness.
  strategy=Use only raw u, g, r, i, z magnitudes and treat the SDSS-like band pivot wavelengths as fixed constants in Å: λu=3562, λg=4686, λr=6165, λi=7481, λz=8931. For each row compute first-order color descriptors on the fixed ordering and selected longer baselines: u−g, g−r, r−i, i−z, u−r, g−i, g−z, and u−z. Convert adjacent pair colors to wavelength-normalized slopes to remove non-uniform band spacing effects: s_ug=(u−g)/(ln λg−ln λu), s_gr=(g−r)/(ln λr−ln λg), s_ri=(r−i)/(ln λi−ln λr), s_iz=(i−z)/(ln λz−ln λi). Compute curvature on magnitudes to capture bends in the SED over successive triples: c_ugr=u−2g+r, c_gri=g−2r+i, c_riz=r−2i+z, and a broader c_uz=u−2r+z. Add corresponding slope-scale curvature terms to avoid aliasing from uneven wavelength spacing: k_ugr=(s_gr−s_ug)/(0.5*(ln λr−ln λu)), k_gri=(s_ri−s_gr)/(0.5*(ln λi−ln λg)), k_riz=(s_iz−s_ri)/(0.5*(ln λz−ln λr)). Keep values deterministic, per-row, and untransformed (including negative and large values); do not bin, rank, target-encode, or combine with labels. If any operand used in a specific derived column is non-finite, emit null for that column only and keep all other derived columns available.
  expected_signal=Color differences remove most overall flux-level effects and are known to separate stellar, galactic, and quasar populations better than raw magnitudes, while explicit slope and curvature terms summarize continuum slope changes and breaks that map to stellar temperature sequences, galaxy populations, and quasar outlier structure, which should help improve balanced multiclass discrimination.
  risk=The engineered descriptors are intentionally redundant with raw bands and with existing categorical context, so marginal gain may be limited; some quasar redshift regions have near-overlap in color space with stars; and formulas relying on u-band and endpoint combinations can amplify photometric noise for faint/high-error objects.

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