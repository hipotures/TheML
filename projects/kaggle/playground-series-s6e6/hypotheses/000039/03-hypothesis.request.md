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

Hypothesis ID: 000039

Previous versions:

- rev 1: Local principal-color manifold residuals
  group_name=local_principal_color_residuals
  family=locus_geometry
  summary=Measure each object's signed and standardized deviation from local SDSS principal-color planes, conditioned on redshift and coarse sky/catalog context, so class distinctions are expressed as controlled departures from stellar or contamination-leaning manifolds rather than raw color magnitudes.
  strategy=Compute orthogonal stellar-manifold coordinates in each source from fixed SDSS/SEGUE linear combinations: s = -0.249*u + 0.794*g - 0.555*r + 0.234; w = -0.227*g + 0.792*r - 0.567*i + 0.050; x = 0.707*g - 0.707*r - 0.988; y = -0.270*r + 0.800*i - 0.534*z + 0.059; l = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984. Build context bins from: (1) redshift bin with fixed edges [-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0], (2) sky cell from alpha/delta using coarse grids (e.g., 12 bins in alpha and 6 bins in delta), and (3) the pair (spectral_type, galaxy_population). For each context, compute median and MAD of each principal-color value using training data only; derive signed residuals c_res = c - median_c and robust scores z_c = (c - median_c)/(1e-6 + 1.4826*MAD_c). Emit per-object features: z_s, z_w, z_x, z_y, z_l, abs(z_*) values, max_abs_z = max(|z_s|,|z_w|,|z_x|,|z_y|,|z_l|), l2_z = sqrt(z_s^2 + z_w^2 + z_x^2 + z_y^2 + z_l^2), an all_close flag where all |z| < 0.8, and sign-code bits for the sign pattern of (z_s, z_w, z_x, z_y). If a context has <300 rows, back off to redshift+tag only, then tag only, then global medians to avoid noisy baselines; clip all z_c to [-10,10]. References informing coefficients: SDSS QA principal colors and SEGUE l-color definitions at https://www.astro.princeton.edu/~strauss/DR3/QA.html and https://www.sdss3.org/dr8/algorithms/segueii/segue_target_selection.php.
  expected_signal=Stars remain close to the principal-color loci while quasars and galaxies more frequently produce structured perpendicular excursions that vary with redshift and tag context, so standardized residual geometry provides a compact, class-aware morphology signal that is less sensitive to absolute photometric level shifts than raw magnitudes alone.
  risk=Feature redundancy is likely with other manifold/locus groups and may offer diminishing returns; sky-cell medians can become unstable in sparse angular/redshift regions, causing noisy residuals unless fallback is robust, and fixed coefficients are tied to SDSS calibration conventions so any photometric-system mismatch could attenuate separability.

- rev 2: Context-stabilized principal-color residual manifold
  group_name=local_principal_color_residuals
  family=locus_geometry
  summary=Model each object as an orthogonal displacement from a local stellar-locus manifold using robust, context-stratified normalization so class-specific deviations and contamination patterns are expressed as standardized geometric outliers rather than raw magnitude differences.
  strategy=Compute the five principal-color coordinates from observed ugriz in every row, using the canonical SDSS/SEGUE definitions: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.983, y = -0.270*r + 0.800*i - 0.534*z + 0.059, and l = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984. Use l only when 0.5 <= g-r <= 0.8 (SEGUE-valid support); otherwise flag l_outside_domain=1 and treat residual as missing for l features. Build context bins from training data only: redshift bins with edges [-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0], sky cells from a 12-bin alpha × 6-bin delta grid, and the categorical pair (spectral_type, galaxy_population). For each train-context-cell compute median m_c and MAD d_c per color, then robust scale s_c = max(1.4826 * MAD_c, eps_c) with eps_c set to 0.02 for s,w,x,y and 0.03 for l. For each object compute signed residual r_c = c - m_c and robust score z_c = (r_c / s_c) for each valid color, then clip z_c to [-10, 10]. To avoid sparse-cell noise, apply hierarchical backing-off: (1) full context, (2) redshift+sky only, (3) redshift+tags only, (4) redshift only, (5) global train medians. Use the first level with at least 300 rows for the required color and 500 for all colors; if none satisfy, use global. Emit signed z_c, absolute z_c, max_abs_z = max(|z|), l2_z = sqrt(sum z^2), a tail_count of coordinates with |z|>2, a tail_ratio = tail_count/available_colors, and in_locus = 1{|all |z|<0.75|}, plus a 4-bit sign code for (s,w,x,y) and binary indicators for each coordinate missing/not-reliable. All context statistics are frozen from training and applied deterministically to train/validation/test with no label usage.
  expected_signal=Class separation is likely strongest in orthogonal manifold offsets: stars stay near locally centered locus planes whereas galaxies and especially quasars produce structured excursions whose direction and magnitude vary with redshift and observing context, and the robust local baseline plus scale normalization suppresses absolute-photometric and field-level offsets that would otherwise mask this signal.
  risk=Residual features can overlap with other locus-based groups and may be redundant, sparse sky/redshift cells can produce unstable baselines if fallback is insufficiently aggressive, and conditioning on l with a narrow validity domain may reduce coverage unless missingness is modeled explicitly.

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