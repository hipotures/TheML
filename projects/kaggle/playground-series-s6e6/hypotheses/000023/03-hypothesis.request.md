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

Hypothesis ID: 000023

Previous versions:

- rev 1: AIDE redshift-bin color residuals
  group_name=aide_redshift_bin_color_residuals
  family=aide_photometric_redshift
  summary=Recreate the AIDE empirical redshift-bin residual features by comparing each object's colors and color curvatures against objects in the same quantile redshift regime.

  strategy=Using clipped nonnegative redshift, assign rows to approximately 24 quantile bins, with duplicate edges handled safely. For each bin, compute the mean and standard deviation of AIDE color-shape columns u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, u-2*r+i, and g-2*i+z. For each row, emit signed deltas from the bin mean and standardized z-scores for each color feature, then aggregate those residuals into L2 norm, absolute L1 norm, mean signed delta, delta standard deviation, and maximum absolute z-score. The transform must be deterministic and must not use the target.
  expected_signal=Stars, galaxies, and quasars occupy different color manifolds at similar observed redshift values; residuals within redshift regimes can expose objects whose colors are inconsistent with their local redshift context.
  risk=This overlaps conceptually with photoz_color_consistency, and quantile bins can be sparse at distribution edges, so the added signal may be redundant or noisy.

- rev 2: Redshift-stratified residualization of AIDE color-shape channels
  group_name=aide_redshift_bin_color_residuals
  family=aide_photometric_redshift
  summary=The feature group characterizes how far each object’s photometric color and color-curvature profile departs from the expected profile at similar redshift, so model decisions can use class-specific deviations from the local manifold rather than raw color trends dominated by redshift effects.
  strategy=Use redshift clipped to the physically valid support z0 = min(max(redshift, 0.0), 7.01). On training rows only, build 24 quantile breakpoints for binning: e_k = quantile(z0_train, k/24), k=0..24. Remove duplicate breakpoints; if the resulting number of edges is too small (for example less than 13), fall back to 24 equal-width edges over [min(z0_train), max(z0_train)] to keep deterministic bin counts. Let edges be E[0..m], with m-1 bins. Assign each row to bin b = clip(searchsorted(E, z0, side='right') - 1, 0, m-2). For each of the fixed nine AIDE color-shape channels (u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, u-2*r+i, g-2*i+z), compute per-bin count n_b, mean mu_b_c, and standard deviation sigma_b_c (ddof=1), plus global mu_all_c and sigma_all_c. For sparse bins, apply shrinkage: w_b = min(1, n_b/(2*200)); mu_tilde_b_c = w_b*mu_b_c + (1-w_b)*mu_all_c; sigma_tilde_b_c = w_b*sigma_b_c + (1-w_b)*sigma_all_c, with sigma_tilde floor at 1e-6. For each row and channel, compute residual d_i_c = x_i_c - mu_tilde_{b_i,c} and standardized residual s_i_c = d_i_c / sigma_tilde_{b_i,c}. Emit aggregated row-level features: root sum squares of d_i_c, mean absolute d_i_c, mean d_i_c, standard deviation of d_i_c, mean s_i_c, and max absolute s_i_c.
  expected_signal=At fixed redshift, stars, galaxies, and quasars occupy different trajectories in color space, so subtracting the local redshift-conditioned center exposes class- and subtype-specific offsets, while normalized residuals reduce dominance of broad redshift drift and amplify physically relevant curvature differences that are useful for balanced multiclass discrimination.
  risk=Using noisy redshift as the conditioning variable can cause boundary effects and unstable bin statistics, especially where bins are sparse; shrinkage lowers variance but may still under- or over-smooth class signal, and the resulting features are likely to overlap substantially with other color-based families.

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