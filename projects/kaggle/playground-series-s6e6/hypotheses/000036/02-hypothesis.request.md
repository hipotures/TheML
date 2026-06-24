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

Hypothesis ID: 000036

Previous versions:

- rev 1: SEGUE stellar atmosphere color diagnostics
  group_name=segue_stellar_atmospheric_indices
  family=atmospheric_color_diagnostics
  summary=Build SDSS color-coordinate diagnostics that encode stellar temperature, metallicity, and line-blanketing structure from u,g,r,i,z so that true stellar continua and non-stellar SEDs are separated by how tightly they conform to known stellar-atmosphere manifolds.
  strategy=Compute ug=u-g, gr=g-r, ri=r-i, iz=i-z from magnitudes. Form three SDSS-defined indices: p1=0.91*ug+0.415*gr-1.280, v=0.283*ug-0.354*gr+0.455*ri+0.766*iz, and l=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984 (all numerically in magnitudes). Bin redshift into quantile bins (e.g., deciles), then for each of p1, v, l compute robust z-scores within each (redshift_bin, spectral_type) cell using median and 1.4826*MAD, falling back to (redshift_bin) and then global statistics when cell counts are too small; clamp MAD to a small epsilon to avoid division by zero. Add geometry margins: p1_interval_violation = max(0, min(p1+0.25, -0.7-p1)), l_low_tail = max(0, 0.07-l), l_mid_tail = max(0, l-0.135), and v_abs = |v|. Aggregate scores such as p1_v_l_outlier = mean(|z_p1|, |z_v|, |z_l|) and optionally interaction terms v_abs*z_p1 and v_abs*z_l. Edge handling: clamp all standardized/interaction features to [-8,8] and keep all outputs finite by replacing null bins with global medians and zero-variance fallback to zero.
  expected_signal=These indices are designed to track stellar metallicity/atmospheric structure and therefore provide a compact, physically anchored separation signal for STAR objects, while galaxies and quasars are expected to show larger departures from this manifold; that helps reduce confusion in overlapping blue-color regions and improves class discrimination where raw color bands are insufficient.
  risk=The projections are stellar-focused by design and can be noisy for non-stellar objects, so signal may be weaker for galaxy/quasar subclasses at faint magnitudes; some features may be redundant with existing locus/continuum groups, and hard-coded threshold constants can induce brittleness if the target sample distribution shifts.

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