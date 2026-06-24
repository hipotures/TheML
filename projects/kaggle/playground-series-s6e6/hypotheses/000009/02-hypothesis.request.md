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

Hypothesis ID: 000009

Previous versions:

- rev 1: Canonical SDSS Locus Coordinates
  group_name=canonical_locus_coordinates
  family=locus_projection
  summary=Project each object into fixed SDSS-derived coordinate systems that measure its signed position and normalized departure from the canonical stellar locus and the red-galaxy locus, so the classifier sees whether the photometry looks star-core, galaxy-track, or locus-outlier like in an astrophysically meaningful frame.
  strategy=First compute the ordered colors ug = u-g, gr = g-r, ri = r-i, and iz = i-z. Then compute the four SDSS stellar-locus principal colors perpendicular to the locus: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.988, and y = -0.270*r + 0.800*i - 0.534*z + 0.054. Also compute the corresponding along-locus coordinates used to determine where each principal color is valid: p1_s = 0.910*u - 0.495*g - 0.415*r - 1.280, p1_w = 0.928*g - 0.556*r - 0.372*i - 0.425, p1_x = r-i, and p1_y = 0.895*r - 0.448*i - 0.447*z - 0.600. Standardize the perpendicular coordinates by the published intrinsic stellar-locus widths to get ns = s/0.031, nw = w/0.025, nx = x/0.042, and ny = y/0.023, and clip each standardized value to [-12, 12] to limit leverage from rare photometric extremes. Mark an axis active only when its published validity window is satisfied: for s use r <= 19.0 and -0.2 <= p1_s <= 0.8; for w use r <= 20.0 and -0.2 <= p1_w <= 0.6; for x use r <= 19.0 and 0.8 <= p1_x <= 1.6; for y use r <= 19.5 and 0.1 <= p1_y <= 1.2. From the active standardized deviations derive aggregate stellar-locus affinity columns: active_axis_count, min absolute deviation, mean absolute deviation, max absolute deviation, count within 1-sigma, count within 2-sigma, and count beyond 4-sigma; if no axis is active, compute those aggregates over all four standardized deviations and set active_axis_count to 0. In the same group, compute the SDSS red-galaxy locus coordinates c_perp = ri - gr/4 - 0.18 and c_par = 0.7*gr + 1.2*(ri - 0.18), plus a simple red-galaxy-track proximity feature abs(c_perp). Keep all raw principal coordinates, standardized deviations, aggregate stellar-locus affinity measures, and the galaxy-locus coordinates as the output of this single feature group.
  expected_signal=Stars should cluster tightly near the stellar-locus tube, quasars are often strong stellar-locus outliers in SDSS color space, and galaxies tend to separate through both weaker stellar-locus affinity and structured position along the galaxy color track, so these coordinates can improve balanced accuracy by making STAR recall and QSO recall more explicit while still giving GALAXY a dedicated manifold signal.
  risk=The coefficients and widths were derived for SDSS-style dereddened photometry and mainly for normal stars plus red-galaxy tracks, so any calibration mismatch, extinction residue, or magnitude-definition mismatch can shift the locus coordinates; the group is also partly redundant with generic color features and may underrepresent blue or unusual galaxies that do not lie near the red-galaxy track.

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