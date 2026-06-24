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

Hypothesis ID: 000057

Previous versions:

- rev 1: Quasar-Star Locus Outlier and Degeneracy Geometry
  group_name=quasar_stellar_locus_conflict
  family=locus_conflict_geometry
  summary=Model each object’s color-space geometry against the SDSS stellar manifold using two three-band color cubes while adding redshift-linked quasar-inclusion corridors and ambiguity-buffer penalties, so the classifier sees both strong quasar outlier evidence and the known redshift regions where quasar and stellar populations become confounded.
  strategy=Compute c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z for every row. Form two 3D color vectors v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). Fit a robust mean and covariance for each vector set on the full available set (train+test), then extract the first principal axis for each cube; define orthogonal residual vectors by subtracting the axis component for each row and compute Mahalanobis orthogonal distances d1,d2 using residual covariance with a color floor added per component (equivalent to adding 0.03^2 in each color term). Create soft outlier scores s1=max(d1-4,0) and s2=max(d2-4,0), and also keep standardized versions d1/z_scale and d2/z_scale. Add three regime features: (a) UV-like regime score favoring very blue c_ug in a bright point-source band, (b) a mid-z bridge score that peaks inside 0.65<c_ug<1.5 and 0<c_gr<0.2 and decays smoothly outside, and (c) a high-z regime score keyed by strong c_ug redness, outlierness in v2, and i-magnitude gating near i<20.2. Add suppressor penalties for known quasar-mimicking contaminant strips defined by white-dwarf/A-star-like color windows from the SDSS target-selection tables, then combine by subtraction so these regions reduce effective quasar support. All scores are clipped to [0,1], all denominators are clamped to avoid divide-by-zero, and invalid rows are deterministically filled with 0.0.
  expected_signal=Separates many quasars that are true color outliers from the thin stellar manifold while explicitly preserving and down-weighting overlap zones where quasars near z≈2.7–2.8 and known contamination strips resemble stars, which should improve minority-class recall without collapsing balanced accuracy into the dominant class boundary.
  risk=Relies on SDSS-era, color-threshold logic that may be mismatched to this dataset’s calibration and label definitions; fixed inclusion/exclusion heuristics can over-penalize legitimate rare quasars near boundaries and may be redundant with existing manifold/locus groups, so incremental gain can be modest if those groups already capture the same geometry.

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