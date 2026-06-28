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

Hypothesis ID: 000014

Previous versions:

- rev 1: Empirical color-redshift consistency
  group_name=photoz_color_consistency
  family=photometric_redshift
  summary=Encode whether an object's observed redshift is consistent with the redshift normally implied by its broadband color location, treating disagreement and local color-redshift ambiguity as class-separating signals.
  strategy=From the original ugriz magnitudes compute adjacent colors u-g, g-r, r-i, and i-z, clipping only non-finite intermediate values to training-column finite bounds if ever encountered. Using training predictors only, build deterministic lookup tables that summarize redshift behavior in color space: a primary grid over quantile-binned g-r, r-i, and r magnitude, plus fallback grids over g-r/r-i, u-g/g-r, and r-i/i-z. Use fixed bin edges learned from the training data, require a minimum cell count such as 30 for the primary grid and 50 for pairwise fallback grids, and store each valid cell's median redshift, interquartile redshift spread, and row count. For each row, retrieve the most specific valid cell available; if no cell is valid, fall back to the global training median redshift and global interquartile spread. Derive features for signed redshift residual, absolute residual, residual divided by max(local IQR, 1e-3), local log count, local IQR, and an indicator for whether the value came from the primary grid, pairwise fallback, or global fallback.
  expected_signal=Stars should cluster near zero redshift even when their colors mimic parts of the galaxy or quasar locus, galaxies should follow smoother low-to-moderate color-redshift relations, and quasars often occupy color regions with multimodal or high-redshift behavior, so redshift residual size and local ambiguity can improve balanced separation of minority classes.
  risk=The lookup can become sparse in rare color regions and may duplicate information already captured by raw redshift-color interactions; if computed with self-inclusion on the same rows being scored it can make training diagnostics look cleaner than test behavior, so the transform should avoid row self-reference when estimating development performance.

- rev 2: Residual-to-local-color redshift consistency
  group_name=photoz_color_consistency
  family=photometric_redshift
  summary=Characterize how plausible each object's measured redshift is relative to the local redshift expectation of its color-magnitude neighborhood so class boundaries are exposed by systematic departures from the normal photometric redshift manifold.
  strategy=Construct color features from ugriz as c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z. Fit color and redshift clipping limits only on the training set (0.1 and 99.9 percentiles) and clamp colors and redshift to those limits before bin assignment. Build deterministic quantile-binned lookup tables from training rows only: primary table on (c_gr, c_ri, r) using 9, 9, and 10 equal-frequency bins respectively; accept a primary cell only if it has at least 40 rows and store z_med, q25, q75 (for IQR), MAD, and n. Build three fallback tables on (c_gr, c_ri), (c_ug, c_gr), and (c_ri, c_iz), each with 10x10 equal-frequency bins and minimum n=60, storing the same statistics. For each row, retrieve expectation statistics by first attempting its exact primary cell, then its immediate primary-neighbor cells by Chebyshev radius 1 then 2 with n>=40, then exact fallback cells in the same three orderings, then each fallback’s radius-1/2 expansion with n>=60; if none apply, use global training z_med and global IQR. Generate deterministic features from the chosen table: residual z_res=z-z_med, |z_res|, calibrated residual z_res/max(IQR,1e-3), local_log_count=log1p(n), local_IQR=IQR, local_MAD, ambiguity=IQR/(global_IQR+1e-6), and a small integer source_code indicating resolution level (exact primary, expanded primary, exact fallback, expanded fallback, global). For validation, construct tables only on each training fold and transform held-out fold rows with fold-specific tables to avoid self-reference artifacts.
  expected_signal=Stars and quasars often violate the dominant galaxy-like color-redshift trend in different ways: stars can have low true redshift with misleading colors, while quasars can generate large or multimodal local residuals at high redshift, so residual magnitude, scale-adjusted residual, and local ambiguity are expected to sharpen balanced separation beyond raw features alone.
  risk=The hierarchy can overfit if cell definitions are too fine or if too many expansions collapse to global behavior, and sparse tail color regions may still be noisy; despite leakage protections, the signals can be redundant with direct redshift-color interactions and add dimensionality/cost, so aggressive regularization or feature pruning may be required.

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