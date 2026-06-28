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

Hypothesis ID: 000022

Previous versions:

- rev 1: AIDE sky-cell local residuals
  group_name=aide_sky_cell_local_residuals
  family=aide_spatial_context
  summary=Encode local empirical sky context from the AIDE top-5 solutions: half-degree sky-cell density, 3x3 neighbor density, concentration, and within-cell residual statistics for magnitudes, redshift, and color-shape features.

  strategy=Build a deterministic sky key from alpha and delta by binning alpha at two cells per degree and delta+90 at two cells per degree, matching the coarse AIDE sky-cell construction. For each row, add the count of objects in its cell, the total count in the surrounding 3x3 cell neighborhood, and the ratio between the cell count and neighbor count as a local concentration feature. Within each sky cell, compute mean and standard-deviation residuals for u, g, r, i, z, and clipped redshift, producing both signed deltas and z-scores. Repeat the same residual construction for AIDE color-shape columns such as u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, u-2*r+i, and g-2*i+z. Add aggregate residual summaries such as L2 norm, absolute L1 norm, mean signed color delta, color-delta standard deviation, and maximum absolute local z-score. Do not use target labels.
  expected_signal=The synthetic catalog may preserve sky-dependent survey depth, calibration, or selection structure. Objects with unusual photometry relative to nearby sky cells can separate stars, galaxies, and quasars beyond pure angular coordinates.
  risk=Cell-level statistics can be noisy in sparse regions and may partially duplicate existing sky-position hypotheses; if train and test sky distributions differ, empirical cell residuals can become brittle.

- rev 2: Boundary-safe sky-cell residual harmonization
  group_name=aide_sky_cell_local_residuals
  family=aide_spatial_context
  summary=Create spatial-context features by expressing each object’s magnitudes, redshift, and color-shape descriptors relative to the local sky-cell population, then summarizing the strength and shape of that local deviation as a robust proxy for survey-systematic structure linked to class differences.
  strategy=Define a fixed equal-area-ish tessellation at 0.5-degree resolution in both coordinates: ra_bin = floor(mod(alpha, 360) * 2), dec_bin = floor((delta + 90) * 2), clipped to [0,719]×[0,337], with RA bins wrapped modulo 720 and DEC neighbors clipped at domain boundaries. Assign each row to one sky cell id, then for each row gather the 3x3 neighborhood (9-cell block, RA-wrapped) and compute n_cell, n_neigh, concentration = n_cell/(n_neigh + 1e-6), plus cell-centered local coordinates if needed. For each numeric feature set x in {u,g,r,i,z,redshift_clipped, and all explicit color/shape features u−g, g−r, r−i, i−z, u−i, u−r, g−z, r−z, u−2r+i, g−2i+z}, compute cell-wise mean μ_cell(x) and scale σ_cell(x) on training data only, plus global μ_global(x), σ_global(x). Use shrinkage for instability: μ*(x)=w μ_cell(x)+(1−w)μ_global(x), σ*(x)=w σ_cell(x)+(1−w)σ_global(x), with w=n_cell/(n_cell+20) and a floor σ=0.1-th percentile clamp per feature, then residual Δx=x−μ*(x) and robust z_x=Δx/(σ*(x)+1e-6). Add per-row statistics over x: mean Δx, mean |Δx|, L1 and L2 residual norms, mean z_x, max|z_x|, and residual standard deviation across color/shape features. For sparse cells, also append fallback features from nearest non-empty cell in great-circle RA/DEC Manhattan neighborhood only when n_cell is below threshold to reduce noise; when any feature is undefined, exclude it from that row’s aggregate pool.
  expected_signal=Astrophysical class boundaries often depend on subtle local photometric offsets caused by spatially varying depth, extinction, and calibration, so residualizing to the local sky environment can isolate class-relevant morphology in color-space that is obscured by global feature values.
  risk=Residuals can become brittle where local support is very low, can amplify catalog artifacts if cell definitions mismatch the hidden test distribution, and can overlap with other spatial or photometric feature groups unless de-duplicated or regularized.

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