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

Hypothesis ID: 000059

Previous versions:

- rev 1: Redshift-Binned Tag Compatibility Residuals
  group_name=tag_redshift_compatibility_residuals
  family=tag_manifold_consistency
  summary=Model how well each object's ugriz color pattern matches the joint catalog-tag regime (spectral_type and galaxy_population) implied by its redshift, and capture the mismatch margin against competing tag regimes as a compact class-disambiguation signal.
  strategy=Compute four adjacent colors C = [u-g, g-r, r-i, i-z]. Define t = log1p(max(redshift, 1e-6)) and bin t globally into quantile bins, then merge adjacent bins until each bin has at least a minimum support (for example >=150 objects) to avoid unstable estimates. Form eight tag states from spectral_type x galaxy_population. For each occupied (tag_state, z_bin), estimate robust color-center mu and robust diagonal/covariance Sigma from training data using median and MAD; enforce a floor on variance before inversion. For each row: (1) assign its own state and redshift bin; (2) compute self distance d_self = sqrt((C-mu)^T Sigma^{-1}(C-mu)) with fallback to coarser support if cell is sparse (state-only bin, then global), (3) compute distances to all alternative states at the same z_bin (or nearest populated bin), and take d_alt1, d_alt2 as the closest and second-closest alternatives; and (4) emit margins m1 = d_alt1-d_self and m2 = d_alt2-d_self, plus standardized color residuals (C-mu)/sqrt(diag(Sigma)) for self and winner-alternative states. Clip all distances to bounded percentiles (for example p1,p99) to reduce outlier blowups.
  expected_signal=This yields a physically interpretable contradiction score where objects with colors that are inconsistent with their provided tags at the same redshift—e.g., quasar-like or galaxy-like colors paired with unlikely tag combinations—receive strong positive residuals, helping separate classes that overlap in raw color space.
  risk=Relies on the quality and consistency of catalog tags; if tag assignments are noisy or distribution-shifted in test, residuals can become misleading, and small-cell sparsity can add instability unless smoothing/fallback logic is carefully applied.

- rev 2: Redshift-binned tag compatibility residual margins
  group_name=tag_redshift_compatibility_residuals
  family=tag_manifold_consistency
  summary=Generate features that quantify how well an object's broadband color pattern aligns with the color manifold implied by its provided catalog tags at similar redshift, and how strongly it is contradicted by competing tag regimes.
  strategy=Compute the 4-vector color response C=[u-g, g-r, r-i, i-z] for every row. Transform redshift with t=log1p(max(redshift,0)) and form initial redshift bins from global training quantiles (e.g., 200 quantiles), then greedily merge adjacent bins so each final bin has at least Bmin_train=3000 rows, capping the number of bins at a small stable value. Define the eight catalog tag states s = spectral_type × galaxy_population. For each state-bined cell (s,b) in training, estimate a robust center μ_s,b by median(C), residuals r=C-μ, componentwise scales σ_s,b=1.4826×MAD(r) with lower floor σ_floor = 1e-3, and a robust covariance Σ_s,b from winsorized residuals (componentwise clipping to [P1, P99] within cell, then empirical covariance). Stabilize Σ_s,b by adding λ·diag(median σ_train^2) with λ=0.1 and 1e-6 on the diagonal before inversion. At score time, for a given object and candidate state j, select profile (j,b*) by first trying the target redshift bin, then nearest neighboring bins b±1, b±2, b±3, b±4 until count(s,b*) ≥ 150, then fallback to all-redshift profile for that state, and finally to the global all-state profile if still unavailable. Compute Mahalanobis distance d_ij^2=(C-μ_j,b*)^T Σ^{-1}_{j,b*}(C-μ_j,b*) and clip d_ij^2 to training P1-P99 for all such distances. Let own state be the object’s catalog tags; denote d_self for own state and d_1,d_2 as nearest and second-nearest distances among the three alternative states. Emit bounded margin features m_1=d_1-d_self and m_2=d_2-d_self, raw scores d_self,d_1,d_2, z-score residual vectors r_self=(C-μ_self)/σ_self and for argmin alternative r_alt1, and a fallback-level indicator (direct/neighbor/state/global) plus a short compatibility prior feature p_self_class = count_class(state-bin, class=self_state?)/count(state-bin) using training-class frequencies inside (s,b) with the same fallback chain.
  expected_signal=Objects whose visible colors are internally consistent with their provided tags at a given redshift should have small d_self and large positive mismatch margins, while quasar-like or mis-tagged sources will show inflated self-distance and small or negative margins, giving a class-discriminative signal in overlaps where raw colors alone are ambiguous.
  risk=This group can be brittle when tag assignments in train and test are noisy or distribution-shifted, so profile mismatch features may become systematic error rather than signal; sparse or highly uneven bins can force fallback and reduce granularity; covariance inversion and clipping choices may introduce hyperparameter sensitivity; and the engineered residuals can be partially redundant with direct color and redshift features, increasing model complexity without guarantee of generalization.

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