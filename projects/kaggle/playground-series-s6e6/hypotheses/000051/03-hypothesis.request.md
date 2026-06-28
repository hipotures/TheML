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

Hypothesis ID: 000051

Previous versions:

- rev 1: Redshift-anchored K-correction residual signatures
  group_name=k_correction_residual_manifold
  family=k_correction_consistency
  summary=Estimate a compact redshift- and color-driven K-correction surface from existing photometry and convert each object to pseudo-rest-frame magnitudes, then use deviations from local regime-consistent corrected-Sed behavior to expose class-specific SED-family mismatch.
  strategy=For each band q in {u,g,r,i,z}, fit a low-order polynomial surface in redshift and one adjacent color using all rows: K_q(z, c_q)=Σ_{a=1..2} Σ_{b=0..2} β_{a,b}^{(q)} z^a c_q^b with no constant term (so K_q(0,·)=0), where c_u=u−g, c_g=g−r, c_r=r−i, c_i=i−z, and c_z=g−r. Fit these surfaces in three redshift regimes: [0,0.5), [0.5,2), and [2,∞) to avoid strong extrapolation, using robust regression and winsorizing coefficients if needed. Compute pseudo-rest magnitudes M_q = m_q − K_q(z,c_q) for each object, plus regime support/edge flags (inside regime-fit hull, clipped-by-edge, low-support bin). Build residual features by comparing M_q against regime- and c_q-local expectations: for each q, M_q_resid = M_q − median(M_q | z-bin, c_q-bin), and a MAD-scaled version M_q_zscore = M_q_resid / MAD(M_q in same bin). Add corrected-color slopes and smoothness scores on M_q, such as (M_u−M_g)_resid, (M_g−M_r)_resid, (M_r−M_i)_resid, (M_i−M_z)_resid, and the dispersion of adjacent corrected slopes, then keep only these deterministic columns.
  expected_signal=A physically plausible galaxy-like SED should admit a relatively stable rest-frame correction and low, structured residuals across bands, while many stellar SEDs near redshift zero and quasar SEDs with strong non-thermal or line-driven behavior should show regime-dependent departures, giving class-separating structure beyond raw magnitudes and redshift.
  risk=High-redshift and low-support bins require edge handling and may produce unstable corrections, especially where polynomial extrapolation is used; this can add noise and occasionally confuse rare true objects that lie off the fitted manifold, and the family fit quality is sensitive to regression stability choices.

- rev 2: Stratified piecewise redshift-color K-correction residual manifold
  group_name=k_correction_residual_manifold
  family=k_correction_consistency
  summary=Fit a deterministic manifold that expresses each observed band through redshift- and color-conditioned correction behavior and use departures from that manifold as residual-geometry signals, so classes are separated by physically meaningful SED mismatch patterns rather than raw photometric scale alone.
  strategy=For each row compute c_u=u-g, c_g=g-r, c_r=r-i, c_i=i-z, and c_z=g-r. Split rows by descriptive strata h=(spectral_type, galaxy_population) and redshift regimes r in { [0,0.2), [0.2,0.7), [0.7,1.6), [1.6,7.0) } with clipping outside support to nearest boundary; if a h x r slice has fewer than 5,000 rows, fallback to a broader slice (merge adjacent regimes, then remove stratum conditioning if still sparse). For each band q in {u,g,r,i,z}, fit in every available h,r a robust Huber polynomial surface K_hat_{q,h,r}(z,c_q)=beta0+beta1 z+beta2 z^2+beta3 c_q+beta4 c_q^2+beta5 z*c_q, where c_q is the chosen adjacent color above. Anchor the correction so that near-zero redshift is neutral by subtracting the 50th percentile of K_hat at z<0.02 within the same h, and optionally clip coefficient magnitudes to median absolute deviation-based bounds before use. Evaluate K^* per object: use regime-local fit in r, but in a blend zone of width delta_z=0.03*(1+z) around regime boundaries linearly interpolate neighboring regime predictions; set edge flags for blended or out-of-hull cases. Compute pseudo-rest magnitudes M_q=m_q-K^*. Build 2D local support bins on (z,c_q) using 30 quantile bins in z and 12 quantile bins in c_q for each h,r. In each bin store median and MAD of each M_q and of slope-like quantities d1=M_u-M_g, d2=M_g-M_r, d3=M_r-M_i, d4=M_i-M_z; produce residuals and MAD-scaled z-scores for all these quantities. Also compute curvature-like terms s1=d1-d2, s2=d2-d3, s3=d3-d4 and derive their residuals/z-scores against the same bin baselines. If a bin has fewer than 200 rows, borrow baseline from distance-weighted 3x3 neighboring bins in quantile-index space; if still empty, use global h,r baseline and emit low-support flags.
  expected_signal=The corrected-rest manifold captures the dominant color-redshift continuum in photometric data, while class-specific departures from that manifold reveal residual structure from stellar-at-z~0 continua and quasar spectral features, so this representation should provide sharper class separation for GALAXY, QSO, and STAR under balanced accuracy.
  risk=Aggressive slicing by stratum and regime can create unstable local fits in sparse high-redshift or rare-type regions, and heavy blending or fallback pooling can damp real minority-class structure; the residual columns are also derived from original magnitudes, so redundancy can inflate model sensitivity to noise unless later learners apply proper regularization.

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