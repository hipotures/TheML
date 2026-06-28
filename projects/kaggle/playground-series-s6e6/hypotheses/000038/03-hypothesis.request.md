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

Hypothesis ID: 000038

Previous versions:

- rev 1: Redshift-Trajectory Geometry Residuals
  group_name=photoz_trajectory_geometry
  family=trajectory_consistency
  summary=Model the SDSS color vector as a smooth function of redshift and encode both deviation from, and geometric relation to, the local color-redshift trajectory so that objects with non-physical evolutionary motion in color space are separated from the dominant manifolds.
  strategy=Create colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. Build a redshift grid using quantiles of redshift (for example 0.5th to 99.5th percentile in bins of fixed width or 160 quantile knots) and fit a robust smooth function to each color over redshift (e.g., LOWESS or monotonic cubic spline on the training distribution, and apply the same transform to test). For every redshift node compute the local mean vector μ(z)=[μ1,μ2,μ3,μ4], robust covariance Sc(z) of colors, and first and second derivatives v(z)=dμ/dz and a(z)=d²μ/dz² by central differences. For each row, interpolate μ(z), v(z), a(z), Sc(z); define residual r=c-μ(z). Output features from geometry: orthogonal offset d_orth = r^T S_c(z)^{-1/2}(I - u u^T)S_c(z)^{1/2}r (with u = S_c(z)^{-1/2}v / ||S_c(z)^{-1/2}v||), tangential signed offset d_tan = r·u, curvature mismatch m_curv = (r·a) / (||a||+ε), and a pair of scale-normalized versions d_orth/σ_orth(z), d_tan/σ_tan(z). Clip z to grid endpoints with expanded-neighbor fallback for sparse bins; when n(z-bin)<minimum, widen the bin and smooth with neighbor bins, and for z<0.01 or extremely high-z tails use global spline fallback with reduced derivative weight.
  expected_signal=This turns the broad color-redshift relation into structured geometric features where nearby redshifted stars/galaxies/quasars are constrained differently: quasars with emission-line-driven color drift create larger manifold-orthogonal residual structure, galaxies typically stay closer to smoother evolutionary tracks, and stars near z≈0 are forced to low-variation trajectories, improving discrimination in ambiguous regions and likely stabilizing balanced-class recall.
  risk=If redshift values are noisy or biased for a substantial fraction of objects, tangent and derivative estimates can become unstable and inject noise into features; extrapolation beyond sparse redshift extremes may produce unreliable derivatives, and the spline/quantile-knot choice can overfit sparse regions, requiring careful regularization and fallback logic.

- rev 2: Adaptive Redshift-Manifold Color Geometry Features
  group_name=photoz_trajectory_geometry
  family=trajectory_consistency
  summary=Model the expected redshift evolution of SDSS color space as a smooth manifold and encode each object by deterministic distances to this manifold, including displacement off the manifold and residual behavior along and across its local trajectory, so class-dependent departures become explicit predictive signals.
  strategy=Use only train rows to fit the trajectory model. Define colors c=[c1,c2,c3,c4]=[u-g, g-r, r-i, i-z]. Build a redshift grid G={g_k} from quantiles of redshift at levels 0.5%..99.5% with K=180 knots. For each knot g_k, fit local estimates with weights w_i(g_k)=exp(-(z_i-g_k)^2/(2h^2)), h=0.08. Require at least N_min=1500 effective weighted samples; if not satisfied, repeatedly double h (up to 1.0) and recompute; if still insufficient, fallback to a global robust fit with derivative suppressed. Compute weighted Huber location μ(g_k)=HuberMean(c | weights), weighted robust covariance Σ(g_k)=HuberCov(c | weights), then shrink Σ(g_k): Σ_s(g_k)=0.85Σ(g_k)+0.15·I*(tr(Σ(g_k))/4) and floor eigenvalues at 1e-6. Fit each color component μ_j(z), each Σ_s entry, and first/second derivatives via monotone cubic smoothing splines over k in z to obtain μ̃(z), v(z)=dμ̃/dz, a(z)=d²μ̃/dz², Σ̃(z), and local scales σ_t(z),σ_e(z),σ_m(z) for projected quantities. For any row (c,z), interpolate at z' = clip(z, g_1, g_K); if z outside [min(train z),max(train z)] then set a=0 and damp all derivative-based components by α=min(1,|z-z_edge|/0.03). Compute residual r=c-μ̃(z'). Let L(z') be Cholesky of Σ̃(z'); y=L^{-1}(z') r, t=L^{-1}(z') v(z'), u=t/||t||, and a_w=L^{-1}(z') a(z'). Define tangential signed offset T=y·u, orthogonal energy E=||y||^2-(y·u)^2, and curvature mismatch M=(r^T Σ̃^{-1} a)/(||a||+1e-6); emit {T, E, M, T/σ_t(z'), E/σ_e(z'), M/σ_m(z')}. For z<0.01, further multiply T,M,E by z/0.01 to avoid unstable derivative behavior in near-zero-redshift regime.
  expected_signal=The feature family turns a noisy multiband snapshot into redshift-normalized manifold diagnostics, so quasars with emission-line-driven color drift, galaxies following smoother spectral tracks, and nearby low-variation stars become separable by different residual geometry patterns, which should improve recall balance across GALAXY, QSO, and STAR.
  risk=A small bandwidth or poorly conditioned covariance can overfit local noise and destabilize derivative features, while large smoothing can erase class-specific local curvature cues; the approach is compute-heavy for spline fitting and per-row linear-algebra operations, and may be partially redundant with other redshift- or color-derived features if not tuned.

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