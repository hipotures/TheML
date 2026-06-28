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

Hypothesis ID: 000046

Previous versions:

- rev 1: Adaptive local manifold codimension and extrapolation signatures
  group_name=local_manifold_codimension
  family=manifold_geometry
  summary=Model the neighborhood geometry of each source in joint photometric-color-plus-redshift space to express whether it sits on a thin, well-supported manifold branch or in an unsupported transition region where class-specific physical interpretation is more likely to differ from nearby stellar-like populations.
  strategy=For every row, build four colors C = [u-g, g-r, r-i, i-z], then form a standardized feature vector x = [C, redshift] with robust scaling (MAD or IQR). For each row, find k nearest neighbors in x (k=64, k' = min(k, available-1) near sparse edges). Compute neighbor centroid m_i, covariance Σ_i, and ordered eigenvalues λ1≥λ2≥...≥λ4 from the 4D color+redshift neighborhood. Derive: (1) local codimension ratio r2 = (λ1+λ2)/(λ1+...+λ4), (2) manifold residual e2 = (x_i-m_i)^T(I−V2V2^T)(x_i-m_i)/(λ3+λ4+1e-12), where V2 are top-2 eigenvectors of Σ_i, (3) local anisotropy and complexity scores from (λ1/λ2), (λ2/λ3), and (λ4/λsum), (4) local support metrics d_k, median radius r_med, and density proxy 1/(d_k+1e-12), plus in-box support = fraction of coordinates of x_i that lie within the [5th,95th] percentile envelope of its neighbors per dimension. Repeat once on colors alone (4D, without redshift) and once on raw magnitudes [u,g,r,i,z] with MAD scaling, then concatenate corresponding features; if a neighborhood covariance is near-singular, replace unstable ratios with global fallback stats and set a binary fallback flag.
  expected_signal=Quasars and many galaxies deviate from the thin local structure and support of the dominant stellar photometric manifold, especially near class boundary regions, while true stars remain in locally low-complexity, high-support neighborhoods, so these local linearity/extrapolation features provide a direct geometric separator that complements global color cuts and helps balanced accuracy on ambiguous regions.
  risk=Nearest-neighbor graph construction is computationally heavy at this dataset size and requires controlled approximations to stay practical; neighborhood features can be sensitive to scaling choices and nearest-neighbor anisotropy, and using transductive neighborhood estimates over unlabeled test points may introduce subtle distribution-aware behavior that must be kept explicit and stable.

- rev 2: Adaptive local manifold codimension and support diagnostics
  group_name=local_manifold_codimension
  family=manifold_geometry
  summary=The core idea is to characterize each object by how well it sits on a locally linear, high-support manifold in standardized photometric space versus how much it departs into sparse, high-curvature regions, using that geometric mismatch as a class-sensitive signal beyond global color patterns.
  strategy=Compute color features c1=u-g, c2=g-r, c3=r-i, c4=i-z. Build two numeric spaces: S1=[c1,c2,c3,c4,redshift] and S2=[u,g,r,i,z]. For each numeric column in each space, fit robust scaling on train only with median m and MAD s, with s clipped below 1e-6, then transform x <- (x-m)/s. Set k=64 and for each row define neighbor budget k_i=min(k,|P|-1), where P is candidate pool in train. Candidate pool is first filtered by exact spectral_type and galaxy_population; if |P|<2k_i fall back to full train for that row. For each space separately, query k_i nearest neighbors using a fixed-seed ANN/Euclidean index (test points only query against train, not vice versa) and then compute centroid μ, covariance Σ on centered neighbors, then Σ←0.5(Σ+Σ^T)+1e-6 I_d. If eigenvalue rank<d or trace<=1e-12, use global fallback covariance/eigen stats and set degenerate=1. Sort eigenvalues λ1…λd and top2 eigenvectors V2. Emit: r2=(λ1+λ2)/sum(λ), residual2=(x-μ)^T(I−V2V2^T)(x-μ)/(λ3+…+λd+1e-12), a12=log((λ1+1e-12)/(λ2+1e-12)), a23=log((λ2+1e-12)/(max(λ3,1e-12))), tail=λd/sum(λ) where d is dimensionality of the space, and dim95=min m with cumulative λ1..λm /sum(λ) >=0.95. For neighborhood support, compute distances to each neighbor, then r_k=max distance, r_med=median distance, r_mean=mean distance, density_proxy=1/(r_med+1e-12), r_ratio=r_k/(r_med+1e-12), and frac_in_box = mean over dimensions of x_j being within per-dimension [5th,95th] percentile of neighbor values. Add local asymmetry stats: median absolute deviation of x from neighbor median (and its ratio to neighbor MAD). Concatenate the same feature set from S1 and S2, and add absolute deltas between matched metrics across spaces (e.g., |r2_S1-r2_S2|, |residual2_S1-residual2_S2|, |density_S1-density_S2|). Use a second-pass guard for very sparse regions: if k_i<6 replace distance/eigen metrics with fallback quantiles from global train neighborhoods and keep sparse_flag=1.
  expected_signal=Stars are expected to occupy dense, low-curvature local manifolds aligned with the main photometric loci, while quasars and many galaxies should present as local geometric outliers with higher codimension residuals or weaker support, especially where class boundaries are blurred; these local shape-and-support descriptors can therefore raise recall on minority classes without sacrificing GALAXY separation, improving balanced accuracy.
  risk=High computational cost and ANN approximation error at full train-plus-test scale can add variance to metrics, and conditioning-dependent eigen-ratio features may be unstable in very sparse or duplicated neighborhoods; restricting neighborhoods by auxiliary categories can help homogeneity but may overfit or create sparsity artifacts, and fallback substitution must be carefully fixed to avoid leakage and distribution-sensitive behavior.

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