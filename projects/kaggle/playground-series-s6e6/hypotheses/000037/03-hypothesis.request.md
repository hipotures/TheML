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

Hypothesis ID: 000037

Previous versions:

- rev 1: Redshift-dependent galaxy-population color manifold concordance
  group_name=population_color_manifold_drifts
  family=tag_conditioned_color_geometry
  summary=Build redshift-evolving red-sequence and blue-cloud color manifolds in ugriz space from the given population tags and penalize objects that are structurally inconsistent with both the manifold they claim via tag and the opposite manifold, with strong, signed mismatch margins as direct class-separation signals.
  strategy=For every object compute c_ug = u-g, c_gr = g-r, c_ri = r-i, c_iz = i-z, and c_ur = u-r. Bin redshift into fixed quantiles (for example 24 equal-count bins on log1p(redshift), clamped to [0, max z]) and, within each bin, compute robust reference statistics separately for galaxy_population=Red_Sequence and galaxy_population=Blue_Cloud using training-only statistics: median vector μ_j and robust scale s_j for each color (use 1.4826*MAD), and an optional diagonal covariance proxy. For each object, form standardized residuals r_{j,f} = (c_f - μ_{bin,j,f})/(1.4826*s_{bin,j,f}) with absolute-value clipping (e.g., at 8), then compute robust squared-distance features D_assigned = Σ_f r_{bin,tag,f}^2 and D_opposite = Σ_f r_{bin,other,f}^2. Add signed color-margin features against the red/blue split proxy using the bin-wise gap g_gap = median(c_ug | bin, Red_Sequence) - median(c_ug | bin, Blue_Cloud), and signed offsets m_ug = c_ug - (2.22 + g_gap), m_gr, m_ri, m_iz to the two population-specific medians. If a (bin, population) cell has < N_min points, shrink the residuals toward global medians/scales with weight w = n/(n+N_min), and if a color scale is near-zero or missing, set its contribution to zero.
  expected_signal=It should separate galaxies that truly follow a coherent red-sequence/blue-cloud trajectory from stars and quasars whose broadband colors are often off-manifold or inconsistent with the supplied population label once redshift evolution is accounted for, improving minority-class recall where color overlap is otherwise severe.
  risk=The population tags are weak indicators and can be noisy for non-galaxy objects, so aggressive manifold distances may over-penalize noisy but valid labels; sparse redshift bins can overfit and become unstable unless shrinkage is strong, and strong overlap in color distributions will still cause leakage-like confusion rather than clean class boundaries.

- rev 2: Redshift-Conditioned Galaxy-Manifold Deviation Features
  group_name=population_color_manifold_drifts
  family=tag_conditioned_color_geometry
  summary=Learn redshift-dependent color manifolds separately for the two population tags and represent each object by how strongly it aligns or diverges from its tagged manifold versus the alternative manifold, using robust, bin-local geometry to surface class-consistent offsets in overlap-heavy color regions.
  strategy=Compute a fixed color vector per object using ugri z-band contrasts (u-g, g-r, r-i, i-z, u-r). Bin redshift using training data only into B=24 equal-frequency bins on t=log1p(redshift), with all bin edges taken from training and clipped so every test object maps deterministically. For every (bin b, population p) cell, estimate robust location mu_{b,p,f} and robust scale sigma_{b,p,f}=1.4826*MAD for each color f from training rows. If a cell has n_{b,p}<N_min (N_min=300) or sigma_{b,p,f} is zero/near-zero, shrink toward global training statistics: w=min(1,n_{b,p}/N_min); mũ=w*mu_{b,p,f}+(1-w)*mu_global_f, sigmã=w*sigma_{b,p,f}+(1-w)*sigma_global_f. If both nearby bins also lack mass, fallback to adjacent-bin averaging with count-weighted interpolation before global fallback. For each object in bin b with tag p, compute standardized residuals z_{b,p,f}=clip((c_f-mũ_{b,p,f})/sigmã_{b,p,f},-8,8), and D_assigned=mean_f(z_{b,p,f}^2). Compute D_opposite from the opposite population statistics in the same bin using the same clipping and scaling. Add cross-manifold discrimination features: D_diff=log1p(D_opposite)-log1p(D_assigned) and D_ratio=D_opposite/(D_assigned+1e-6). Also add per-color relative manifold features Delta_f=(c_f-mũ_{b,Red,f})/sigmã_{b,Red,f} - (c_f-mũ_{b,Blue,f})/sigmã_{b,Blue,f}. Build red/blue divider features per color: mid_f=0.5*(mũ_{b,Red,f}+mũ_{b,Blue,f}), gap_f=mũ_{b,Red,f}-mũ_{b,Blue,f}, and m_f=(c_f-mid_f)*sign(gap_f), where sign indicates side of the bin-wise separator and magnitude indicates margin-to-boundary. For any unavailable color contribution after shrink/fallback, set its term to zero and emit a missing contribution indicator.
  expected_signal=The resulting features make the model sensitive to the expected redshift evolution of red-sequence versus blue-cloud trajectories while letting stars and quasars be penalized for inconsistent manifold fit, which should improve boundary quality where raw magnitudes overlap and help balanced accuracy by giving clearer minority-class separation.
  risk=If population tags are noisy for non-galactic objects, manifold-distance penalties can introduce structured errors, sparse bin estimation can still be unstable even with shrinkage, hard bin edges may create discontinuities near cut points, and diagonalized robust distances can miss useful color covariance unless compensated by downstream learners.

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