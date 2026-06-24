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

Hypothesis ID: 000041

Previous versions:

- rev 1: Relative-flux likelihood margins for class discrimination
  group_name=xdqso_inspired_flux_density_scores
  family=probabilistic_flux_density_scoring
  summary=Create a compact probabilistic representation of each object by modeling its i-band-normalized flux shape in redshift-aware, magnitude-binned flux space and encoding class membership strength through class-marginal likelihood margins.
  strategy=Convert magnitudes to linear fluxes with f_b = 10^{-0.4 m_b}. Build the 4D relative-flux vector r = (f_u/f_i, f_g/f_i, f_r/f_i, f_z/f_i) (use f_i as reference; this avoids ratio blowups from u-band nonlinearity). Bin objects into overlapping i-band bins of width 0.2 mag with 0.1 mag step across [17.7, 22.5], and coarse redshift slabs e.g. z<0.8, 0.8<=z<2.5, 2.5<=z<4.0, z>=4.0. For each class c in {GALAXY, QSO, STAR} and each (i-bin, z-slab), fit a 4D Gaussian mixture on r from train only (for instance 20 components, matching the classical XDQSO setup) using class-labeled points. During scoring, evaluate class-conditional densities L_c(r | i,z) = prior_c(i,z)*sum_k pi_ck N(r; mu_ck, Sigma_ck + Sigma_proxy), where Sigma_proxy is a small data-driven positive-definite proxy error covariance (for example, robust diagonal variance of each ratio within the same i-bin; add a floor to avoid singularity). Smoothly handle empty/low-count bins by backing off to adjacent i-bins and then to global class models. Convert to pseudo-posteriors p_c = L_c / (L_GALAXY + L_QSO + L_STAR + eps) and emit: p_GALAXY, p_STAR, p_QSO, pairwise log-odds features (log p_QSO - log p_STAR, log p_QSO - log p_GALAXY, log p_STAR - log p_GALAXY), top-class confidence, and posterior entropy.
  expected_signal=This directly captures class overlap structure in a noise-aware generative manifold rather than relying on handcrafted color cuts, which is especially valuable where quasar and stellar loci intersect at medium redshift, so it should improve recall balance for the minority and ambiguous classes under balanced accuracy.
  risk=Density fitting is compute-heavy and sensitive to sparse bins (especially for rare high-redshift objects), and synthetic uncertainty proxies may bias likelihood calibration if noise is misestimated, requiring aggressive regularization and binning fallback to control instability.

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