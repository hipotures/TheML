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

Hypothesis ID: 000058

Previous versions:

- rev 1: Deconvolved Flux-Branch Posteriors
  group_name=redshift_branch_deconvolved_flux_posteriors
  family=probabilistic_density_geometry
  summary=Model each object’s relative ugriz flux pattern with uncertainty-smoothed, class- and redshift-branch-conditioned generative densities and use the resulting posterior mass, margins, and ambiguity signals as compact features that resolve quasar-star overlap while preserving galaxy cues.
  strategy=Convert ugriz magnitudes to linear fluxes F_b = 10^{-0.4 m_b} and form a 4D relative-flux vector x = [log(F_u/F_i), log(F_g/F_i), log(F_r/F_i), log(F_z/F_i)] plus optional additive color fallback features [u-g, g-r, r-i, i-z] for robustness. Assign each row to overlapping i-band windows (width 0.8 mag, step 0.4); for quasars also split a redshift branch variable r in {z<2.2, 2.2<=z<=3.5, z>3.5}, while galaxies and stars use a shared non-branch mode. In every populated (i-window, class, branch) stratum, fit a fixed-K Gaussian mixture on x using only training rows of that stratum (for example K=10-20, diagonal floor on covariance, equalized component weight initialization). Estimate a per-window empirical noise matrix V from robust MAD of colors in that i-window, and deconvolve-noise likelihoods by evaluating each component with Sigma_tilde = Sigma_k + V. For each query row in stratum s, compute deconvolved class likelihood L_c,s = Σ_k π_ck N(x|μ_ck, Σ_tilde_ck), branch likelihoods for QSO_low/QSO_mid/QSO_high likewise, then normalize to posterior masses P_c and P_qso,* using smoothed class priors from train counts in the same i-window (backoff to nearest populated window if empty). Emit compact features: posterior logits log((P_c+1e-8)/(1-P_c+1e-8)) for STAR, GALAXY, QSO and for each QSO branch, pairwise margins log((P_qso+1e-8)/(P_star+1e-8)) and log((P_qso+1e-8)/(P_galaxy+1e-8)), branch concentration max(P_qso_branch)/sum(P_qso_*), branch entropy over the three QSO branches, and branch-count indicator (number of branches with P_qso_branch > 0.5*max_branch), with all probabilities clipped to [1e-8, 1-1e-8] and all log features truncated to finite range.
  expected_signal=This approach converts the physical separation logic behind SDSS quasar-stellar-galaxy color behavior into smooth likelihoods, which should better capture medium-z quasar ambiguity near the stellar-overlap region, preserve galaxy probability structure, and provide calibrated uncertainty-sensitive signals that are more informative for balanced accuracy than hard color cuts.
  risk=High-dimensional mixture fitting can overfit in sparse bins (especially high-redshift quasars), branch priors may drift if the test distribution differs from training, and using magnitude-derived noise proxies instead of true photometric errors may miscalibrate likelihood sharpness near faint limits.

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