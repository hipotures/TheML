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

Hypothesis ID: 000043

Previous versions:

- rev 1: Adaptive redshift-stratified SED manifold residuals
  group_name=redshift_partitioned_sed_pca_residuals
  family=manifold_projection
  summary=Model each object’s normalized ugriz flux pattern against a redshift-local low-dimensional spectral manifold so that quasar-like departures from the dominant smooth spectral envelope and band-profile anomalies become explicit, class-discriminative signals.
  strategy=Convert each band to linear flux with a stable log-scale form f_b = 10^{-0.4 m_b}, then form scale-free per-object shape coordinates p_b = f_b / (Σ f_b + eps) for b in {u,g,r,i,z}. Partition training rows into fixed redshift strata (e.g., equal-mass quantile bins over redshift or fixed-width bins with minimum-size fallback); for each populated stratum, fit PCA on the 5-D vector p and keep the first three components. For each row, using its assigned stratum’s mean and eigenvectors, compute the first-three projection scores c1..c3, reconstruct p̂ from those scores, and emit residual-based features: ||p - p̂||_2, ||p - p̂||_2 / ||p||_2, residual energy fraction (sum residual^2 / sum p^2), and signed residual coordinates (p - p̂)·e_b for selected band basis directions to preserve where mismatch occurs in color profile. If a row falls into an under-supported bin (or boundary edge), back off to the nearest sufficiently populated bin or a global PCA fit, and cap/clip redshift to training min/max only for binning.
  expected_signal=A small number of PCA coordinates captures dominant smooth continuum geometry, while residual magnitude and direction expose high-frequency spectral irregularity and redshift-driven profile twists that are expected for quasars and some AGN-like galaxies but less for ordinary stars, especially improving separation in ambiguous redshift regions.
  risk=Component estimates can be noisy when bin support is low, residual features may correlate strongly with existing curvature/locus-based groups, and extreme faint/noisy points (notably in u and z) can inflate residual metrics unless clipping and robust scaling are handled carefully.

- rev 2: Redshift-Partioned SED Residual Manifold
  group_name=redshift_partitioned_sed_pca_residuals
  family=manifold_projection
  summary=Model the typical local shape of galaxy SEDs in redshift-sliced color space and use each object’s orthogonal departure from that manifold to expose class-specific spectral irregularities that are not captured by raw magnitudes alone.
  strategy=Convert ugriz magnitudes to flux using f_b = 10^{-0.4 m_b}, then build a scale-free 5-band shape vector p=(f_u,f_g,f_r,f_i,f_z)/(Σf + 1e-12) in double precision. Build redshift bins from training data via fixed quantile edges at 0,0.1,...,1.0 of train redshift and merge neighboring bins until every bin has at least m samples where m=max(4000, floor(0.004 * N_train / nbins)); if any bins remain sparse, remove them from local fitting and mark for fallback reassignment. For each surviving bin, z-score p within-bin (mean/std from train rows in that bin, with std floor) and fit PCA on the standardized vectors; retain the first three components (or full rank if lower) and store mean, std, loadings, explained-variance profile, and optional whitening scale. At inference, clip object redshift to [min_train, max_train], assign to nearest populated bin by midpoint distance; if no direct bin match, use nearest surviving bin; if still unavailable, fall back to a global PCA fit on all training rows using the same preprocessing. Compute scores c = W^T (p-z) and reconstruction p̂ = W c + μ, then residual r = p - p̂. Emit deterministic features: residual L2 norm ||r||_2, relative residual ||r||_2 / ||p||_2, residual energy fraction Σr^2 / Σp^2, orthogonal residual component energy (variance outside retained PCs), signed directional residuals r·w_1..r·w_3, per-band signed residual components r_b, and bin-local reconstruction-error quantile (rank of ||r||_2 inside the fitted bin). Apply a global cap on residual magnitudes per band to avoid u/z noise spikes from dominating and set a fixed epsilon for all divisions.
  expected_signal=Within narrow redshift windows, galaxies and many stars occupy a low-dimensional smooth spectral manifold, so residual-based distance and residual-direction descriptors provide a stable view of continuum mismatch; quasars, with shifted line features and other narrow-band anomalies, should produce more structured and larger out-of-manifold deficits, improving separability where plain color summaries are blurred.
  risk=Local PCA quality depends on bin support and can become unstable near sparsely populated redshift tails, fallback reassignment can smooth over subtle local class structure, residuals in faint/noisy bands can still dominate despite clipping, and these features may overlap strongly with existing color-curvature features, increasing correlation and overfitting risk without regularization.

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