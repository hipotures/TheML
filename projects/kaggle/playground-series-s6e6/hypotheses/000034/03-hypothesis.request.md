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

Hypothesis ID: 000034

Previous versions:

- rev 1: Blackbody-Like Continuum Compatibility
  group_name=blackbody_continuum_distance
  family=physically_informed_shape_model
  summary=Project each source’s five-band photometric SED onto a low-dimensional physical blackbody-manifold, then encode how closely its normalized continuum shape matches a stellar-like thermal spectrum and where it fails in a way that is distinct from simple color differences.
  strategy=For each row, convert ugriz magnitudes to positive linear flux proxies using f_b = 10^(-0.4*m_b) and normalize to shape-only flux fractions p_b = f_b / (f_u + f_g + f_r + f_i + f_z). Let effective wavelengths be λ = [3551, 4686, 6165, 7481, 8931] Å. Build a rest-frame variant by dividing λ by (1 + z_clip), where z_clip = max(redshift, 0.0). For a fixed log-spaced temperature grid T ∈ [1500 K, 50000 K] (e.g., 140 points), compute normalized Planck templates B_b(T) ∝ (1/λ'^5)/(exp(14387.7/(λ' * T)) − 1) for each band b, then for each T solve the least-squares scale a_T = (Σ_b p_b B_b)/(Σ_b B_b^2) and residual SSR_T = Σ_b (p_b − a_T B_b)^2. Select T* with minimum SSR_T, and compute features: bb_logT = log10(T*), bb_ssr = SSR_T*, bb_gap = sqrt(SSR_T*) , bb_second_margin = log(1 + SSR_2nd_best - SSR_min), and per-band signed residual contrasts r_b = p_b − a_{T*}B_b(T*) for b∈{u,g,r,i,z}; clip any non-finite redshift inputs and clamp fluxes with eps = 1e-12 before log/normalization to avoid division-by-zero instabilities.
  expected_signal=This group gives a direct physical proxy for photospheric thermal shape: stars are expected to sit near the blackbody manifold with low residuals and interpretable temperature estimates, while quasars’ power-law/line-enhanced continua and galaxies’ composite spectra with breaks should systematically increase mismatch, improving separation especially in color-overlap regions that hurt linear color features.
  risk=The fixed blackbody prior can over-constrain intrinsically nonthermal objects and may underperform where dust reddening, calibration offsets, or strong line contributions dominate broad-band shape, and the per-row template search is heavier than simple arithmetic transforms (so it increases feature-generation cost and should be implemented with vectorized grid evaluation).

- rev 2: Rest-frame Blackbody Continuum Distance
  group_name=blackbody_continuum_distance
  family=physically_informed_shape_model
  summary=This group maps each object's normalized ugriz spectral shape onto a thermal continuum manifold in rest-frame wavelength space and derives compact distance-to-manifold signals that should distinguish near-blackbody stellar spectra from quasar and galaxy continua.
  strategy=For each row, compute flux proxies from magnitudes as `f_b = 10^{-0.4 m_b}` for bands b∈{u,g,r,i,z}; if a magnitude is non-finite, replace it with 30 before exponentiation (equivalent to a flux floor of 10^-12) so the proxy is finite. Form the normalized shape vector `p_b = f_b / (Σ_b f_b)`; if `Σ_b f_b <= 1e-30`, set `p_b = 0.2` for all bands. Use effective wavelengths `λ = [3551,4686,6165,7481,8931] Å` and redshift-corrected wavelengths `λ'_b = λ_b / (1 + z_eff)` with `z_eff = max(redshift, 0)` and non-finite redshift mapped to 0. Build a logarithmic temperature grid `T_j = 10^{linspace(log10(1500), log10(50000), J)}` with `J=180`. For each T_j, evaluate template shape `B_b(T_j) = (1 / λ'^5_b) / (exp(14387.7/(λ'_b * T_j)) - 1)` using a numerically stable `expm1` form and clip argument to `≤700` before exponentiation; normalize each template as `t_b(T_j)=B_b/Σ_b B_b`. Compute least-squares scale `a_j = (Σ_b p_b t_b(T_j)) / (Σ_b t_b(T_j)^2)` and residual `SSR_j = Σ_b (p_b - a_j t_b(T_j))^2`. Let `j* = argmin SSR_j`; optionally refine `log10(T*)` by quadratic interpolation over `(log10 T)` across `j*-1, j*, j*+1` and clamp within grid bounds. Emit deterministic outputs: best-fit `log10(T*)`, `SSR_min`, `sqrt(SSR_min)`, residual gap to second best `SSR_2nd - SSR_min`, local curvature from second finite-difference in log-space, and signed residuals `r_b = p_b - a_{j*} t_b(T*)` for each band. Implement all operations in vectorized matrix form over T to keep runtime tractable and deterministic.
  expected_signal=The procedure yields an explicit physical shape-distance representation: stars are expected to show low, smooth mismatch to a thermal manifold while galaxies and quasars, which often include nonthermal continua, breaks, and strong emission structures, should incur larger manifold residuals, especially where raw magnitudes and color ratios alone blur class boundaries.
  risk=The manifold prior is intentionally physics-driven and can over-penalize heavily reddened stars, line-dominated quasars, and objects with calibration artifacts, and grid-based fitting can produce boundary effects or instability at extreme redshift unless interpolation and numeric guards are used; generated features are also partially redundant with color-based predictors, so downstream regularization/selection is advised.

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