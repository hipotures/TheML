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

Hypothesis ID: 000048

Previous versions:

- rev 1: Luptitude regime-aware flux re-parameterization
  group_name=luptitude_regime_flux_features
  family=asinh_flux_domain
  summary=Transform SDSS asinh magnitudes into a flux-domain representation that explicitly models the low-signal linear regime and AB calibration idiosyncrasies, then describe each object’s spectral shape and color behavior with regime-aware flux-based descriptors to stabilize class clues near the photometric detection floor.
  strategy=Apply the SDSS u-band AB correction u_ab = u - 0.04 first. Convert each band magnitude m_b in {u_ab,g,r,i,z} to linear flux proxy f_b with the inverse Luptitude relation f_b = 2*b_b*sinh(-0.4*ln(10)*m_b - ln(b_b)), using SDSS softening constants b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10 (f0 normalized to 1). For each band define a reliability regime flag s_b = 1(|f_b| <= 10*b_b), and a soft-clipped flux \tilde f_b = sign(f_b)*max(|f_b|, b_b). Build features from \tilde f_b: flux-log colors for adjacent bands, second-order differences across log10 wavelength order, and a least-squares slope/curvature of log(\tilde f) along {u,g,r,i,z}; compute magnitude-vs-flux color mismatch for each adjacent pair, e.g. delta_c_ug = (u_ab-g) - log(\tilde f_u/\tilde f_g), similarly for other pairs. Add shape summary features on soft-clipped normalized flux allocation (sum abs \tilde f, max share, entropy-like concentration, L2 concentration), and regime descriptors per row: soft_count = sum(s_b), soft_fraction = soft_count/5, negative_flux_count = sum(f_b<0), and a soft_regime_weight = 1 - soft_fraction. Interact slope/curvature features with (1-soft_fraction) and soft_fraction to separate high-SNR and low-SNR regime behavior. For all divisions include epsilon-safe clipping to avoid instability.
  expected_signal=By replacing noisy magnitude-domain colors with physical flux-domain structure near the asinh linearization break, the group can better preserve true SED ordering for faint objects and explicitly flag when color geometry is dominated by measurement floor effects, which should help disentangle quasars, galaxies, and stars in overlapping color-redshift regions and reduce ambiguity-driven boundary errors.
  risk=The transformation is SDSS-specific and tied to published b constants and u-band offset, so performance can degrade if the dataset’s photometry comes from different processing or zero-point conventions; clipping and ratio construction can still amplify noise for extreme values, the feature set may overlap with prior flux/locus groups, and additional nonlinear operations add some compute overhead for very large tables.

- rev 2: Regime-gated luptitude flux geometry descriptors
  group_name=luptitude_regime_flux_features
  family=asinh_flux_domain
  summary=This feature family reconstructs each object's five-band spectral shape in luptitude-derived flux space while explicitly modeling the asinh softening regime so that slope and curvature cues used for class separation are preserved when measurements sit near the photometric floor.
  strategy=For each row use u, g, r, i, z and apply SDSS asinh photometric offsets before inversion with fixed defaults u_corr = u - 0.04 and z_corr = z + 0.02, while g_corr = g, r_corr = r, i_corr = i (all offsets are constants to be toggled by data provenance). Use b = {u:1.4e-10, g:0.9e-10, r:1.2e-10, i:1.8e-10, z:7.4e-10} and compute inverse-luptitude flux proxies in maggy-normalized units as f_b = 2*b_b*sinh(-(ln(10)/2.5)*m_corr,b - ln(b_b)). Use eps=1e-12 and define absf_b=|f_b|. Define floor regime flags floor1_b = 1(absf_b <= 2*b_b) and floor2_b = 1(absf_b <= 0.2*b_b), then soft_count = Σ floor1_b, floor_frac = soft_count/5, ultra_floor_count = Σ floor2_b, neg_flux_count = Σ(f_b<0), and soft_regime_weight = floor_frac. Define soft-clipped flux f̃_b = sign(f_b)*max(absf_b, b_b). Build safe log flux coordinates v_b = log10(abs(f̃_b) + eps) and adjacent luptitude-flux colors d̂_ug = v_u-v_g, d̂_gr = v_g-v_r, d̂_ri = v_r-v_i, d̂_iz = v_i-v_z. Compute weighted spectral geometry by fitting v_b against log10(λ_b) with X=[1,t,t^2], t_b in Å = [3551,4686,6165,7481,8931], weights w_b = 1 + 0.5*floor1_b: fit v = β0 + β1*t + β2*t^2 to obtain slope=β1 and curvature=β2. Also compute a pure finite-difference curvature proxy κ = d̂_ug - 2*d̂_gr + d̂_ri and second-order finite-difference on adjacent bands analogously. For each adjacent pair (u,g),(g,r),(r,i),(i,z), define mag_color_ab = m_corr,a - m_corr,b and flux_color_ab = 2.5*(log10(abs(f_a)+eps)-log10(abs(f_b)+eps)); mismatch_ab = mag_color_ab - flux_color_ab, and mismatch̃_ab computed using f̃_b similarly. Add shape-summary descriptors from the signed-safe shares p̃_b = abs(f̃_b)/(Σ abs(f̃)+eps): max_share, entropy_sh = -Σ p̃_b log(p̃_b+eps), l2_share = Σ p̃_b^2, and gini-like concentration = 1 - l2_share; include sign entropy from the binary sign histogram across bands with eps smoothing. Create regime-interacted variants slope_hi = slope*(1-floor_frac), slope_lo = slope*floor_frac, curvature_hi, curvature_lo, and mismatch_hi/mismatch_lo with the same gating so models can separate high-SNR and floor-dominated behavior.
  expected_signal=Replacing raw magnitude-space colors with luptitude-flux shape features and explicitly gating them by floor regime should reduce sensitivity to low-SNR nonlinearity and negative-flux noise, giving more stable SED slope, curvature, and mismatch indicators for borderline objects where STAR, QSO, and GALAXY occupy overlapping photometric regions, which is exactly where balanced_accuracy on rare-class boundaries is usually lost.
  risk=The inversion constants assume SDSS-style asinh calibration; if the dataset uses a different photometric zero-point or nonstandard preprocessing, flux reconstruction can be biased and degrade all dependent features, and many engineered descriptors overlap existing color-based variables so overfitting or collinearity is possible without strong regularization and validation; ratio and log operations also need strict epsilon/clipping to avoid instability when flux values are extremely close to zero.

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