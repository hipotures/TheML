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

Hypothesis ID: 000047

Previous versions:

- rev 1: Flux-Allocation Entropy and Concentration in ugriz
  group_name=flux_allocation_entropy
  family=spectral_energy_allocation
  summary=Represent each object as a normalized five-band flux-allocation profile and model its information concentration and wavelength spread to separate sources by how their intrinsic energy is distributed across the ugriz SED instead of using raw magnitudes or pairwise colors alone.
  strategy=For each row, convert magnitudes to linear flux proxies f_u, f_g, f_r, f_i, f_z = 10^(-0.4*m). Add a tiny floor eps = 1e-12 * max(1, median(f_u..f_z)) to each flux, then normalize p_k = f_k / sum(f_k) so p lies on a simplex. Create deterministic shape descriptors: Shannon entropy H = -sum(p_k*ln p_k), normalized entropy Hn = H / ln(5), Gini-style concentration G = 1 - sum(p_k^2), Simpson concentration S = sum(p_k^2), and top-bucket ratio p_max / (p_second + eps). Add wavelength-location descriptors using fixed effective SDSS central wavelengths λ = [3551,4686,6165,7481,8931] Å: μ = sum(p_k*ln λ_k), v = sum(p_k*(ln λ_k-μ)^2), and 3rd/4th centered moments m3 = sum(p_k*(ln λ_k-μ)^3), m4 = sum(p_k*(ln λ_k-μ)^4). Compute the same set once more using rest-frame-adjusted axis values ln(λ_k/(1+z+1e-6)) to capture coarse redshift-aware redistribution. Add robust contrast proxies using cumulative shares: blue_flux = p_u+p_g, red_flux = p_i+p_z, center_flux = p_r, and differences blue_flux-red_flux, p_u-p_i, p_g-p_r, p_z-p_g. For numerical safety, clip each moment to finite range by replacing NaN/inf with training-stable neutral values before downstream use.
  expected_signal=The feature group captures global SED allocation patterns that are invariant to distance and overall brightness and can reveal class-specific energy concentration behavior (e.g., red-dominated, peaked profiles versus flatter, broader profiles), adding orthogonal information to color-slope/curvature features and improving separation where quasar/galaxy/star colors partially overlap.
  risk=These descriptors can be highly correlated with existing shape and color-transformed groups, and concentration metrics may become noisy for low-SNR points, so gains may be modest or unstable in the faintest or photometrically problematic regions.

- rev 2: Rest-frame-aware Flux Allocation and Concentration
  group_name=flux_allocation_entropy
  family=spectral_energy_allocation
  summary=Represent each object as a normalized five-band energy-share profile and summarize how compact, uneven, and wavelength-biased that profile is, with and without approximate rest-frame correction, to capture class-specific SED shape signatures that are insensitive to total brightness scale.
  strategy=For each row, start from magnitudes m_u,m_g,m_r,m_i,m_z and build a stable share vector p by log-domain softmax: l_k=-0.4*m_k, l_k:=l_k-max(l), q_k=exp(ln(10)*l_k), and p_k=(q_k+α)/(Σ q_j+5α) with α=1e-12. If any row contains non-finite values or if Σ q_k<=0, force p=[0.2,0.2,0.2,0.2,0.2] and set an invalid-share flag for downstream audit. Sortability, concentration, and contrast features are then computed from p only: entropy H=-Σ p_k ln p_k, normalized entropy Hn=H/ln(5), Simpson concentration Sc=Σ p_k^2, Gini-style concentration G=1-Sc, effective band count Neff=1/(Sc+1e-12), largest-share ratio R12=p_(1)/ (p_(2)+1e-12), top-share gap p_(1)-p_(2), cumulative buckets B=p_u+p_g, M=p_g+p_r, R=p_i+p_z, and differences ΔBR=B-R, ΔUI=p_u-p_i, ΔGR=p_g-p_r, ΔZI=p_z-p_g. Add wavelength-distribution shape statistics using canonical SDSS centers λ=[355.1,468.6,616.5,748.1,893.1] nm and x=ln λ: μ=Σ p_k x_k, v=Σ p_k (x_k-μ)^2, σ=max(√v,1e-12), skew=Σ p_k (x_k-μ)^3/σ^3, kurtosis=kurt=Σ p_k (x_k-μ)^4/σ^4-3, plus centroid ratio μ/μ_max_band and scale ratio v/( (max(x)-min(x))^2 ). For redshift-aware descriptors, compute λ_rf_k=λ_k/(1+z_clamp), x_rf=ln λ_rf_k with z_clamp=clip(z,-0.99,20), then μ_rf,v_rf,skew_rf,kurt_rf analogously, and include Δμ=μ-μ_rf and Δv=v-v_rf. Replace any NaN/inf from arithmetic with 0 and clip skew and kurtosis to [-8,8] before use.
  expected_signal=A flux-share formulation removes overall magnitude scaling while preserving the relative spectral-shape distribution, which can distinguish galaxy/star/QSO populations that overlap in raw magnitudes but differ in how optical energy is concentrated across bands; coupling these concentration/asymmetry descriptors with coarse rest-frame shift-aware moments adds discriminatory structure where redshifted features blur absolute-color rules and therefore should improve balanced accuracy.
  risk=These engineered descriptors are likely correlated with existing color and slope-based groups, can become unstable in extreme low-SNR or near-degenerate-band regimes despite the numeric guards, and the rest-frame moment terms can partially encode redshift-class priors, so overfitting and weak generalization are possible without validation controls.

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