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

Hypothesis ID: 000049

Previous versions:

- rev 1: Asinh-band censoring and dropout regime geometry
  group_name=asinh_censoring_regime_geometry
  family=detection_regime_geometry
  summary=Represent each object by whether each ugriz measurement is in a reliable detection regime or in the SDSS low-signal asinh regime, and encode the resulting band-visibility and dropout geometry so class-separating patterns tied to faint-band suppression are made explicit.
  strategy=For each band b in {u,g,r,i,z}, invert the SDSS asinh magnitude m_b using the published relation m = -2.5/ln(10) * [asinh((f/f0)/(2 b_b)) + ln b_b], where b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10, then compute f_b/f0 = 2 b_b sinh(-(m_b * ln(10)/2.5) - ln b_b) and q_b = (f_b/f0)/(2 b_b). Create per-band regime features: detect_10b_b = 1[f_b/f0 > 10 b_b] (equiv m_b < m10b_b), neg_b = 1[f_b/f0 < 0], and delta_b = clip((m_b - m10b_b)/(m0_b - m10b_b), 0, 1), with m10b_b={22.12,22.60,22.29,21.85,20.32} and m0_b={24.63,25.11,24.80,24.36,22.83} for u,g,r,i,z respectively. Add joint descriptors: count of detected bands, count of non-detected redder-than-blue bands, longest contiguous missing-band run in wavelength order, blue-dropout pattern flags (e.g., u-drop, g-drop, u+g-drop while r/i/z detected), detected-band-only color slopes and curvatures from f_b/f0, and mass-center entropy of abs(f_b/f0) after zeroing not-detected bands; clip f_b/f0 and delta_b to training quantile tails to avoid single-object blow-up and keep edge handling stable when very bright or extreme values appear.
  expected_signal=Because asinh magnitudes compress faint flux, raw magnitude differences can hide true non-detections; regime-aware features expose low-SNR structure and dropout signatures (especially blue-band suppression at specific redshifts and low-confidence detections) that are known to be relevant for distinguishing stars from galaxies and quasars, which should improve class separation in the ambiguous boundary regions measured by balanced accuracy.
  risk=Reliance on published softening constants and threshold magnitudes creates calibration sensitivity if the competition data were reduced with different photometric conventions, and regime flags are intrinsically noisy near boundaries, so they can inject instability or duplicate existing flux-asinh or depth-based features unless regularized.

- rev 2: Asinh regime topology with censored-band geometry
  group_name=asinh_censoring_regime_geometry
  family=detection_regime_geometry
  summary=Represent each source by a wavelength-ordered topology of SDSS asinh-photometric detection regimes that explicitly separates high-SNR bands, low-SNR asinh bands, and negative/very-faint bands, and summarizes how the pattern of censored measurements reshapes SED shape descriptors used for class separation.
  strategy=For each band b in {u,g,r,i,z}, invert the SDSS asinh system using m_b = -(2.5/ln10)*[asinh((f_b/f0)/(2*b_b)) + ln(b_b)], therefore x_b = f_b/f0 = 2*b_b*sinh(-(ln(10)/2.5)*m_b - ln(b_b)). Use b_b = {u:1.4e-10, g:0.9e-10, r:1.2e-10, i:1.8e-10, z:7.4e-10}. Winsorize x_b to training-only quantiles [q0.1,q99.9] per band before any slope/diversity calculations. Define fixed SDSS depth anchors m10_b={22.12,22.60,22.29,21.85,20.32} and m0_b={24.63,25.11,24.80,24.36,22.83}. Then create three deterministic regime states per band: R_b=2 if m_b<=m10_b (high-SNR/Pogson-like), R_b=1 if m10_b<m_b<=m0_b (low-SNR asinh regime), and R_b=0 if m_b>m0_b (negative/near-or-below noise regime). Add a smooth boundary confidence c_b = sigmoid((m10_b - m_b)/0.10) and a censor-depth ratio d_b = clip((m_b - m10_b)/(m0_b - m10_b), 0, 1) so threshold uncertainty is represented continuously. Build a single topology block in band order u->z from R_b: counts of {R=2, R=1, R=0}, number of late-dropouts conditioned on earlier detections (e.g., u-drop and g-drop cases, and u+g dropout while any of r/i/z are detected), longest contiguous run of R<=1 and longest contiguous run of R==0, first detected band index, last detected band index, and counts of non-detected redder-than-blue bands. For shape features, compute color geometry only on bands with R>0 using x_b: adjacent log-slope s_k=log10((x_{k+1}+ε)/(x_k+ε)), adjacent curvature κ_k=s_{k+1}-s_k, and a mass-centered summary of valid flux allocation where p_b = |x_b| / Σ_{R>0}|x_b| if Σ|x_b|>0 else 0, plus normalized entropy H= -Σ p_b log(p_b) and centroid μ=Σ p_b*position_b; set shape terms to 0 when fewer than two valid adjacent bands exist. Handle edge cases with ε=1e-12 in logs and always use train-only statistics for any winsorization so there is no target leakage.
  expected_signal=This makes low-SNR and dropout behavior a first-order object descriptor rather than an implicit side-effect of raw magnitudes, so the model can better capture blue/UV suppression and wavelength-localized censoring patterns consistent with quasar/galaxy/stellar boundaries at the faint end where class confusion is highest under balanced accuracy.
  risk=Hard-coded SDSS asinh softening and depth anchors can be miscalibrated if preprocessing conventions differ from the source convention, and regime boundaries are inherently noisy near m10 and m0; adding hard one-hot states can duplicate information from existing asinh/flux features and amplify instability unless aggressively regularized and kept to a compact topology block.

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