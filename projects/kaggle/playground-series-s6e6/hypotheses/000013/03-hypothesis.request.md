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

Hypothesis ID: 000013

Previous versions:

- rev 1: Redshifted Emission-Line Band Resonance
  group_name=emission_line_bandpass_resonance
  family=line_filter_geometry
  summary=Encode whether known galaxy and quasar emission lines should land inside specific SDSS ugriz passbands at the object's redshift and whether the corresponding observed broadband flux shows a local line-like excess rather than a smooth continuum.
  strategy=Use fixed SDSS filter central wavelengths u=3551, g=4686, r=6166, i=7480, z=8932 Angstrom from https://www.sdss4.org/instruments/camera/ and fixed common SDSS spectral-line wavelengths/weights from https://classic.sdss.org/dr5/algorithms/linestable.php. Convert ugriz magnitudes to relative fluxes f_b=10^(-0.4*m_b), then normalize by the median finite band flux per row. For each row and each selected emission line, compute observed wavelength lambda_obs=lambda_rest*(1+redshift). For each band, compute a resonance kernel exp(-0.5*(log(lambda_obs/lambda_band)/0.08)^2), set kernels to 0 when redshift<0 or lambda_obs is outside 3000-10000 Angstrom, and weight by separate fixed galaxy-line and quasar-line strengths. Create coherent summaries only: total quasar resonance, total galaxy resonance, maximum single-band resonance, nearest resonant band index, quasar-minus-galaxy resonance, resonance-weighted observed flux excess, and resonance-weighted signed excess where band excess is log(f_band) minus a linear interpolation of neighboring log fluxes across wavelength; for edge bands use the closest two interior bands as the continuum estimate and clip nonpositive fluxes with a tiny epsilon. Optionally include coarse bins for whether the strongest resonance falls in u/g/r/i/z or outside coverage, but do not learn any statistics from labels or test targets.
  expected_signal=Quasars and emission-line galaxies can have strong spectral lines that shift through SDSS filters with redshift and alter broadband colors; SDSS documentation notes quasar selection depends on ugri/griz color behavior, while filter-gap studies show strong lines such as MgII, CIV, and Ly-alpha can significantly affect colors, so this group gives the classifier a physically aligned explanation for redshift-dependent broadband bumps that smooth color or luminosity features may blur.
  risk=This may be partially redundant with existing color, spectral-break, and rest-frame landmark features, and the line-band kernels are an approximation because true filter throughput widths, line equivalent widths, and photometric errors are unavailable; if the provided redshift is noisy for stars or catalog-derived in a class-dependent way, resonance features may add unstable or overly direct class cues.

- rev 2: Redshifted Emission-Line Passband Resonance Contrast
  group_name=emission_line_bandpass_resonance
  family=line_filter_geometry
  summary=Create features that capture how redshifted strong galaxy and quasar emission-line systems align with SDSS ugri z passbands and whether those aligned bands show local broadband flux departures from adjacent continuum, preserving an explicit physical interpretation of line-driven color shifts.
  strategy=For each row, compute flux proxies f_b = max(10^{-0.4 m_b}, 1e-12) for b in {u,g,r,i,z}, then set y_b = log(f_b) and center each row by y'_b = y_b - median(y_u, y_g, y_r, y_i, y_z) to remove distance scale effects. Use fixed SDSS effective passband centers λ_b = {3551,4686,6166,7480,8931} Å. Use fixed SDSS spectro1d line wavelengths in vacuum with separate class-specific importance sets: quasar lines (non-zero weight): {Lyα 1215.24, N V 1240.81, Si IV+O IV 1399.8, C IV 1549.48, C III]1908.734, Mg II 2799.117, [O II] 3727.092, Hβ 4862.68, [O III] 5008.240, Hα 6564.614}, galaxy lines (non-zero weight): {[O II] 3727.092, Hδ 4102.89, Hγ 4341.68, Hβ 4862.68, [O III] 4960.295, [O III] 5008.240, Hα 6564.614, [N II] 6585.27, [S II] 6718.29, [S II] 6732.67}. For each line λ0 with redshift z, compute λ_obs = λ0(1+z); if z < 0 or λ_obs not in [3000,10000], set that line’s contribution to zero for all bands. Otherwise, for each band define a resonance kernel k_{b,l} = exp[-0.5*(ln(λ_obs/λ_b)/σ_b)^2] with fixed σ_b = 0.06 in log-wavelength; optionally zero out k_{b,l} < 1e-4 for numerical sparsity. Compute local continuum residuals e_b on the row-centered log-flux scale: for interior bands interpolate linearly in log λ using neighboring points, and for edges use the closest two interior bands (u uses g and r; g uses u and r; r uses g and i; i uses r and z; z uses i and r). Define e_b = y'_b - ŷ_b. Build deterministic class summaries: T_qso = Σ_l w_qso(l) Σ_b k_{b,l}, T_gal = Σ_l w_gal(l) Σ_b k_{b,l}; E_qso = Σ_l w_qso(l) Σ_b k_{b,l} * e_b, E_gal = Σ_l w_gal(l) Σ_b k_{b,l} * e_b; peak band indices B_qso = argmax_b Σ_l w_qso(l)k_{b,l}, B_gal = argmax_b Σ_l w_gal(l)k_{b,l}; peak strengths S_qso = max_b Σ_l w_qso(l)k_{b,l}, S_gal = max_b Σ_l w_gal(l)k_{b,l}; margin features M_t = T_qso - T_gal, M_e = E_qso - E_gal, M_s = S_qso - S_gal. Add four-way binning for each class of strongest-resonant band among u,g,r,i,z and an outside-coverage bin when all k_{b,l}=0. All quantities are computed purely from input columns without target-derived fitting.
  expected_signal=Redshifted emission lines produce class- and redshift-dependent bumps that move through SDSS filters in a physically deterministic way; this gives the classifier a stable signal where quasars and star-forming/AGN-like galaxies depart from smooth continuum colors, especially in overlap regions where raw color features can blur distinctions.
  risk=This group may over-trust the provided redshift and can create spurious signals when redshift is noisy or systematically wrong for some classes, especially for stars with catalog-like values; fixed central wavelengths and Gaussian log-kernels approximate real throughput and line profiles and ignore equivalent width variation, while added class-specific resonance features may duplicate existing color/continuum features and encourage redshift-bin overfitting without regularization.

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