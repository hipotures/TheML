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

Hypothesis ID: 000019

Previous versions:

- rev 1: Observed Continuum Shape Moments
  group_name=observed_sed_continuum_moments
  family=spectral_shape
  summary=Represent each object's observed ugriz photometry as a low-order broadband continuum shape so the model can separate smooth stellar spectra, galaxy break-like curvature, and quasar power-law or line-disturbed continua.
  strategy=Use fixed SDSS effective wavelengths for u,g,r,i,z, for example 3543, 4770, 6231, 7625, and 9134 Angstrom, convert each magnitude to relative log-flux as -0.4*magnitude, subtract the object's five-band mean log-flux to remove absolute brightness, and fit an equal-weight quadratic curve against centered log-wavelength. Emit the fitted linear slope, quadratic curvature, residual RMS, maximum absolute residual, blue-side slope from u-g-r, red-side slope from r-i-z, red-minus-blue slope break, and signed residuals at the central and edge bands after the quadratic fit. Since the supplied data have no missing values, only guard non-finite intermediate values by replacing them with zero after deterministic clipping of input magnitudes to a broad physical range such as [-1, 35].
  expected_signal=SDSS class separation is strongly expressed in multi-band color space: stars tend to occupy smooth nearly one-dimensional continua, galaxies often show stronger broad curvature from population breaks, and QSOs can look closer to power laws with localized broadband excesses, so compact continuum-shape moments may improve balanced accuracy without copying a full locus model.
  risk=The group is partially redundant with raw magnitudes and simple colors, and the polynomial residual terms may learn survey-specific calibration artifacts or rare magnitude pathologies if extreme photometric values are not clipped consistently.

- rev 2: Observed SED Continuum Moments with Robust Residual Diagnostics
  group_name=observed_sed_continuum_moments
  family=spectral_shape
  summary=Create a compact shape descriptor of the five-band observed spectral energy distribution that captures large-scale continuum slope and curvature together with localized deviations from a smooth trend while removing object-by-object brightness scaling.
  strategy=For each object, first enforce deterministic validity checks on u,g,r,i,z by replacing any non-finite value with the training-set per-band median and clamping each remaining magnitude to train-derived bandwise 0.1%/99.9% quantiles (with an explicit fallback physical cap of [-1,35] if quantiles are unavailable). Convert to relative log-flux as l_b=-0.4*m_b, then remove absolute level by centering: l'_b=l_b-mean_b(l_b). Use fixed SDSS pivot wavelengths λ=[3543,4770,6231,7625,9134] Å, define t_b=log10(λ_b)-mean_b(log10 λ_b), and fit a quadratic y(t)=β0+β1 t+β2 t^2 to {t_b,l'_b} by closed-form OLS over the five bands. Compute residuals e_b=l'_b-y(t_b), and emit deterministic residual descriptors: β1, β2, sqrt(mean(e_b^2)), mean(|e_b|), max(|e_b|), signed residuals at the edge bands e_u,e_z, and a signed break index Δblue−red=((l'_u-l'_g)/(t_u-t_g))-(l'_r-l'_i)/(t_r-t_i). If X'X is singular/ill-conditioned or any intermediate becomes non-finite, emit zeros for β1,β2 and residual features for that row to ensure stable behavior.
  expected_signal=The class priors differ primarily in broadband continuum morphology: stellar loci are usually smoothly varying and temperature-driven, galaxies often show stronger curvature from broad stellar-population breaks, and quasars are closer to power-law-like continua with characteristic local departures; fitting a normalized quadratic trajectory plus residual structure should isolate these class-separating traits more directly than raw magnitudes alone and can improve balanced-accuracy by reducing dependence on absolute flux scale.
  risk=The shape descriptors are strongly correlated with simple colors and can still learn survey-specific photometric systematics (extinction, calibration stripes, saturated/edge cases), and a quadratic model over five points may miss fine line-driven features that matter for some quasars, so aggressive clipping or ill-conditioned fallback can suppress real signal on outlier objects while reducing instability.

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