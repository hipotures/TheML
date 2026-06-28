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

Hypothesis ID: 000055

Previous versions:

- rev 1: asinh-Jacobian weighted SED shape
  group_name=asinh_jacobian_weighted_shape
  family=flux_uncertainty_geometry
  summary=Model the five-band photometry in the SDSS asinh flux domain and encode continuum-shape features after down-weighting each band by its SDSS asinh low-signal confidence, so color relations from noisy bands do not dominate class boundaries.
  strategy=For each band b in {u,g,r,i,z}, recover normalized flux from the SDSS asinh relation using published softening parameters (b_u=1.4e-10, b_g=0.9e-10, b_r=1.2e-10, b_i=1.8e-10, b_z=7.4e-10): q_b = 2*b_b*sinh(-(ln 10/2.5)*m_b - ln b_b), where m_b is the provided magnitude and q_b = f/f0. Compute low-signal index x_b = q_b/(2b_b) and a continuous reliability score w_b = |x_b|/(1+|x_b|), clipped to [0,1]. Use these as weights to build one-band and cross-band shape features: (1) weighted mean flux mu = Σ(w_b*q_b)/Σw_b, (2) weighted orthogonal polynomial coefficients (slope and curvature) from least-squares fitting q_b-mu against band order index t∈{0,1,2,3,4}, (3) reliability-weighted flux contrasts c_ug, c_gr, c_ri, c_iz where each contrast is scaled by sqrt((w_a w_b)/(w_a+w_b)) to suppress uncertain endpoints, and (4) regime descriptors n_eff=Σw_b, frac_reliable = count(w_b>0.6)/5, weak_band=min(w_b), and reliable_band_count=count(w_b>0.2). If n_eff is near zero, fall back to robustly imputed values from the corresponding global redshift-bin medians and emit a binary low-confidence flag so the model can separate extrapolated values.
  expected_signal=Because SDSS magnitudes are asinh-based and u/z become progressively noisy at survey limits, this group should better preserve physically meaningful continuum differences for faint quasars and galaxies while suppressing spurious color excursions from low-SNR bands that blur class boundaries in raw magnitude-space differences.
  risk=High overlap with existing asinh/flux-domain transforms is possible, so gains may be incremental; the method depends on fixed b coefficients from SDSS documentation and can become unstable for extreme magnitudes, requiring clipping/fallback handling to avoid numerical outliers and risking residual leakage of noise patterns as signal.

- rev 2: Asinh-flux reliability-weighted continuum shape encoding
  group_name=asinh_jacobian_weighted_shape
  family=flux_uncertainty_geometry
  summary=Map ugriz magnitudes into luptitude-derived flux space and derive a low-SNR-aware five-band SED shape representation that emphasizes continuum geometry (level, tilt, curvature, and controlled-color differences) rather than raw magnitude offsets.
  strategy=Use the SDSS luptitude inverse for each band b in {u,g,r,i,z}: q_b = (1/(2*b_b)) * sinh(-(ln(10)/2.5)*m_b - ln(b_b)), with b_b={1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10}; clip the argument of sinh to [-35,35] for numeric safety. Compute reliability weight w_b = |q_b|/(|q_b|+2*b_b), then n_eff = Σ w_b. If n_eff <= 1e-6 (or any non-finite q_b), replace q_b by deterministic train-derived medians for the matching redshift-bin × spectral_type × galaxy_population cell (if empty, then training global medians); set low_confidence=1 else low_confidence=0. Take band order t = {-2,-1,0,1,2} and fit q_b = s0 + s1*t + s2*t^2 by weighted least squares with weights w_b, and emit coefficients s0,s1,s2 plus weighted residual RMS and weighted MAE from fitted values. Build weighted adjacent contrasts c_ug,c_gr,c_ri,c_iz where c_ab = sqrt(w_a*w_b/(w_a+w_b))*(q_a-q_b), and weighted curvature triplets d1 = sqrt(w_u*w_g/(w_u+w_g))*(q_u-2*q_g+q_r), d2 = sqrt(w_g*w_r/(w_g+w_r))*(q_g-2*q_r+q_i), d3 = sqrt(w_r*w_i/(w_r+w_i))*(q_r-2*q_i+q_z). Add reliability descriptors frac_hi=count(w_b>0.6)/5, frac_mid=count(w_b>0.2)/5, min_w=min(w_b), max_w=max(w_b), and mean_adj_gap = mean(|q_b-q_{b+1}| for adjacent pairs where both w_b,w_{b+1} > 0.1).
  expected_signal=The luptitude-to-flux map preserves ordering at low or negative flux while allowing finite operations in low-SNR tails, and the Jacobian-based reliability weights prevent unstable bands from dominating color geometry; polynomial-plus-contrast shape descriptors should separate quasar UV excess and line-affected slopes from stellar/galaxy continua more cleanly in classes with overlapping raw magnitudes.
  risk=Mis-calibration risk is non-trivial: if the actual photometric release uses different softening constants or implicit zero-points, all derived q_b and derived geometry can be biased; aggressive clipping and fallback imputation can also compress rare but informative outliers, and any fallback statistics must be computed strictly from training-only data to avoid leakage in validation.

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