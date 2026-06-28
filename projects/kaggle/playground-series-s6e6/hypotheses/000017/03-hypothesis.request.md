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

Hypothesis ID: 000017

Previous versions:

- rev 1: Cosmological luminosity plausibility
  group_name=redshift_luminosity_plausibility
  family=absolute_photometry
  summary=Transform apparent ugriz brightness through the object's reported redshift into redshift-scaled intrinsic luminosity plausibility signals, so the classifier can distinguish nearby stellar objects from intrinsically luminous galaxies and quasars.
  strategy=For each row, compute a deterministic luminosity-distance proxy from redshift using a fixed flat cosmology approximation with H0=70 km/s/Mpc, Omega_m=0.3, and Omega_lambda=0.7; clip redshift only for distance calculation to z_eff=max(redshift, 1e-4), and retain signed indicators for redshift<=0 and redshift<0.01. Convert each apparent magnitude band m in u,g,r,i,z to a pseudo absolute magnitude M=m-5*log10(D_L_Mpc)-25 without k-correction. Derive coherent summaries over these pseudo absolute magnitudes: central intrinsic brightness level using r and i bands, brightest and faintest pseudo absolute band, band-to-band absolute-magnitude spread, mean pseudo absolute magnitude, and margins to fixed astrophysical luminosity regimes such as stellar-like faint intrinsic scale, normal-galaxy scale, and quasar-like extreme luminosity scale using thresholds around M_r=-10, -18, -23, and -27. For numerical stability, cap luminosity distance and absolute-magnitude outputs at broad finite limits after calculation, and use the same fixed constants for train and test.
  expected_signal=Balanced accuracy may improve because the same observed colors and apparent magnitudes imply very different object classes depending on distance scale: stars should be compatible with near-zero redshift and modest intrinsic luminosity, galaxies occupy intermediate absolute magnitudes, and QSOs are often only plausible when the redshift-scaled luminosity is extreme.
  risk=The reported redshift may already dominate class separation, and pseudo absolute magnitudes without k-correction can be physically crude at high redshift, so the group may be partly redundant with existing redshift and photometric features or unstable for near-zero and slightly negative redshifts.

- rev 2: Redshift-normalized absolute-luminosity regime features
  group_name=redshift_luminosity_plausibility
  family=absolute_photometry
  summary=Convert observed ugriz magnitudes into redshift-scaled intrinsic-brightness proxies and summarize their cross-band behavior to provide physically grounded class separation cues between nearby stars, intermediate-luminosity galaxies, and very luminous quasars.
  strategy=For every row, compute a deterministic luminosity-distance proxy from redshift under a fixed flat cosmology (H0=70 km/s/Mpc, Ωm=0.3, ΩΛ=0.7) with c=299792.458 km/s, using z_eff = max(redshift, 1e-4) for distance math. Calculate D_L via D_C=(c/H0)*∫_0^{z_eff}(1/sqrt(Ωm(1+z)^3+ΩΛ))dz and D_L=(1+z_eff)*D_C, using the same numerical approximation at train and test; retain flags z_nonpos=(redshift<=0), z_low=(redshift<0.01), and z_hi=(redshift>=3). For each band b in {u,g,r,i,z}, derive M_b = m_b - 5*log10(D_L_Mpc*1e6) + 5 (equivalently m_b - 5*log10(D_L_Mpc) - 25), then clip each M_b to [-35, 10]. Build deterministic aggregates: M_mean, M_median, M_iqr, M_sd, M_min, M_max, M_rg = M_r - M_g, M_ri = M_r - M_i, and pairwise absolute spread (max-min). Add threshold-margin and bin features on M_r: Δ_star = M_r - (-10), Δ_midgal = M_r - (-18), Δ_brightgal = M_r - (-23), Δ_qso = M_r - (-27), plus binary flags for intervals M_r <= -27, -27 < M_r <= -23, -23 < M_r <= -18, -18 < M_r <= -10, and M_r > -10. Add scale-preserving cross terms M_r * z_eff and M_ri * z_eff to reduce instability where redshift dominates and to keep relative structure visible at high z.
  expected_signal=The derived features resolve ambiguity where apparent magnitudes alone are class-confounding, because they encode whether an observed brightness is physically plausible at the implied distance; stars are expected to occupy low-redshift, moderate-faint absolute regimes while galaxies occupy a mid-luminosity band and quasars align with extreme-luminosity regime flags, improving balanced-class separability.
  risk=The absolute-magnitude transform is an astrophysical approximation that omits extinction and k-corrections, and negative/near-zero redshift handling is heuristic, so very low-z noise can create artifacts despite clipping and flags; the constructs may also be partially redundant with raw redshift and photometric features, which can overfit if model capacity is not regularized.

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