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

Hypothesis ID: 000050

Previous versions:

- rev 1: Main-Sequence Photometric Parallax Plausibility
  group_name=main_sequence_parallax_plausibility
  family=photometric_parallax
  summary=Build features from a calibrated main-sequence distance interpretation of each object's ugriz colors, so the model receives an explicit signal for how physically plausible its photometry is as a star versus a non-stellar galaxy/quasar-like source.
  strategy=Compute x = g - i and clip it into [0.2, 4.0] with two edge flags for lower/upper saturation. Evaluate the main-sequence absolute-magnitude surrogate M_r(x) = -5.06 + 14.32x - 12.97x^2 + 6.127x^3 - 1.267x^4 + 0.0967x^5. Define distance modulus mu = r - M_r(x), implied distance d_pc = 10^((mu + 5)/5), and log10 distance log_d = log10(max(d_pc, 1)). Derive implied absolute magnitudes in all other bands using M_u = u - mu, M_g = g - mu, M_i = i - mu, M_z = z - mu. Partition x into fixed quantile bins (for example 80 equal-count bins), compute per-bin median_mu and MAD_mu from the same pipeline context, then add standardized offset mu_z = (mu - median_mu_bin) / (MAD_mu_bin + 1e-6) and mu_percentile rank within the x-bin. Keep deterministic handling: if a bin has too few samples, back off to neighboring bins with a fallback global median/MAD.
  expected_signal=Objects whose colors and brightness are consistent with the calibrated main-sequence distance mapping should keep coherent mu, d_pc, and implied absolute colors, while galaxies and quasars should produce inconsistent photometric distances or out-of-manifold abs-magnitude signatures, improving class separation where raw colors alone overlap.
  risk=The adopted relation is for main-sequence stars; giants, unresolved blends, very red/blue extremes, metallicity/extinction shifts, and noisy faint measurements can produce spurious distances. Clipping x can blur edge objects, and the per-bin normalization may overcorrect if the global population is highly imbalanced across classes or if sampling density is low in some color tails.

- rev 2: Robust Main-Sequence Parallax Plausibility Residuals
  group_name=main_sequence_parallax_plausibility
  family=photometric_parallax
  summary=Create features that quantify how closely each object's ugriz photometry matches a main-sequence-derived distance model, treating deviations from this physically grounded absolute-magnitude manifold as a direct cue for non-stellar objects.
  strategy=For each row compute x = g - i. Create deterministic edge flags x_below = (x < 0.20), x_above = (x > 4.00), x_turnoff_zone = (x < 0.45) as a turnoff-risk indicator, and define x_c = clip(x, 0.20, 4.00). Compute M_r(x_c) with the calibrated relation M_r = -5.06 + 14.32*x_c - 12.97*x_c^2 + 6.127*x_c^3 - 1.267*x_c^4 + 0.0967*x_c^5, then mu = r - M_r, d_pc = 10^((mu + 5)/5), and log_d = log10(clamp(d_pc, 1, 1e10)). Compute absolute-magnitude counterparts M_u = u - mu, M_g = g - mu, M_i = i - mu, M_z = z - mu and at least these four absolute colors: M_u - M_g, M_g - M_r, M_r - M_i, M_i - M_z (M_r - M_i is included as a compact internal consistency check). From training data only, build 80 equal-frequency bins of x_c, and for each bin store robust statistics (median and MAD_scaled = 1.4826 * median(|x - median(x)|)) for mu, log_d, each absolute color, and each absolute-magnitude delta of interest; require min_bin_n = 500. For scoring, assign each row to a bin; if count < min_bin_n, expand to adjacent bins iteratively (±1, ±2, ±3) and aggregate until min_bin_n is met; if still unavailable, fall back to global training medians/MADs. Then emit standardized residuals z_mu, z_log_d, and z_abscolor_i = (feature - feature_bin_median)/(feature_bin_mad + 1e-6), with all z-scores clipped to [-10, 10], plus within-bin percentile rank of mu. Persist and apply bin edges and statistics from training fold/model fitting only (no refit on validation/test at inference).
  expected_signal=The same observed colors can map to overlapping raw feature values for GALAXY, STAR, and QSO, but stars should remain near a coherent main-sequence absolute-magnitude/distance manifold while galaxies and quasars generally produce implausible implied distances or absolute-color structures, so this residual-based representation provides a stable non-linear separation signal for balanced multiclass classification.
  risk=The mapping is trained on main-sequence assumptions and is less reliable for turnoff stars, giants, unresolved blends, very red/blue extremes, extinction shifts, and low-SNR photometry, so valid-looking stars can be mislabeled as implausible; sparse tail bins can create unstable normalization if fallback logic is weak, and if statistics are learned on full data without proper fold-scoped fitting it can induce subtle selection-shift leakage into validation.

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