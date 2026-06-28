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

Hypothesis ID: 000010

Previous versions:

- rev 1: Rest-frame filter landmark alignment
  group_name=rest_frame_filter_landmarks
  family=spectral_redshift_geometry
  summary=Map each object's fixed ugriz filter measurements into rest-frame wavelength space and summarize how the available bands align with major astrophysical spectral landmarks rather than treating the bands as the same physical wavelengths for every redshift.
  strategy=Use fixed SDSS effective wavelengths in Angstroms for u,g,r,i,z as 3543, 4770, 6231, 7625, and 9134, compute rest wavelengths as lambda_rest=lambda_obs/(1+z_safe) where z_safe=max(redshift,0), and derive one coherent set of deterministic landmark-alignment features: counts of bands with rest wavelengths below 1216, 2500, 4000, between 4000 and 7000, and above 7000; signed log-distance from each rest wavelength to 1216, 2500, and 4000; the nearest observed band index to each landmark; the magnitude at that nearest band; adjacent color contrast around that landmark when both neighboring bands exist; and binary flags for whether each landmark lies inside the observed rest-frame wavelength span. For negative redshift values, use z_safe=0 for wavelength mapping and add a negative-redshift flag; when a landmark falls outside the ugriz span, set nearest-band summaries to the closest endpoint and set the inside-span flag to 0.
  expected_signal=The same observed color pattern can correspond to different physical continuum regions at different redshifts, so landmark alignment may help separate near-zero-redshift stars from galaxies with 4000 Angstrom break structure and QSOs whose ultraviolet or Lyman-region behavior shifts through the optical bands, improving minority-class balanced accuracy.
  risk=This may partially overlap with existing redshift-regime and spectral-break ideas, and hardcoded wavelength landmarks can be brittle for noisy or synthetic photometry if the catalog redshift is unreliable or if the labels were generated from simpler rules.

- rev 2: Rest-frame landmark alignment with boundary-aware passband context
  group_name=rest_frame_filter_landmarks
  family=spectral_redshift_geometry
  summary=Construct a compact representation that expresses how each object’s ugriz measurements shift into rest-frame wavelength space and where key spectral landmarks fall relative to that shifted coverage, turning raw observed colors into physically aligned redshift-aware descriptors.
  strategy=Use fixed SDSS effective wavelengths λobs = [3543, 4770, 6231, 7625, 9134] Å for u,g,r,i,z. For each row define z_safe = max(redshift, 0), z_neg_flag = I(redshift < 0), and compute λrest_j = λobs_j / (1 + z_safe) for j=1..5 (with band order preserved). Compute span_min = min(λrest), span_max = max(λrest), and span_width = span_max - span_min. Build ordered occupancy counts over deterministic rest-frame bins that partition the domain: n_lt_912, n_912_1216, n_1216_2500, n_2500_3646, n_3646_4000, n_4000_7000, n_gt_7000, plus a consistency feature total_bands = 5; add a normalized entropy over these bin counts (only nonzero bins) to capture how broadly a spectrum probes rest-frame regions. For each landmark L in {912, 1216, 2500, 3646, 4000, 7000} compute: in_span = I(span_min <= L <= span_max), below_span = I(L < span_min), above_span = I(L > span_max), and nearest_band_idx k = argmin_j |λrest_j - L| (tie resolved to smaller j). Define signed_log_dist_L = sign(λrest_k - L) * log1p(|λrest_k - L|), and abs_log_dist_L = log1p(|λrest_k - L|). Return nearest_band_mag_L = observed magnitude at k and nearest_band_id_L = k. If k is interior and both adjacent bands straddle L, return left_right_delta_L = (mag_k - mag_{k-1}) and (mag_{k+1} - mag_k); if exactly one neighbor exists inside the sampled set, return only the available one. If L is strictly between λrest_k and λrest_{k+1}, compute linear_predict_at_L from mags k and k+1 and return landmark_residual_L = mag_k - linear_predict_at_L (same formula with k and k-1 when interpolation from below is used), else set landmark_residual_L = 0 when outside span. For outside-span landmarks, clamp endpoint_band = 1 or 5 (nearest endpoint), set outside_distance_L = log1p(|L - clamp(L, span_min, span_max)|), and expose outside_side_L = I(L < span_min) else 1 for above, enabling explicit handling of hard saturation at edges.
  expected_signal=By explicitly encoding whether physically meaningful breaks and UV discontinuities are entering, inside, or missed by the observed passbands, the model can discriminate classes that share coarse colors but differ by redshifted spectral content, especially separating low-redshift stellar continua from galaxies with a 4000 Å/Balmer region signature and quasars with high-redshift Lyman-region behavior.
  risk=This family depends on redshift precision and fixed landmark positions, so noisy or systematically biased redshifts, wide broad-band photometry, and coarse 5-point sampling can produce unstable hard-threshold decisions and duplicate information already present in raw colors and redshift, increasing overfitting risk on minority classes without strong additional signal.

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