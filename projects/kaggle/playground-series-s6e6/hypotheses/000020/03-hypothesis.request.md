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

Hypothesis ID: 000020

Previous versions:

- rev 1: Survey Depth Limit Margins
  group_name=survey_depth_limit_margins
  family=selection_depth_geometry
  summary=Encode each object's apparent brightness as signed proximity to known SDSS galaxy and quasar spectroscopic depth regimes, so the classifier can recognize whether the source lies in bright galaxy-like, classical quasar-like, faint BOSS-like, or edge-of-survey magnitude territory.
  strategy=Using only the observed ugriz magnitudes, create deterministic signed margin features to published SDSS-style depth thresholds: r - 17.77 for the main galaxy bright limit; i - 15.0, i - 19.1, i - 20.2, i - 20.4, i - 21.3, and i - 22.45 for classical, high-redshift, extended, catalog, and faint quasar regimes; g - 22.0 and r - 21.85 for BOSS-like faint quasar targeting depth; and r - 19.0 and r - 19.5 for deeper low-redshift galaxy extensions. Add interval-membership indicators and signed interval margins for [15.0,19.1], [15.0,20.2], [17.75,22.45] in i, [17.77,19.0] and [19.0,19.5] in r, plus a compact faintness profile consisting of the minimum signed distance to any quasar depth boundary, the minimum signed distance to any galaxy depth boundary, and the difference between those two minima. All magnitudes are finite in the supplied data; if a future value is non-finite, set its derived margins to 0 and add the corresponding interval indicators as 0.
  expected_signal=The labels are likely shaped partly by SDSS selection regimes: nearby main galaxies concentrate around bright r limits, quasars are often selected in deeper i/g/r regimes, and stars can overlap quasar colors but differ in how they populate bright and faint survey-depth boundaries, which can improve balanced accuracy especially around QSO versus STAR and bright GALAXY boundaries.
  risk=These thresholds are survey-procedure proxies rather than intrinsic astrophysical quantities, so they may be redundant with raw magnitudes or brittle if the train/test generation only loosely follows SDSS targeting rules; approximate use of a generic r magnitude for Petrosian-style galaxy limits can also add noisy boundary signals.

- rev 2: SDSS Selection-Depth Boundary Margins
  group_name=survey_depth_limit_margins
  family=selection_depth_geometry
  summary=Represent each object by its signed distance to canonical SDSS galaxy and quasar targeting depth landmarks in ugriz space so the learner can capture how class likelihood shifts when an object sits near historical spectroscopic magnitude boundaries.
  strategy=Compute deterministic signed margin features from available magnitudes: for each threshold L in each relevant band, use margin_b_L = mag_b - L (negative means brighter than the cutoff, positive means fainter). Use galaxy-relevant r-band limits L = 17.77, 19.2, 19.5, and use quasar-relevant i-band limits L = 15.0, 19.1, 20.2, 20.4, 21.3, 22.45 plus g=22.0 and r=21.85. Build interval-structure features with inclusive bounds by computing both membership and distance-to-interval-boundary: in_[a,b] = 1{a <= mag_b <= b}; dist_to_interval_[a,b] = 0 if inside interval, else -(a-mag_b) if mag_b < a, else +(mag_b-b) if mag_b > b. Recommended intervals: i in [15.0,19.1], i in [15.0,20.2], i in [20.2,21.3], i in [21.3,22.45], r in [17.77,19.2], and r in [19.2,19.5]. Add compact regime descriptors: d_qso = min absolute margin over all quasar thresholds (per object), d_gal = min absolute margin over {17.77,19.2,19.5} in r, nearest_qso_band = argmin threshold distance among {i-15.0, i-19.1, i-20.2, i-20.4, i-21.3, i-22.45, g-22.0, r-21.85}, nearest_gal_band = analogous over galaxy r thresholds, and boundary_gap = d_gal - d_qso. Optionally include signs of the nearest qso and nearest gal margins to distinguish brighter-side versus fainter-side excursions. All features are pure deterministic transforms; if a magnitude is missing/non-finite, set all derived features from that band to 0 and associated interval flags for that band to 0.
  expected_signal=These features inject prior structure from known survey selection geometry that is often reflected in class composition: bright flux-limited cuts align strongly with nearby galaxies, very faint point-like objects align with quasar-style targeting depth, and star/qso separators are most ambiguous near shared color regimes; explicit boundary-aware features can therefore improve balanced accuracy at the difficult QSO versus STAR and bright/faint GALAXY transition regions.
  risk=The margins are proxy features tied to historical SDSS targeting rules, so if the train/test generation does not follow those specific boundaries (different tile geometry, photometric calibration, or newer selection logic), they can become noisy, redundant with raw magnitudes and color features, and may overfit to release- or chunk-specific artifacts rather than intrinsic class structure.

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