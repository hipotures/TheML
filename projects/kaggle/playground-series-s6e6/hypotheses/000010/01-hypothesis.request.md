You are designing feature-group hypothesis.

# Task

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

# Prior Root Groups

Use this only to avoid obvious repetition. Do not imitate the best previous group.
- broadband_color_shape (photometric_sed): Transform the five ugriz magnitudes into color slopes and curvature features that describe each object's broadband spectral energy distribution shape.
- redshift_regime_catalog_consistency (astrophysical_consistency): Create redshift-regime features and cross them with the provided catalog tags to capture whether an object looks locally stellar, galactic, or quasar-like.
- galactic_sightline_context (sky_position): Use the object's sky location to express whether the line of sight passes through Milky Way-dominated regions or cleaner extragalactic fields.
- distance_corrected_luminosity_signature (luminosity_distance): Convert apparent broadband brightness into a redshift-adjusted intrinsic-luminosity signature so the model can compare whether an object is physically star-like, galaxy-like, or quasar-like rather than only how bright it looks from Earth.
- catalog_template_residuals (template_mismatch): Quantify how far each object's observed ugriz photometric pattern departs from the typical broadband locus implied by its own catalog tags, so that agreement or disagreement with the stated object template becomes a direct class signal.
- bandpass_break_localization (spectral_break): Capture where along the ordered ugriz band sequence each object shows its dominant broadband discontinuity or turnover, because the location and sharpness of a photometric break can separate smooth stellar spectra from galaxy continuum breaks and re…
- band_order_topology (ordinal_sed): Represent each object by the ordinal topology of its ugriz brightness sequence so the model can distinguish monotone stellar-like continua from single-turnover galaxy-like patterns and more irregular quasar-like reorderings without relying on absolu…
- survey_manifold_rarity (density_rarity): Estimate how typical or unusual each object is inside the unlabeled survey manifold formed by coarse redshift regime, catalog tags, and broad photometric energy allocation, so the model can use rarity itself as a class cue.
- canonical_locus_coordinates (locus_projection): Project each object into fixed SDSS-derived coordinate systems that measure its signed position and normalized departure from the canonical stellar locus and the red-galaxy locus, so the classifier sees whether the photometry looks star-core, galaxy…

# Web search

Live web search is available and should be used before proposing the hypothesis.
Use the `web_search` tool for both stages:
- Search queries: use `web_search` to find feature engineering ideas,
  preprocessing directions, data representations, validation traps, simple
  baseline patterns, public Kaggle discussions/notebooks, package
  documentation, and domain or method background relevant to this task.
- Page retrieval: use `web_search` again to open/fetch the most relevant result
  pages or direct URLs and read their page content.

Analyze the retrieved information before using it. Do not rely only on search
result titles, snippets, or URLs. Convert externally found techniques into one
generalized, leakage-safe, budget-compatible semantic feature-group idea. Do not
copy public solutions directly. If `web_search` is not useful or not available,
continue normally without inventing sources.

# Rules

- Create exactly one new semantic feature group.
- Do not create a full preprocessing pipeline.
- Do not modify existing groups.
- Do not discuss model families, ensembling, validation folds, training time, hyperparameters, or AutoGluon configuration.
- The hypothesis must later generate exactly one feature-group function.
- The group may contain multiple derived columns only when they belong to one coherent semantic idea.
- A hypothesis must represent one homogeneous group-level change. Multiple implementation details are allowed only when they are necessary parts of that same semantic change.
- Use prior root groups only to avoid obvious repetition.
- must use `depends_on: []`.
- Do not output code.

Return valid JSON only. Do not use markdown fences.

Return exactly this JSON object:

{
  "title": "short descriptive title",
  "group_name": "concise_snake_case_group_name",
  "family": "compact feature-family label",
  "summary": "one comprehensive sentence describing the semantic feature-group idea; do not list formulas, exact feature names, or implementation details here",
  "depends_on": [],
  "strategy": "concrete deterministic feature logic, formulas, bins, statistics, and edge handling",
  "expected_signal": "why this group may improve or clarify the metric",
  "risk": "specific overfitting, leakage, cost, redundancy, or instability risk"
}