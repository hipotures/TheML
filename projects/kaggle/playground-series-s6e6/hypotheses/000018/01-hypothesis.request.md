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
- bandpass_break_localization (spectral_break): Capture where along the ordered ugriz band sequence each object shows its dominant broadband discontinuity or turnover, because the location and sharpness of a photometric break can separate smooth stellar spectra from galaxy continuum breaks and re…
- band_order_topology (ordinal_sed): Represent each object by the ordinal topology of its ugriz brightness sequence so the model can distinguish monotone stellar-like continua from single-turnover galaxy-like patterns and more irregular quasar-like reorderings without relying on absolu…
- survey_manifold_rarity (density_rarity): Estimate how typical or unusual each object is inside the unlabeled survey manifold formed by coarse redshift regime, catalog tags, and broad photometric energy allocation, so the model can use rarity itself as a class cue.
- canonical_locus_coordinates (locus_projection): Project each object into fixed SDSS-derived coordinate systems that measure its signed position and normalized departure from the canonical stellar locus and the red-galaxy locus, so the classifier sees whether the photometry looks star-core, galaxy…
- rest_frame_filter_landmarks (spectral_redshift_geometry): Map each object's fixed ugriz filter measurements into rest-frame wavelength space and summarize how the available bands align with major astrophysical spectral landmarks rather than treating the bands as the same physical wavelengths for every reds…
- lrg_target_cut_margins (target_selection_geometry): Approximate how close each object lies to the documented SDSS luminous-red-galaxy photometric target-selection surfaces, turning a known red-galaxy color-magnitude selection geometry into direct class-separation signals.
- faint_blue_galaxy_wedge_margins (blue_galaxy_selection): Represent each object by how deeply it falls inside or outside a deterministic SDSS-style faint blue galaxy photometric selection region, capturing galaxy-like color-magnitude geometry that is distinct from stellar and quasar loci.
- emission_line_bandpass_resonance (line_filter_geometry): Encode whether known galaxy and quasar emission lines should land inside specific SDSS ugriz passbands at the object's redshift and whether the corresponding observed broadband flux shows a local line-like excess rather than a smooth continuum.
- photoz_color_consistency (photometric_redshift): Encode whether an object's observed redshift is consistent with the redshift normally implied by its broadband color location, treating disagreement and local color-redshift ambiguity as class-separating signals.
- redshift_template_domain_margins (redshift_geometry): Encode where each object falls relative to the physically and procedurally distinct redshift domains used for stellar, galaxy, and quasar spectral interpretation, so the classifier sees redshift as class-feasibility geometry rather than only as a ra…
- sky_frame_position_geometry (spatial_geometry): Express each object's celestial position in physically meaningful Galactic and SDSS survey-aligned coordinate frames so the classifier can distinguish true astrophysical sky-population gradients from raw right-ascension and declination artifacts.
- redshift_luminosity_plausibility (absolute_photometry): Transform apparent ugriz brightness through the object's reported redshift into redshift-scaled intrinsic luminosity plausibility signals, so the classifier can distinguish nearby stellar objects from intrinsically luminous galaxies and quasars.

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