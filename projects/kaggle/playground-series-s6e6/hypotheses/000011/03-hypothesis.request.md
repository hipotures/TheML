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

Hypothesis ID: 000011

Previous versions:

- rev 1: SDSS LRG Selection Margins
  group_name=lrg_target_cut_margins
  family=target_selection_geometry
  summary=Approximate how close each object lies to the documented SDSS luminous-red-galaxy photometric target-selection surfaces, turning a known red-galaxy color-magnitude selection geometry into direct class-separation signals.
  strategy=Using only raw g, r, and i magnitudes, compute g_minus_r = g - r and r_minus_i = r - i, then form the SDSS LRG rotated color coordinates c_perp = r_minus_i - g_minus_r / 4 - 0.177 and c_parallel = 0.7 * g_minus_r + 1.2 * (r_minus_i - 0.177), following the DR2-and-later LRG target-selection equations documented by SDSS at https://www.sdss4.org/dr17/algorithms/legacy_target_selection/. Emit signed margins to the available photometric parts of Cut I: abs(c_perp) - 0.2, r - 19.2, and r - (13.116 + c_parallel / 0.3); emit signed margins to Cut II: (0.449 - g_minus_r / 6) - c_perp, (1.296 + 0.25 * r_minus_i) - g_minus_r, and r - 19.5; also emit compact pass scores equal to the maximum signed violation for Cut I and Cut II, where negative means all available color-magnitude tests pass. Ignore morphology and surface-brightness requirements because PSF/model and size columns are unavailable; use finite arithmetic directly because there are no missing values, and clip only extreme final margins to a wide fixed range such as [-20, 20] to prevent a few pathological magnitudes from dominating.
  expected_signal=Objects close to or inside the LRG cuts should be strongly galaxy-like in a way that is more specific than generic color slopes, while large positive violations can help distinguish stars and quasars that share some broadband colors but fail the red-galaxy color-magnitude geometry.
  risk=The official cuts use extinction-corrected model/Petrosian magnitudes plus morphology and surface-brightness filters that are not present here, so these proxy margins may be noisy, partly redundant with existing color features, and less stable if the competition data were synthetically generated from a distribution that only loosely follows SDSS targeting rules.

- rev 2: LRG DR2/DR3 Cut-Margin Geometry
  group_name=lrg_target_cut_margins
  family=target_selection_geometry
  summary=This group converts SDSS LRG target-selection geometry into continuous margin features by projecting color information into the classic rotated LRG basis and measuring how strongly each object satisfies or violates each cut-family boundary.
  strategy=For every row, compute gmr = g - r and rmi = r - i. Derive DR2/DR3 rotated LRG coordinates c_perp = rmi - gmr / 4 - 0.177 and c_par = 0.7 * gmr + 1.2 * (rmi - 0.177). Use r as a proxy for r_Petro (because Petrosian magnitudes are not available). Build signed cut margins where negative means pass and positive means violation. Cut I margins: vI1 = r - (13.116 + c_par / 0.3), vI2 = r - 19.2, vI3 = abs(c_perp) - 0.2; score_cutI = max(vI1, vI2, vI3), which is <= 0 when Cut I is passed. Cut II margins: vII1 = (0.449 - gmr / 6) - c_perp, vII2 = (1.296 + 0.25 * rmi) - gmr, vII3 = r - 19.5; score_cutII = max(vII1, vII2, vII3), which is <= 0 when Cut II is passed. Emit score_cutI, score_cutII, and score_lrg_any = clip(min(score_cutI, score_cutII), -20, 20) so negative indicates membership in at least one LRG region and larger positive values indicate stronger rejection distance; optionally emit cut-pass indicators iI = 1 if score_cutI <= 0 else 0 and iII = 1 if score_cutII <= 0 else 0. No imputation is required because the provided columns have zero missing values; only final signed margins are clipped to [-20, 20] to suppress outlier leverage.
  expected_signal=Objects passing either LRG cut are strongly associated with red, extended galaxies, so these geometry margins provide a calibrated, physics-motivated separation signal that can outperform raw colors by explicitly encoding whether the object sits on, inside, or outside known LRG boundaries that also screen out many compact blue contaminants, helping GALAXY vs STAR/QSO discrimination.
  risk=The implementation is a proxy of the true SDSS target logic because r_Petro, extinction-corrected model magnitudes, PSF/model morphology, and surface-brightness filters are not in the table, so margins can be biased and may overfit if the dataset is not SDSS-distributed; fixed numeric boundaries may also be brittle under synthetic color shifts and are partly redundant with base u,g,r,i,z color features.

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