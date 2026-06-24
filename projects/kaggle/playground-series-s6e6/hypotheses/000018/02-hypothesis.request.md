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

Hypothesis ID: 000018

Previous versions:

- rev 1: Catalog Tag Color Concordance
  group_name=catalog_tag_color_concordance
  family=tag_photometry_consistency
  summary=Measure whether each object's broadband color temperature agrees with the semantic ordering implied by its provided spectral-type and galaxy-population catalog tags, treating contradictions as informative astrophysical ambiguity.
  strategy=Compute adjacent SDSS colors u-g, g-r, r-i, and i-z, then form a deterministic observed redness score as the mean of clipped color components: clip((u-g+0.5)/3.5,0,1), clip((g-r+0.5)/2.0,0,1), clip((r-i+0.5)/1.5,0,1), and clip((i-z+0.5)/1.5,0,1). Map spectral_type along the hot-to-cool stellar sequence O/B=0.00, A/F=0.33, G/K=0.67, M=1.00, and map galaxy_population as Blue_Cloud=0.00 and Red_Sequence=1.00. Derive signed and absolute differences between observed redness and each mapped expectation, the difference between the two tag expectations, coarse redness bins [0,0.33), [0.33,0.67), [0.67,1], and binary contradiction indicators for hot-tag-red-color, cool-tag-blue-color, red-sequence-blue-color, and blue-cloud-red-color. Clip all color-derived values before scoring; if an unexpected category appears, use neutral 0.5 for its expectation and leave contradiction indicators off. Background used: SDSS documents that ugriz colors are used for stellar and quasar photometric transformations and classification context (https://classic.sdss.org/dr5/algorithms/sdssUBVRITransform.php, https://www.sdss3.org/dr10/algorithms/redshifts.php), while astronomy references describe O/B/A/F/G/K/M as a temperature-color sequence and galaxies as red-sequence versus blue-cloud populations (https://lweb.cfa.harvard.edu/~pberlind/atlas/htmls/note.html, https://academic.oup.com/mnras/article/481/1/1183/5078380).
  expected_signal=The categorical tags are likely strong but noisy summaries; converting them into agreement and contradiction geometry can help separate ordinary stars and galaxies from QSOs or mislabeled-borderline objects whose colors conflict with the simple stellar-temperature or galaxy-population expectation, improving balanced accuracy on minority and ambiguous classes.
  risk=This may be partly redundant with raw categorical features and existing color-derived groups, and fixed color cutoffs could be brittle for unusual redshifted objects whose observed colors violate local stellar or galaxy expectations for legitimate physical reasons.

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