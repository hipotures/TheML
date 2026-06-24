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

Hypothesis ID: 000030

Previous versions:

- rev 1: SDSS quasar target-surface geometry
  group_name=sdss_quasar_targeting_surface
  family=astrophysical_selection_geometry
  summary=Reconstruct the legacy SDSS quasar-selection decision geometry in ugriz color space by encoding both locus-outlier behavior in the ugri and griz photometric cubes and the known inclusion/exclusion region logic that controls how quasars were accepted near the stellar-locus crossings, so class boundaries are represented as geometry against documented selection surfaces.
  strategy=Create colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. In cube A=(c1,c2,c3), bin objects by c3 (for example 0.08-mag bins between global 5th and 95th percentiles, with endpoints clamped), and in cube B=(c2,c3,c4), bin by c2 with the same style. For each occupied bin compute robust centroid μ and robust scale s (MAD, clamped minimum 0.01), then compute a local 1D ridge direction t via principal component on standardized colors. For each object define orthogonal residual distances d_A and d_B as Mahalanobis-like distances to the ridge in each cube, d = sqrt(sum((r/s)^2)), where r is the perpendicular residual after removing projection onto t; clip d to [0,20] for stability. Build candidate-surface scores: S_A=clip((d_A-4)/2,0,1) and S_B=clip((d_B-4)/2,0,1), reflecting the historical 4σ locus-outlier logic. Add inclusion-surface terms from documented legacy cuts: MZ=1 if 0.65<=c1<=1.5 and 0<=c2<=0.2 (mid-z bridge support), UVX=1 if c1<=0.6 (optional with i-band fading weight w=clip((20.0-i)/1.0,0,1)), HIZ=1 if redshift>3 and i<20 and c1>1.5, and BLUE_REJECT=1 if c1<0.9 and c2<0.8 and i>19 (negative evidence). Add exclusion-surface penalties from known color-rejection boxes: WD_REJ for approximately -0.8<g-r<-0.2, -0.6<r-i<-0.2, -1.0<i-z<0; MWD_REJ for 0.0<g-r<1.6 and 0.6<r-i<2.0; A_REJ for 0.9<u-g<1.5 and -0.35<g-r<0. These are signed margin features so the group remains continuous near boundaries. Combine scores with redshift gating (e.g., weight MZ for 2.3<=redshift<=3.2 and HIZ for redshift>=3) without using any target labels.
  expected_signal=Quasars are historically selected as broad outliers from the stellar locus with specific redshift-dependent exception surfaces, so these geometry features directly encode decision boundaries and ambiguities (especially the 2.7-2.8 crossing and high-redshift color regions), providing separability signals where raw color features alone are less stable and helping reduce confusion with stars and compact galaxies.
  risk=The constants come from SDSS historical selection logic and may be mildly misaligned with this dataset’s exact photometric calibration and redshift uncertainties, especially for faint objects, and hard cut-derived surfaces can be brittle in out-of-distribution color tails; if the underlying survey systematics differ, these features may overemphasize legacy-selection artifacts and reduce robustness.

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