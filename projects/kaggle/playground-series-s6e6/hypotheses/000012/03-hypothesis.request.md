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

Hypothesis ID: 000012

Previous versions:

- rev 1: Faint Blue Galaxy Wedge Margins
  group_name=faint_blue_galaxy_wedge_margins
  family=blue_galaxy_selection
  summary=Represent each object by how deeply it falls inside or outside a deterministic SDSS-style faint blue galaxy photometric selection region, capturing galaxy-like color-magnitude geometry that is distinct from stellar and quasar loci.
  strategy=Compute ug=u-g, gr=g-r, ri=r-i, iz=i-z, then form signed margins for the faint-blue galaxy wedge: gr-(0.40+0.6*ug), (1.7-0.1*ug)-gr, ug+0.5, 3.0-ug, gr-0.0, 1.8-gr, ri+0.5, 1.5-ri, iz+1.0, 1.5-iz, u-18.0, 24.0-u, g-18.0, 21.5-g, r-17.8, 19.5-r, i-16.5, 20.5-i, z-16.0, and 20.0-z; derive one overall inside score as the minimum of all margins, separate minimum color-only and magnitude-only margins, a violation count over margins below zero, and the SDSS sampling-intensity proxy exp(0.1411*(gr-(0.40+0.6*ug))) clipped to a finite upper bound such as 10; all inputs are finite in the provided data, but any nonfinite intermediate should be replaced with 0 after margin construction.
  expected_signal=Objects satisfying this blue-galaxy color-magnitude wedge should be more likely to be GALAXY even when their colors overlap UV-excess quasars or hot stars, improving class balance by adding a targeted galaxy-specific contrast not supplied by generic broadband colors alone.
  risk=The wedge comes from a survey targeting rule rather than this exact dataset, so it may be redundant with raw magnitudes, colors, and galaxy_population, and hard boundaries could be unstable for objects near the cuts.

- rev 2: Robust Soft-Wedge Margin Stack
  group_name=faint_blue_galaxy_wedge_margins
  family=blue_galaxy_selection
  summary=This feature group encodes how strongly each object lies inside, near, or outside a deterministic faint-blue color-magnitude envelope by combining interior depth and violation-count signals so class boundaries between galaxies, stars, and quasars are expressed more explicitly than raw broadband features alone.
  strategy=For each object compute ug = u - g, gr = g - r, ri = r - i, iz = i - z, and clamp ug to [-0.60, 3.20] only for line-based wedge calculations. Define signed color margins m_c = [m1=gr-(0.40+0.60*ug), m2=(1.70-0.10*ug)-gr, m3=ug+0.50, m4=3.00-ug, m5=ri+0.50, m6=1.80-ri, m7=iz+1.00, m8=1.50-iz] and magnitude margins m_m = [n1=u-18.0, n2=24.0-u, n3=g-18.0, n4=21.5-g, n5=r-17.8, n6=19.5-r, n7=i-16.5, n8=20.5-i, n9=z-16.0, n10=20.0-z]. Replace any non-finite derived value with 0, then clip every margin to [-10, 10]. Normalize color margins by [0.35,0.35,0.50,0.50,0.35,0.35,0.35,0.35] and magnitude margins by [3.0,3.0,2.0,2.0,1.7,1.7,1.4,1.4,1.4,1.4] to equalize scale. Create deterministic features: color_depth_min=min(m_c_norm), color_depth_q20=20th percentile of m_c_norm, color_depth_soft = -logsumexp(-8*m_c_norm)/8, color_viol=sum(m_c_norm<0). For magnitudes: mag_depth_min=min(m_m_norm), mag_depth_q20=20th percentile of m_m_norm, mag_depth_soft = -logsumexp(-8*m_m_norm)/8, mag_viol=sum(m_m_norm<0). Create a combined score = 0.60*color_depth_soft + 0.40*mag_depth_soft - 0.25*color_viol - 0.10*mag_viol. Add a bounded sampling-intensity term s = clip(exp(0.1411*(gr-(0.40+0.60*ug))), 0, 10) and include both a color-only version of score and the full score + s as final engineered signals.
  expected_signal=Replacing single hard boundaries with clipped, normalized depth/violation statistics stabilizes the wedge signal for near-boundary objects, preserves a graded notion of membership, and helps recover galaxy-like patterns where stars and quasars leak into similar broad color space by adding explicit magnitude-constrained support information.
  risk=The wedge constants remain tied to legacy SDSS-style cuts and may not perfectly match this dataset’s exact photometric system, and any percentile/softmin summarization can shift emphasis toward the dominant class regions unless tuned and validated under balanced-accuracy splits.

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