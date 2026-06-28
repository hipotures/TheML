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

Hypothesis ID: 000032

Previous versions:

- rev 1: Redshift-anchored 4000A break curvature
  group_name=redshifted_4000a_break_curvature
  family=restframe_break_geometry
  summary=Project the known 4000 Angstrom continuum break into the observed ugriz frame using redshift and encode how strongly each object bends at that physically expected transition, which is especially diagnostic of galaxy stellar populations versus quasar and star-like continua.
  strategy=Define SDSS effective band centers as lambda_u=3562, lambda_g=4686, lambda_r=6165, lambda_i=7481, lambda_z=8931. Compute lambda_break = 4000*(1+redshift). Also compute color triplets c1=u-g, c2=g-r, c3=r-i, c4=i-z. Assign each row to one active break regime by lambda_break:
- G regime: 3562 <= lambda_break < 4686 (redshift < 0.17) or equivalently u-g region for near-zero z,
- GR regime: 4686 <= lambda_break < 6165 (approx 0.17 <= z < 0.54),
- RI regime: 6165 <= lambda_break < 7481 (approx 0.54 <= z < 0.87),
- IZ regime: 7481 <= lambda_break < 8931 (approx 0.87 <= z < 1.23).
Use these as one-hot flags plus out_of_band = 1 when z < 0 or z >= 1.23, with break outputs set to 0 there. For each active regime, compute position p = clip((lambda_break-lambda_left)/(lambda_right-lambda_left),0,1) and a boundary confidence weight w = 4*p*(1-p) so estimates are suppressed near filter edges.
- G regime: break_excess = (u-g) - 0.5*((g-r)+(u-g? if unavailable due edge, use 0)) equivalent to local jump relative to adjacent slope; also compute asym = ((u-g)-(g-r)) - ((g-r)-(r-i)).
- GR regime: break_excess = (g-r) - 0.5*((u-g)+(r-i)); asym = ((g-r)-(u-g)) - ((r-i)-(g-r)).
- RI regime: break_excess = (r-i) - 0.5*((g-r)+(i-z)); asym = ((r-i)-(g-r)) - ((i-z)-(r-i)).
- IZ regime: break_excess = (i-z) - (r-i); asym = (i-z)-(r-i).
Emit weighted outputs w*break_excess and w*asym, plus raw regime position p and selected regime flags so a model can learn smooth, physically localized transitions.
  expected_signal=Because the 4000A break shifts from g to r to i across 0.17<=z<1.23, this captures a redshift-dependent shape cue that is tightly linked to stellar-population-dominated galaxy continua but much less consistent for quasars and main-sequence stars, reducing confusion in mixed-color redshift regions.
  risk=Signal weakens when redshift noise, dust effects, or emission features distort broad-band colors, and for high-z objects where the break is beyond z-band the feature provides little information; there is moderate redundancy with other spectral-break shape groups and potential instability on very noisy faint photometry.

- rev 2: Rest-frame 4000Å break curvature with smooth redshift gating
  group_name=redshifted_4000a_break_curvature
  family=restframe_break_geometry
  summary=This group encodes how the broadband color trajectory bends near the observed-frame position of the 4000 Angstrom rest-frame continuum break as redshift shifts it through ugriz, providing a redshift-localized shape signature that distinguishes galaxy-like stellar-population continua from quasar and stellar SED patterns.
  strategy=Use SDSS effective wavelengths λu=3551, λg=4686, λr=6165, λi=7481, λz=8931 Å. For each object, compute λ_break = 4000*(1+redshift). Define colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. Define regime flags by intervals of λ_break: G in [3551,4686), GR in [4686,6165), RI in [6165,7481), IZ in [7481,8931), and out_of_band=1 otherwise. Set all regime outputs to 0 for out_of_band. For active regime j, compute t=(λ_break-λ_left)/(λ_right-λ_left), p=clip(t,0,1), and w=4*p*(1-p). Compute regime-specific local jump and asymmetry, then multiply both by w and by the corresponding regime flag. For G: jump_G = c1 - c2 and asym_G = (c1-c2) - (c2-c3). For GR: jump_GR = c2 - 0.5*(c1+c3) and asym_GR = (c2-c1) - (c3-c2). For RI: jump_RI = c3 - 0.5*(c2+c4) and asym_RI = (c3-c2) - (c4-c3). For IZ: jump_IZ = c4 - c3 and asym_IZ = (c4-c3) - (c3-c2). Emit 12 numeric features: weighted break terms w*jump_G, w*asym_G, w*jump_GR, w*asym_GR, w*jump_RI, w*asym_RI, w*jump_IZ, w*asym_IZ, plus w and p, and one-hot regime flags; all inactive regime values stay 0.
  expected_signal=The 4000Å break is a physically interpretable anchor that shifts from u/g to z with increasing redshift, so weighting a redshift-localized second-difference/curvature representation should isolate galaxy-specific continuum structure while suppressing global color drift, which helps resolve class confusion in overlapping color regions and should improve balanced accuracy on a class-imbalanced multiclass split.
  risk=The hypothesis depends on redshift precision and the assumed fixed filter centers, so noisy redshift estimates can misassign regimes; it contributes weakly at very low or high redshift where the break is outside ugriz coverage, and broad emission features, dust, or heavy photometric noise can mimic or mask the expected curvature, creating redundancy with other spectral-shape feature groups and potential overfitting to specific redshift-color patterns.

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