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

Hypothesis ID: 000028

Previous versions:

- rev 1: Dust-aware dereddened SED geometry
  group_name=foreground_reddening_geometry
  family=reddening_invariant
  summary=This group models line-of-sight Galactic dust effects explicitly and splits each object's observed photometric shape into dust-amplitude and dust-invariant components so class boundaries reflect intrinsic stellar and extragalactic signatures rather than foreground reddening.
  strategy=Estimate line-of-sight color excess E(B-V) from RA/DEC by converting to Galactic coordinates and querying a fixed Schlegel-Finkbeiner-Davis based map with Schlafly-Finkbeiner recalibration; for points with missing lookup, use the same dataset's 10-degree (l, b) bin median fallback E(B-V), then clip to [0, 1.2] and add a missing-indicator feature. Convert to per-band extinction using fixed SDSS coefficients A_u=5.155*E(B-V), A_g=3.793*E(B-V), A_r=2.751*E(B-V), A_i=2.086*E(B-V), A_z=1.479*E(B-V), then create dereddened magnitudes u0..z0. Build observed color vector C=[u-g, g-r, r-i, i-z] and dereddened color vector C0=[u0-g0, g0-r0, r0-i0, i0-z0]. Use the SDSS reddening color direction d=[1.362, 1.042, 0.665, 0.607] and compute projections t=(C·d)/||d||^2 and t0=(C0·d)/||d||^2, dust-induced shift Δt=t-t0, plus orthogonal residue R=C0-(t0*d), keeping the first three components of R. Add context features: |b|, b-band of E(B-V) quantized into 6 bins, interactions Δt*|b| and bin_EBV*redshift, plus clipped versions of all new columns (0.5/99.5 percentiles) to stabilize tails.
  expected_signal=Foreground extinction can move stars, galaxies, and quasars along a known reddening direction in color space, so separating this dust-driven axis from the orthogonal color geometry should reduce confusion in dusty sightlines while preserving class-discriminative structure from intrinsic SED shape and redshift-dependent behavior.
  risk=Accuracy depends on the quality and calibration consistency of the external extinction map with this survey photometric scale; map mismatch or poor coverage can add systematic noise, and these dust-aware projections may overlap with existing sky-position and catalog-consistency groups, reducing marginal gain.

- rev 2: Dust-vector decomposition with extinction-invariant colors
  group_name=foreground_reddening_geometry
  family=reddening_invariant
  summary=This feature family makes classifying stars, galaxies, and quasars more robust by separating foreground dust-driven color shifts from intrinsic broadband color structure so model boundaries reflect intrinsic SED and redshift information rather than line-of-sight extinction.
  strategy=For each object, convert (alpha, delta) in degrees to Galactic longitude/latitude (l, b) using a fixed sky-coordinate transform, then estimate E(B-V) from a preloaded SFD98 dust map rescaled by the Schlafly–Finkbeiner correction (multiply by 0.86). If no map value is available, impute from the median E(B-V) in the same 5°×5° (l,b) bin built on train-only rows; if still empty, use the nearest non-empty bin within 10° great-circle distance, else backfill with the global train median. Clamp E(B-V) to [0, 1.2], output ebv_clipped and a binary is_ebv_imputed flag. Compute A_u=5.155E, A_g=3.793E, A_r=2.751E, A_i=2.086E, A_z=1.479E and dereddened magnitudes u0..z0 by subtracting each extinction from the observed band. Form observed colors C=[u−g, g−r, r−i, i−z] and dereddened colors C0=[u0−g0, g0−r0, r0−i0, i0−z0]. Define reddening direction d=(A_u−A_g, A_g−A_r, A_r−A_i, A_i−A_z)/||·||2 and compute projections t=C·d, t0=C0·d, and dust displacement Δt=t−t0. Compute orthogonal residual R0=C0−(t0d), keep R0[0],R0[1],R0[2], and ||R0||2. Add context features |b|, latitude bin (train-fit quantile/decile bins), Δt×|b|, redshift×E(B-V), and E(B-V)×|b|. Clip all newly created continuous features to train-derived 0.5th and 99.5th percentiles.
  expected_signal=Foreground extinction moves objects along a nearly fixed direction in color space, so removing this axis and using orthogonal residuals should reduce class confusion in dusty regions while preserving discriminative structure from intrinsic SED differences and redshifted continua that are expected to be more informative for GALAXY/STAR/QSO separation.
  risk=The approach can misfire where the external dust map is biased or inconsistent with this photometric campaign, and fixed SDSS coefficients may not perfectly match all object depths/regions; additional projected features may be redundant with base colors and sky-position signals, so net benefit can be small when the underlying model already internalizes those correlations.

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