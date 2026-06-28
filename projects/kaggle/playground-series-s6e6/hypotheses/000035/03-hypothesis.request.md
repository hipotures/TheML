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

Hypothesis ID: 000035

Previous versions:

- rev 1: eBOSS PTF c1/c3 contamination-aware color geometry
  group_name=eboss_ptf_c1_c3_geometry
  family=color_plane_selection_geometry
  summary=Compute rotated SDSS color-plane geometry used by eBOSS-style quasar targeting to separate true quasar candidates from stellar and low-redshift contaminant manifolds using only ugriz colors, then encode both boundary distance and explicit bright-object rejection-pocket proximity.
  strategy=Build adjacent colors c_ug = u - g, c_gr = g - r, c_ri = r - i, c_iz = i - z. In each row, clip each color to a safe finite range (for example [-10, 10]) before projection. Compute c1 = 0.95*c_ug + 0.31*c_gr + 0.11*c_ri and c3 = -0.39*c_ug + 0.79*c_gr + 0.47*c_ri. Compute boundary margins d1 = (1.4 - 0.55*c1) - c3 and d2 = (0.3 - 0.1*c1) - c3; output signed core score S = min(d1, d2), plus upper-violation magnitudes V1 = max(-d1, 0), V2 = max(-d2, 0), and S_abs = |S|. Add a bright rejection-pocket indicator using the same coordinates: P = 1 if (r < 20.5 AND 0.85 <= c1 <= 1.35 AND c3 > -0.2), else 0; also output pocket distances R1 = max(0.85 - c1, 0, c1 - 1.35) and R2 = max(-c3 - 0.2, 0) to encode how far from the forbidden pocket. Optionally produce two binned versions of S (e.g., quantiles q1<0, q2≈0, q3>0 and a signed ternary bin in {below/near/above}) to stabilize tree partitioning.
  expected_signal=Quasars typically sit away from the stellar/low-z continuum manifold in this rotated space while many stars and unresolved galaxies cluster near or inside the reject pocket, so these signed distances and pocket flags can provide strong class-separation information independent of absolute brightness and help resolve ambiguous color overlap.
  risk=Constants are drawn from SDSS-era target-selection conventions and may be sensitive to photometric-system offsets or magnitude-type mismatch, so the margins can drift and over-prune valid objects if this survey’s preprocessing differs; this transform is also partially redundant with other quasar-selection-surface geometry features and may saturate marginal gain when combined.

- rev 2: Robust eBOSS c1/c3 wedge and bright-pocket geometry
  group_name=eboss_ptf_c1_c3_geometry
  family=color_plane_selection_geometry
  summary=This feature group encodes the rotated ugri color-manifold geometry used in quasar targeting by measuring signed distances to the c1/c3 selection wedge and the proximity of objects to a known bright-object contamination pocket, yielding a compact contamination-aware representation for multiclass separation.
  strategy=Create adjacent colors from clipped magnitudes to reduce outlier sensitivity: c_ug = u-g, c_gr = g-r, c_ri = r-i, with each band first clipped to training-set robust bounds (p0.5 and p99.5, and fallback to [-30, 35] per band if bounds are missing). Compute c1 = 0.95*c_ug + 0.31*c_gr + 0.11*c_ri and c3 = -0.39*c_ug + 0.79*c_gr + 0.47*c_ri. Define wedge responses d1 = (1.4 - 0.55*c1) - c3 and d2 = (0.3 - 0.1*c1) - c3. Emit signed separation: S1 = max(d1, 0), S2 = max(d2, 0), core = min(d1, d2), V1 = max(-d1, 0), V2 = max(-d2, 0), soft_gap = max(V1, V2), and normalized_gap = soft_gap / (1 + abs(c1)) to avoid magnitude dependence at large c1. Emit bright-pocket coordinates with explicit finite handling: pocket_c1 = max(0.85 - c1, 0, c1 - 1.35), pocket_c3 = max(-c3 - 0.2, 0), pocket_flag = 1 if (r < 20.5 and pocket_c1 == 0 and pocket_c3 == 0), else 0; pocket_distance = max(pocket_c1, pocket_c3). Add stable discretized companions using training-time quantiles only: q_core in {below, near, above} from core values (split at two symmetric quantiles around zero, with tie-safe fallback to sign-based bins), c1_bin as train tertiles, and soft_gap_bin from train quantiles [q0.33, q0.67], while clamping all edge/inf values into nearest finite bin.
  expected_signal=The rotated-plane formulation directly reflects quasar-targeting structure, so distance-to-boundary and boundary-violation features distinguish QSO-like points from the dense stellar/galaxy manifold even when raw colors are similar, and the bright-pocket terms add a calibrated penalty for low-r, color-restricted contaminants, likely improving minority-class recall balance in the multiclass setting.
  risk=The linear boundaries and pocket constants are derived from a specific SDSS-like selection context and may be offset if this dataset’s photometric calibration, extinction treatment, or magnitude conventions differ, which can create systematic false positives/negatives; percentile-based clipping and binning can also dampen true-but-rare extremes and add redundancy with other engineered color-ratio features.

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