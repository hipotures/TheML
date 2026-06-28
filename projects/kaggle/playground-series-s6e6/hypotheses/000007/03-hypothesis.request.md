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

Hypothesis ID: 000007

Previous versions:

- rev 1: Band-Order Topology Signature
  group_name=band_order_topology
  family=ordinal_sed
  summary=Represent each object by the ordinal topology of its ugriz brightness sequence so the model can distinguish monotone stellar-like continua from single-turnover galaxy-like patterns and more irregular quasar-like reorderings without relying on absolute brightness.
  strategy=Use the wavelength-ordered magnitudes [u,g,r,i,z]. Assign within-object ranks 1 through 5, where 1 is the brightest band (smallest magnitude), breaking exact ties by wavelength order. Derive the rank of each band, the wavelength index of the brightest band, the wavelength index of the faintest band, the total number of pairwise inversions in the observed rank sequence, and the inversion counts relative to the strictly increasing and strictly decreasing wavelength-order templates. Compute adjacent differences d1=g-u, d2=r-g, d3=i-r, d4=z-i. With epsilon=0.02 mag, label each difference as +1 if >epsilon, -1 if <-epsilon, and 0 otherwise; replace any 0 by the nearest nonzero sign if one exists, else keep 0. From that sign path, derive a fully monotone flag, the number of sign changes, a flag for exactly one sign change, and the position of the first sign change using values 0 through 3, with 0 meaning no change. Add flags for whether the brightest band lies in the interior set {g,r,i} or at an edge {u,z}, and the same for the faintest band. If max(m)-min(m)<=0.02 across all five bands, mark the object as flat and set inversion and sign-change counts to 0.
  expected_signal=These ordinal features are invariant to any global brightness offset and less sensitive to scale than raw magnitudes, so they can preserve class structure when absolute flux and simple colors are ambiguous: stars often follow smooth monotone temperature-driven orderings, galaxies frequently show one interior turnover from broadband continuum breaks, and QSOs more often create non-monotone band reorderings as strong features shift through the filters.
  risk=Because this group intentionally discards amplitude information, it may be partly redundant with color-based groups while also becoming unstable for faint objects whose adjacent magnitudes are nearly tied, creating brittle discrete patterns from small photometric noise.

- rev 2: Robust Ordinal Band Topology Signature
  group_name=band_order_topology
  family=ordinal_sed
  summary=Encode each object by the shape of its ugriz brightness ordering and local monotonicity pattern so the model can separate smooth stellar continua, galaxy-like single turnovers, and more irregular quasar band patterns while remaining insensitive to absolute flux scale.
  strategy=Let m=[u,g,r,i,z] and τ=0.02 mag. Compute spread Δ=max(m)-min(m). If Δ<=2τ, mark flat_topology=1 and set rank/turn features to neutral defaults (monotone=1, inversion counts=0, sign_changes=0, first_change=0, peak/valley positions=0). Otherwise set flat_topology=0 and proceed. Form a deterministic ordered-band sequence by sorting indices by ascending magnitude (brighter first), breaking exact equality or |m_a-m_b|<=τ ties by wavelength index u<g<r<i<z. For this sequence derive: (a) position of brightest and faintest bands (1..5), (b) whether each is interior {g,r,i} vs edge {u,z}, (c) total tie-block count, (d) pairwise inversion count versus template [u,g,r,i,z], and (e) pairwise inversion count versus reverse template [z,i,r,g,u], both restricted to non-tied pairs. Compute adjacent differences d1=g-u,d2=r-g,d3=i-r,d4=z-i and initial signs qk=sign(dk) where sign=+1 if dk>τ, -1 if dk< -τ, else 0. Impute zeros deterministically: scan from left to right to fill each interior zero with nearest non-zero sign; if still missing use nearest on the right, and if both neighbors non-zero and conflicting keep 0 (ambiguous inflection). From imputed q derive monotone flag (all non-zero signs equal), number of sign changes over consecutive q entries, a one_turn flag, first_change position in {0..3} (0 when none), and second_change if present. Map the imputed sign pattern into 4 buckets: strictly_monotone, single_turn, multi_turn, and flat/ambiguous. These structural outputs are the hypothesis features.
  expected_signal=By encoding only relative order and local slope signs, the representation is robust to magnitude zero-point offsets and global extinction-like shifts, while still preserving the class-distinctive topology cues (smooth monotonicity for many stars, one interior turnover for many galaxies, and mixed turnovers for many QSOs) that can improve recall balance on minority structure-heavy classes under balanced_accuracy.
  risk=The discrete topology features can be brittle when adjacent bands differ by tiny amounts, so small photometric noise may flip tie/turn states; aggressive tie smoothing may also over-stabilize genuinely informative micro-structure, and several signals may be correlated with explicit color features, reducing marginal gain versus noise.

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