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

Hypothesis ID: 000006

Previous versions:

- rev 1: Bandpass break localization
  group_name=bandpass_break_localization
  family=spectral_break
  summary=Capture where along the ordered ugriz band sequence each object shows its dominant broadband discontinuity or turnover, because the location and sharpness of a photometric break can separate smooth stellar spectra from galaxy continuum breaks and redshifted quasar features.
  strategy=Treat the bands in wavelength order u,g,r,i,z and convert each magnitude m to log-flux l = -0.4*m so only relative spectral shape matters. Compute the four adjacent log-flux differences d_ug = l_u - l_g, d_gr = l_g - l_r, d_ri = l_r - l_i, and d_iz = l_i - l_z. Define the dominant break index k as the position with maximum absolute adjacent difference; if there is a tie, choose the bluest position. Create deterministic features from this single concept: the signed dominant break amplitude d_k, its absolute value, a 4-level one-hot encoding of k, the mean adjacent difference on the blue side of k and on the red side of k using 0 when a side has no remaining segments, a local contrast term equal to d_k minus the average of those side means, a sharpness share equal to |d_k| divided by the sum of all |d| values plus 1e-8, and a turnover indicator that is 1 when the sign of adjacent differences changes across the dominant break and 0 otherwise. No missing-value handling is needed from the provided summary; keep all calculations finite with the 1e-8 denominator guard only.
  expected_signal=Galaxies often show a broad continuum break, quasars can show a strong redshifted break or emission-line-driven jump landing in different bands, and stars tend to have smoother monotone spectra, so explicitly localizing the strongest band-to-band discontinuity should help the classifier distinguish the three classes and improve balanced accuracy on harder minority-class boundaries.
  risk=This group is partially redundant with generic color-slope features and may become unstable for faint noisy objects where small photometric errors flip the argmax break position, so the discrete break-location features could overfit rare edge cases unless the underlying signal is strong.

- rev 2: Bandpass break localization with confidence-aware soft assignment
  group_name=bandpass_break_localization
  family=spectral_break
  summary=Represent each object by the location of its strongest broadband discontinuity across ugriz and how strong, asymmetric, and reliable that discontinuity is, so class-specific spectral-shape signatures become easier to separate.
  strategy=Use only bands u,g,r,i,z. Compute l_u,l_g,l_r,l_i,l_z as log-flux proxies with l_b=-0.4*m_b. Let d_ug=l_u-l_g, d_gr=l_g-l_r, d_ri=l_r-l_i, d_iz=l_i-l_z, and a_uj=|d_j| for j in {ug,gr,ri,iz}. Let A=a_ug+a_gr+a_ri+a_iz and eps=1e-8. If A<1e-6 (near-flat spectrum), emit break_present=0, break_position=-1, and set all break-derived numeric outputs to 0 with a dedicated flag. Otherwise set break_present=1 and choose dominant break index k=argmax_j a_j with tie broken by smallest j (bluest) for deterministic behavior. For all j define soft posterior p_j=a_j/(A+eps), entropy H=−∑ p_j log(max(p_j,eps)), and expected position k_soft=∑ p_j·idx(j) where idx maps ug/gr/ri/iz to 1..4. For chosen k, compute d_k and s_k=sign(d_k); blue_mean=mean(a_ug..a_{prev(k)-1}) (0 if k=1), red_mean=mean(a_{k+1}..a_iz) (0 if k=4), and continuity_break= d_k - (blue_mean+red_mean)/2. Define asymmetry=blue_mean-red_mean, sharpness=|d_k|/(A+eps), and turnover_count = I(k>1 and sign(d_{k-1})!=s_k)+I(k<4 and sign(d_{k+1})!=s_k). Optionally add confidence bins by applying equal-frequency cutpoints on sharpness and H computed on training data (e.g., quartiles) and record low/med/high bins to avoid brittle behavior in near-tie cases.
  expected_signal=The approach gives the model a compact description of where and how abrupt the largest spectral break is, which is physically aligned with galaxy continuum breaks, redshifted quasar features moving across ugriz, and the smoother monotonic trends of stars, so it should improve recall-balanced discrimination where global colors alone are ambiguous.
  risk=Because magnitudes can be noisy at faint flux levels, argmax selection may flip between adjacent bands and inflate discrete break-location features; entropy and soft-weights reduce but do not remove this, and these descriptors are partially redundant with adjacent-color slopes, so careless use can still overfit survey-specific photometric quirks.

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