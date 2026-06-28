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

Hypothesis ID: 000052

Previous versions:

- rev 1: SEGUE-inspired stellar subtype plane margins
  group_name=segue_stellar_subtype_margins
  family=stellar_color_geometry
  summary=Translate legacy SDSS-SEGUE color-selection boundaries for specific stellar subtypes into signed distance features so each object carries how strongly it conforms to or departs from known stellar loci in calibrated color index space as a direct complement to generic broad-band shape features.
  strategy=Compute base colors ug=u-g, gr=g-r, ri=r-i, iz=i-z, gi=g-i and linear principal colors l_color=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984, s_color=-0.249*u+0.794*g-0.555*r+0.234, p1=0.91*(u-g)+0.415*(g-r)-1.280, and v=0.283*(u-g)-0.354*(g-r)+0.455*(r-i)+0.766*(i-z). Define a helper margin function for an interval [a,b]: I(x)=0 if a<=x<=b else min(|x-a|,|x-b|), and for one-sided cut x<=k/x>=k use the positive excess beyond the bound. For each SEGUE-style prototype define a normalized score as the weighted sum of clause margins across its color inequalities (weights equal or scaled by clause width): white_dwarf: gr in [-1,-0.2], ug in [-1,0.7], ug+2*gr<-0.1; cool_white_dwarf: 14.5<r<20.5 and gi in [-2,1.7] with branch adjustment; A/BHB: 0.8<ug<1.5 and -0.5<gr<0.2 plus v around 0 (e.g., |v-0|/1.0); K_giants: 0.35<gr<0.8, 0.15<ri<0.6, l_color>0.07; low_metallicity: -0.5<gr<0.75, 0.6<ug<3.0, l_color>0.135; M_subdwarf: ri<0.9, ri<0.787*gr-0.356, 1.8<gi<2.4; main_sequence_white_dwarf_pairs: ug<2.25, -0.2<gr<1.2, 0.5<ri<2.0, gr>-19.78*ri+11.13, gr<0.95*ri+0.5, plus iz side conditions by ri branch. Also produce aggregate features: best_star_margin=min(all prototype margins), second_best_margin, margin_gap=second_best-best, and count_inside_prototypes (non-negative margins). Normalize each clause by a robust MAD/scale from training and clip margins to a finite range (for stability) before aggregation.
  expected_signal=This adds explicit, survey-proven stellar-template geometry that is complementary to smooth manifold or flux-shape features and should raise separation in regions where stars mimic galaxies or quasars, because many non-stellar classes remain farther from these subtype envelopes while true stars frequently lie close to one of them.
  risk=These are legacy targeting cuts, not universal physical laws; they can be shifted by calibration, extinction treatment, or sample-selection bias and may add redundancy with existing color-geometry families, causing misclassification of exotic quasars/galaxies with stellar-like colors or reduced performance outside SDSS-like photometric conditions.

- rev 2: Robust SEGUE subtype margins with normalized prototype proximity features
  group_name=segue_stellar_subtype_margins
  family=stellar_color_geometry
  summary=Model canonical stellar subtype regions from calibrated color-space geometry as smooth signed-distance-to-envelope features, so each object carries structured evidence of how closely it matches multiple known stellar loci versus how far it deviates.
  strategy=Compute base colors ug=u-g, gr=g-r, ri=r-i, iz=i-z, gi=g-i. Compute the SEGUE-style linear transforms on magnitudes: l_color=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984, s_color=-0.249*u+0.794*g-0.555*r+0.234, p1=0.91*ug+0.415*gr-1.280, and v=0.283*ug-0.354*gr+0.455*ri+0.766*iz. For each primitive term t, estimate training-only robust scale s_t as max(MAD(t),0.01) and clip raw t to the train 0.5th/99.5th percentile range before normalization. Use penalties: I_int(x,[a,b])=0 if a<=x<=b else min(|x-a|,|x-b|)/s_x, I_hi(x,k)=max(0,x-k)/s_x, I_lo(x,k)=max(0,k-x)/s_x. For each prototype p, define a weighted sum P_p of clause penalties (weights proportional to inverse interval width) with explicit bounds: white_dwarf: gr in [-1,-0.2], ug in [-1,0.7], and ug+2*gr <= -0.1; cool_white_dwarf: r in [14.5,20.5], gi in [-2,1.7], and if gi<0.6 then ug in [-0.4,1.0], else ug in [-0.2,1.4]; A_BHB: ug in [0.8,1.5], gr in [-0.5,0.2], and v in [-0.15,0.15]; K_giants: gr in [0.35,0.8], ri in [0.15,0.6], and l_color>=0.07 (I_lo form); low_metallicity: gr in [-0.5,0.75], ug in [0.6,3.0], and l_color>=0.135; M_subdwarf: ri<=0.9, ri<=0.787*gr-0.356, gi in [1.8,2.4]; main_sequence_white_dwarf_pairs: ug<=2.25, gr in [-0.2,1.2], ri in [0.5,2.0], gr>-1.0*ri+1.13, gr<0.95*ri+0.50, and if ri<1 use iz in [-0.5,1.0], else iz in [-0.2,0.7]. Apply a redshift trust gate g_z = 1 + max(0, redshift-0.30)/0.30 and use P_p = g_z * P_p. Emit aggregated outputs: star_margin_best=min(P_p), star_margin_second=min2(P_p), star_margin_gap=star_margin_second-star_margin_best, star_inlier_count=Σ(P_p<=0.02), and star_confidence=exp(-star_margin_best).
  expected_signal=The model receives continuous proximity-to-locus and margin-gap structure instead of brittle binary cuts, which should better preserve star-like manifold geometry and expose ambiguous regions where galaxies or quasars mimic one stellar cut but fail a full subtype envelope, improving balanced separability for the STAR class.
  risk=SEGUE-style legacy cuts are sensitive to photometric calibration, extinction treatment, and selection shifts; boundary hardening can overfit to this survey epoch, and quasar/galaxy contaminants with low redshift and star-like colors may still produce small prototype penalties, especially when transformed scales change across subsets.

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