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

Hypothesis ID: 000053

Previous versions:

- rev 1: Dual Rest-frame Break Alignment Signal
  group_name=dual_restframe_break_alignment
  family=break_competition_geometry
  summary=Model the rest-frame placement and strength of both the Lyman and 4000 Å continuum breaks against each object's observed ugriz trend so the class signal comes from whether high-z blueward dropout dominates (quasar-like) or a Balmer/4000 Å jump dominates (galaxy-like) versus weak break structure (star-like).
  strategy=For each object, work from magnitudes m_u,m_g,m_r,m_i,m_z and redshift z (clipped to [0, 7]). Convert to flux space f_b=10^{-0.4 m_b} only if needed for slope checks, but keep jump calculations in magnitudes. Use SDSS effective wavelengths λ=[3551, 4670, 6170, 7480, 8930] Å. For each target break λ_break in {1216, 4000} Å, compute λ_rest=λ_break·(1+z) and identify the adjacent observed band pair (j, j+1) with λ_j < λ_rest ≤ λ_{j+1}; if no such pair (object too low or too high z), emit 0 and a regime-missing flag for that break. For a valid pair, define observed local break jump J_obs = m_j - m_{j+1}. Estimate the expected smooth-continuum jump J_exp by local slope extrapolation from the two redder neighbouring points: if j+2 exists, use k=((m_{j+2}-m_{j+1})/(log λ_{j+2}-log λ_{j+1})) and m̂_j = m_{j+1} - k( log λ_j - log λ_{j+1}); else, if j==0 use the pair (j+1,j+2) not available fallback and set J_exp = mean(m_{j+1},m_{j+2})-m_{j+1}; if neither exists, fallback to 0. Define normalized residual R=(J_obs-J_exp)/( |m_{j+1}-m_{j+2}| + 1e-6 ) (with j+2 fallback as above). Create six deterministic features: (1) Lya_raw = J_obs(1216), (2) Lya_norm = R_1216, (3) Balmer_raw = J_obs(4000), (4) Balmer_norm = R_4000, (5) Break_balance = sign-safe ratio R_1216/(|R_4000|+1), and (6) Break_regime_flags for valid/invalid break placement and whether redshift falls in the expected Lya-visible regime (1.9<z<6.5) and 4000-visible regime (−0.1<z<1.3). Clip residual-derived features to a robust range, e.g. [-8, 8], with missing variants set to 0 and flags capturing missingness.
  expected_signal=High-redshift quasars should align with strong positive Lyman-break residuals while remaining relatively weak in the 4000 Å jump, whereas normal galaxies tend to show stronger 4000 Å structure in the low-z regime; this relative-break contrast gives a compact geometric cue that separates the class manifolds beyond raw colors alone.
  risk=The regime assignment is sensitive near break-transition redshifts, so redshift noise or systematic offsets can flip the selected band pair; coarse band spacing means jumps can be emission-line contaminated or degenerate in low-SNR objects, and using only one pair extrapolation may be unstable when nearby magnitudes are noisy or nearly flat.

- rev 2: Regime-Weighted Dual Rest-Frame Break Residuals
  group_name=dual_restframe_break_alignment
  family=break_competition_geometry
  summary=This feature family encodes how strongly each object exhibits a 1216 Å and 4000 Å rest-frame discontinuity when mapped into ugriz, by measuring break jump residuals against a local smooth-continuum expectation so class differences from Lyman-dropout-like quasars versus 4000 Å-dominated galaxies versus weak-break stellar SEDs are expressed in a single geometric signal.
  strategy=Let ugriz magnitudes be m=[m_u,m_g,m_r,m_i,m_z] with effective wavelengths λ=[3551,4670,6170,7480,8930] Å and z̃=clip(redshift, 0, 7). For each target rest-frame break λ_b in {1216,4000}, compute λ_b^obs=λ_b*(1+z̃) and locate the unique interval j ∈ {0,1,2,3} where λ_j ≤ λ_b^obs < λ_{j+1}; if none exists (below u band or above z band), emit all break-related features as 0 and set miss_b=1 and skip residual computation for that break. For valid intervals, define fractional position Δt_b=(ln λ_b^obs−ln λ_j)/(ln λ_{j+1}−ln λ_j), observed jump J_obs,b = m_j−m_{j+1}, and observed slope S_obs,b=(m_j−m_{j+1})/(ln λ_j−ln λ_{j+1}). Estimate local smooth continuum slope S_cont,b via linear fit in ln λ: if j=1 or 2, fit through points (ln λ_{j-1},m_{j-1}) and (ln λ_{j+2},m_{j+2}); if j=0, fit through (ln λ_1,m_1) and (ln λ_2,m_2); if j=3, fit through (ln λ_2,m_2) and (ln λ_4,m_4). Then J_exp,b = S_cont,b·(ln λ_j−ln λ_{j+1}) and residual R_b=(J_obs,b−J_exp,b)/S_loc,b, where S_loc,b is the median absolute local color scale over available adjacent colors around the interval, specifically S_loc,b=median(|m_0−m_1|, |m_1−m_2|, |m_2−m_3|, |m_3−m_4|)+1e-6; clip R_b to [−8,8]. Add interval-confidence C_b=clip(1−2·|Δt_b−0.5|, 0, 1) so C_b downweights placements near band boundaries; generate miss_b, boundary_b=(C_b<0.2), and regime flags lya_regime=(z̃∈[1.9,6.5]), balmer_regime=(z̃∈[0,1.3]). Output deterministic features per break: raw jump J_obs,b, normalized residual R_b, confidence-weighted jump J_obs,b·C_b, confidence-weighted residual R_b·C_b, and miss/border/regime flags. Build competition features: Break_balance=((R_1216·C_1216)−(R_4000·C_4000)), Break_abs_diff=|R_1216·C_1216|−|R_4000·C_4000|, and sign-consistent variants that suppress invalid breaks by multiplying with (1−miss_b).
  expected_signal=Because quasars at suitable redshift should show strong positive 1216 Å-break residuals with weak competing 4000 Å structure while low-redshift galaxies usually show opposite balance and stellar SEDs remain smoother across both windows, these confidence-weighted cross-break residual contrasts should separate the three classes in a regime-aware, redshift-consistent manner and reduce dependence on raw color magnitude alone.
  risk=The feature signal is sensitive to redshift-dependent regime assignment, so objects near band-edge transitions or with systematic redshift offsets can misplace breaks and distort residuals; broad-band filters and emission-line features can mimic continuum jumps, and the scheme may be less stable for intrinsically smooth or noisy SEDs where local slopes are poorly defined despite clipping.

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