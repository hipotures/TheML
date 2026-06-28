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

Hypothesis ID: 000033

Previous versions:

- rev 1: Rest-frame SED family fit residuals
  group_name=restframe_sed_family_fit
  family=spectral_family
  summary=Compare each object's rest-frame ugriz continuum against competing power-law and curved continuum family signatures to encode whether its broadband shape is quasar-like, star-like, or galaxy-like, using physically motivated rest-frame shape diagnostics that are orthogonal to absolute brightness.
  strategy=For each object, convert magnitudes to flux proxies f_b=10^{-0.4 m_b} for b in {u,g,r,i,z}, then compute x_b=log10(\lambda_b/(1+z_plus)) and y_b=log10(f_b), where \lambda=[3551,4686,6165,7481,8931] Å and z_plus=max(redshift,0). Fit two deterministic models to (x_b,y_b): linear model y=a+b\,x (power-law surrogate), and quadratic model y=c+d\,x+e\,x^2 (curved/thermal surrogate). Derive SSE_linear and SSE_quad from least-squares residuals, R2_linear and R2_quad, curvature gain g=(SSE_linear-SSE_quad)/SSE_linear, and signed curvature coefficient e. Add an endpoint slope-contrast feature s=(y_u-y_g)/(x_u-x_g)-(y_i-y_z)/(x_i-x_z). To reduce Lyman-forest/absorbed-band distortion at high redshift, compute a second weighted variant with weights w_b=1 if x_b_rest>1216 Å, 0.5 if 912 Å<\lambda_rest<=1216 Å, 0.2 if \lambda_rest<=912 Å, where \lambda_rest=\lambda_b/(1+z_plus); compute the same model statistics on weighted data and keep a binary fallback flag for the weighted fit when all weights are 1 (low-z regime), otherwise both sets are returned.
  expected_signal=Quasars are expected to be better explained by linear continua in log-log space (small curvature gain) while stars and many galaxies show stronger smooth curvature or break-driven slope asymmetry, so family-residual features should sharpen class boundaries near stellar and quasar overlaps and improve balanced discrimination.
  risk=Rest-frame warping amplifies redshift noise and any photometric-system calibration mismatch, especially for negative or noisy redshifts and very high-redshift objects with heavily absorbed blue bands; this can inject unstable curvature estimates and creates overlap with existing break/continuum groups, so feature utility may weaken if redshift quality is poor.

- rev 2: Lyman-aware rest-frame SED family fit with stability-aware dual residual diagnostics
  group_name=restframe_sed_family_fit
  family=spectral_family
  summary=This feature family characterizes each object by how well its rest-frame ugriz continuum is represented by a low-order power-law-like shape versus a curved continuum family, using redshift-normalized wavelength geometry and absorption-aware weighting to separate quasar-like, stellar, and galaxy-like spectral morphology.
  strategy=For each object define zc = max(redshift, 0), and use rest-frame effective wavelengths x_b = log10(lambda_b/(1 + zc)) with lambda=[3551, 4686, 6165, 7481, 8931] Å for b in {u,g,r,i,z}. Convert magnitudes to relative log-flux y_b = -0.4 * m_b (equivalent to log10 of flux up to a constant) and center y by subtracting its object-wise mean so absolute scale is removed. Fit weighted linear model y = a + b*x' and weighted quadratic model y = a + b*x' + c*x'^2 where x' is x centered per object to reduce conditioning issues, using weighted normal equations with explicit weights. Produce one feature block with weights w=1 for all bands (base model), and one block with Lyman-aware weights v: v=1 if lambda_b/(1+zc) > 1216, v=0.5 if 912 < lambda_b/(1+zc) <= 1216, and v=0 otherwise, to de-emphasize IGM-absorbed regions. For each block compute RSS, R2, slope b, curvature c, curvature magnitude |c|, and curvature gain g = (RSS_linear - RSS_quadratic) / RSS_linear (only when both fits are estimable); add a signed asymmetry slope feature s = (y_u - y_g)/(x_u - x_g) - (y_i - y_z)/(x_i - x_z) with a deterministic fallback pair (g-r) minus (r-i) if either endpoint is ineligible due heavy masking. Define fit-feasibility flags by point support: require at least 2 non-negligible bands for linear and 3 for quadratic; if a band is downweighted to zero and a fit is infeasible, set its statistics to NaN and use linear/unweighted estimates for that branch with a binary instability flag. Add branch-difference features: ΔR2 = R2_weighted - R2_unweighted, Δb = b_weighted - b_unweighted, Δc = c_weighted - c_unweighted, plus a flag for no_uv_downweight (all v=1). If redshift is clamped to 0 (negative inputs), set z_low_flag to indicate observer-frame fallback is used for this object.
  expected_signal=The quasar-like class should retain near-power-law behaviour once shifted toward rest-frame and after suppressing UV-absorbed points, while many stars and galaxies should show stronger curvature and slope asymmetry from stellar/galaxy continua and breaks, so the contrast between linear and quadratic fit quality plus asymmetry under masked versus unmasked regimes should improve class boundary separation where colors alone are ambiguous.
  risk=At very high redshift or noisy redshift values, rest-frame compression and band suppression can leave too few informative points, causing unstable curvature estimates and increased missing-flag rate; the weighted masking can also overlap with intrinsic class-correlated redshift distribution, creating potential overfitting to the redshift-induced regime unless stability flags and branch-differences are treated with care.

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