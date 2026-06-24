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

Hypothesis ID: 000040

Previous versions:

- rev 1: Bayesian class posterior from redshift- and brightness-conditioned color density bins
  group_name=class_conditional_color_density_posteriors
  family=bayesian_density_scoring
  summary=Builds class-compatible score surfaces for galaxy, quasar, and star classes by estimating class-conditional color-density and class priors in coarse redshift and apparent-magnitude strata, then emitting posterior-style margins as features so ambiguous color overlaps are resolved through class prevalence and local manifold support.
  strategy=Compute the color vector x=[u-g, g-r, r-i] and keep redshift and i as conditioning coordinates only. Bin redshift into fixed astrophysical intervals (for example 0–0.3, 0.3–1.2, 1.2–2.6, 2.6–4.0, 4.0–7.0) and i into fixed 0.5-mag bins across the training range. For each class c in {GALAXY, QSO, STAR}, each redshift bin z_k and i-bin i_m, estimate a smoothed 3D color histogram H_{c,k,m}(bin_u-g, bin_g-r, bin_r-i) using global color edges from training (e.g., 14 equiprobable bins per color, clipped at training min/max). Use additive count smoothing α: P_c(x|k,m) = (H_{c,k,m}+α)/(sum_all_bins H_{c,k,m}+α*Bcells), with Bcells the total 3D bins. Class prior in each stratum is π_c(k,m)=(N_{c,k,m}+β)/(N_{k,m}+3β). Score each row as s_c = log π_c(k,m)+log P_c(x|k,m). If a stratum-cell combination is under-supported, back off: first to neighboring color cells (Chebyshev radius 1), then to the stratum-aggregated density for (k,m), then to global class prior-only baseline. Emit three raw scores s_galaxy, s_qso, s_star, two margins s_qso-s_star and s_star-s_galaxy, plus softmax entropy over s_c as additional features.
  expected_signal=This group adds calibrated, class-aware compatibility estimates that directly encode where each object sits relative to empirical class manifolds under each redshift/brightness regime, reducing confusion where raw colors overlap and improving balanced performance by down-weighting dense-star regions and rewarding high-probability quasar and galaxy color-redshift pockets.
  risk=Histogram binning is lossy and can introduce boundary artifacts, especially for sparse high-redshift QSO cells, so nearest-neighbor-backed smoothing is required; because priors are class- and stratum-dependent, fit-time leakage can occur if class statistics are computed from test-visible targets, so the model must be fit only on training folds.

- rev 2: Hierarchical Redshift-Brightness Bayesian Color Posteriors
  group_name=class_conditional_color_density_posteriors
  family=bayesian_density_scoring
  summary=The group builds class-conditional posterior-like compatibility scores by comparing each object’s colors against empirical class manifolds estimated in redshift and i-band brightness slices, with staged fallback to broader distributions when local support is weak.
  strategy=For each fold, build color coordinates c=[u−g, g−r, r−i] from the available training fold only. Use this same mapping for train and test rows. Define redshift strata with explicit bounds [-0.01, 0.2, 0.7, 1.3, 2.0, 3.2, 4.8, 7.01] (inclusive of the upper edge in the last bin). Define i-band bins as fixed 0.50-mag intervals over the fold i-range, with hard clipping of any value outside range to endpoint bins. For each color component, compute fold quantile cut edges using 16 equiprobable bins after clipping at the 0.5 and 99.5 percentiles, then append two overflow bins to guarantee bounded indexing for all rows. Let n^c_{k,m,b} be class-c counts in redshift bin k, i bin m, and 3D color cell b; let N^c_{k,m}=Σ_b n^c_{k,m,b}, N_{k,m}=Σ_c N^c_{k,m}, and B be total color cells. Apply Dirichlet smoothing with α=1 and class-prior smoothing β=0.5: π^c_{k,m}=(N^c_{k,m}+β)/(N_{k,m}+3β), P^c_{k,m}(b)=(n^c_{k,m,b}+α)/(N^c_{k,m}+αB). Define a support floor n_min (e.g., 200). For each sample at cell b in stratum (k,m), score class c as s_c=log(π+ε)+log(p+ε), where ε=1e-12 for numerical safety, but before using p run a deterministic backoff chain: if N_{k,m}<n_min or the target cell is empty/unstable, use Chebyshev-radius-1 neighborhood averaging with weights 1, 1/2, 1/4, then radius-2 if needed; if still unsupported, use redshift-marginal P^c_k(b), then global P^c(b), and if all else fails use global π^c only. After final backoff, emit s_GALAXY, s_QSO, s_STAR, margins s_QSO−s_STAR and s_STAR−s_GALAXY, and normalized softmax entropy of s_c.
  expected_signal=Conditioning on redshift and brightness directly encodes expected astrophysical evolution in color distributions while preserving class-level prevalence, so ambiguous colors are resolved using local class manifold density instead of global similarity, improving discrimination of minority structures such as high-z quasars and reducing balanced-accuracy loss from overlap between classes.
  risk=Sparsity in fine strata can still produce noisy estimates if thresholds are set too low, bin boundaries may bias nearby points near transitions, and the additional 3D histogram tables increase memory/time unless built sparsely and cached, while any hyperparameter tuning of binning/backoff done outside cross-validation risks optimistic bias.

- rev 3: Fold-wise Bayesian color posteriors with hierarchical stratum backoff
  group_name=class_conditional_color_density_posteriors
  family=bayesian_density_scoring
  summary=This group generates class-aware posterior-like compatibility features by estimating how likely each object’s colors are under galaxy, quasar, and stellar color distributions conditioned on redshift and i-band brightness, then emitting calibrated class score and margin signals.
  strategy=For each model-training fold, build all statistics from the training partition only and reuse the same transforms for validation and test rows. Construct colors c=[u−g, g−r, r−i]. Clip redshift to [−0.01, 7.01] and clip each color to foldwise 0.5th/99.5th percentile bounds, keeping two overflow bins so binning is always valid. Use fixed redshift intervals z_breaks=[−0.01, 0.20, 0.60, 1.05, 1.70, 2.45, 3.60, 5.00, 7.01] and fixed 0.50-mag i-band intervals with hard clipping to edge bins. For each color component, use 16 equiprobable bins within the clipped range plus two overflow bins; combine into 3D cells b=(b1,b2,b3) and keep sparse counts n_{c,k,m,b} for class c∈{GALAXY,QSO,STAR}, redshift bin k and i-bin m. Apply Dirichlet smoothing α=1 and class-prior smoothing β=0.5: π_{c,k,m}=(N^c_{k,m}+β)/(N_{k,m}+3β), P_{c,k,m}(b)=(n_{c,k,m,b}+α)/(N^c_{k,m}+α·B^3), where B is number of 1D color bins including overflow bins and N^c_{k,m}=∑_b n_{c,k,m,b}. Precompute marginals P_{c,k}(b) (sum over m), P_c(b) (sum over k,m), and global π_c from the same fold. Score each sample deterministically with a strict backoff chain: if N_{k,m} and the target cell support are above n_min=180, use P_{c,k,m}(b); else smooth within the same stratum with Chebyshev-neighborhood averaging first radius 1 then radius 2 using weights [1, 1/2, 1/4, ...], renormalized; if still sparse use P_{c,k}(b), then P_c(b), and finally π_c. Use s_c=log(p_support+1e-12)+log(π_support+1e-12), where p_support is the chosen likelihood level and π_support is corresponding prior at that level. Output dense features s_GALAXY, s_QSO, s_STAR, margins s_QSO−s_STAR and s_STAR−s_GALAXY, and softmax entropy of [s_GALAXY,s_QSO,s_STAR].
  expected_signal=Conditioning color distributions on both redshift and i-band brightness aligns with known astrophysical shifts in object loci, so score contrasts resolve overlap regions with lower ambiguity and preserve class prevalence in each physical regime, which should improve balanced recall of minority structures such as high-redshift QSOs without forcing a rigid global decision boundary.
  risk=Although staged backoff reduces brittle sparsity effects, the fixed stratum and bin edges can still induce boundary sensitivity and noisy estimates in poorly populated regions if tuned on full data, and maintaining many 3D sparse cubes can increase memory/time unless explicitly implemented with sparse structures and strict fold-based fitting.

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