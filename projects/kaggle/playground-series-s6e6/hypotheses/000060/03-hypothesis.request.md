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

Hypothesis ID: 000060

Previous versions:

- rev 1: Context-anchored class prior encoding
  group_name=metadata_redshift_magnitude_class_priors
  family=conditional_target_prior
  summary=Create class-priority features by learning how class prevalence changes with catalog tags, redshift regime, and brightness regime, so color-overlap regions are resolved using empirically supported astrophysical context rather than raw photometry alone.
  strategy=For each training row, build deterministic stratification cells on `(spectral_type, galaxy_population, redshift_bin, i_bin)`, where redshift_bin uses fixed astrophysically interpretable breaks such as [-0.01, 0.1, 0.4, 1.0, 2.0, 2.8, 3.5, 4.5, 7.02] and i_bin uses fixed magnitude bands over the observed range, e.g. [-inf,17,18,19,20,21,22,23,24,inf]. For every cell c, compute class counts n_c(GALAXY), n_c(QSO), n_c(STAR) and n_c(nonstratified) from training labels only. Emit smoothed class posteriors with shrinkage to the global priors π via n̂_c(k) = (n_c(k)+α·π_k)/(n_c+α), where k ∈ {GALAXY,QSO,STAR} and α is a fixed constant (e.g., 20–40). Generate derived columns: three class posteriors, three centered-logit margins against the non-k mass (log((n̂_c(k)+ε)/(1-n̂_c(k)+3ε))), class-entropy, and top-two gap. For empty or very sparse cells, back off hierarchically to `(spectral_type, galaxy_population, redshift_bin)` then `(spectral_type, galaxy_population)` then global priors; implement backoff by weighted interpolation with cell-count-dependent weights, so no divide-by-zero and no undefined outputs. Any unexpected categorical level in deployment is mapped to global priors.
  expected_signal=This group adds calibrated, context-dependent prevalence signals where photometric features are ambiguous, especially in regions where quasar, galaxy, and star colors overlap, improving balanced recall by separating instances that share similar ugriz patterns but occur in very different known metadata regimes.
  risk=It is a target-derived encoding and can overfit if class proportions drift between training and test contexts; sparse strata may produce unstable priors, and overly sharp α or hard bins can amplify sampling noise, so shrinkage and backoff are essential.

- rev 2: Hierarchical class-prior encoding from spectral, population, redshift, and magnitude context
  group_name=metadata_redshift_magnitude_class_priors
  family=conditional_target_prior
  summary=This feature group injects empirical, context-conditioned class prevalence signals derived from astrophysical metadata so that the classifier can distinguish otherwise overlapping photometric patterns by the expected target mix in each metadata regime.
  strategy=Fit this encoder exclusively from train labels. First estimate global priors π_k for k in {GALAXY, QSO, STAR}. Bin redshift into fixed edges [-0.01, 0.15, 0.40, 1.0, 1.4, 2.0, 2.8, 3.5, 4.5, 7.02] and i-band into fixed edges [-inf, 17, 18, 19, 20, 21, 22, 23, 24, 25, inf]. For each row define a hierarchical key chain: L4=(spectral_type, galaxy_population, redshift_bin, i_bin), L3=(spectral_type, galaxy_population, redshift_bin), L2=(spectral_type, galaxy_population), L1=global. For every observed key at each level compute class counts n_c(k), total n_c, and Dirichlet-smoothed posteriors p_c(k)=(n_c(k)+α·π_k)/(n_c+α) with α=30. Let λ_c=n_c/(n_c+β), β=120. For a row with counts at each level, blend priors by descending backoff: p*=λ_4 p_4 + (1-λ_4)[λ_3 p_3 + (1-λ_3)(λ_2 p_2 + (1-λ_2)π)]. If any key is unseen, skip its level and continue from the next parent; if no counts exist at all, use π. Clamp final p*_k to [1e-6,1-1e-6], then emit: p*_GALAXY, p*_QSO, p*_STAR, class entropy H=-Σ p*log p*, max probability, top-two gap, and one-vs-rest logit-like margins log((p*+eps)/(1-p*+3eps)) for each class. Map unexpected categorical levels at inference time directly to global priors before applying interpolation.
  expected_signal=The priors explicitly encode how class balance changes with redshift and brightness within known astrophysical metadata strata, which helps balanced_accuracy by improving recall in feature-overlap regions where color-only boundaries between stars, galaxies, and quasars are weak or unstable.
  risk=Because the features are target-derived, aggressive binning or too little smoothing can overfit sparse strata and create context-specific noise; in deployment, shifts in class prevalence by redshift, spectral_type, or i-band range would degrade calibration, so the hierarchical backoff and fixed shrinkage are required to avoid unstable outputs, and out-of-fold generation is needed if this is used inside internal validation.

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