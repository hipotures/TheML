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

Hypothesis ID: 000024

Previous versions:

- rev 1: AIDE catalog rank-frequency context
  group_name=aide_catalog_rank_frequency_context
  family=aide_catalog_context
  summary=Add the AIDE catalog-frequency and distribution-rank context: category frequencies, spectral-population cross frequency, global percentile ranks, quantile-bin frequencies, and within-category numeric ranks.

  strategy=From spectral_type and galaxy_population, create normalized frequency encodings for each category and for their crossed pair. For u, g, r, i, z, clipped redshift, and log1p clipped redshift, compute global percentile rank, a 12-bin quantile index, and the empirical frequency of that quantile bin. Also compute each numeric column's percentile rank within spectral_type and within galaxy_population. Add compact summaries across numeric ranks such as mean and standard deviation. All encodings are deterministic statistics over the available predictor rows and must not read the target.
  expected_signal=Rare catalog tags, rare tag combinations, and unusual numeric ranks within a catalog tag can distinguish minority classes and reproduce the AIDE solutions' frequency/rank context without requiring model-level ensembling.
  risk=Frequency and rank features can be sensitive to whether they are estimated on train only or combined train/test predictors, and they may duplicate signals already learnable from raw categoricals in strong tree models.

- rev 2: Catalog-Conditioned Frequency and Rank Context
  group_name=aide_catalog_rank_frequency_context
  family=aide_catalog_context
  summary=Create a compact context profile for each object that combines catalog-tag prevalence with global and catalog-conditioned rank/quantile position of key numeric observables, enabling the model to use how common or rare its metadata and value locations are.
  strategy=Fit all contextual statistics on training predictors only using train set rows after dropping the target. Compute unigram frequencies for spectral_type and galaxy_population and their 2-way joint frequencies using Laplace smoothing with alpha=1 on counts, then map every row to these frequencies and log-frequency values. For each numeric feature in {u, g, r, i, z, redshift}, define redshift_clip = clip(redshift, 0, max(0, train redshift 99.5% quantile)) and redshift_log = log1p(redshift_clip). For each base feature f in {u, g, r, i, z, redshift_clip, redshift_log}, compute: (1) global empirical percentile rank r_global=(rank(f)-0.5)/(N_train-0.5) using tie-average ranks on train and then mapped to test with train quantiles; (2) quantile-bin index q12 via 12 equal-frequency bins from train quantiles (q=[0..1] step 1/12, resolving duplicate edges by shrinked bins); if binning collapses to <12 bins, keep available bins and use the mapped train frequencies per final bin; (3) bin frequency f_bin = count(train rows in bin)/N_train; (4) percentile rank within each spectral_type, within each galaxy_population, and within each (spectral_type, galaxy_population) pair computed analogously to global rank with group-local ranks. For any test row in a single-sample or unseen group, fall back to global ranks for that feature. Add row-level summaries: mean and standard deviation of the seven global ranks, mean of the three within-group rank sets, and count of low-frequency bins where f_bin falls below the global 10th percentile. Concatenate all context fields with raw inputs only.
  expected_signal=Rare and atypical combinations of catalog tags and quantile-context frequently align with minority classes (especially QSO and STAR tails), and rank-based contrasts capture subtle survey-selection structure that raw magnitudes/redshift alone can miss, which should improve recall-sensitive classes under balanced_accuracy.
  risk=Residual distribution shift between train and test can make percentile/bin frequencies poorly calibrated, and tiny within-group samples for crossed categories can produce unstable ranks, so smoothing and fallback to global ranks are required; these engineered variables are also correlated with each other, adding redundancy and potential overfit in small-capacity models if regularization is weak.

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