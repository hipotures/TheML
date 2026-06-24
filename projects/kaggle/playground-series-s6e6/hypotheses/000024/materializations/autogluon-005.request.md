Write preprocessing feature-group code for an AutoGluon wrapper.

Your output must define one semantic feature group for preprocessing.
The fixed wrapper imports this file, runs `FEATURE_GROUPS`, logs group timing,
renames returned columns, assembles the final DataFrame, and handles all
non-preprocessing work.

Do not write the wrapper.

# Data Overview

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

# Hypothesis
- title: Catalog-Conditioned Frequency and Rank Context
- group_name: aide_catalog_rank_frequency_context
- family: aide_catalog_context
- summary: Create a compact context profile for each object that combines catalog-tag prevalence with global and catalog-conditioned rank/quantile position of key numeric observables, enabling the model to use how common or rare its metadata and value locations are.
- strategy: Fit all contextual statistics on training predictors only using train set rows after dropping the target. Compute unigram frequencies for spectral_type and galaxy_population and their 2-way joint frequencies using Laplace smoothing with alpha=1 on counts, then map every row to these frequencies and log-frequency values. For each numeric feature in {u, g, r, i, z, redshift}, define redshift_clip = clip(redshift, 0, max(0, train redshift 99.5% quantile)) and redshift_log = log1p(redshift_clip). For each base feature f in {u, g, r, i, z, redshift_clip, redshift_log}, compute: (1) global empirical percentile rank r_global=(rank(f)-0.5)/(N_train-0.5) using tie-average ranks on train and then mapped to test with train quantiles; (2) quantile-bin index q12 via 12 equal-frequency bins from train quantiles (q=[0..1] step 1/12, resolving duplicate edges by shrinked bins); if binning collapses to <12 bins, keep available bins and use the mapped train frequencies per final bin; (3) bin frequency f_bin = count(train rows in bin)/N_train; (4) percentile rank within each spectral_type, within each galaxy_population, and within each (spectral_type, galaxy_population) pair computed analogously to global rank with group-local ranks. For any test row in a single-sample or unseen group, fall back to global ranks for that feature. Add row-level summaries: mean and standard deviation of the seven global ranks, mean of the three within-group rank sets, and count of low-frequency bins where f_bin falls below the global 10th percentile. Concatenate all context fields with raw inputs only.
- expected_signal: Rare and atypical combinations of catalog tags and quantile-context frequently align with minority classes (especially QSO and STAR tails), and rank-based contrasts capture subtle survey-selection structure that raw magnitudes/redshift alone can miss, which should improve recall-sensitive classes under balanced_accuracy.
- risk: Residual distribution shift between train and test can make percentile/bin frequencies poorly calibrated, and tiny within-group samples for crossed categories can produce unstable ranks, so smoothing and fallback to global ranks are required; these engineered variables are also correlated with each other, adding redundancy and potential overfit in small-capacity models if regularization is weak.

# Group Code Contract

Return only Python code. Do not use markdown fences.

Define semantic feature-group functions and `FEATURE_GROUPS`.

Generate only the feature-group module: Python definitions for feature-group
preprocessing. A separate fixed runtime wrapper imports this module and is
responsible for logging, timing, dependency ordering, output-column renaming,
final DataFrame assembly, and all non-preprocessing work.

Each feature function must use this signature:

```python
def add_group_name(raw, deps, aux):
    ...
    return new_features
```

Rules:
- `raw` is the raw/base train+test covariate frame without target labels. It includes ID columns.
- `deps` is a dict of dependency outputs by logical group name. Use it only when this group declares dependencies.
- `aux` is an auxiliary DataFrame when available, otherwise empty.
- Return a pandas DataFrame containing only new local feature columns with `index=raw.index`.
- Preserve row count, row order, and index exactly.
- Do not return raw/input columns.
- Do not mutate `raw`, `deps`, or `aux` in place.
- Use clear local feature names. The executor will rename returned columns after the function finishes.
- Outputs may be numeric, boolean, categorical, or string scalar columns. Do not return nested lists, dicts, tuples, or sets.
- You may compute covariate-only train+test statistics from `raw`; do not use target labels, validation labels, model outputs, or leaderboard feedback.
- Do not read project data files, write files, train models, create `main()`, concatenate final blocks, or implement orchestration.
- Do not implement timing decorators or logging wrappers. The group executor logs every group call and duration.
- Top-level code may contain only imports, function definitions, literal constants, and `FEATURE_GROUPS`.
- Do not call functions in top-level assignments. For example, do not write `EDGES = np.array(...)`, `CUTS = pd.IntervalIndex(...)`, or any other assignment whose right-hand side calls a function or constructor.
- If a constant needs conversion to a NumPy/Pandas object, store it as a literal tuple/list at module level and convert it inside the feature function.

Register groups like this:

```python
FEATURE_GROUPS = [
    {
        "name": "group_name",
        "fn": add_group_name,
        "depends_on": [],
        "description": "One sentence describing this feature group.",
    }
]
```

AutoGluon wrapper boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not define `preprocess(df)`.
- Do not read project data files or `./input`.
- The fixed AutoGluon wrapper handles all of those steps outside this generated file.