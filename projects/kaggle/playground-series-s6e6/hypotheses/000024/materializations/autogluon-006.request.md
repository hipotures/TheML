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
- title: Catalog-Conditioned Rank and Frequency Context
- group_name: aide_catalog_rank_frequency_context
- family: aide_catalog_context
- summary: Create unsupervised catalog-context features that describe how common each object's metadata tags are and how unusual its photometric and redshift values are globally and within comparable catalog groups.
- strategy: Build a reusable encoder using predictor columns only. For validation, fit all statistics inside each training fold and transform the held-out fold; for final submission, refit on all labeled training predictors and transform test. Encode spectral_type, galaxy_population, and their crossed pair with smoothed normalized counts: freq=(count+1)/(N+K), plus log_freq=log(freq), using the global smoothed minimum for unseen categories. For numeric context, use u, g, r, i, z, redshift_clip, and redshift_log, where redshift_clip=clip(redshift, 0, train_99_5_percentile_redshift) and redshift_log=log1p(redshift_clip). For each numeric feature, compute global empirical percentile rank from the fitted rows with average ties and map new rows by search-sorted train values, clipped to [0,1]. Also assign a quantile-bin id from up to 12 equal-frequency train quantile bins; duplicate quantile edges are collapsed, and each row receives the empirical fitted-row frequency and log-frequency of its final bin. Repeat percentile-rank mapping within spectral_type, within galaxy_population, and within the crossed pair, requiring a minimum fitted group size of 30; smaller or unseen groups fall back to the corresponding global percentile rank and receive a binary fallback indicator. Add compact row summaries over the seven numeric features: mean, standard deviation, minimum, maximum, and range of global ranks; mean absolute difference between global rank and each available group-conditioned rank family; mean quantile-bin frequency; and count of numeric features whose bin frequency is below the fitted 10th percentile of bin frequencies. Concatenate these context features with the raw predictors without using class labels.
- expected_signal: Balanced accuracy can benefit because rare catalog tags, rare tag combinations, and objects that are photometrically or redshift-wise atypical within their catalog context may correspond to underrepresented STAR or QSO regions that raw values and simple categorical splits do not isolate cleanly.
- risk: The features are highly correlated with raw magnitudes, redshift, and each other, so weakly regularized models may overfit; fold-inconsistent fitting can inflate validation scores; small crossed groups can produce noisy conditional ranks, requiring minimum-size fallback indicators; and train-test distribution shift can make fitted quantiles and frequencies less calibrated on the test set.

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