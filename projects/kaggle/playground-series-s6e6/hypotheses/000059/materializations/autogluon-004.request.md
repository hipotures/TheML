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
- title: Redshift-Binned Tag Compatibility Margins
- group_name: tag_redshift_compatibility_residuals
- family: tag_manifold_consistency
- summary: Quantify whether an object's broadband color pattern is typical for its catalog tag combination at comparable redshift, and whether competing tag regimes provide a more plausible explanation.
- strategy: Compute adjacent color vector C=[u-g, g-r, r-i, i-z]. Fit all bin edges and profile statistics on training data only. Transform redshift as t=log1p(max(redshift,0)) and create quantile redshift bins on train, starting from about 100-200 candidate bins and merging adjacent bins until each final bin has at least 3000 total training rows; assign test rows by these fixed edges with boundary clipping. Define tag state s=spectral_type x galaxy_population, giving 8 possible states. For every occupied (s,bin), estimate a robust profile: center mu as componentwise median(C), scale sigma as max(1.4826*MAD, sigma_floor) with sigma_floor taken as a small fraction of the global training color scale, and covariance from residuals after componentwise winsorization to the cell P1-P99 range. Stabilize covariance by shrinkage toward the global diagonal color covariance plus a small diagonal jitter before inversion. Also fit state-level all-redshift profiles and one global profile as fallbacks. For each row and for every candidate state j, select the profile in this order: exact (j,current_bin) if count>=150, nearest neighboring bin for state j within an expanding window if count>=150, state-level all-redshift profile if count>=150, otherwise global profile. Compute squared Mahalanobis distance d_j2 and diagonal standardized residuals z_j=(C-mu_j)/sigma_j for all 8 candidate states, clipping distances to the train-fitted P0.5-P99.5 range for numerical stability. Let j_self be the row's observed tag state, d_self=d_j_self, d_best_alt and d_second_alt be the smallest and second-smallest distances among the other 7 states, and j_best be the best alternative state. Emit d_self, d_best_alt, d_second_alt, margins d_best_alt-d_self and d_second_alt-d_self, rank of self among all 8 distances, distance entropy or softmin concentration over states, the four z_j_self residuals, the four z_j_best residuals, and fallback-depth indicators for self and best alternative. If target-class frequency priors by tag-state/bin are included, compute them out-of-fold for training rows and from full training data only for test rows, using the same fallback chain and smoothed class counts.
- expected_signal: Balanced accuracy may improve because the features expose cases where the provided catalog tags and redshift imply one color manifold but the observed colors fit another, which is especially useful for separating rare QSO and STAR examples from the dominant galaxy-like regions.
- risk: The features can overfit tag-redshift artifacts if train/test tag quality differs, and class-frequency priors are leakage-prone unless computed out-of-fold for validation; sparse state-bin profiles require careful fallback, covariance estimates may be unstable without shrinkage, and the resulting distances may be partly redundant with raw colors and redshift.

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