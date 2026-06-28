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
- title: Redshift-Local SED PCA Residuals
- group_name: redshift_partitioned_sed_pca_residuals
- family: manifold_projection
- summary: Represent each object's normalized ugriz spectral-energy shape by how well it lies on a redshift-local low-dimensional photometric manifold, exposing class-specific continuum mismatches and band-profile anomalies.
- strategy: Convert ugriz magnitudes to linear positive fluxes in double precision with f_b = 10^(-0.4 * m_b), optionally clipping magnitudes only to broad train-derived percentiles before exponentiation to avoid numerical domination by extreme values. Form the scale-free shape vector p_b = f_b / (sum_b f_b + 1e-12). Build label-blind redshift bins from training rows using quantile edges, starting with about 20 bins, then merge adjacent sparse bins until each fitted bin has at least max(3000, 0.25% of N_train) rows; keep a global fallback PCA fit over all training rows. For each populated bin, compute train-only mean and standard deviation of the five p components with a small std floor, standardize x = (p - mu) / sigma, fit PCA on x, retain K = min(3, rank) leading components, and keep the remaining orthogonal components when available. At transform time, clip redshift only for bin assignment to the train redshift range, assign to the matching populated interval or nearest populated bin midpoint, standardize with that bin's stored statistics, project onto retained PCs, reconstruct x_hat, and compute residual e = x - x_hat plus original-scale residual r = sigma * e. Emit retained PC scores, reconstruction error L2 in standardized space, relative standardized error, residual energy fraction sum(e^2)/(sum(x^2)+eps), omitted-PC signed coordinates for PC4 and PC5 when present, per-band original-scale signed residuals r_u..r_z, absolute per-band residuals, max absolute residual band index/value, and the empirical train-bin percentile rank of standardized reconstruction error. Winsorize emitted residual magnitudes using train-only per-feature caps and use fixed epsilons for all divisions.
- expected_signal: Balanced accuracy may improve because redshift-local normalization separates ordinary smooth color evolution from unusual SED shape departures, giving the classifier explicit distance and direction information for quasars and ambiguous stars or galaxies that raw magnitudes, colors, and redshift alone may blur together.
- risk: The features can be redundant with existing color and curvature descriptors, sparse redshift tails may produce unstable local bases despite merging, faint-band photometric noise can inflate residuals, and train-bin percentile features must be computed strictly from training folds during validation to avoid optimistic leakage.

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