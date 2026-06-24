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
- title: Redshift-stratified residualization of AIDE color-shape channels
- group_name: aide_redshift_bin_color_residuals
- family: aide_photometric_redshift
- summary: The feature group characterizes how far each object’s photometric color and color-curvature profile departs from the expected profile at similar redshift, so model decisions can use class-specific deviations from the local manifold rather than raw color trends dominated by redshift effects.
- strategy: Use redshift clipped to the physically valid support z0 = min(max(redshift, 0.0), 7.01). On training rows only, build 24 quantile breakpoints for binning: e_k = quantile(z0_train, k/24), k=0..24. Remove duplicate breakpoints; if the resulting number of edges is too small (for example less than 13), fall back to 24 equal-width edges over [min(z0_train), max(z0_train)] to keep deterministic bin counts. Let edges be E[0..m], with m-1 bins. Assign each row to bin b = clip(searchsorted(E, z0, side='right') - 1, 0, m-2). For each of the fixed nine AIDE color-shape channels (u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, u-2*r+i, g-2*i+z), compute per-bin count n_b, mean mu_b_c, and standard deviation sigma_b_c (ddof=1), plus global mu_all_c and sigma_all_c. For sparse bins, apply shrinkage: w_b = min(1, n_b/(2*200)); mu_tilde_b_c = w_b*mu_b_c + (1-w_b)*mu_all_c; sigma_tilde_b_c = w_b*sigma_b_c + (1-w_b)*sigma_all_c, with sigma_tilde floor at 1e-6. For each row and channel, compute residual d_i_c = x_i_c - mu_tilde_{b_i,c} and standardized residual s_i_c = d_i_c / sigma_tilde_{b_i,c}. Emit aggregated row-level features: root sum squares of d_i_c, mean absolute d_i_c, mean d_i_c, standard deviation of d_i_c, mean s_i_c, and max absolute s_i_c.
- expected_signal: At fixed redshift, stars, galaxies, and quasars occupy different trajectories in color space, so subtracting the local redshift-conditioned center exposes class- and subtype-specific offsets, while normalized residuals reduce dominance of broad redshift drift and amplify physically relevant curvature differences that are useful for balanced multiclass discrimination.
- risk: Using noisy redshift as the conditioning variable can cause boundary effects and unstable bin statistics, especially where bins are sparse; shrinkage lowers variance but may still under- or over-smooth class signal, and the resulting features are likely to overlap substantially with other color-based families.

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