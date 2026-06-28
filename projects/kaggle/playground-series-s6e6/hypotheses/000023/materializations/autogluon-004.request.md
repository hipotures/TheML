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
- title: Shrunk redshift-bin color residuals
- group_name: aide_redshift_bin_color_residuals
- family: aide_photometric_redshift
- summary: This feature group measures how unusually each object’s color and color-curvature profile behaves relative to the typical profile of training objects at similar redshift, exposing local photometric deviations that raw colors and redshift alone may not separate cleanly.
- strategy: Derive the fixed ten AIDE color-shape channels for every row: u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, u-2*r+i, and g-2*i+z. Fit all bin edges and statistics on training rows only, and in cross-validation refit them inside each training fold. Define z0 = clip(redshift, 0.0, 7.01). Build 24 quantile redshift bins from z0_train using quantiles k/24 for k=0..24, drop duplicate edges, and require at least 12 usable bins; otherwise use 24 equal-width edges over [min(z0_train), max(z0_train)]. Assign train and test rows with b = clip(searchsorted(edges, z0, side='right') - 1, 0, n_bins - 1), so values at or outside fitted limits map deterministically to the nearest edge bin. For each bin b and channel c, compute count n_b, mean mu_b_c, and sample standard deviation sd_b_c from training rows; also compute global training mean mu_all_c and sample standard deviation sd_all_c. Stabilize sparse bins with shrinkage weight w_b = min(1.0, n_b / 400.0): mu_hat_b_c = w_b * mu_b_c + (1 - w_b) * mu_all_c and sd_hat_b_c = max(w_b * sd_b_c + (1 - w_b) * sd_all_c, 1e-6). If a bin has fewer than 2 rows, use the global standard deviation before shrinkage. For each row, emit per-channel signed residuals d_c = x_c - mu_hat_b_c and standardized residuals s_c = clip(d_c / sd_hat_b_c, -8.0, 8.0). Also emit aggregate residual summaries across the ten channels: sqrt(sum(d_c^2)), mean(abs(d_c)), mean(d_c), standard deviation of d_c, mean(s_c), standard deviation of s_c, and max(abs(s_c)). The transform is deterministic, target-free, and uses only fitted training statistics when applied to test rows.
- expected_signal: Balanced accuracy can improve because the classes are not only separated by absolute color and redshift, but also by how their photometric shape departs from the local redshift-conditioned manifold; per-channel residuals retain directional class cues, while aggregate residual norms capture broad color inconsistency patterns that may distinguish quasars, stars, and galaxies in minority-sensitive scoring.
- risk: The features can be redundant with raw color, redshift, and other photometric-consistency groups, and redshift noise or edge-bin sparsity can create unstable residuals; fitting bin statistics outside a training fold would leak validation distribution information, so fold-local fitting is required for honest validation.

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