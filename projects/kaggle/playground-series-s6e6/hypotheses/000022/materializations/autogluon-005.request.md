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
- title: Fold-safe hierarchical sky-cell residuals
- group_name: aide_sky_cell_local_residuals
- family: aide_spatial_context
- summary: Encode each object by how dense and photometrically unusual its local sky neighborhood is, using smoothed sky-cell context to capture spatial survey effects without using labels.
- strategy: Build deterministic sky cells from coordinates with ra_bin = floor(mod(alpha, 360) * 2) and dec_bin = floor((delta + 90) * 2), clipped to valid DEC bins and wrapping RA modulo 720. For each row, compute n_cell, n_neigh over the RA-wrapped 3x3 neighborhood, log1p counts, concentration = n_cell / (n_neigh + 1e-6), and within-cell fractional offsets from the cell center. Construct base numeric descriptors u, g, r, i, z, clipped redshift, colors u-g, g-r, r-i, i-z, u-i, u-r, g-z, r-z, and curvature-like shapes u-2*r+i and g-2*i+z. For validation, fit all sky statistics only on the training fold; for final inference, fit unsupervised statistics on the available training predictors and apply the same mapping to test, optionally including test predictors only if the competition protocol permits transductive feature construction. For each descriptor, compute cell and 3x3-neighborhood means and standard deviations, then shrink them toward global statistics with w = n_ref / (n_ref + 30), where n_ref is the supporting count for that statistic. Clamp standard deviations by a small per-feature floor such as the 1st percentile of nonzero cell scales or 1e-3, whichever is larger. Add signed residuals x - mu_shrunk and standardized residuals residual / sigma_shrunk for both cell and neighborhood references. Summarize residuals across magnitude, color, and shape subsets separately using mean signed residual, mean absolute residual, L1 norm, L2 norm, standard deviation, maximum absolute z-score, and count of valid residuals. For cells absent from the fitted reference, fall back to the 3x3 neighborhood if populated, otherwise to global statistics, and mark the fallback level with a small integer feature.
- expected_signal: Balanced accuracy may improve because class separability can depend on whether an object is locally bright, color-shifted, or redshift-unusual relative to nearby observations, which can expose spatial calibration, depth, extinction, and selection patterns not captured by raw photometry alone.
- risk: The features can overfit spatial artifacts or validation leakage if sky statistics are computed across held-out rows, may be unstable in sparse cells despite shrinkage, and can be redundant with raw coordinates or existing color features unless the model regularizes correlated predictors.

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