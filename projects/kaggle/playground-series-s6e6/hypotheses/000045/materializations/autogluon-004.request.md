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
- title: Intrinsic-Rank Pairwise Color Lattice Residuals
- group_name: pairwise_color_lattice_residuals
- family: pairwise_color_geometry
- summary: Characterize each object by redshift-local residual geometry of the complete ugriz pairwise color lattice, using robust centering and intrinsic low-rank orthogonal directions to expose non-adjacent color-manifold departures relevant to class separation.
- strategy: Compute the 10 pairwise color differences from magnitudes in fixed order: u-g, u-r, u-i, u-z, g-r, g-i, g-z, r-i, r-z, i-z. Fit all statistics on training data only within each validation fold, and on full training data for final test generation. Define redshift bins from training redshift quantiles with nq=32, then merge adjacent sparse bins until every final bin has at least m_min=max(1500, ceil(0.001*|train|)) rows; freeze these boundaries and assign rows by clipped lookup, including negative redshift values in the lowest applicable bin. For each bin, compute component-wise medians mu_b and MAD scales sigma_b over the 10-color vectors, replacing sigma_b<1e-6 with 1e-6. Create standardized residual vector s=clip((x-mu_b)/sigma_b, -8, 8) and emit its 10 components. Because the 10 pairwise colors are linearly dependent, compute orthogonal geometry in the empirical intrinsic subspace rather than inverting an unstable full 10x10 covariance: run deterministic PCA/SVD on the bin's standardized residual matrix, keep components with eigenvalue greater than max(1e-8, 1e-6*largest_eigenvalue), and cap retained components at 4, the maximum rank implied by five magnitudes. Emit the first 3 signed-deterministic principal residual scores, padding missing components with 0 if fewer than 3 valid axes exist. Also emit a regularized PCA-space Mahalanobis distance sqrt(sum_j score_j^2 / (lambda_j + ridge_b)) over retained axes, where ridge_b=1e-6*mean(lambda_j). If a row's bin is missing, too small after merging, or has no valid PCA axis, use the nearest populated bin by redshift midpoint, then global training statistics as final fallback.
- expected_signal: The full pairwise lattice makes long-baseline color contrasts explicit while redshift-local residualization removes broad magnitude-color trends that can blur class boundaries; using the intrinsic color subspace should preserve useful orthogonal deviation signals without making the geometry numerically dominated by the known linear dependencies among pairwise colors.
- risk: The added residuals are still highly correlated with raw magnitudes and simple colors, so flexible models may overfit redundant signal; redshift-bin boundaries can introduce discontinuities near cut points; sparse high-redshift bins may yield noisy PCA axes despite merging; incorrect fold handling would leak validation distribution information into the residual statistics.

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