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
- title: Smoothed Redshift-Conditioned Population Color Manifold Deviations
- group_name: population_color_manifold_drifts
- family: tag_conditioned_color_geometry
- summary: Represent each object by how well its broadband color geometry agrees with the redshift-local color manifold implied by its supplied galaxy-population tag compared with the alternative population manifold.
- strategy: Compute color contrasts u-g, g-r, r-i, i-z, u-r, g-i, and r-z from the magnitude columns. Transform redshift as t=log1p(max(redshift,-0.009999)) and build B=24 equal-frequency bins using training rows only; store bin edges and bin centers, then assign train/test rows deterministically with clipping to the edge range. For each population p in {Red_Sequence, Blue_Cloud} and each redshift bin b, estimate robust per-color medians mu_{b,p,f} and robust scales sigma_{b,p,f}=1.4826*MAD_{b,p,f}. Also estimate a local robust correlation matrix from clipped standardized color residuals when the cell has enough rows; shrink this matrix toward identity with lambda=N_min/(n_{b,p}+N_min), using N_min=300, and always add a small diagonal ridge. For cells with n<N_min or near-zero scale, shrink medians and scales toward the corresponding global population-level statistics, then toward global all-training statistics if the population-level fallback is still unstable. To reduce hard bin discontinuities, compute each row's reference statistics by linear interpolation between the two nearest bin centers for the same population; at the redshift extremes use the nearest bin only. For an object with population tag p, compute clipped standardized residual vectors against both the assigned population manifold and the opposite population manifold, with clipping to [-8,8]. Emit diagonal robust distance summaries for assigned and opposite manifolds, shrinkage-covariance Mahalanobis distances for assigned and opposite manifolds, their log distance difference and ratio, and the minimum distance across the two manifolds. Emit per-color signed separator margins using the interpolated midpoint between Red_Sequence and Blue_Cloud medians and the sign of their median gap, plus absolute residuals to each population median. Add reliability features such as the effective cell counts, interpolation weight, and shrinkage weight so the downstream model can discount distances derived from sparse manifold estimates.
- expected_signal: These features should clarify whether an object follows a plausible galaxy-population color trajectory at its redshift or instead has colors more typical of stars or quasars, improving class boundaries in regions where raw magnitudes and redshift alone overlap and supporting balanced accuracy through better minority-class recall.
- risk: The supplied population tags may be weak or synthetic for non-galaxy objects, so tag-conditioned deviations can encode misleading structure; redshift-local statistics may overfit dense training artifacts despite shrinkage; covariance estimates add redundancy and instability in sparse bins; and these features may duplicate signal already captured by strong tree models from raw colors and redshift.

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