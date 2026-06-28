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
- title: Boundary-safe smooth spline SED interactions
- group_name: aide_smooth_spline_sed_interactions
- family: aide_smooth_basis
- summary: Represent smooth nonlinear structure across sky position, photometric SED shape, redshift, and a small set of color-redshift manifolds using robust spline bases plus compact tensor interactions.
- strategy: Compute all transformation statistics on the training predictors only and reuse them unchanged for validation/test. Derive color indices c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z, c_ur=u-r, c_gi=g-i, c_rz=r-z, and c_uz=u-z. For alpha, wrap values with alpha % 360 and build a cyclic cubic spline basis over the fixed [0, 360) domain with 7 degrees of freedom, dropping any constant or linearly redundant column so continuity is preserved at the 0/360 boundary. For delta, u, g, r, i, z, redshift, and all derived colors, impute unexpected missing values with the corresponding training median, then clip to robust training bounds before expansion. Use [q0.001, q0.999] bounds for magnitudes and colors to retain rare but plausible photometric extremes, [q0.001, q0.999] for delta, and for redshift use lower_bound=max(0, q0.001(redshift)) and upper_bound=q0.999(redshift) so tiny negative measurement noise is removed without discarding the high-redshift QSO tail too aggressively. Fit non-cyclic cubic B-spline bases with 5 degrees of freedom per clipped non-periodic variable using training quantile knots, then retain the non-constant basis columns after centering each emitted column by its training mean. To keep the group compact, create tensor-product interactions only from the first two retained basis columns of redshift crossed with c_ug, c_gr, and c_ri, plus c_gr crossed with c_ri; name and order these deterministically. Do not use target labels, class frequencies, predictions, fold-specific outcomes, or model artifacts when constructing the features.
- expected_signal: Spline bases expose smooth curved class boundaries in magnitude, color, sky-position, and redshift space, while the targeted tensor terms make redshift-dependent color separation easier for downstream learners, which can help balanced accuracy especially for QSO and STAR regions that are not well represented by raw threshold splits alone.
- risk: The expansion introduces correlated columns that may be redundant for strong tree models and can increase variance for linear or distance-based learners; quantile clipping and fixed low-order interactions may also underrepresent rare extreme-redshift or unusual-color objects if those tails contain class-specific signal.

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