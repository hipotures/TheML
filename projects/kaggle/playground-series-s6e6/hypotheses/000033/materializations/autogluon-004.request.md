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
- title: Stability-aware Lyman-weighted rest-frame SED family residuals
- group_name: restframe_sed_family_fit
- family: spectral_family
- summary: Characterize each object by the relative adequacy of power-law-like and smoothly curved rest-frame ugriz continuum families, with absorption-aware variants that preserve shape information while reducing sensitivity to absolute brightness.
- strategy: For each object use SDSS-like effective wavelengths lambda=[3551,4686,6165,7481,8931] Angstrom for bands {u,g,r,i,z}. Set zc=max(redshift,0) and record z_low_flag=1 when redshift<0. Compute rest wavelengths lambda_rest_b=lambda_b/(1+zc), log-wavelength x_b=log10(lambda_rest_b), and relative log-flux y_b=-0.4*m_b. Center x and y within the object as x0_b=x_b-mean(x_b) and y0_b=y_b-mean(y_b), so the fits describe spectral shape rather than brightness. Fit two nested weighted least-squares models for each weighting branch: linear y0=a+b*x0 and quadratic y0=a+b*x0+c*x0^2. Use a base branch with all weights equal to 1 and a Lyman-aware branch with weights w_b=1 for lambda_rest_b>1216, w_b=0.5 for 912<lambda_rest_b<=1216, and w_b=0.15 for lambda_rest_b<=912, keeping a no_uv_downweight flag when all Lyman weights are 1. Use the same deterministic solver for both branches; if the weighted design matrix is rank deficient, fall back to the lower-rank estimable model, set unavailable higher-order statistics to NaN, and emit linear_feasible, quadratic_feasible, and lyman_instability flags. For each branch output linear RSS, quadratic RSS, weighted R2 for both models, fitted slope b, fitted curvature c, abs(c), signed curvature c, curvature gain clipped to [-1,1] as (RSS_linear-RSS_quadratic)/max(RSS_linear,eps), residual standard deviation, and maximum absolute standardized residual across bands. Add local shape contrasts computed on centered values: blue slope (u-g), red slope (i-z), endpoint slope contrast blue_slope-red_slope, and middle curvature contrast (g-r)-(r-i), using the same branch weights but reporting NaN plus a support flag if either required pair has negligible combined support. Finally add branch-difference features for R2, RSS ratio, slope, curvature, curvature gain, endpoint slope contrast, and maximum residual between Lyman-aware and unweighted fits.
- expected_signal: Quasars should often retain a relatively linear log-flux versus log-wavelength continuum after rest-frame shifting and UV absorption downweighting, while stars and galaxies should more often require curvature or show asymmetric breaks; balanced accuracy can improve because these residual and branch-difference diagnostics expose class morphology not fully captured by raw colors or redshift alone.
- risk: The features partly depend on redshift and Lyman-regime membership, so they may overfit class-correlated redshift distributions rather than intrinsic SED shape; high-redshift objects can have weak support in several bands, making curvature and branch-difference estimates unstable despite flags, and the group may be redundant with simpler color, color-ratio, or spectral-break feature groups.

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