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
- title: Rest-frame anchored SED slope and break profile
- group_name: restframe_anchor_sed_shape
- family: restframe_shape
- summary: Create rest-frame descriptors of broadband spectral shape on fixed physical wavelength anchors so continuum slope, curvature, and stellar-break structure are compared consistently across redshift rather than through drifting observed-frame filters.
- strategy: For each object, use ugriz magnitudes and effective wavelengths λu=3551, λg=4686, λr=6165, λi=7481, λz=8931 Å. Convert magnitudes to log-flux surrogates y_b=-0.4*m_b and compute rest wavelengths λ'_b=λ_b/max(1+redshift, 1.0e-3), with x_b=log10(λ'_b). Sort the five (x_b,y_b) pairs and linearly interpolate y as a function of x in log-wavelength space. Evaluate anchors A=[1500,2200,2900,3650,4500,6200,7600] Å only when log10(A) lies within the observed rest-frame wavelength span; otherwise set the anchor value to NaN. Emit one binary coverage mask per anchor, n_anchor, frac_anchor, and min/max covered anchor wavelength. For defined anchors, create relative anchor levels yrel_a=y_a−median(y_defined) to remove absolute brightness scale while preserving SED shape. Compute adjacent slopes s_1500_2200 through s_6200_7600 as Δy/Δlogλ when both anchors exist, with segment masks. Derive curvature terms from consecutive available slopes, especially uv_curv=s_2200_2900−s_1500_2200, near_break_curv=s_3650_4500−s_2900_3650, red_curv=s_6200_7600−s_4500_6200, and aggregate mean, standard deviation, and maximum absolute adjacent-slope change over all valid neighboring segments. Compute a 3650 Å break residual b3650=y_3650−linear_prediction(y_3650 from anchors 2900 and 4500), plus a broader break contrast (mean yrel over anchors >=4500)−(mean yrel over anchors <=2900) when both sides are present. Fit a centered global OLS line y=a*x+c over all defined anchors when n_anchor>=2 and output only shape terms: global slope a, RMSE, mean absolute residual, max absolute residual, and residual at 3650 if available. Fit separate centered blue and red lines for anchors <=3650 and >=3650 when each side has at least two points; output blue_slope, red_slope, slope_jump=red_slope−blue_slope, side RMSEs, and break_sharpness as the difference between the two side-line predictions at 3650. All undefined quantities remain NaN and are accompanied by prerequisite/coverage masks rather than imputed constants.
- expected_signal: Balanced accuracy may improve because galaxies should show stronger rest-frame optical break and curvature structure, quasars should often resemble smoother broad power-law continua, and stars should have smoother stellar-continuum slopes; using physical wavelength anchors makes these patterns comparable across redshift while masks let the model learn when the estimate is reliable.
- risk: At high redshift or very low redshift many anchors may be outside the ugriz rest-frame span, making the feature group sparse and sensitive to interpolation endpoints. Because redshift strongly affects anchor coverage, the masks and break features can learn redshift-selection artifacts as well as intrinsic spectral shape. The features are also correlated with color and redshift-derived groups, so overly flexible downstream models may overfit unless validation preserves the train-test distribution.

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