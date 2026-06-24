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
- summary: Create rest-frame, interpolation-based descriptors of the broadband spectral shape at fixed physical wavelengths so class-specific continuum slope, curvature, and 3650 Å-break behavior are compared on a common spectral grid rather than in observed-frame filter space.
- strategy: For each object, use ugriz magnitudes m_b and effective wavelengths λu=3551, λg=4686, λr=6165, λi=7481, λz=8931 Å. Set den= max(1+z, 1.0e-3) to avoid division-by-near-zero for the rare negative-redshift tail and compute rest wavelengths λ′b = λb/den. Convert each magnitude to a log-flux surrogate y_b = -0.4·m_b (absolute scale irrelevant for shape); set x_b = log10(λ′b). Sort the five (x_b, y_b) pairs by x_b and build piecewise-linear interpolation y(x). Define fixed anchors A={1500,2200,2900,3650,4500,6200,7600} Å with xa=log10(a). For each anchor, if xa is inside [min(x_b), max(x_b)] evaluate y_a = y(xa), else set y_a=NaN; also create anchor_mask_a∈{0,1}, n_anchor and frac_anchor as coverage indicators. Compute segment slopes only from adjacent defined anchors: s_i = Δy/Δx for the six possible adjacent anchor intervals, and carry per-segment masks. Define curvature deltas with available slopes as Δs_uv=(s2200-2900)-(s1500-2200) and Δs_opt=(s3650-4500)-(s2900-3650), setting NaN if any endpoint is missing. Compute a 3650-Å break residual b4000 = y(3650)−ŷ where ŷ is linear interpolation of y at 3650 from defined anchors at 2900 and 4500; if either bracketing anchor is missing set b4000=NaN. Fit a global OLS line y = a_g x + b_g over all defined anchors (require n_anchor≥2), then output a_g, b_g, RMSE_global and mean absolute residual over defined anchors. Split anchors at xa=3650: UV side x≤3650 and optical side x>3650. Fit separate OLS lines when each side has at least 2 points; output a_blue, b_blue, RMSE_blue, a_red, b_red, RMSE_red, and a break-sharpness metric b_break = ŷ_blue(3650)−ŷ_red(3650); set all side metrics to NaN if prerequisites fail. Add stability features as mean and max of |y_a−ŷ_global(xa)| over defined anchors and a no-data flag (n_anchor<2) to keep downstream handling explicit.
- expected_signal: With anchors anchored in physical wavelength, slope and curvature descriptors become directly comparable across redshift, allowing the model to capture the stronger rest-frame 3650 Å-break curvature typical of galaxies, the smoother broken power-law-like continua of quasars, and the smoother, weaker-curvature profile of stars despite the wide z span in the sample.
- risk: Anchor-wise interpolation can become unstable when only 2–3 ugriz points are valid in rest-frame at extreme redshift, and side-specific breaks may collapse into missing values, increasing sparsity; if class is partially confounded with redshift sampling, rest-frame shape features can absorb redshift-correlated artifacts instead of intrinsic spectral differences, and they may be redundant with existing color/redshift features unless downstream regularization handles correlation.

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