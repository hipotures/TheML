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
- title: Lyman-aware rest-frame SED family fit with stability-aware dual residual diagnostics
- group_name: restframe_sed_family_fit
- family: spectral_family
- summary: This feature family characterizes each object by how well its rest-frame ugriz continuum is represented by a low-order power-law-like shape versus a curved continuum family, using redshift-normalized wavelength geometry and absorption-aware weighting to separate quasar-like, stellar, and galaxy-like spectral morphology.
- strategy: For each object define zc = max(redshift, 0), and use rest-frame effective wavelengths x_b = log10(lambda_b/(1 + zc)) with lambda=[3551, 4686, 6165, 7481, 8931] Å for b in {u,g,r,i,z}. Convert magnitudes to relative log-flux y_b = -0.4 * m_b (equivalent to log10 of flux up to a constant) and center y by subtracting its object-wise mean so absolute scale is removed. Fit weighted linear model y = a + b*x' and weighted quadratic model y = a + b*x' + c*x'^2 where x' is x centered per object to reduce conditioning issues, using weighted normal equations with explicit weights. Produce one feature block with weights w=1 for all bands (base model), and one block with Lyman-aware weights v: v=1 if lambda_b/(1+zc) > 1216, v=0.5 if 912 < lambda_b/(1+zc) <= 1216, and v=0 otherwise, to de-emphasize IGM-absorbed regions. For each block compute RSS, R2, slope b, curvature c, curvature magnitude |c|, and curvature gain g = (RSS_linear - RSS_quadratic) / RSS_linear (only when both fits are estimable); add a signed asymmetry slope feature s = (y_u - y_g)/(x_u - x_g) - (y_i - y_z)/(x_i - x_z) with a deterministic fallback pair (g-r) minus (r-i) if either endpoint is ineligible due heavy masking. Define fit-feasibility flags by point support: require at least 2 non-negligible bands for linear and 3 for quadratic; if a band is downweighted to zero and a fit is infeasible, set its statistics to NaN and use linear/unweighted estimates for that branch with a binary instability flag. Add branch-difference features: ΔR2 = R2_weighted - R2_unweighted, Δb = b_weighted - b_unweighted, Δc = c_weighted - c_unweighted, plus a flag for no_uv_downweight (all v=1). If redshift is clamped to 0 (negative inputs), set z_low_flag to indicate observer-frame fallback is used for this object.
- expected_signal: The quasar-like class should retain near-power-law behaviour once shifted toward rest-frame and after suppressing UV-absorbed points, while many stars and galaxies should show stronger curvature and slope asymmetry from stellar/galaxy continua and breaks, so the contrast between linear and quadratic fit quality plus asymmetry under masked versus unmasked regimes should improve class boundary separation where colors alone are ambiguous.
- risk: At very high redshift or noisy redshift values, rest-frame compression and band suppression can leave too few informative points, causing unstable curvature estimates and increased missing-flag rate; the weighted masking can also overlap with intrinsic class-correlated redshift distribution, creating potential overfitting to the redshift-induced regime unless stability flags and branch-differences are treated with care.

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