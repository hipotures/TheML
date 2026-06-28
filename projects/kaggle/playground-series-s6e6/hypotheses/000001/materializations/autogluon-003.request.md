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
- title: Observed-frame ugriz SED shape diagnostics
- group_name: broadband_color_shape
- family: photometric_sed
- summary: Represent each object’s observed optical spectral-energy-distribution shape through deterministic broadband color, slope, and curvature descriptors that emphasize continuum geometry over absolute brightness.
- strategy: Use only the raw u, g, r, i, and z magnitudes, in fixed band order, with SDSS-like pivot wavelengths treated as constants in Angstroms: lambda_u=3562, lambda_g=4686, lambda_r=6165, lambda_i=7481, lambda_z=8931. Compute adjacent colors color_u_g=u-g, color_g_r=g-r, color_r_i=r-i, and color_i_z=i-z; add longer-baseline colors color_u_r=u-r, color_u_i=u-i, color_u_z=u-z, color_g_i=g-i, color_g_z=g-z, and color_r_z=r-z to summarize broad continuum tilt. For each adjacent pair, compute wavelength-normalized observed-frame slopes using natural-log wavelength spacing: slope_u_g=(u-g)/(ln(lambda_g)-ln(lambda_u)), slope_g_r=(g-r)/(ln(lambda_r)-ln(lambda_g)), slope_r_i=(r-i)/(ln(lambda_i)-ln(lambda_r)), and slope_i_z=(i-z)/(ln(lambda_z)-ln(lambda_i)). Capture local SED bends with both simple magnitude second differences, curve_u_g_r=u-2*g+r, curve_g_r_i=g-2*r+i, curve_r_i_z=r-2*i+z, and spacing-aware curvature from adjacent slopes: k_u_g_r=(slope_g_r-slope_u_g)/(0.5*(ln(lambda_r)-ln(lambda_u))), k_g_r_i=(slope_r_i-slope_g_r)/(0.5*(ln(lambda_i)-ln(lambda_g))), and k_r_i_z=(slope_i_z-slope_r_i)/(0.5*(ln(lambda_z)-ln(lambda_r))). Add one broad curvature term curve_u_r_z=u-2*r+z to capture large-scale blue-to-red bending. Keep all values per-row, deterministic, and unbinned; do not rank, target-encode, fit normalization parameters, or use class labels. Preserve signed values, including negative colors and extreme values. If any operand required for a derived feature is non-finite, emit null only for that affected feature while leaving all other computable features unchanged.
- expected_signal: Balanced accuracy may improve because colors convert magnitudes into approximate flux-ratio information, reducing dependence on distance and exposure-level brightness while retaining astrophysical shape cues. Adjacent colors and wavelength-normalized slopes should separate stellar temperature sequences, galaxy red/blue continua, and quasar UV-excess patterns, while curvature terms can expose spectral breaks or broad-band bends that raw magnitudes and simple categorical predictors may not isolate.
- risk: The features are highly correlated with the original magnitudes and with each other, so tree models may gain little while linear or distance-based models may need regularization. Observed-frame colors can overlap for stars and quasars at some redshifts, and endpoint or u-band-heavy descriptors may amplify photometric noise for faint objects or outliers.

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