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
- title: Rest-Frame Blackbody Shape Residuals
- group_name: blackbody_continuum_distance
- family: physically_informed_shape_model
- summary: This feature group represents each object's normalized ugriz continuum by its closest match to a redshift-corrected thermal blackbody shape and by the structured residual pattern left after that fit.
- strategy: For each row, convert ugriz magnitudes to finite positive flux proxies with f_b = 10^(-0.4 * m_b), replacing non-finite magnitudes with 30 and clipping resulting fluxes to at least 1e-12. Normalize to a shape vector p_b = f_b / sum_b f_b; if the sum is non-finite or <= 0, use p_b = 0.2 for all five bands. Use SDSS-like effective wavelengths lambda = [3551, 4686, 6165, 7481, 8931] Angstrom and define z_eff = clip(redshift, 0, 7.5), mapping non-finite redshift to 0. Work in rest-frame wavelengths lambda'_b = lambda_b / (1 + z_eff). Evaluate a log-spaced temperature grid T_j from 1500 K to 50000 K, for example J = 180 points, using the wavelength-form Planck proxy B_bj = lambda'_b^-5 / expm1(143877000 / (lambda'_b * T_j)), where lambda is in Angstrom and the exponent is clipped to [1e-8, 700] for numerical stability. Normalize each template to unit sum across bands, t_bj = B_bj / sum_b B_bj. Because both p and t are shape-normalized, compute the primary distance without a free amplitude scale: SSR_j = sum_b (p_b - t_bj)^2, and select j_star = argmin_j SSR_j. Emit best-fit log10(T_j_star), SSR_min, sqrt(SSR_min), the residuals r_u through r_z = p_b - t_b,j_star, the maximum absolute residual, the signed residual slope across wavelength from a five-point linear fit of r_b on log(lambda_b), and a confidence margin defined as the minimum SSR outside a +/-2 grid-index neighborhood around j_star minus SSR_min so the margin is not dominated by adjacent grid duplicates. Optionally add a local curvature feature SSR_{j-1} - 2*SSR_j + SSR_{j+1}, using 0 at grid boundaries. Implement in deterministic chunks over rows and temperature grid so memory use stays bounded.
- expected_signal: Stars should more often have smooth broad-band continua that lie close to a thermal shape, while galaxies and QSOs should leave larger and more structured residuals because of composite stellar populations, spectral breaks, nonthermal continua, and emission-line contamination; these residual-distance features can help balanced accuracy by separating classes in regions where ordinary colors overlap.
- risk: A single-temperature blackbody is only an approximation to real stellar and extragalactic spectra, so reddened stars, unusual stars, and smooth galaxy continua may be misrepresented; the features are also correlated with raw colors and redshift-derived separation, and the per-row grid search is computationally heavier than simple color arithmetic, requiring chunked vectorized implementation and validation against overfitting.

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