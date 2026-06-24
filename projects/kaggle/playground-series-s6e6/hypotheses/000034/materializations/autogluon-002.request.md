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
- title: Rest-frame Blackbody Continuum Distance
- group_name: blackbody_continuum_distance
- family: physically_informed_shape_model
- summary: This group maps each object's normalized ugriz spectral shape onto a thermal continuum manifold in rest-frame wavelength space and derives compact distance-to-manifold signals that should distinguish near-blackbody stellar spectra from quasar and galaxy continua.
- strategy: For each row, compute flux proxies from magnitudes as `f_b = 10^{-0.4 m_b}` for bands b∈{u,g,r,i,z}; if a magnitude is non-finite, replace it with 30 before exponentiation (equivalent to a flux floor of 10^-12) so the proxy is finite. Form the normalized shape vector `p_b = f_b / (Σ_b f_b)`; if `Σ_b f_b <= 1e-30`, set `p_b = 0.2` for all bands. Use effective wavelengths `λ = [3551,4686,6165,7481,8931] Å` and redshift-corrected wavelengths `λ'_b = λ_b / (1 + z_eff)` with `z_eff = max(redshift, 0)` and non-finite redshift mapped to 0. Build a logarithmic temperature grid `T_j = 10^{linspace(log10(1500), log10(50000), J)}` with `J=180`. For each T_j, evaluate template shape `B_b(T_j) = (1 / λ'^5_b) / (exp(14387.7/(λ'_b * T_j)) - 1)` using a numerically stable `expm1` form and clip argument to `≤700` before exponentiation; normalize each template as `t_b(T_j)=B_b/Σ_b B_b`. Compute least-squares scale `a_j = (Σ_b p_b t_b(T_j)) / (Σ_b t_b(T_j)^2)` and residual `SSR_j = Σ_b (p_b - a_j t_b(T_j))^2`. Let `j* = argmin SSR_j`; optionally refine `log10(T*)` by quadratic interpolation over `(log10 T)` across `j*-1, j*, j*+1` and clamp within grid bounds. Emit deterministic outputs: best-fit `log10(T*)`, `SSR_min`, `sqrt(SSR_min)`, residual gap to second best `SSR_2nd - SSR_min`, local curvature from second finite-difference in log-space, and signed residuals `r_b = p_b - a_{j*} t_b(T*)` for each band. Implement all operations in vectorized matrix form over T to keep runtime tractable and deterministic.
- expected_signal: The procedure yields an explicit physical shape-distance representation: stars are expected to show low, smooth mismatch to a thermal manifold while galaxies and quasars, which often include nonthermal continua, breaks, and strong emission structures, should incur larger manifold residuals, especially where raw magnitudes and color ratios alone blur class boundaries.
- risk: The manifold prior is intentionally physics-driven and can over-penalize heavily reddened stars, line-dominated quasars, and objects with calibration artifacts, and grid-based fitting can produce boundary effects or instability at extreme redshift unless interpolation and numeric guards are used; generated features are also partially redundant with color-based predictors, so downstream regularization/selection is advised.

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