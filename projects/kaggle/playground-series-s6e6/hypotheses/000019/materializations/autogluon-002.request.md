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
- title: Observed SED Continuum Moments with Robust Residual Diagnostics
- group_name: observed_sed_continuum_moments
- family: spectral_shape
- summary: Create a compact shape descriptor of the five-band observed spectral energy distribution that captures large-scale continuum slope and curvature together with localized deviations from a smooth trend while removing object-by-object brightness scaling.
- strategy: For each object, first enforce deterministic validity checks on u,g,r,i,z by replacing any non-finite value with the training-set per-band median and clamping each remaining magnitude to train-derived bandwise 0.1%/99.9% quantiles (with an explicit fallback physical cap of [-1,35] if quantiles are unavailable). Convert to relative log-flux as l_b=-0.4*m_b, then remove absolute level by centering: l'_b=l_b-mean_b(l_b). Use fixed SDSS pivot wavelengths λ=[3543,4770,6231,7625,9134] Å, define t_b=log10(λ_b)-mean_b(log10 λ_b), and fit a quadratic y(t)=β0+β1 t+β2 t^2 to {t_b,l'_b} by closed-form OLS over the five bands. Compute residuals e_b=l'_b-y(t_b), and emit deterministic residual descriptors: β1, β2, sqrt(mean(e_b^2)), mean(|e_b|), max(|e_b|), signed residuals at the edge bands e_u,e_z, and a signed break index Δblue−red=((l'_u-l'_g)/(t_u-t_g))-(l'_r-l'_i)/(t_r-t_i). If X'X is singular/ill-conditioned or any intermediate becomes non-finite, emit zeros for β1,β2 and residual features for that row to ensure stable behavior.
- expected_signal: The class priors differ primarily in broadband continuum morphology: stellar loci are usually smoothly varying and temperature-driven, galaxies often show stronger curvature from broad stellar-population breaks, and quasars are closer to power-law-like continua with characteristic local departures; fitting a normalized quadratic trajectory plus residual structure should isolate these class-separating traits more directly than raw magnitudes alone and can improve balanced-accuracy by reducing dependence on absolute flux scale.
- risk: The shape descriptors are strongly correlated with simple colors and can still learn survey-specific photometric systematics (extinction, calibration stripes, saturated/edge cases), and a quadratic model over five points may miss fine line-driven features that matter for some quasars, so aggressive clipping or ill-conditioned fallback can suppress real signal on outlier objects while reducing instability.

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