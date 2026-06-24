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
- title: Dust-vector decomposition with extinction-invariant colors
- group_name: foreground_reddening_geometry
- family: reddening_invariant
- summary: This feature family makes classifying stars, galaxies, and quasars more robust by separating foreground dust-driven color shifts from intrinsic broadband color structure so model boundaries reflect intrinsic SED and redshift information rather than line-of-sight extinction.
- strategy: For each object, convert (alpha, delta) in degrees to Galactic longitude/latitude (l, b) using a fixed sky-coordinate transform, then estimate E(B-V) from a preloaded SFD98 dust map rescaled by the Schlafly–Finkbeiner correction (multiply by 0.86). If no map value is available, impute from the median E(B-V) in the same 5°×5° (l,b) bin built on train-only rows; if still empty, use the nearest non-empty bin within 10° great-circle distance, else backfill with the global train median. Clamp E(B-V) to [0, 1.2], output ebv_clipped and a binary is_ebv_imputed flag. Compute A_u=5.155E, A_g=3.793E, A_r=2.751E, A_i=2.086E, A_z=1.479E and dereddened magnitudes u0..z0 by subtracting each extinction from the observed band. Form observed colors C=[u−g, g−r, r−i, i−z] and dereddened colors C0=[u0−g0, g0−r0, r0−i0, i0−z0]. Define reddening direction d=(A_u−A_g, A_g−A_r, A_r−A_i, A_i−A_z)/||·||2 and compute projections t=C·d, t0=C0·d, and dust displacement Δt=t−t0. Compute orthogonal residual R0=C0−(t0d), keep R0[0],R0[1],R0[2], and ||R0||2. Add context features |b|, latitude bin (train-fit quantile/decile bins), Δt×|b|, redshift×E(B-V), and E(B-V)×|b|. Clip all newly created continuous features to train-derived 0.5th and 99.5th percentiles.
- expected_signal: Foreground extinction moves objects along a nearly fixed direction in color space, so removing this axis and using orthogonal residuals should reduce class confusion in dusty regions while preserving discriminative structure from intrinsic SED differences and redshifted continua that are expected to be more informative for GALAXY/STAR/QSO separation.
- risk: The approach can misfire where the external dust map is biased or inconsistent with this photometric campaign, and fixed SDSS coefficients may not perfectly match all object depths/regions; additional projected features may be redundant with base colors and sky-position signals, so net benefit can be small when the underlying model already internalizes those correlations.

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