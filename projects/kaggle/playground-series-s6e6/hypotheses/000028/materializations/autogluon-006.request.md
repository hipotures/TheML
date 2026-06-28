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
- title: Calibrated dust-vector color decomposition
- group_name: foreground_reddening_geometry
- family: reddening_invariant
- summary: This feature group decomposes broadband photometry into foreground-dust amplitude, dereddened color geometry, and sky-extinction context so classification relies more on intrinsic SED differences than on line-of-sight reddening.
- strategy: Convert alpha and delta from ICRS degrees to Galactic longitude and latitude using a fixed deterministic coordinate transform. Query a preloaded SFD98 E(B-V) map rescaled by the Schlafly-Finkbeiner factor 0.86; if lookup fails, impute from train-fitted 5 degree by 5 degree Galactic-coordinate bin medians among rows with valid map values, then from the nearest non-empty train bin within 10 degrees, then from the global train median. Store ebv_raw, ebv_clipped=clip(ebv_raw,0,1.2), and is_ebv_imputed. Use SDSS extinction coefficients [A_u,A_g,A_r,A_i,A_z]=[5.155,3.793,2.751,2.086,1.479]*ebv_clipped and compute dereddened magnitudes u0,g0,r0,i0,z0 by subtracting extinction from observed magnitudes. Form observed colors C=[u-g,g-r,r-i,i-z] and dereddened colors C0=[u0-g0,g0-r0,r0-i0,i0-z0]. Define the color-excess vector k=[1.362,1.042,0.665,0.607] and unit reddening direction d=k/||k||2. Add parallel_obs=C dot d, parallel_dered=C0 dot d, dust_parallel=parallel_obs-parallel_dered, and orthogonal dereddened residuals R=C0-(parallel_dered*d), keeping R_ug,R_gr,R_ri,R_iz and ||R||2. Add compact context terms abs_b=|b|, signed_b=b, train-fitted decile bins of ebv_clipped and abs_b, plus interactions dust_parallel*abs_b, ebv_clipped*redshift, and ebv_clipped*galaxy_population encoded as two binary interactions. Fit all percentile cut points and bin edges on training data only, then clip continuous generated features to train-derived 0.5th and 99.5th percentiles before applying the same transform to test rows.
- expected_signal: Balanced accuracy may improve because dust reddening shifts objects along a predictable color direction that can otherwise blur STAR, GALAXY, and QSO boundaries, while the orthogonal dereddened residuals retain class-specific SED curvature and redshift-related color structure.
- risk: External dust-map calibration, coordinate-transform mistakes, or extinction coefficients mismatched to the survey passbands can inject systematic noise; the dust amplitude and latitude features may also duplicate information already learnable from colors, redshift, and sky position, so validation must check for marginal gain rather than assumed benefit.

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