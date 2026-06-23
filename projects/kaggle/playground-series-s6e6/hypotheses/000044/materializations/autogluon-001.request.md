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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

# Hypothesis
- title: Redshift-tracked absorption trough residuals in ugriz
- group_name: redshifted_absorption_trough_residuals
- family: absorption_line_geometry
- summary: Model the broadband imprint of age/metallicity-sensitive stellar and galaxy absorption complexes by measuring how much flux is locally suppressed when known absorption-line rest-frame wavelengths are shifted into the observed ugriz bands, so class separation uses spectroscopic-style line-blanketing patterns in addition to global colors.
- strategy: Convert ugriz magnitudes to linear flux proxies f_u..f_z = 10^-0.4*m. Define band effective wavelengths λ = [3551, 4686, 6165, 7481, 8931] Å and their contiguous passband spans (u 3000-4100, g 4100-5500, r 5500-6900, i 6900-8200, z 8200-9200 Å approximations). For each object compute observed line centers λ_obs = λ_rest*(1+z) for a fixed set of absorption-sensitive features: Ca II K (3933.7), Ca II H (3968.5), G-band (4304), Hδ (4102), Hγ (4341), Hβ (4861), Mg b triplet proxy 5175, and Na I D 5893. If λ_obs lies inside any ugriz span, estimate smooth local continuum at λ_obs by log-flux interpolation between the two neighboring filters straddling λ_obs; if at an edge, use a 5-point robust linear fit in log f vs log λ to predict f_cont. Compute signed residual r = (f_cont - f_band)/(f_band + eps), with eps small (e.g. 1e-12), then split into deficit d = max(r,0) and excess e = max(-r,0). Build grouped columns: total_blue_abs_deficit = mean(d for blue lines), total_red_abs_deficit = mean(d for Hβ/Mg b/Na D), absorption_excess_ratio = (sum(e))/ (sum(d)+1e-6), metal_trough_balance = total_blue_abs_deficit - total_red_abs_deficit, and line_family_contrast = CaK+CaH combined deficit vs Hδ+Hγ+G-band deficit. Add visibility masks per line (1 if in-band, 0 else) and per-redshift-regime masks (z<0.4, 0.4-1.2, 1.2-2.0, 2.0-7.0) so regime-limited aggregates are zeroed when no line is visible.
- expected_signal: Quasar continua are comparatively smooth and line-deficit patterns in broad bands are weaker or irregular, while stars and galaxies more often show redshift-dependent absorption-blanketing structure; tracking these signed deviations at physically meaningful rest-frame line positions should help disentangle ambiguous QSO-vs-galaxy and STAR-vs-QSO regions that are often separable only by subtle spectral-shape perturbations.
- risk: Broadband integration strongly dilutes narrow-line effects, so signal can be weak or noisy in faint objects and at high redshift where key lines redshift out of ugriz; approximate band pass boundaries or single-value continuum proxies may inject systematic bias, and the features are partially redundant with other continuum/break geometry groups, increasing overfit risk unless regularized.

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