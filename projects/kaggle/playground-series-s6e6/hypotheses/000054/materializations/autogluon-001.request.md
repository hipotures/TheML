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
- title: Restframe anchored SED slope and curvature profiles
- group_name: restframe_anchor_sed_shape
- family: restframe_shape
- summary: Transform each object into a redshift-normalized five-point SED traced at fixed rest-frame wavelength anchors and expose explicit UV-to-optical slope and curvature signals so class-specific continuum behavior is compared on a shared physical wavelength basis rather than fixed observed filters.
- strategy: For each row, convert ugriz magnitudes to linear fluxes F_b = 10^{-0.4 m_b}. Use rest-frame effective wavelengths λ'_b = λ_b / (1 + redshift) with λ_u=3551, λ_g=4686, λ_r=6165, λ_i=7481, λ_z=8931 Å (use 1+redshift clipped at a small positive floor, e.g. 0.1, only if redshift < -0.9). Sort the five (log10 λ', log10 F) pairs by wavelength and build a monotonic piecewise-linear continuum in log-space. Evaluate interpolated rest-frame flux values at fixed anchors A=[1500,2200,2900,3650,4500,6200,7600] Å; if an anchor is outside the interpolated range, set the anchor value to missing and emit a valid-mask feature for each anchor plus a count of usable anchors. From available anchors, compute adjacent segment slopes s1=ΔlogF/Δlogλ in 1500–2200, 2200–2900, 2900–3650, 3650–4500, 4500–6200, 6200–7600 Å, and derive curvature deltas d1=s2-s3 (UV bend around metal-line regime), d2=s4-s5 (optical curvature), and a 3650-Å break residual b4000 = logF(3650) - linear prediction from anchors at 2900 and 4500. Add mismatch terms as RMS residuals of (a) global 1-slope power-law fit over all available anchors and (b) a constrained 2-slope piecewise fit with an enforced break at 3650 Å to quantify break sharpness; replace undefined fits with NaN and carry coverage-count masks.
- expected_signal: Stars are expected to show smooth blackbody-like rest-frame slopes with modest curvature, galaxies should exhibit stronger positive/negative curvature near the 3650–4500 Å region from stellar population breaks, and quasars should remain closer to a broken/power-law continuum with weaker broad-band stellar-break structure; anchoring to physical wavelengths removes redshift-induced feature drift and makes these class differences directly comparable across the full z range.
- risk: Interpolation is unstable when a redshift places most anchors outside observed bands, causing many missing values at extremes, and the method inherits redshift noise; if redshift errors or selection artifacts correlate with class, piecewise fit scores could overfit those artifacts, and the extra non-linear features can become correlated with other redshift-based geometry groups.

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