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
- title: Redshifted Emission-Line Passband Resonance Contrast
- group_name: emission_line_bandpass_resonance
- family: line_filter_geometry
- summary: Create features that capture how redshifted strong galaxy and quasar emission-line systems align with SDSS ugri z passbands and whether those aligned bands show local broadband flux departures from adjacent continuum, preserving an explicit physical interpretation of line-driven color shifts.
- strategy: For each row, compute flux proxies f_b = max(10^{-0.4 m_b}, 1e-12) for b in {u,g,r,i,z}, then set y_b = log(f_b) and center each row by y'_b = y_b - median(y_u, y_g, y_r, y_i, y_z) to remove distance scale effects. Use fixed SDSS effective passband centers λ_b = {3551,4686,6166,7480,8931} Å. Use fixed SDSS spectro1d line wavelengths in vacuum with separate class-specific importance sets: quasar lines (non-zero weight): {Lyα 1215.24, N V 1240.81, Si IV+O IV 1399.8, C IV 1549.48, C III]1908.734, Mg II 2799.117, [O II] 3727.092, Hβ 4862.68, [O III] 5008.240, Hα 6564.614}, galaxy lines (non-zero weight): {[O II] 3727.092, Hδ 4102.89, Hγ 4341.68, Hβ 4862.68, [O III] 4960.295, [O III] 5008.240, Hα 6564.614, [N II] 6585.27, [S II] 6718.29, [S II] 6732.67}. For each line λ0 with redshift z, compute λ_obs = λ0(1+z); if z < 0 or λ_obs not in [3000,10000], set that line’s contribution to zero for all bands. Otherwise, for each band define a resonance kernel k_{b,l} = exp[-0.5*(ln(λ_obs/λ_b)/σ_b)^2] with fixed σ_b = 0.06 in log-wavelength; optionally zero out k_{b,l} < 1e-4 for numerical sparsity. Compute local continuum residuals e_b on the row-centered log-flux scale: for interior bands interpolate linearly in log λ using neighboring points, and for edges use the closest two interior bands (u uses g and r; g uses u and r; r uses g and i; i uses r and z; z uses i and r). Define e_b = y'_b - ŷ_b. Build deterministic class summaries: T_qso = Σ_l w_qso(l) Σ_b k_{b,l}, T_gal = Σ_l w_gal(l) Σ_b k_{b,l}; E_qso = Σ_l w_qso(l) Σ_b k_{b,l} * e_b, E_gal = Σ_l w_gal(l) Σ_b k_{b,l} * e_b; peak band indices B_qso = argmax_b Σ_l w_qso(l)k_{b,l}, B_gal = argmax_b Σ_l w_gal(l)k_{b,l}; peak strengths S_qso = max_b Σ_l w_qso(l)k_{b,l}, S_gal = max_b Σ_l w_gal(l)k_{b,l}; margin features M_t = T_qso - T_gal, M_e = E_qso - E_gal, M_s = S_qso - S_gal. Add four-way binning for each class of strongest-resonant band among u,g,r,i,z and an outside-coverage bin when all k_{b,l}=0. All quantities are computed purely from input columns without target-derived fitting.
- expected_signal: Redshifted emission lines produce class- and redshift-dependent bumps that move through SDSS filters in a physically deterministic way; this gives the classifier a stable signal where quasars and star-forming/AGN-like galaxies depart from smooth continuum colors, especially in overlap regions where raw color features can blur distinctions.
- risk: This group may over-trust the provided redshift and can create spurious signals when redshift is noisy or systematically wrong for some classes, especially for stars with catalog-like values; fixed central wavelengths and Gaussian log-kernels approximate real throughput and line profiles and ignore equivalent width variation, while added class-specific resonance features may duplicate existing color/continuum features and encourage redshift-bin overfitting without regularization.

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