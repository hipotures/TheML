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
- title: Normalized Emission-Line Passband Resonance Contrast
- group_name: emission_line_bandpass_resonance
- family: line_filter_geometry
- summary: Encode how physically important rest-frame emission lines for quasars and galaxies redshift into SDSS ugriz passbands and whether the matched bands show local broadband excesses relative to a smooth adjacent continuum.
- strategy: Use only input columns and fixed astrophysical constants. Convert magnitudes to positive relative flux proxies f_b = max(10^(-0.4*m_b), 1e-12) for bands b in u,g,r,i,z, then work on log flux y_b = log(f_b) centered per row by subtracting median(y_u,y_g,y_r,y_i,y_z). Use fixed SDSS effective passband centers lambda_b = {u:3551,g:4686,r:6166,i:7480,z:8932} Angstrom. Define two fixed rest-frame emission-line templates: a quasar template with Ly-alpha 1215.24, N V 1240.81, Si IV/O IV 1399.8, C IV 1549.48, C III] 1908.734, Mg II 2799.117, [O II] 3727.092, H-beta 4862.68, [O III] 5008.240, and H-alpha 6564.614; and a galaxy template with [O II] 3727.092, H-delta 4102.89, H-gamma 4341.68, H-beta 4862.68, [O III] 4960.295, [O III] 5008.240, H-alpha 6564.614, [N II] 6585.27, [S II] 6718.29, and [S II] 6732.67. Assign deterministic nonnegative line weights reflecting broad relative importance, then normalize weights to sum to one within each template so quasar and galaxy totals are comparable rather than dominated by template length. For each row, line, and band, compute lambda_obs = lambda_rest*(1+redshift); if redshift < 0 or lambda_obs is outside [3000,10000] Angstrom, set that contribution to zero. Otherwise compute k_{b,l} = exp(-0.5*(ln(lambda_obs/lambda_b)/0.08)^2), optionally truncating values below 1e-4 to zero. Estimate local continuum residual e_b on centered log flux: for g,r,i interpolate linearly in log wavelength between the adjacent bands; for u use the g-r slope extrapolated to u; for z use the r-i slope extrapolated to z. Define e_b = centered_y_b - continuum_hat_b and clip e_b to a reasonable fixed range such as [-5,5] to limit extreme magnitude artifacts. Create summary features for each template: total geometric resonance T = sum_l w_l sum_b k_{b,l}, residual-weighted resonance E = sum_l w_l sum_b k_{b,l}*e_b, positive-excess resonance E_pos using max(e_b,0), negative-residual resonance E_neg using min(e_b,0), peak band strength S = max_b sum_l w_l*k_{b,l}, and strongest band as one categorical value among u,g,r,i,z,outside where outside is used when S=0. Add contrast features T_qso - T_gal, E_qso - E_gal, E_pos_qso - E_pos_gal, and S_qso - S_gal. All bins and weights are fixed before training and no label, fold, or test-target statistics are learned.
- expected_signal: Balanced accuracy can benefit if minority QSO and STAR/GALAXY boundary cases are separated by redshift-dependent broadband bumps that ordinary color differences treat as smooth variation; these features give the model compact, physically aligned summaries of when quasar-like or galaxy-like emission lines should perturb specific SDSS bands.
- risk: The signal depends heavily on the reliability of the provided redshift and may create misleading resonance patterns for stars or objects with noisy catalog redshifts; fixed filter centers, approximate Gaussian kernels, and fixed line weights ignore true throughput curves, equivalent widths, absorption, and photometric uncertainty, so the features may be redundant with colors or encourage redshift-pattern overfitting if the downstream model is too flexible.

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