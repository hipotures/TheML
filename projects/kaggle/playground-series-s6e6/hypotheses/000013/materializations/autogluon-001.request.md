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
- title: Redshifted Emission-Line Band Resonance
- group_name: emission_line_bandpass_resonance
- family: line_filter_geometry
- summary: Encode whether known galaxy and quasar emission lines should land inside specific SDSS ugriz passbands at the object's redshift and whether the corresponding observed broadband flux shows a local line-like excess rather than a smooth continuum.
- strategy: Use fixed SDSS filter central wavelengths u=3551, g=4686, r=6166, i=7480, z=8932 Angstrom from https://www.sdss4.org/instruments/camera/ and fixed common SDSS spectral-line wavelengths/weights from https://classic.sdss.org/dr5/algorithms/linestable.php. Convert ugriz magnitudes to relative fluxes f_b=10^(-0.4*m_b), then normalize by the median finite band flux per row. For each row and each selected emission line, compute observed wavelength lambda_obs=lambda_rest*(1+redshift). For each band, compute a resonance kernel exp(-0.5*(log(lambda_obs/lambda_band)/0.08)^2), set kernels to 0 when redshift<0 or lambda_obs is outside 3000-10000 Angstrom, and weight by separate fixed galaxy-line and quasar-line strengths. Create coherent summaries only: total quasar resonance, total galaxy resonance, maximum single-band resonance, nearest resonant band index, quasar-minus-galaxy resonance, resonance-weighted observed flux excess, and resonance-weighted signed excess where band excess is log(f_band) minus a linear interpolation of neighboring log fluxes across wavelength; for edge bands use the closest two interior bands as the continuum estimate and clip nonpositive fluxes with a tiny epsilon. Optionally include coarse bins for whether the strongest resonance falls in u/g/r/i/z or outside coverage, but do not learn any statistics from labels or test targets.
- expected_signal: Quasars and emission-line galaxies can have strong spectral lines that shift through SDSS filters with redshift and alter broadband colors; SDSS documentation notes quasar selection depends on ugri/griz color behavior, while filter-gap studies show strong lines such as MgII, CIV, and Ly-alpha can significantly affect colors, so this group gives the classifier a physically aligned explanation for redshift-dependent broadband bumps that smooth color or luminosity features may blur.
- risk: This may be partially redundant with existing color, spectral-break, and rest-frame landmark features, and the line-band kernels are an approximation because true filter throughput widths, line equivalent widths, and photometric errors are unavailable; if the provided redshift is noisy for stars or catalog-derived in a class-dependent way, resonance features may add unstable or overly direct class cues.

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