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
- title: Quasar Small-Bump Bandpass Contrast
- group_name: quasar_small_bump_bandpass_contrast
- family: quasar_continuum_geometry
- summary: Represent whether the redshifted SDSS bands sample the broad quasar Fe II and Balmer-continuum small-blue-bump region and whether the observed ugriz flux shape shows the expected localized pseudo-continuum excess.
- strategy: Use fixed SDSS central wavelengths u=3551, g=4686, r=6166, i=7480, z=8932 Angstrom and map each band to rest-frame wavelength lambda_rest=lambda_obs/(1+z_safe), with z_safe clipped only to keep the denominator positive. Convert ugriz magnitudes to relative log fluxes and fit a simple log-flux versus log-wavelength power-law continuum using bands outside the bump when available, falling back to all five bands with low bump weights otherwise. Define smooth triangular membership weights for the quasar small-bump core around 3000 Angstrom, its UV Fe II side from roughly 2200-3050 Angstrom, its Balmer-continuum side from roughly 3000-4000 Angstrom, and blue/red side continua around 1450-1900 and 4200-5500 Angstrom. Emit coverage and alignment descriptors for how much of the five-band system samples those rest-frame regions, weighted residual contrasts of bump-region log flux against the fitted continuum, UV-versus-Balmer asymmetry, side-continuum balance, and the strongest aligned observed band encoded ordinally. If a region has negligible total weight, set its contrast to 0 and retain the corresponding coverage feature so absence of wavelength support is explicit. Clip final log-ratio residuals to a finite mag-equivalent range to limit isolated photometric outliers. Motivation sources: https://www.sdss4.org/instruments/camera/ and https://www.researchgate.net/publication/230932119_Colors_of_2625_Quasars_at_0_ITALCLCzCLCITAL_5_Measured_in_the_Sloan_Digital_Survey_Photometric_System
- expected_signal: SDSS quasar-color studies note that quasar broadband colors follow a redshift-structured relation and are materially affected by the lambda-3000 small blue bump and strong spectral features, so isolating this rest-frame pseudo-continuum excess may add QSO-specific evidence in color regions where stars and compact galaxies are otherwise similar.
- risk: The five broad bands give only coarse wavelength sampling, so the signal may be redundant with existing rest-frame SED, emission-line, and quasar-selection groups, and it can become misleading when the supplied redshift is noisy or when synthetic photometry does not preserve real quasar small-bump behavior.

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