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
- title: UV Blanketing Stellar Plausibility
- group_name: uv_blanketing_metallicity_plausibility
- family: stellar_photometric_parameters
- summary: Represent whether an object's ugr colors are physically plausible as an F/G main-sequence star by translating u-band line blanketing at fixed optical color into stellar temperature and metallicity support, so quasar or galaxy impostors with impossible stellar parameters become explicit.
- strategy: Compute u-g and g-r, then form an F/G calibration support gate that is 1 inside 0.2 <= g-r <= 0.6, tapers linearly to 0 over 0.0-0.2 and 0.6-0.8, and is 0 outside; estimate stellar temperature with log10(Teff/K)=3.872-0.264*(g-r) inside the core and a broad fallback log10(Teff/K)=3.882-0.316*(g-r)+0.0488*(g-r)^2+0.0283*(g-r)^3 clipped to the documented -0.3 < g-r < 1.3 range; estimate UV-blanketing metallicity in the hotter high-confidence window 0.2 < g-r < 0.4 and 0.8 < u-g < 1.4 using [Fe/H]=-21.88+47.39*(u-g)-35.50*(u-g)^2+9.018*(u-g)^3, and also compute the simpler low-metallicity proxy [Fe/H]=5.14*(u-g)-6.10 when 0.8 < u-g < 1.0; derive clipped margins to plausible stellar metallicity bands such as [-2.5,0.3], halo-like [-2.2,-1.0], and disk-like [-1.0,0.3], plus validity distances from the u-g and g-r calibration windows, UV-excess conflict margins for u-g < 0.6, and support-weighted versions of the temperature and metallicity scores; invalid or out-of-domain estimates are retained as finite clipped boundary values with separate support indicators rather than missing values. This uses published SDSS F/G photometric metallicity and temperature relations from Ivezic et al. 2008 (https://www.aoc.nrao.edu/~akimball/Publications/ivezic_et_al_2008.pdf) and SDSS UV-excess quasar-selection context (https://classic.sdss.org/dr7/products/spectra/special.php).
- expected_signal: Real STAR objects, especially F/G-like sources, should occupy physically plausible UV-blanketing and temperature ranges, while QSOs and some compact galaxies can share broad ugr colors but imply extreme or invalid stellar metallicities, improving separation in the known star-quasar overlap region without using labels.
- risk: The signal is narrow because the metallicity calibration is intended for dereddened main-sequence F/G stars, so it may be redundant with existing u-band anomaly or stellar-locus features and can become noisy for faint objects or non-stellar sources whose u-band colors violate the calibration assumptions.

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