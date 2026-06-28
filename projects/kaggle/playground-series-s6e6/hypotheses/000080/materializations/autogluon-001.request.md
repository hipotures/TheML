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
- title: Balmer Jump Contaminant Geometry
- group_name: balmer_jump_contaminant_geometry
- family: hydrogen_break_geometry
- summary: Encode whether each object’s broadband continuum shows a hydrogen Balmer discontinuity at the observed wavelength implied by redshift and catalog temperature context, separating A-star/white-dwarf-like stellar contaminants and post-starburst compact galaxies from smoother quasars and ordinary galaxies.
- strategy: Use SDSS central wavelengths u=3551, g=4686, r=6166, i=7480, z=8932 Angstrom from https://www.sdss4.org/instruments/camera/ and convert magnitudes to relative fluxes with f_band=10^(-0.4*(mag_band-median_ugriz)). Set the Balmer edge location to 3646*(1+max(redshift,0)) Angstrom, motivated by the Balmer jump reference https://en.wikipedia.org/wiki/Balmer_jump. For each row, compute soft log-wavelength weights for bands blueward and redward of that edge, a red/blue flux discontinuity log((weighted_red_flux+eps)/(weighted_blue_flux+eps)), the same discontinuity after subtracting a linear log-flux continuum fit across wavelength, the local adjacent-pair contrast for the filter interval containing the edge, coverage flags for edge below u, between each adjacent band pair, and above z, plus near-zero-redshift observed-frame Balmer contrast using u-g versus g-r. Add smooth gates for spectral_type A/F and O/B, near-zero redshift abs(redshift)<0.01, and galaxy_population Red_Sequence versus Blue_Cloud by multiplying the core Balmer-strength and coverage values by these gates; clip flux ratios to finite ranges, use eps=1e-9, and set out-of-coverage strength values to 0 with explicit coverage indicators.
- expected_signal: SDSS quasar targeting notes identify A stars, white dwarfs, white-dwarf/M-dwarf pairs, compact blue galaxies, and E+A galaxies with Balmer-break-like colors as major quasar contaminants (https://classic.sdss.org/dr7/products/general/edr_html/node54.php), so an explicit Balmer-edge geometry can clarify STAR-QSO and GALAXY-QSO confusion that raw colors and redshift may leave entangled.
- risk: The signal may be partly redundant with u-g, redshift, and prior spectral-break features, and broad ugriz bands only weakly resolve a 3646 Angstrom discontinuity; if spectral_type already encodes the same physics, gains may be small or unstable.

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