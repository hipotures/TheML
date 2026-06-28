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
- title: Seeing-Limited Resolvability Proxy
- group_name: seeing_limited_resolvability_proxy
- family: morphology_proxy
- summary: Estimate whether each object's redshift, apparent brightness, and catalog population imply an angularly resolved galaxy-like source or an unresolved point-source-like object under SDSS seeing, adding a surrogate for missing morphology information.
- strategy: Use a fixed flat LCDM approximation with H0=70, Omega_m=0.3, Omega_L=0.7, and c=299792.458 to compute luminosity distance and angular-diameter distance from redshift clipped to [1e-4, 7], while adding validity gates for redshift <= 0.005 and negative redshift. Compute pseudo absolute r magnitude from r and distance modulus for valid cosmological redshifts. Map galaxy_population to a canonical half-light-size relation: Red_Sequence uses a larger, steeper early-type size scale and Blue_Cloud uses a disk-like size scale, both brightness-scaled from pseudo absolute r magnitude and clipped to a plausible 0.3-30 kpc range. Convert this physical size to an angular half-light radius in arcsec, compare it with the SDSS median r-band seeing of 1.32 arcsec, and derive continuous log size-to-seeing margins, soft resolved/unresolved indicators at 0.5x, 1x, and 2x seeing, a compact-bright flag for r < 15 with predicted radius < 2 arcsec, and an approximate half-light surface-brightness margin using r + 2.5*log10(2*pi*theta_e^2) against 23.0 and 24.5 mag arcsec^-2. For invalid cosmological redshifts, set angular-size and surface-brightness values to neutral sentinels and rely on the explicit invalid/local-redshift gates. Sources consulted for the SDSS morphology and seeing rationale: https://www.sdss4.org/dr17/algorithms/classify/, https://www.sdss4.org/dr17/imaging/other_info/, and https://ned.ipac.caltech.edu/level5/March02/Sahni/Sahni4_5.html.
- expected_signal: SDSS star-galaxy separation is fundamentally morphology-driven, so a deterministic proxy for whether a source should be resolved at its redshift and brightness may help separate true galaxies from point-like stars and QSOs, especially in low-redshift or compact regimes where color and redshift cues can overlap.
- risk: The physical-size constants are approximate and may be redundant with existing redshift, luminosity, and survey-depth features; compact galaxies, Seyfert-like nuclei, and synthetic catalog artifacts could make the proxy misleading or unstable.

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