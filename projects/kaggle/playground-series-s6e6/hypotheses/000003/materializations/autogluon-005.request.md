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
- title: Galactic Sightline Context
- group_name: galactic_sightline_context
- family: sky_position
- summary: Represent each object's sky position as a physically meaningful Galactic line-of-sight context so the model can learn foreground-star density and extragalactic visibility patterns tied to the Milky Way disk and poles.
- strategy: Convert alpha and delta from degrees to radians after wrapping alpha into [0, 360). Build an equatorial unit vector x_eq = cos(delta) * cos(alpha), y_eq = cos(delta) * sin(alpha), z_eq = sin(delta), then transform it to Galactic Cartesian coordinates with the standard J2000/ICRS-to-Galactic rotation matrix: [[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445], [-0.8676661490, -0.1980763734, 0.4559837762]]. Recover Galactic longitude and latitude as l = atan2(y_gal, x_gal) wrapped to [0, 360) degrees and b = asin(clip(z_gal, -1, 1)) in degrees. Derive continuous features from signed b, abs_b = |b|, sign_b, sin(l), cos(l), sin(b), cos(b), and the Galactic Cartesian components x_gal, y_gal, z_gal. Add physically interpretable angular distances in degrees: distance to the Galactic plane as abs_b; distance to the Galactic center using arccos(clip(cos(b_rad) * cos(l_rad), -1, 1)); distance to the anticenter using arccos(clip(-cos(b_rad) * cos(l_rad), -1, 1)); distance to the north Galactic pole as 90 - b; and distance to the south Galactic pole as 90 + b. Add coarse regime features by binning abs_b into [0,5), [5,10), [10,20), [20,30), [30,45), [45,60), [60,75), [75,90] with the final bin including 90, and binning l into 12 equal 30-degree sectors with l == 360 treated as 0. Include the latitude-band id, longitude-sector id, their crossed band-sector id, and simple hemisphere interactions such as sign_b crossed with latitude band and longitude sector. Encode regime variables as categorical or one-hot features according to the downstream model, while keeping all transformations deterministic and fitted without target information.
- expected_signal: Balanced accuracy may improve because stellar contamination and source detectability vary strongly with Galactic latitude and direction: STAR examples should be relatively enriched near lower-latitude Milky Way sightlines, while GALAXY and QSO objects are more likely in cleaner high-latitude regions. The signed, cyclic, distance, and coarse-regime features expose spatial structure that raw alpha and delta can hide, especially around the alpha wraparound, and may help recover minority-class recall in regions where photometry and redshift features overlap.
- risk: The group can learn survey footprint, depth, masking, or targeting artifacts rather than stable astrophysical signal, so it may overfit if the train and test sky coverage are not sampled similarly. It is also partly redundant with alpha and delta and may add little if color and redshift dominate separation. Incorrect angle wrapping, matrix orientation, or bin-edge handling could create artificial discontinuities, especially near l = 0/360 degrees and at high Galactic latitude.

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