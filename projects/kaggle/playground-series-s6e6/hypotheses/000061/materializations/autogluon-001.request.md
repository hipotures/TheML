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
- title: Redshift-Slice Angular Environment
- group_name: redshift_slice_angular_environment
- family: spatial_environment
- summary: Represent each object by whether it lies in a sky-local overdensity of objects at comparable redshift and coherent broadband color, exposing galaxy-cluster and large-scale-structure context distinct from foreground stars and isolated quasar candidates.
- strategy: Using only predictor columns, fit an unsupervised neighbor reference on training rows with class removed; for training rows exclude the row itself, and for test rows query the frozen training reference. Convert alpha and delta to unit-sphere coordinates and use great-circle distances. For each object define a redshift half-window dz=max(0.01,0.03*(1+max(redshift,0))) and select neighbors with absolute redshift difference <= dz. For positive redshift above 0.003, compute approximate angular-diameter distance under a fixed flat cosmology with H0=70 and Omega_m=0.3, then evaluate spherical-cap counts and nearest-neighbor distances at physical transverse radii of 0.5, 1, 3, and 10 Mpc with angular radii clipped to [0.02,2.0] degrees; for redshift <= 0.003 use fixed angular radii of 0.05, 0.2, 1.0, and 2.0 degrees plus an invalid-physical-scale indicator. Convert counts to log overdensities against annular backgrounds at 3 to 6 times each radius and against all-redshift angular counts, add nearest-neighbor rank ratios for the 1st, 5th, and 20th neighbors where available, and add neighborhood color-coherence summaries from the median and MAD of neighbor differences in g-r and r-i with empty-neighbor statistics set to zero and paired missingness flags. Motivating references opened: https://www.sdss4.org/dr17/tutorials/lss_galaxy/ and https://astrobites.org/2012/03/27/the-red-sequence-method-for-galaxy-cluster-detection/.
- expected_signal: Galaxies, especially red-sequence and cluster-associated populations, should appear in same-redshift angular overdensities with coherent colors, while stars mostly reflect foreground sky density without cosmological slice coherence and QSOs are often sparse high-redshift objects; making this projection contrast explicit may reduce galaxy-star and galaxy-QSO confusions and improve balanced accuracy.
- risk: The signal can be partly survey-footprint or targeting-mask artifact, sparse redshift slices may produce noisy neighbor statistics for rare regimes, close-neighbor searches add implementation cost, and the group may overlap with existing sky-density features if redshift-slice contrast is weak.

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