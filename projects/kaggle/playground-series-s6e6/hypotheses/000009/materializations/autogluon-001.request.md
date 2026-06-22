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
- title: Canonical SDSS Locus Coordinates
- group_name: canonical_locus_coordinates
- family: locus_projection
- summary: Project each object into fixed SDSS-derived coordinate systems that measure its signed position and normalized departure from the canonical stellar locus and the red-galaxy locus, so the classifier sees whether the photometry looks star-core, galaxy-track, or locus-outlier like in an astrophysically meaningful frame.
- strategy: First compute the ordered colors ug = u-g, gr = g-r, ri = r-i, and iz = i-z. Then compute the four SDSS stellar-locus principal colors perpendicular to the locus: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.988, and y = -0.270*r + 0.800*i - 0.534*z + 0.054. Also compute the corresponding along-locus coordinates used to determine where each principal color is valid: p1_s = 0.910*u - 0.495*g - 0.415*r - 1.280, p1_w = 0.928*g - 0.556*r - 0.372*i - 0.425, p1_x = r-i, and p1_y = 0.895*r - 0.448*i - 0.447*z - 0.600. Standardize the perpendicular coordinates by the published intrinsic stellar-locus widths to get ns = s/0.031, nw = w/0.025, nx = x/0.042, and ny = y/0.023, and clip each standardized value to [-12, 12] to limit leverage from rare photometric extremes. Mark an axis active only when its published validity window is satisfied: for s use r <= 19.0 and -0.2 <= p1_s <= 0.8; for w use r <= 20.0 and -0.2 <= p1_w <= 0.6; for x use r <= 19.0 and 0.8 <= p1_x <= 1.6; for y use r <= 19.5 and 0.1 <= p1_y <= 1.2. From the active standardized deviations derive aggregate stellar-locus affinity columns: active_axis_count, min absolute deviation, mean absolute deviation, max absolute deviation, count within 1-sigma, count within 2-sigma, and count beyond 4-sigma; if no axis is active, compute those aggregates over all four standardized deviations and set active_axis_count to 0. In the same group, compute the SDSS red-galaxy locus coordinates c_perp = ri - gr/4 - 0.18 and c_par = 0.7*gr + 1.2*(ri - 0.18), plus a simple red-galaxy-track proximity feature abs(c_perp). Keep all raw principal coordinates, standardized deviations, aggregate stellar-locus affinity measures, and the galaxy-locus coordinates as the output of this single feature group.
- expected_signal: Stars should cluster tightly near the stellar-locus tube, quasars are often strong stellar-locus outliers in SDSS color space, and galaxies tend to separate through both weaker stellar-locus affinity and structured position along the galaxy color track, so these coordinates can improve balanced accuracy by making STAR recall and QSO recall more explicit while still giving GALAXY a dedicated manifold signal.
- risk: The coefficients and widths were derived for SDSS-style dereddened photometry and mainly for normal stars plus red-galaxy tracks, so any calibration mismatch, extinction residue, or magnitude-definition mismatch can shift the locus coordinates; the group is also partly redundant with generic color features and may underrepresent blue or unusual galaxies that do not lie near the red-galaxy track.

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