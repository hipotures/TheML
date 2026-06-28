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
- title: Galactic and SDSS Sky Geometry
- group_name: sky_frame_position_geometry
- family: spatial_geometry
- summary: Represent sky location in physically aligned Galactic and survey-coordinate frames so the classifier can use continuous celestial geometry, Milky Way structure, and SDSS footprint context instead of discontinuous raw equatorial angles.
- strategy: Use only alpha and delta. Wrap alpha_deg = alpha modulo 360 into [0,360), clip delta_deg to [-90,90], convert both to radians, and form the column unit vector p = [cos(delta)*cos(alpha), cos(delta)*sin(alpha), sin(delta)], renormalized by max(norm(p), 1e-12). Add wrap-safe equatorial geometry features sin(alpha), cos(alpha), sin(delta), cos(delta), and the three components of p. Convert p to Galactic coordinates with the standard J2000 equatorial-to-Galactic rotation for column vectors: [[-0.0548755604,-0.8734370902,-0.4838350155],[0.4941094279,-0.4448296300,0.7469822445],[-0.8676661490,-0.1980763734,0.4559837762]]. From g = M p, compute l = atan2(g_y,g_x) wrapped to [0,2*pi) and b = asin(clamp(g_z,-1,1)); add sin(l), cos(l), sin(b), cos(b), abs(b), pi/2 - abs(b), acos(clamp(cos(b)*cos(l),-1,1)) for distance to the Galactic center, and acos(clamp(-cos(b)*cos(l),-1,1)) for distance to the anticenter. For SDSS survey coordinates, use the documented anchors from https://www.sdss4.org/dr16/algorithms/surveycoords/: X = unit_vector(RA=185, Dec=32.5), Y = unit_vector(RA=275, Dec=0), Z = normalize(cross(X,Y)); compute eta = asin(clamp(dot(p,Z),-1,1)) and lambda = atan2(dot(p,Y), dot(p,X)) in [-pi,pi]. Add sin(lambda), cos(lambda), sin(eta), cos(eta), abs(eta), distances to the two SDSS node directions at RA 95 and 275 degrees on Dec 0, and stripe-phase terms based on the 2.5 degree eta width: phase = (((eta_deg + 90) modulo 2.5) / 2.5), then sin(2*pi*phase), cos(2*pi*phase), and a scaled stripe index floor((eta_deg + 90)/2.5) clipped to [0,71] and centered by subtracting 35.5. Clip all inverse-trig inputs to [-1,1], keep angular outputs in radians except the stripe-width calculation, and replace any numerical NaN or Inf with deterministic finite defaults.
- expected_signal: Stars are local Milky Way objects whose density varies strongly with Galactic latitude and direction, while galaxies and QSOs are extragalactic but still inherit SDSS footprint and targeting anisotropy; these continuous frame-aware features can improve balanced accuracy by giving the model positional priors that are less discontinuous and easier to learn than raw RA/Dec.
- risk: The group can overfit survey footprint, targeting, or synthetic split artifacts rather than astrophysical class structure, especially through stripe-phase and node-distance features; it is partly redundant with raw alpha and delta, and any mismatch between the assumed SDSS/J2000 coordinate conventions and the data generation process could introduce misleading high-confidence spatial priors.

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