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
- title: Galactic sightline morphology features
- group_name: galactic_sightline_context
- family: sky_position
- summary: Represent each object by physically meaningful Galactic-sky morphology features that capture how its line of sight aligns with the Milky Way disk and high-latitude regions, enabling the model to distinguish foreground-star-dense directions from cleaner extragalactic sightlines.
- strategy: From alpha and delta, compute radians: α = deg2rad(alpha mod 360), δ = deg2rad(delta). Compute Galactic longitude and latitude using IAU J2000 formulas: sinb = sinδ·sinδ_NGP + cosδ·cosδ_NGP·cos(α−α_NGP), b = asin(clamp(sinb, -1, 1)); l = (l_asc − atan2(cosδ·sin(α−α_NGP), cosδ_NGP·sinδ − sinδ_NGP·cosδ·cos(α−α_NGP))) mod 2π, with α_NGP = 192.85948°, δ_NGP = 27.12825°, l_asc = 122.93192°. Convert b and l to degrees after wrapping l into [0, 360). Also build a parallel unit-vector path via x = cosδ cosα, y = cosδ sinα, z = sinδ and rotate to Galactic Cartesian to reduce singularity risk, then recover l, b if needed from the rotated vector. Derive deterministic features: b_abs = |b|, sign_b, sin(b), cos(b), sin(l), cos(l), d_plane = b_abs, angular distances to Galactic center d_gc = arccos(clamp(cos(b_rad)·cos(l_rad), -1, 1)), anticenter d_ac = arccos(clamp(-cos(b_rad)·cos(l_rad), -1, 1)), north pole d_np = arccos(clamp(sin(b_rad), -1, 1)), south pole d_sp = arccos(clamp(-sin(b_rad), -1, 1)); all inverse-trig inputs must be clipped to [-1, 1] and all angular outputs in degrees. Bin b_abs into nine bands: [0,5), [5,10), [10,20), [20,30), [30,40), [40,50), [50,60), [60,75), [75,90] with the final bin closed; bin l into 12 equal 30° sectors [0,30), ... [330,360), plus explicit treatment for l==360 as 0; create crossed regime feature band_id×sector_id and include b_sign×band and b_sign×sector interactions. Include these as integer/categorical encodings plus optional one-hot expansion in modeling.
- expected_signal: Spatial prior captures class-specific anisotropy in the sky that is not available from photometry alone: stars are more frequent near the Galactic plane while extragalactic sources are comparatively less detectable there, so this geometry channel should improve discrimination where u/g/r/i/z and redshift signatures are overlapping, especially by improving STAR recall and reducing confusion in low-latitude/high contamination regions.
- risk: Strongly spatial features can absorb survey footprint artifacts (masked fields, depth variation, target-selection boundaries) and overfit if train/test coverage differs, they may remain noisy around pole/crossing boundaries if angle normalization is mishandled, and they may provide only incremental benefit because color-based features already dominate class separability.

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