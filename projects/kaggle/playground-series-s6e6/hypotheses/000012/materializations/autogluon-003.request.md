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
- title: Robust Soft-Wedge Margin Stack
- group_name: faint_blue_galaxy_wedge_margins
- family: blue_galaxy_selection
- summary: This feature group encodes how strongly each object lies inside, near, or outside a deterministic faint-blue color-magnitude envelope by combining interior depth and violation-count signals so class boundaries between galaxies, stars, and quasars are expressed more explicitly than raw broadband features alone.
- strategy: For each object compute ug = u - g, gr = g - r, ri = r - i, iz = i - z, and clamp ug to [-0.60, 3.20] only for line-based wedge calculations. Define signed color margins m_c = [m1=gr-(0.40+0.60*ug), m2=(1.70-0.10*ug)-gr, m3=ug+0.50, m4=3.00-ug, m5=ri+0.50, m6=1.80-ri, m7=iz+1.00, m8=1.50-iz] and magnitude margins m_m = [n1=u-18.0, n2=24.0-u, n3=g-18.0, n4=21.5-g, n5=r-17.8, n6=19.5-r, n7=i-16.5, n8=20.5-i, n9=z-16.0, n10=20.0-z]. Replace any non-finite derived value with 0, then clip every margin to [-10, 10]. Normalize color margins by [0.35,0.35,0.50,0.50,0.35,0.35,0.35,0.35] and magnitude margins by [3.0,3.0,2.0,2.0,1.7,1.7,1.4,1.4,1.4,1.4] to equalize scale. Create deterministic features: color_depth_min=min(m_c_norm), color_depth_q20=20th percentile of m_c_norm, color_depth_soft = -logsumexp(-8*m_c_norm)/8, color_viol=sum(m_c_norm<0). For magnitudes: mag_depth_min=min(m_m_norm), mag_depth_q20=20th percentile of m_m_norm, mag_depth_soft = -logsumexp(-8*m_m_norm)/8, mag_viol=sum(m_m_norm<0). Create a combined score = 0.60*color_depth_soft + 0.40*mag_depth_soft - 0.25*color_viol - 0.10*mag_viol. Add a bounded sampling-intensity term s = clip(exp(0.1411*(gr-(0.40+0.60*ug))), 0, 10) and include both a color-only version of score and the full score + s as final engineered signals.
- expected_signal: Replacing single hard boundaries with clipped, normalized depth/violation statistics stabilizes the wedge signal for near-boundary objects, preserves a graded notion of membership, and helps recover galaxy-like patterns where stars and quasars leak into similar broad color space by adding explicit magnitude-constrained support information.
- risk: The wedge constants remain tied to legacy SDSS-style cuts and may not perfectly match this dataset’s exact photometric system, and any percentile/softmin summarization can shift emphasis toward the dominant class regions unless tuned and validated under balanced-accuracy splits.

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