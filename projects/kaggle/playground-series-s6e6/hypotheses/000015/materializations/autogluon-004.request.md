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
- title: Redshift Template Feasibility Margins
- group_name: redshift_template_domain_margins
- family: redshift_geometry
- summary: Encode redshift as class-feasibility geometry by measuring each object against near-zero stellar, low-redshift galaxy, and broad quasar template domains rather than relying on redshift only as an unconstrained scalar.
- strategy: Using only raw redshift z, define fixed inclusive template domains: STAR S=[-1200/299792.458, 1200/299792.458], GALAXY G=[-0.01, 1.00], and QSO Q=[0.0333, 7.00]. For each domain D=[lo, hi], compute an in_domain flag 1[lo <= z <= hi], lower signed margin z-lo, upper signed margin hi-z, outside signed distance where values below lo are z-lo, values above hi are z-hi, and zero otherwise, plus outside absolute gap max(lo-z, 0) + max(z-hi, 0). Add redshift sign, absolute redshift, and a star_velocity_scaled feature z / (1200/299792.458) to make the narrow stellar Doppler-scale region explicit. Add ordered breakpoint geometry for T=[-0.01, 0.0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0, 7.0]: interval_id indicating which adjacent interval contains z, inclusive lower-edge distances z-t for each t, upper-edge distances t-z for each t, nearest_breakpoint_distance=min_t abs(z-t), and nearest_breakpoint_index. For all continuous margin and distance features, provide both raw clipped values using a symmetric finite cap such as [-10, 10] and compressed log_abs_signed=sign(x)*log1p(abs(x)); preserve exact boundary equality as in-domain, avoid divisions except the fixed stellar scale, and replace any non-finite result with the corresponding finite clipped value.
- expected_signal: Balanced accuracy may improve because the classes have strongly different redshift feasibility profiles: stars should cluster close to zero redshift, galaxies are mostly plausible in a low-to-moderate positive range, and quasars occupy a much broader positive range with important ambiguity near low-redshift boundaries; explicit margins can help both tree models and linear models learn these asymmetric class regions.
- risk: These features are largely deterministic transforms of raw redshift and may be redundant for high-capacity learners; fixed SDSS-inspired cutoffs can overfit survey-specific or synthetic-generator artifacts, and hard boundary features may miscalibrate examples near overlap regions where spectral_type, colors, or population features should dominate.

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