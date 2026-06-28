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
- title: Calibrated Faint Blue Wedge Margins
- group_name: faint_blue_galaxy_wedge_margins
- family: blue_galaxy_selection
- summary: Encode graded membership in a faint blue galaxy color-magnitude selection envelope by measuring normalized interior depth, boundary proximity, and directional violations against galaxy-targeting cuts.
- strategy: Compute colors ug=u-g, gr=g-r, ri=r-i, iz=i-z. Use ug_line=clip(ug,-0.60,3.20) only in sloped gr boundary formulas, while using raw ug for explicit ug lower and upper bounds. Define signed color margins where positive means inside: gr-(0.40+0.60*ug_line), (1.70-0.10*ug_line)-gr, ug+0.50, 3.00-ug, ri+0.50, 1.80-ri, iz+1.00, 1.50-iz. Define signed magnitude margins: u-18.0, 24.0-u, g-18.0, 21.5-g, r-17.8, 19.5-r, i-16.5, 20.5-i, z-16.0, 20.0-z. Replace nonfinite margins with 0, clip raw margins to [-10,10], then create both raw directional margin features and normalized summaries. Normalize color margins by [0.35,0.35,0.50,0.50,0.35,0.35,0.35,0.35] and magnitude margins by [3.0,3.0,2.0,2.0,1.7,1.7,1.4,1.4,1.4,1.4]. For color and magnitude subsets separately compute minimum normalized margin, 20th percentile margin, soft minimum -logsumexp(-8*m)/8, mean positive interior margin, total negative violation count, and total violation depth sum(max(0,-m)). Add combined_wedge_depth = 0.65*color_softmin + 0.35*mag_softmin, combined_wedge_penalty = combined_wedge_depth - 0.20*color_violation_count - 0.08*mag_violation_count - 0.15*color_violation_depth - 0.05*mag_violation_depth, and binary indicators for all-inside-color, all-inside-magnitude, and all-inside-full. Include bounded sloped-boundary intensity proxy clip(exp(0.1411*(gr-(0.40+0.60*ug_line))),0,10) plus log1p of that proxy.
- expected_signal: Balanced accuracy may improve because these features give the model an explicit, smooth galaxy-selection geometry with directional information about which boundary failed, helping separate faint blue galaxies from quasars and stars in regions where raw colors and magnitudes alone overlap.
- risk: The fixed wedge constants are external survey-style cuts and may be partly redundant with raw photometry, spectral_type, and galaxy_population; hard thresholds and hand-chosen normalizers can misrepresent this dataset near boundaries or bias the model toward the majority galaxy region if not validated with stratified balanced-accuracy folds.

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