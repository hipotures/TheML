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
- title: Fold-safe SEGUE atmosphere residuals
- group_name: segue_stellar_atmospheric_indices
- family: atmospheric_color_diagnostics
- summary: Construct SDSS principal-color residual features that measure how tightly each object's broadband photometry follows a robust stellar-atmosphere color locus after accounting for redshift, spectral type, and population context.
- strategy: Fit the feature transformer only on the current training fold during validation and on full train for final submission; do not use test rows or target labels when estimating preprocessing statistics. From u,g,r,i,z compute ug=u-g, gr=g-r, ri=r-i, iz=i-z, then p1=0.91*ug+0.415*gr-1.280, v=0.283*ug-0.354*gr+0.455*ri+0.766*iz, and l=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984. Build redshift decile edges from the fitting data, collapse duplicate edges, set outer edges to -inf and +inf, and assign zbin with out-of-range clipping. For each x in {p1,v,l}, estimate robust center and scale at four context levels: global, spectral_type, zbin+spectral_type, and zbin+spectral_type+galaxy_population. A cell statistic is usable only when n>=500; otherwise use the next coarser level. Center is median_x; scale is max(1.4826*median(abs(x-median_x)), (q75_x-q25_x)/1.349 when available, global_scale_x, 1e-6). Emit the raw indices, contextual z_x values from the deepest usable level, coarser global/spectral z_x values, abs(z_x), and context deltas z_x_global-z_x_contextual. Let z_p1,z_v,z_l denote the deepest contextual z-scores for composite features. Add p1_low=max(0,-0.70-p1), p1_high=max(0,p1+0.25), p1_band_violation=p1_low+p1_high, l_low=max(0,0.07-l), l_high=max(0,l-0.135), l_band_violation=l_low+l_high, v_abs=abs(v), abs_z_mean=(abs(z_p1)+abs(z_v)+abs(z_l))/3, abs_z_max=max(abs(z_p1),abs(z_v),abs(z_l)), locus_dist=sqrt(z_p1^2+z_v^2+z_l^2), signed_sum=z_p1+z_v+z_l, tail_count=I(abs(z_p1)>3)+I(abs(z_v)>3)+I(abs(z_l)>3), cross_vp=v_abs*z_p1, and cross_vl=v_abs*z_l. Clamp z-scores to [-8,8] before composites, winsorize raw indices, violations, v_abs, and interaction terms to fitting-data 0.1/99.9 percentile limits, and replace unseen categorical cells, NaN, or inf with the defined coarser/global fallbacks, using 0 for unavailable standardized residuals.
- expected_signal: The canonical projections encode stellar continuum curvature and line-blanketing behavior, while multi-scale robust residuals distinguish objects that look normal only after contextual normalization from objects that are globally or locally off the stellar locus; this should improve STAR recall and reduce QSO/GALAXY confusion in overlapping color regions, directly supporting balanced_accuracy.
- risk: The hard-coded stellar-locus bands and redshift/category bins may be brittle under distribution shift, robust statistics estimated from all classes can partially center dense non-stellar modes and weaken separation, and the added residual scales may be redundant with raw color or redshift features if the downstream model already captures the same geometry.

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