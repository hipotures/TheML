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
- title: Context-stabilized principal-color residual manifold
- group_name: local_principal_color_residuals
- family: locus_geometry
- summary: Model each object as an orthogonal displacement from a local stellar-locus manifold using robust, context-stratified normalization so class-specific deviations and contamination patterns are expressed as standardized geometric outliers rather than raw magnitude differences.
- strategy: Compute the five principal-color coordinates from observed ugriz in every row, using the canonical SDSS/SEGUE definitions: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.983, y = -0.270*r + 0.800*i - 0.534*z + 0.059, and l = -0.436*u + 1.129*g - 0.119*r - 0.574*i + 0.1984. Use l only when 0.5 <= g-r <= 0.8 (SEGUE-valid support); otherwise flag l_outside_domain=1 and treat residual as missing for l features. Build context bins from training data only: redshift bins with edges [-0.01, 0.2, 0.6, 1.2, 2.4, 4.0, 7.0], sky cells from a 12-bin alpha × 6-bin delta grid, and the categorical pair (spectral_type, galaxy_population). For each train-context-cell compute median m_c and MAD d_c per color, then robust scale s_c = max(1.4826 * MAD_c, eps_c) with eps_c set to 0.02 for s,w,x,y and 0.03 for l. For each object compute signed residual r_c = c - m_c and robust score z_c = (r_c / s_c) for each valid color, then clip z_c to [-10, 10]. To avoid sparse-cell noise, apply hierarchical backing-off: (1) full context, (2) redshift+sky only, (3) redshift+tags only, (4) redshift only, (5) global train medians. Use the first level with at least 300 rows for the required color and 500 for all colors; if none satisfy, use global. Emit signed z_c, absolute z_c, max_abs_z = max(|z|), l2_z = sqrt(sum z^2), a tail_count of coordinates with |z|>2, a tail_ratio = tail_count/available_colors, and in_locus = 1{|all |z|<0.75|}, plus a 4-bit sign code for (s,w,x,y) and binary indicators for each coordinate missing/not-reliable. All context statistics are frozen from training and applied deterministically to train/validation/test with no label usage.
- expected_signal: Class separation is likely strongest in orthogonal manifold offsets: stars stay near locally centered locus planes whereas galaxies and especially quasars produce structured excursions whose direction and magnitude vary with redshift and observing context, and the robust local baseline plus scale normalization suppresses absolute-photometric and field-level offsets that would otherwise mask this signal.
- risk: Residual features can overlap with other locus-based groups and may be redundant, sparse sky/redshift cells can produce unstable baselines if fallback is insufficiently aggressive, and conditioning on l with a narrow validity domain may reduce coverage unless missingness is modeled explicitly.

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