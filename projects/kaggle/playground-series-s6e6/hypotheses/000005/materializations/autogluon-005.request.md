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
- title: Cross-Validated Shrunk Template Color Residuals
- group_name: catalog_template_residuals
- family: template_mismatch
- summary: Measure how atypical each object's ugriz color pattern is relative to the robust catalog-template locus implied by its spectral and population tags, turning tag-conditioned photometric mismatch into class-separation features.
- strategy: Define adjacent colors c1=u-g, c2=g-r, c3=r-i, and c4=i-z. For model validation, fit all template statistics inside each training fold only and apply them to the held-out fold; for the final submission, refit on the full training set. For each color, compute robust median location and MAD scale for every (spectral_type, galaxy_population) cell, every spectral_type group, every galaxy_population group, and the global training set. For an object with exact cell count n_cell, spectral_type count n_sp, and population count n_pop, form a hierarchical template by shrinking the joint-cell profile toward the spectral_type profile, then lightly toward the global profile when support is weak: w_cell=clip((n_cell-500)/1500,0,1), w_sp=clip((n_sp-1000)/3000,0,1). Use mu*_k=w_cell*mu_cell,k+(1-w_cell)*(w_sp*mu_sp,k+(1-w_sp)*mu_global,k), and s*_k=w_cell*s_cell,k+(1-w_cell)*(w_sp*s_sp,k+(1-w_sp)*s_global,k). If spectral_type is unseen, substitute the galaxy_population profile when available; if both tags are unseen, use global statistics. Floor every scale at 0.02. Standardize residuals as z_k=(c_k-mu*_k)/s*_k and clip z_k to [-10,10]. Emit signed z1..z4 plus aggregate mismatch descriptors mean_abs_z, median_abs_z, max_abs_z, l1_abs_z, l2_z, residual_sum_squares, count_abs_z_gt_2, and count_abs_z_gt_3. Apply identical category fallback and clipping rules at inference.
- expected_signal: Balanced accuracy may improve because GALAXY, STAR, and QSO can share overlapping raw magnitudes or colors while differing in how consistently their full color sequence matches the catalog-implied template; robust residual magnitudes can highlight quasars and mislabeled-template edge cases without directly using target labels.
- risk: The features may be redundant with raw colors and tag variables, sparse or shifted tag groups can produce unstable template baselines despite shrinkage, and validation will be optimistic if the residual statistics are fit on held-out folds rather than only on each fold's training split.

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