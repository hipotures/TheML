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
- title: Boundary-safe sky-cell residual harmonization
- group_name: aide_sky_cell_local_residuals
- family: aide_spatial_context
- summary: Create spatial-context features by expressing each object’s magnitudes, redshift, and color-shape descriptors relative to the local sky-cell population, then summarizing the strength and shape of that local deviation as a robust proxy for survey-systematic structure linked to class differences.
- strategy: Define a fixed equal-area-ish tessellation at 0.5-degree resolution in both coordinates: ra_bin = floor(mod(alpha, 360) * 2), dec_bin = floor((delta + 90) * 2), clipped to [0,719]×[0,337], with RA bins wrapped modulo 720 and DEC neighbors clipped at domain boundaries. Assign each row to one sky cell id, then for each row gather the 3x3 neighborhood (9-cell block, RA-wrapped) and compute n_cell, n_neigh, concentration = n_cell/(n_neigh + 1e-6), plus cell-centered local coordinates if needed. For each numeric feature set x in {u,g,r,i,z,redshift_clipped, and all explicit color/shape features u−g, g−r, r−i, i−z, u−i, u−r, g−z, r−z, u−2r+i, g−2i+z}, compute cell-wise mean μ_cell(x) and scale σ_cell(x) on training data only, plus global μ_global(x), σ_global(x). Use shrinkage for instability: μ*(x)=w μ_cell(x)+(1−w)μ_global(x), σ*(x)=w σ_cell(x)+(1−w)σ_global(x), with w=n_cell/(n_cell+20) and a floor σ=0.1-th percentile clamp per feature, then residual Δx=x−μ*(x) and robust z_x=Δx/(σ*(x)+1e-6). Add per-row statistics over x: mean Δx, mean |Δx|, L1 and L2 residual norms, mean z_x, max|z_x|, and residual standard deviation across color/shape features. For sparse cells, also append fallback features from nearest non-empty cell in great-circle RA/DEC Manhattan neighborhood only when n_cell is below threshold to reduce noise; when any feature is undefined, exclude it from that row’s aggregate pool.
- expected_signal: Astrophysical class boundaries often depend on subtle local photometric offsets caused by spatially varying depth, extinction, and calibration, so residualizing to the local sky environment can isolate class-relevant morphology in color-space that is obscured by global feature values.
- risk: Residuals can become brittle where local support is very low, can amplify catalog artifacts if cell definitions mismatch the hidden test distribution, and can overlap with other spatial or photometric feature groups unless de-duplicated or regularized.

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