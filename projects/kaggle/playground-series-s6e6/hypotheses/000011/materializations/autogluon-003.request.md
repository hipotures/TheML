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
- title: SDSS LRG Cut-Margin Proxies
- group_name: lrg_target_cut_margins
- family: target_selection_geometry
- summary: Convert the documented SDSS luminous-red-galaxy color-magnitude selection geometry into continuous proximity and hard-membership signals that describe how strongly each object resembles or violates the red-galaxy target region.
- strategy: Using only g, r, and i, compute gmr = g - r, rmi = r - i, c_perp = rmi - gmr / 4 - 0.177, and c_par = 0.7 * gmr + 1.2 * (rmi - 0.177), following the DR2/DR3 LRG equations documented by SDSS DR17 at https://www.sdss4.org/dr17/algorithms/legacy_target_selection/. Treat the provided r magnitude as the best available proxy for unavailable r_Petro, and treat raw g/r/i values as proxies for extinction-corrected model magnitudes. Use a signed-margin convention where negative means the available photometric cut is satisfied and positive means violation. For Cut I, compute mI_lum = r - (13.116 + c_par / 0.3), mI_r = r - 19.2, and mI_c = abs(c_perp) - 0.2; define score_cutI_raw = max(mI_lum, mI_r, mI_c). For Cut II, compute mII_c = (0.449 - gmr / 6) - c_perp, mII_red = (1.296 + 0.25 * rmi) - gmr, and mII_r = r - 19.5; define score_cutII_raw = max(mII_c, mII_red, mII_r). Emit c_perp, c_par, all six component margins, score_cutI = clip(score_cutI_raw, -20, 20), score_cutII = clip(score_cutII_raw, -20, 20), score_lrg_any = clip(min(score_cutI_raw, score_cutII_raw), -20, 20), and binary pass flags pass_cutI, pass_cutII, and pass_lrg_any computed from the unclipped raw scores using <= 0. Clip only emitted margin-like features, not the raw values used for pass logic; no imputation is needed because the required columns have no missing values.
- expected_signal: The features encode a known oblique red-galaxy selection surface that models might otherwise need to relearn from separate color and magnitude columns, so objects inside or close to either LRG region should become more confidently GALAXY-like while the direction and size of violations help reject red stars, compact quasars, and blue contaminants, potentially improving balanced accuracy through cleaner GALAXY separation.
- risk: The official target selection also used extinction-corrected model colors, Petrosian magnitudes, morphology, surface-brightness, and quality flags that are unavailable here, so these are only proxy margins; fixed SDSS thresholds may be brittle under synthetic calibration shifts, can duplicate information already present in g/r/i, redshift, spectral_type, or galaxy_population, and may overemphasize red galaxies while adding little signal for blue galaxies or QSO-vs-STAR separation.

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