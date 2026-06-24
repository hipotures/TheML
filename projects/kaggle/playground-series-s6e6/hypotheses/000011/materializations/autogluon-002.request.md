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
- title: LRG DR2/DR3 Cut-Margin Geometry
- group_name: lrg_target_cut_margins
- family: target_selection_geometry
- summary: This group converts SDSS LRG target-selection geometry into continuous margin features by projecting color information into the classic rotated LRG basis and measuring how strongly each object satisfies or violates each cut-family boundary.
- strategy: For every row, compute gmr = g - r and rmi = r - i. Derive DR2/DR3 rotated LRG coordinates c_perp = rmi - gmr / 4 - 0.177 and c_par = 0.7 * gmr + 1.2 * (rmi - 0.177). Use r as a proxy for r_Petro (because Petrosian magnitudes are not available). Build signed cut margins where negative means pass and positive means violation. Cut I margins: vI1 = r - (13.116 + c_par / 0.3), vI2 = r - 19.2, vI3 = abs(c_perp) - 0.2; score_cutI = max(vI1, vI2, vI3), which is <= 0 when Cut I is passed. Cut II margins: vII1 = (0.449 - gmr / 6) - c_perp, vII2 = (1.296 + 0.25 * rmi) - gmr, vII3 = r - 19.5; score_cutII = max(vII1, vII2, vII3), which is <= 0 when Cut II is passed. Emit score_cutI, score_cutII, and score_lrg_any = clip(min(score_cutI, score_cutII), -20, 20) so negative indicates membership in at least one LRG region and larger positive values indicate stronger rejection distance; optionally emit cut-pass indicators iI = 1 if score_cutI <= 0 else 0 and iII = 1 if score_cutII <= 0 else 0. No imputation is required because the provided columns have zero missing values; only final signed margins are clipped to [-20, 20] to suppress outlier leverage.
- expected_signal: Objects passing either LRG cut are strongly associated with red, extended galaxies, so these geometry margins provide a calibrated, physics-motivated separation signal that can outperform raw colors by explicitly encoding whether the object sits on, inside, or outside known LRG boundaries that also screen out many compact blue contaminants, helping GALAXY vs STAR/QSO discrimination.
- risk: The implementation is a proxy of the true SDSS target logic because r_Petro, extinction-corrected model magnitudes, PSF/model morphology, and surface-brightness filters are not in the table, so margins can be biased and may overfit if the dataset is not SDSS-distributed; fixed numeric boundaries may also be brittle under synthetic color shifts and are partly redundant with base u,g,r,i,z color features.

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