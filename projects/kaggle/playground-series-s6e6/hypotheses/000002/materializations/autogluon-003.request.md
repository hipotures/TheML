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
- title: Redshift-Regime Catalog Consistency Encoding
- group_name: redshift_regime_catalog_consistency
- family: astrophysical_consistency
- summary: Represent whether an object's redshift-implied distance regime is consistent with its catalog spectral and population tags, emphasizing stellar-like near-zero cases, galaxy-like intermediate cases, and quasar-like high-redshift cases.
- strategy: Use only `redshift`, `spectral_type`, and `galaxy_population`. Compute `abs_redshift = abs(redshift)`, `redshift_negative_flag = 1[redshift < 0]`, `redshift_zero_like_flag = 1[abs_redshift < 0.003]`, and `signed_log_redshift = sign(redshift) * log1p(abs_redshift)`. Bin `abs_redshift` into ordered deterministic regimes: `R0_zero_like` for abs_redshift < 0.003, `R1_very_low` for 0.003 <= abs_redshift < 0.0333, `R2_low` for 0.0333 <= abs_redshift < 0.25, `R3_moderate` for 0.25 <= abs_redshift < 1.0, `R4_high` for 1.0 <= abs_redshift < 3.0, and `R5_extreme` for abs_redshift >= 3.0. Boundary handling is left-closed/right-open, so exact threshold values enter the higher regime. Emit the regime as both an ordered categorical feature and a small ordinal code from 0 to 5. Add crossed categorical features `redshift_regime x spectral_type`, `redshift_regime x galaxy_population`, and `redshift_regime x spectral_type x galaxy_population` to expose catalog-consistency patterns. Preserve all rows, do not clip redshift values, and treat the provided categorical values as fixed known levels.
- expected_signal: Balanced accuracy may improve because the same catalog tag can imply different classes depending on redshift regime: near-zero redshift is strongly stellar-like, moderate redshift is more galaxy-like, and high or extreme redshift is more QSO-like. The crossed features help the model learn rare but class-informative combinations without relying only on a smooth raw redshift split.
- risk: The hand-chosen thresholds may be brittle if redshift calibration differs between train and test, and the three-way cross can create sparse combinations that overfit in high-capacity models. These features may also be partly redundant with raw redshift and categorical encodings, so regularization or validation-based pruning may be needed.

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