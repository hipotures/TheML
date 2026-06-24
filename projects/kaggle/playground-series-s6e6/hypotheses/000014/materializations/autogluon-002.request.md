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
- title: Residual-to-local-color redshift consistency
- group_name: photoz_color_consistency
- family: photometric_redshift
- summary: Characterize how plausible each object's measured redshift is relative to the local redshift expectation of its color-magnitude neighborhood so class boundaries are exposed by systematic departures from the normal photometric redshift manifold.
- strategy: Construct color features from ugriz as c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z. Fit color and redshift clipping limits only on the training set (0.1 and 99.9 percentiles) and clamp colors and redshift to those limits before bin assignment. Build deterministic quantile-binned lookup tables from training rows only: primary table on (c_gr, c_ri, r) using 9, 9, and 10 equal-frequency bins respectively; accept a primary cell only if it has at least 40 rows and store z_med, q25, q75 (for IQR), MAD, and n. Build three fallback tables on (c_gr, c_ri), (c_ug, c_gr), and (c_ri, c_iz), each with 10x10 equal-frequency bins and minimum n=60, storing the same statistics. For each row, retrieve expectation statistics by first attempting its exact primary cell, then its immediate primary-neighbor cells by Chebyshev radius 1 then 2 with n>=40, then exact fallback cells in the same three orderings, then each fallback’s radius-1/2 expansion with n>=60; if none apply, use global training z_med and global IQR. Generate deterministic features from the chosen table: residual z_res=z-z_med, |z_res|, calibrated residual z_res/max(IQR,1e-3), local_log_count=log1p(n), local_IQR=IQR, local_MAD, ambiguity=IQR/(global_IQR+1e-6), and a small integer source_code indicating resolution level (exact primary, expanded primary, exact fallback, expanded fallback, global). For validation, construct tables only on each training fold and transform held-out fold rows with fold-specific tables to avoid self-reference artifacts.
- expected_signal: Stars and quasars often violate the dominant galaxy-like color-redshift trend in different ways: stars can have low true redshift with misleading colors, while quasars can generate large or multimodal local residuals at high redshift, so residual magnitude, scale-adjusted residual, and local ambiguity are expected to sharpen balanced separation beyond raw features alone.
- risk: The hierarchy can overfit if cell definitions are too fine or if too many expansions collapse to global behavior, and sparse tail color regions may still be noisy; despite leakage protections, the signals can be redundant with direct redshift-color interactions and add dimensionality/cost, so aggressive regularization or feature pruning may be required.

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