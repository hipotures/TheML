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
- title: Robust local color-redshift residuals
- group_name: photoz_color_consistency
- family: photometric_redshift
- summary: Measure how unusual an object's observed redshift is relative to the redshift distribution expected for nearby objects in broadband color-magnitude space, using local disagreement and ambiguity as class-separating evidence.
- strategy: From ugriz magnitudes compute adjacent colors c_ug=u-g, c_gr=g-r, c_ri=r-i, and c_iz=i-z. Learn all clipping limits and bin edges from training predictors only: clamp colors to their 0.1/99.9 percentile training limits for stable bin assignment, but keep the original redshift for residual calculation after only replacing non-finite values with the training median if ever encountered. Build deterministic quantile-binned redshift reference tables using training rows only and no target labels. The primary table uses (c_gr, c_ri, r) with 9, 9, and 10 equal-frequency bins; valid cells require n>=40. Fallback tables use (c_gr, c_ri), (c_ug, c_gr), and (c_ri, c_iz), each with 10x10 equal-frequency bins and n>=60. For every valid cell store median redshift, q10, q25, q75, q90, IQR=max(q75-q25, 1e-3), MAD, n, and robust skew proxy (q90+q10-2*median)/IQR. When transforming a row, first use its exact primary cell; if invalid, aggregate valid neighboring primary cells within Chebyshev radius 1 then 2 by n-weighted medians/quantiles; if still invalid, try the exact fallback tables in the listed order, then radius-1 and radius-2 fallback neighborhoods, then global training redshift statistics. Emit features from the selected reference distribution: signed residual z-redshift_median, absolute residual, residual/IQR, absolute residual/IQR, indicators for below-q10 and above-q90, local_log_count=log1p(n), local_IQR, local_MAD, local_skew_proxy, ambiguity=local_IQR/global_IQR, and an ordinal source_code for exact primary, expanded primary, exact fallback, expanded fallback, or global. For model validation, fit these tables separately inside each training fold and transform only the held-out fold; for final test scoring, refit tables on all training rows.
- expected_signal: Balanced accuracy can improve because stars should have redshifts near zero even when their colors overlap non-stellar regions, galaxies should usually follow smoother low-to-moderate color-redshift structure, and quasars often appear as high-redshift or locally ambiguous outliers, so calibrated residuals and local spread expose differences not captured by raw magnitudes alone.
- risk: Fine bins and neighbor aggregation may still be noisy in rare color tails, and the features can be redundant with raw redshift and color interactions in strong nonlinear models. Training features are especially leakage-prone if rows contribute to their own lookup statistics, so out-of-fold construction is required for honest validation; the added lookup logic also increases preprocessing cost and must keep train/test binning strictly fixed.

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