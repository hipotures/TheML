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
- title: AIDE auxiliary distribution distance with robust circular sky geometry
- group_name: aide_aux_reference_distribution_distance
- family: aide_auxiliary_reference
- summary: Construct robust outlier-style signals by comparing each test object against an external reference population in overlapping sky, photometric, and redshift predictor space, using stable marginal and joint distance statistics to capture how typical or atypical its profile is.
- strategy: Load the auxiliary reference table and restrict to predictor columns in the overlap set {alpha, delta, u, g, r, i, z, redshift}; reject any other fields from distance computation. Clean numeric predictors deterministically: values <= -9000 in u, g, r, i, z are set to missing, and redshift < 0 is clipped to 0.0001. Convert alpha to radians and create two deterministic positional components sin_alpha = sin(2*pi*alpha/360) and cos_alpha = cos(2*pi*alpha/360); keep delta as a linear feature. For each used column j, compute reference median m_j and scale s_j from MAD; if MAD is zero, fall back to IQR/1.348, and if still zero, replace with the median positive fallback scale across all columns. For each row x, compute per-column robust z value z_j=(x_j-m_j)/s_j with clipping to [-8,8] when finite, empirical CDF rank p_j using 1-indexed rank ordering on complete reference values (p_j=(rank+1)/(n_ref+1) and clipped to [1/(n_ref+2), 1-1/(n_ref+2)]), and median distance a_j=|x_j-m_j|. Emit a per-feature missing flag and maintain missing_count and available_ratio over overlap columns. Aggregate marginal features as mean_abs_z, max_abs_z, mean_rank_shift = mean(|p_j-0.5|), and mean_abs_median_delta = mean(a_j), each computed over available columns only. Build a joint standardized vector w from [sin_alpha, cos_alpha, delta, u, g, r, i, z, redshift] after excluding unavailable coordinates per row; estimate reference covariance on complete rows after centering and clipping z-like standardized values to [-6,6], then regularize with ridge τ = 1e-6 * trace(C)/d if d>=2 or if covariance rank is deficient. Compute Mahalanobis squared distance d2 = w^T * inv(C_reg) * w, Mahalanobis distance d = sqrt(max(d2,0)), and inlier score s = log1p(d2); if row-wise completeness is insufficient for joint computation, use deterministic row-level fallback by shrinking to available components and a missingness-aware mean distance, but keep availability flags. If auxiliary data is missing or effective overlap is below three usable numeric columns, return explicit neutral constants for all distance features plus full missing/unavailable indicators; never use target labels and never train a downstream classifier in this hypothesis.
- expected_signal: This feature group provides a calibrated notion of how likely each object is under a broad external population; class boundaries are often separable by how far objects move from the joint photometric/redshift manifold (e.g., unusual color-redshift combinations or sky-position concentration), so these robustness-stabilized marginal and Mahalanobis-derived distances add strong discriminative structure for balanced multiclass scoring.
- risk: Potential distributional shift between auxiliary and competition data (calibration, selection, or footprint bias) can introduce systematic ordering errors, and aggressive clipping/regularization may obscure subtle class structure; joint distance estimates are sensitive to low-variance features and incomplete rows, so missingness and fallback paths are essential but may add redundancy or noisy proxies.

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