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
- title: Redshift-binned tag compatibility residual margins
- group_name: tag_redshift_compatibility_residuals
- family: tag_manifold_consistency
- summary: Generate features that quantify how well an object's broadband color pattern aligns with the color manifold implied by its provided catalog tags at similar redshift, and how strongly it is contradicted by competing tag regimes.
- strategy: Compute the 4-vector color response C=[u-g, g-r, r-i, i-z] for every row. Transform redshift with t=log1p(max(redshift,0)) and form initial redshift bins from global training quantiles (e.g., 200 quantiles), then greedily merge adjacent bins so each final bin has at least Bmin_train=3000 rows, capping the number of bins at a small stable value. Define the eight catalog tag states s = spectral_type × galaxy_population. For each state-bined cell (s,b) in training, estimate a robust center μ_s,b by median(C), residuals r=C-μ, componentwise scales σ_s,b=1.4826×MAD(r) with lower floor σ_floor = 1e-3, and a robust covariance Σ_s,b from winsorized residuals (componentwise clipping to [P1, P99] within cell, then empirical covariance). Stabilize Σ_s,b by adding λ·diag(median σ_train^2) with λ=0.1 and 1e-6 on the diagonal before inversion. At score time, for a given object and candidate state j, select profile (j,b*) by first trying the target redshift bin, then nearest neighboring bins b±1, b±2, b±3, b±4 until count(s,b*) ≥ 150, then fallback to all-redshift profile for that state, and finally to the global all-state profile if still unavailable. Compute Mahalanobis distance d_ij^2=(C-μ_j,b*)^T Σ^{-1}_{j,b*}(C-μ_j,b*) and clip d_ij^2 to training P1-P99 for all such distances. Let own state be the object’s catalog tags; denote d_self for own state and d_1,d_2 as nearest and second-nearest distances among the three alternative states. Emit bounded margin features m_1=d_1-d_self and m_2=d_2-d_self, raw scores d_self,d_1,d_2, z-score residual vectors r_self=(C-μ_self)/σ_self and for argmin alternative r_alt1, and a fallback-level indicator (direct/neighbor/state/global) plus a short compatibility prior feature p_self_class = count_class(state-bin, class=self_state?)/count(state-bin) using training-class frequencies inside (s,b) with the same fallback chain.
- expected_signal: Objects whose visible colors are internally consistent with their provided tags at a given redshift should have small d_self and large positive mismatch margins, while quasar-like or mis-tagged sources will show inflated self-distance and small or negative margins, giving a class-discriminative signal in overlaps where raw colors alone are ambiguous.
- risk: This group can be brittle when tag assignments in train and test are noisy or distribution-shifted, so profile mismatch features may become systematic error rather than signal; sparse or highly uneven bins can force fallback and reduce granularity; covariance inversion and clipping choices may introduce hyperparameter sensitivity; and the engineered residuals can be partially redundant with direct color and redshift features, increasing model complexity without guarantee of generalization.

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