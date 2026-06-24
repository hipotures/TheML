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
- title: Stratified piecewise redshift-color K-correction residual manifold
- group_name: k_correction_residual_manifold
- family: k_correction_consistency
- summary: Fit a deterministic manifold that expresses each observed band through redshift- and color-conditioned correction behavior and use departures from that manifold as residual-geometry signals, so classes are separated by physically meaningful SED mismatch patterns rather than raw photometric scale alone.
- strategy: For each row compute c_u=u-g, c_g=g-r, c_r=r-i, c_i=i-z, and c_z=g-r. Split rows by descriptive strata h=(spectral_type, galaxy_population) and redshift regimes r in { [0,0.2), [0.2,0.7), [0.7,1.6), [1.6,7.0) } with clipping outside support to nearest boundary; if a h x r slice has fewer than 5,000 rows, fallback to a broader slice (merge adjacent regimes, then remove stratum conditioning if still sparse). For each band q in {u,g,r,i,z}, fit in every available h,r a robust Huber polynomial surface K_hat_{q,h,r}(z,c_q)=beta0+beta1 z+beta2 z^2+beta3 c_q+beta4 c_q^2+beta5 z*c_q, where c_q is the chosen adjacent color above. Anchor the correction so that near-zero redshift is neutral by subtracting the 50th percentile of K_hat at z<0.02 within the same h, and optionally clip coefficient magnitudes to median absolute deviation-based bounds before use. Evaluate K^* per object: use regime-local fit in r, but in a blend zone of width delta_z=0.03*(1+z) around regime boundaries linearly interpolate neighboring regime predictions; set edge flags for blended or out-of-hull cases. Compute pseudo-rest magnitudes M_q=m_q-K^*. Build 2D local support bins on (z,c_q) using 30 quantile bins in z and 12 quantile bins in c_q for each h,r. In each bin store median and MAD of each M_q and of slope-like quantities d1=M_u-M_g, d2=M_g-M_r, d3=M_r-M_i, d4=M_i-M_z; produce residuals and MAD-scaled z-scores for all these quantities. Also compute curvature-like terms s1=d1-d2, s2=d2-d3, s3=d3-d4 and derive their residuals/z-scores against the same bin baselines. If a bin has fewer than 200 rows, borrow baseline from distance-weighted 3x3 neighboring bins in quantile-index space; if still empty, use global h,r baseline and emit low-support flags.
- expected_signal: The corrected-rest manifold captures the dominant color-redshift continuum in photometric data, while class-specific departures from that manifold reveal residual structure from stellar-at-z~0 continua and quasar spectral features, so this representation should provide sharper class separation for GALAXY, QSO, and STAR under balanced accuracy.
- risk: Aggressive slicing by stratum and regime can create unstable local fits in sparse high-redshift or rare-type regions, and heavy blending or fallback pooling can damp real minority-class structure; the residual columns are also derived from original magnitudes, so redundancy can inflate model sensitivity to noise unless later learners apply proper regularization.

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