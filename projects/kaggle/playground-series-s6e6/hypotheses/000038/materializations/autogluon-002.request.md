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
- title: Adaptive Redshift-Manifold Color Geometry Features
- group_name: photoz_trajectory_geometry
- family: trajectory_consistency
- summary: Model the expected redshift evolution of SDSS color space as a smooth manifold and encode each object by deterministic distances to this manifold, including displacement off the manifold and residual behavior along and across its local trajectory, so class-dependent departures become explicit predictive signals.
- strategy: Use only train rows to fit the trajectory model. Define colors c=[c1,c2,c3,c4]=[u-g, g-r, r-i, i-z]. Build a redshift grid G={g_k} from quantiles of redshift at levels 0.5%..99.5% with K=180 knots. For each knot g_k, fit local estimates with weights w_i(g_k)=exp(-(z_i-g_k)^2/(2h^2)), h=0.08. Require at least N_min=1500 effective weighted samples; if not satisfied, repeatedly double h (up to 1.0) and recompute; if still insufficient, fallback to a global robust fit with derivative suppressed. Compute weighted Huber location μ(g_k)=HuberMean(c | weights), weighted robust covariance Σ(g_k)=HuberCov(c | weights), then shrink Σ(g_k): Σ_s(g_k)=0.85Σ(g_k)+0.15·I*(tr(Σ(g_k))/4) and floor eigenvalues at 1e-6. Fit each color component μ_j(z), each Σ_s entry, and first/second derivatives via monotone cubic smoothing splines over k in z to obtain μ̃(z), v(z)=dμ̃/dz, a(z)=d²μ̃/dz², Σ̃(z), and local scales σ_t(z),σ_e(z),σ_m(z) for projected quantities. For any row (c,z), interpolate at z' = clip(z, g_1, g_K); if z outside [min(train z),max(train z)] then set a=0 and damp all derivative-based components by α=min(1,|z-z_edge|/0.03). Compute residual r=c-μ̃(z'). Let L(z') be Cholesky of Σ̃(z'); y=L^{-1}(z') r, t=L^{-1}(z') v(z'), u=t/||t||, and a_w=L^{-1}(z') a(z'). Define tangential signed offset T=y·u, orthogonal energy E=||y||^2-(y·u)^2, and curvature mismatch M=(r^T Σ̃^{-1} a)/(||a||+1e-6); emit {T, E, M, T/σ_t(z'), E/σ_e(z'), M/σ_m(z')}. For z<0.01, further multiply T,M,E by z/0.01 to avoid unstable derivative behavior in near-zero-redshift regime.
- expected_signal: The feature family turns a noisy multiband snapshot into redshift-normalized manifold diagnostics, so quasars with emission-line-driven color drift, galaxies following smoother spectral tracks, and nearby low-variation stars become separable by different residual geometry patterns, which should improve recall balance across GALAXY, QSO, and STAR.
- risk: A small bandwidth or poorly conditioned covariance can overfit local noise and destabilize derivative features, while large smoothing can erase class-specific local curvature cues; the approach is compute-heavy for spline fitting and per-row linear-algebra operations, and may be partially redundant with other redshift- or color-derived features if not tuned.

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