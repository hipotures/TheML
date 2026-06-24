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
- title: Adaptive local manifold codimension and support diagnostics
- group_name: local_manifold_codimension
- family: manifold_geometry
- summary: The core idea is to characterize each object by how well it sits on a locally linear, high-support manifold in standardized photometric space versus how much it departs into sparse, high-curvature regions, using that geometric mismatch as a class-sensitive signal beyond global color patterns.
- strategy: Compute color features c1=u-g, c2=g-r, c3=r-i, c4=i-z. Build two numeric spaces: S1=[c1,c2,c3,c4,redshift] and S2=[u,g,r,i,z]. For each numeric column in each space, fit robust scaling on train only with median m and MAD s, with s clipped below 1e-6, then transform x <- (x-m)/s. Set k=64 and for each row define neighbor budget k_i=min(k,|P|-1), where P is candidate pool in train. Candidate pool is first filtered by exact spectral_type and galaxy_population; if |P|<2k_i fall back to full train for that row. For each space separately, query k_i nearest neighbors using a fixed-seed ANN/Euclidean index (test points only query against train, not vice versa) and then compute centroid μ, covariance Σ on centered neighbors, then Σ←0.5(Σ+Σ^T)+1e-6 I_d. If eigenvalue rank<d or trace<=1e-12, use global fallback covariance/eigen stats and set degenerate=1. Sort eigenvalues λ1…λd and top2 eigenvectors V2. Emit: r2=(λ1+λ2)/sum(λ), residual2=(x-μ)^T(I−V2V2^T)(x-μ)/(λ3+…+λd+1e-12), a12=log((λ1+1e-12)/(λ2+1e-12)), a23=log((λ2+1e-12)/(max(λ3,1e-12))), tail=λd/sum(λ) where d is dimensionality of the space, and dim95=min m with cumulative λ1..λm /sum(λ) >=0.95. For neighborhood support, compute distances to each neighbor, then r_k=max distance, r_med=median distance, r_mean=mean distance, density_proxy=1/(r_med+1e-12), r_ratio=r_k/(r_med+1e-12), and frac_in_box = mean over dimensions of x_j being within per-dimension [5th,95th] percentile of neighbor values. Add local asymmetry stats: median absolute deviation of x from neighbor median (and its ratio to neighbor MAD). Concatenate the same feature set from S1 and S2, and add absolute deltas between matched metrics across spaces (e.g., |r2_S1-r2_S2|, |residual2_S1-residual2_S2|, |density_S1-density_S2|). Use a second-pass guard for very sparse regions: if k_i<6 replace distance/eigen metrics with fallback quantiles from global train neighborhoods and keep sparse_flag=1.
- expected_signal: Stars are expected to occupy dense, low-curvature local manifolds aligned with the main photometric loci, while quasars and many galaxies should present as local geometric outliers with higher codimension residuals or weaker support, especially where class boundaries are blurred; these local shape-and-support descriptors can therefore raise recall on minority classes without sacrificing GALAXY separation, improving balanced accuracy.
- risk: High computational cost and ANN approximation error at full train-plus-test scale can add variance to metrics, and conditioning-dependent eigen-ratio features may be unstable in very sparse or duplicated neighborhoods; restricting neighborhoods by auxiliary categories can help homogeneity but may overfit or create sparsity artifacts, and fallback substitution must be carefully fixed to avoid leakage and distribution-sensitive behavior.

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