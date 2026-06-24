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
- title: Smoothed SDSS quasar-targeting surface geometry
- group_name: sdss_quasar_targeting_surface
- family: astrophysical_selection_geometry
- summary: Model quasar likelihood as a smooth function of distance from learned stellar-locus tube geometry in two SDSS color manifolds while explicitly injecting the known redshift-dependent quasar inclusion regions and color-rejection penalties that resolve ambiguity where quasar and star colors overlap.
- strategy: Compute base colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. On train, clip each color component to the 1st and 99th percentiles to define robust color bounds and avoid tail leakage. Build two color cubes: A=(c1,c2,c3) and B=(c2,c3,c4). In cube A bin by c3 with fixed width 0.08 mag between clipped bounds; in cube B bin by c2 with the same width and the same boundary rules. For each bin in each cube, compute robust centroid μ_b and MAD scale s_b (per axis, clamped floor=0.015). Compute principal axis t_b from eigen-decomposition of the robust covariance of standardized colors in the bin. Define global fallback statistics μ_0, s_0, t_0 from all train objects in the corresponding cube. Let n_b be bin size and w_b=clamp((n_b-30)/80,0,1); define blended parameters μ'_b=w_b μ_b +(1-w_b) μ_0, s'_b = sqrt(w_b s_b^2 +(1-w_b) s_0^2), t'_b=normalize(w_b t_b +(1-w_b) t_0). If n_b=0, set μ'_b=μ_0, s'_b=s_0, t'_b=t_0. For each object x in a cube, compute standardized residual r=(x-μ'_b)/s'_b, projection p=r·t'_b, orthogonal residue q=r-p t'_b, then d=||q||_2. Convert to soft outlier score O=sigmoid((d-3.9)/0.45). Compute OA and OB from cubes A and B and keep both OA, OB and their max Omax=max(OA,OB). Define sigmoid-box gate g(v;L,U,τ)=σ((v-L)/τ)*σ((U-v)/τ), τ in [0.05,0.12]. Inclusion terms: UVX = σ((20.2-i)/1.0)*σ((0.60-c1)/0.05); mid_z = g(c1,0.65,1.5,0.07)*g(c2,0.0,0.20,0.06)*σ((redshift-2.35)/0.35)*σ((3.20-redshift)/0.35); hi_z = σ((redshift-3.0)/0.35)*σ((c1-1.55)/0.08)*σ((20.2-i)/1.0). Exclusion terms: WD = g(g-r,-0.8,-0.2,0.10)*g(r-i,-0.6,-0.2,0.10)*g(i-z,-1.0,0.0,0.10); MWD = g(g-r,0.0,1.6,0.10)*g(r-i,0.6,2.0,0.10); A = g(u-g,0.9,1.5,0.06)*g(g-r,-0.35,0.0,0.06); BLUE = σ((0.90-c1)/0.08)*σ((0.80-c2)/0.08)*σ((i-19.0)/1.0). Also build population-aware geometry: if a population-specific bin has n_b>=40 compute separate μ_b, s_b, t_b; otherwise use blended μ'_b as above, else fallback global. Output OA,OB,Omax and the smooth inclusion/rejection terms as distinct numeric features.
- expected_signal: The revision keeps the original SDSS-inspired physical selection geometry but removes brittle hard thresholds by using probabilistic tubes and continuous gates, which should preserve the discriminative behavior near the 2.7-2.8 and high-redshift overlap regions while reducing instability from hard bin edges and outliers; this directly targets STAR/QSO confusion and can improve balanced accuracy for minority quasar rows by making the boundary signal more calibratable.
- risk: The constants remain from historical SDSS designs and can be mismatched to this dataset's calibration, color system, and spectroscopic mix; sparse-bin PCA blending can still produce unstable ridge directions in extreme tails, and redshift-magnitude gating may over-penalize noisy faint objects unless model regularization keeps these features from dominating.

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