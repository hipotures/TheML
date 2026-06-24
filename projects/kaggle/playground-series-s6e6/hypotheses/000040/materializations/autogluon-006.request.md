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
- title: Fold-aware hierarchical Bayesian color posteriors
- group_name: class_conditional_color_density_posteriors
- family: bayesian_density_scoring
- summary: This feature group estimates how compatible each object is with class-specific color structure across redshift and i-band regimes, then translates those empirical compatibilities into calibrated class evidence with staged fallback so sparse local cells defer to broader astrophysical context.
- strategy: For each training fold, fit all statistics from that fold only and reuse the same transformed edges and counts for validation and test rows. Define colors as c1=u-g, c2=g-r, c3=r-i, then assign each row to a redshift stratum k using fixed breaks zbreak=[-0.01,0.20,0.60,1.05,1.70,2.45,3.60,5.00,7.01] and to an i-bin m using fixed width 0.50 mag intervals on the fold i-range, with hard clipping to boundary bins. For each color component, compute fold-wise 16 equiprobable bins between the 0.5% and 99.5% quantiles, then append two overflow bins (-inf, +inf), so B=18 bins per component and B^3=5832 3D color cells. Compute sparse counts n_{c,k,m,b}=count(class=c, z-bin=k, i-bin=m, color-cell=b) for c in {GALAXY,QSO,STAR}, along with marginals n_{c,k,m}, n_{c,k}, n_{c,m}, n_c and totals n_{k,m}, n_k, n_m, n. Use α=1 and β=0.5 with formulas π_{c|k,m}=(n_{c,k,m}+β)/(n_{k,m}+3β), π_{c|k}=(n_{c,k}+β)/(n_k+3β), π_{c|m}=(n_{c,m}+β)/(n_m+3β), and π_c=(n_c+β)/(n+3β). Compute conditional likelihoods p_{c,k,m}(b)=(n_{c,k,m,b}+α)/(n_{c,k,m}+αB^3), plus Chebyshev-neighborhood smooths p^{(1)} and p^{(2)} by averaging cells with max-distance 1 or 2 using weights w(d)=2^{-d}, and marginal likelihoods p_{c|k}(b) and p_{c|m}(b), and global p_c(b)= (sum_{k,m} n_{c,k,m,b}+α)/(n_c+αB^3). Select support level with strict ladder for each row and class: if n_{k,m}>=300 and n_{k,m,b}>=5 use p_{c,k,m}(b) with π_{c|k,m}; else if neighborhood mass at radius 1 >=30 use p^{(1)} with π_{c|k,m}; else if radius-2 mass >=80 use p^{(2)} with π_{c|k,m}; else if n_k>=300 use p_{c|k}(b) with π_{c|k}; else if n_m>=300 use p_{c|m}(b) with π_{c|m}; else use p_c(b) with π_c; if no fold data reach this object, use π_c only. Produce class scores s_c=log(max(p+1e-12))+log(max(π_support+1e-12)), plus margins s_QSO-s_STAR and s_STAR-s_GALAXY, and normalized softmax entropy over [s_GALAXY,s_QSO,s_STAR].
- expected_signal: Conditioning color likelihoods on physically meaningful redshift and brightness slices while applying explicit support-aware fallback should preserve strong class-specific priors where data are reliable and reduce brittle local noise in sparse regimes, improving separability in overlapping color regions and likely lifting balanced recall of minority structures such as high-redshift quasars.
- risk: Sparsity and hard stratum boundaries can still destabilize estimates near bin edges, threshold settings may need tuning and can add fold sensitivity, and the sparse 3D histogram storage plus neighbor aggregation can increase memory and engineering complexity if not implemented with compact data structures.

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