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
- title: Hierarchical Redshift-Brightness Bayesian Color Posteriors
- group_name: class_conditional_color_density_posteriors
- family: bayesian_density_scoring
- summary: The group builds class-conditional posterior-like compatibility scores by comparing each object’s colors against empirical class manifolds estimated in redshift and i-band brightness slices, with staged fallback to broader distributions when local support is weak.
- strategy: For each fold, build color coordinates c=[u−g, g−r, r−i] from the available training fold only. Use this same mapping for train and test rows. Define redshift strata with explicit bounds [-0.01, 0.2, 0.7, 1.3, 2.0, 3.2, 4.8, 7.01] (inclusive of the upper edge in the last bin). Define i-band bins as fixed 0.50-mag intervals over the fold i-range, with hard clipping of any value outside range to endpoint bins. For each color component, compute fold quantile cut edges using 16 equiprobable bins after clipping at the 0.5 and 99.5 percentiles, then append two overflow bins to guarantee bounded indexing for all rows. Let n^c_{k,m,b} be class-c counts in redshift bin k, i bin m, and 3D color cell b; let N^c_{k,m}=Σ_b n^c_{k,m,b}, N_{k,m}=Σ_c N^c_{k,m}, and B be total color cells. Apply Dirichlet smoothing with α=1 and class-prior smoothing β=0.5: π^c_{k,m}=(N^c_{k,m}+β)/(N_{k,m}+3β), P^c_{k,m}(b)=(n^c_{k,m,b}+α)/(N^c_{k,m}+αB). Define a support floor n_min (e.g., 200). For each sample at cell b in stratum (k,m), score class c as s_c=log(π+ε)+log(p+ε), where ε=1e-12 for numerical safety, but before using p run a deterministic backoff chain: if N_{k,m}<n_min or the target cell is empty/unstable, use Chebyshev-radius-1 neighborhood averaging with weights 1, 1/2, 1/4, then radius-2 if needed; if still unsupported, use redshift-marginal P^c_k(b), then global P^c(b), and if all else fails use global π^c only. After final backoff, emit s_GALAXY, s_QSO, s_STAR, margins s_QSO−s_STAR and s_STAR−s_GALAXY, and normalized softmax entropy of s_c.
- expected_signal: Conditioning on redshift and brightness directly encodes expected astrophysical evolution in color distributions while preserving class-level prevalence, so ambiguous colors are resolved using local class manifold density instead of global similarity, improving discrimination of minority structures such as high-z quasars and reducing balanced-accuracy loss from overlap between classes.
- risk: Sparsity in fine strata can still produce noisy estimates if thresholds are set too low, bin boundaries may bias nearby points near transitions, and the additional 3D histogram tables increase memory/time unless built sparsely and cached, while any hyperparameter tuning of binning/backoff done outside cross-validation risks optimistic bias.

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