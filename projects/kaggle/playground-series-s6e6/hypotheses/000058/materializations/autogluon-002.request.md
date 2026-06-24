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
- title: Adaptive Deconvolved Flux-Branch Posteriors
- group_name: redshift_branch_deconvolved_flux_posteriors
- family: probabilistic_density_geometry
- summary: Model class-conditional structure of deconvolved relative-flux space as uncertainty-aware generative densities in i-band/ redshift-aware strata to produce posterior-driven features that explicitly encode star--galaxy separation and redshift-dependent quasar ambiguity in a calibrated, uncertainty-aware way.
- strategy: For each object, convert ugriz magnitudes to fluxes F_b = 10^{-0.4 m_b} and build the 4-dim relative vector x = [log(F_u/F_i), log(F_g/F_i), log(F_r/F_i), log(F_z/F_i)] plus fallback color features c=[u-g, g-r, r-i, i-z]. Define overlapping i-band windows I_k = [L_k, L_k+0.8) with step 0.4 covering [min(i_train), max(i_train)]; keep a global window set and for each window require at least N_min=240 training rows by expanding with nearest neighboring windows until coverage reaches N_min, otherwise back off to a global model for that class. Assign each training row to every expanded window it belongs to for density fitting, and for test rows compute features from the same window set using triangular weights w(I_k, i)=1-|i-c_k|/0.4 clipped to [0,1]. Assign QSO branches by observed redshift: B1: z<2.2, B2: 2.2<=z<=3.5, B3: z>3.5; assign galaxies and stars to one non-branch group. For every (window, class, branch) stratum, fit K components by BIC search over K∈{2..8} on x, with floor covariance 1e-6 I and deterministic seed; for sparse strata (<150 samples) shrink toward the corresponding class-wide stratum fit and use fewer components as above. Estimate measurement-noise proxy matrix V per stratum as diagonal in color space from MAD dispersion, then transform to x space; if local MAD is unavailable (e.g., <30 neighbors), use global MAD for that class and window, and if still missing, use a global floor 0.05 per x-dimension. For a query x, compute deconvolved likelihood L_{c,b,w}(x)=Σ_k π_{c,b,w,k} N(x | μ_{c,b,w,k}, Σ_{c,b,w,k}+V). Class priors are smoothed by P(c|w)= (n_{c,w}+α)/(n_w+3α), α=2; branch priors for QSO are similarly smoothed and, when branch sample counts are low, backoff to global class-branch priors with weight λ=n_branch/(n_branch+80). Aggregate across overlapping windows with w(I_k,i), then normalize posterior masses with class probabilities clipped to [1e-8,1-1e-8]: P(C)=P(QSO_all), P(QSO_j) for j in {low,mid,high}. Emit logits for each class, pairwise class margins, per-QSO-branch margins versus STAR and GALAXY, branch concentration max(P(QSO_j))/Σ_j P(QSO_j), branch entropy, and branch-count indicators where P(QSO_j)>0.5*max_j P(QSO_j); clip all log features to [-12,12].
- expected_signal: The revision makes the generative geometry more stable through adaptive window coverage, bounded component complexity, and explicit backoff, so deconvolved class and QSO-branch posteriors remain informative even in sparse regions while retaining redshift-conditioned separation of quasars, which should improve balanced accuracy in overlap regions that dominate confusion.
- risk: Residual overfitting is still possible in thin late-z QSO strata, window mixing can blur sharp local structure, and the proxy noise model may remain misspecified because true photometric errors are absent; the approach is also computationally heavier than hard-cut baselines and should be implemented with care to avoid cross-fold leakage during training-feature generation.

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