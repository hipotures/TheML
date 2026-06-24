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
- title: Adaptive local XDQSO-style flux-density likelihood margins
- group_name: xdqso_inspired_flux_density_scores
- family: probabilistic_flux_density_scoring
- summary: Build a redshift- and magnitude-conditioned probabilistic representation of normalized flux shape from photometry and convert class-specific likelihood margins into posterior-discriminative features that explicitly target overlap regions between GALAXY, QSO, and STAR.
- strategy: For each object, convert magnitudes to linear flux f_b = 10^{-0.4 m_b} for b in {u,g,r,i,z}, use i as reference, and form color-like relative-flux features x = [log(f_u/f_i), log(f_g/f_i), log(f_r/f_i), log(f_z/f_i)] = -0.4[(m_u-m_i),(m_g-m_i),(m_r-m_i),(m_z-m_i)]. Winsorize each component of x to the 0.5%–99.5% train quantiles per component to avoid ratio explosion from extreme magnitudes. Bin two conditioning axes: redshift bins z in { [0,0.8), [0.8,2.5), [2.5,4.0), [4.0,7.0] } and overlapping i bins with centers from min_i_train to max_i_train at 0.2 mag step, each bin width 0.4; assign soft triangular weights across nearest i and z cells so each sample contributes to up to four neighboring (i,z) cells instead of hard assignment. For each class c in {GALAXY, STAR, QSO} and each (i,z) cell with sufficient support, fit a 4D Gaussian mixture on x using train-only class-labeled points in that cell. Use full-covariance components with deterministic count selection by local support n: n>=5000 => 8 comps, 2000<=n<5000 => 4 comps, 500<=n<2000 => 2 comps, n<500 => back off to parent redshift slab; if that is still <200, back off to global class model. Add a diagonal stability term to each component covariance Σ_k: Σ_k <- Σ_k + diag(max(MAD(x_cell,c)^2, 1e-6)); if any eigenvalue is non-positive after fitting, re-regularize with max variance floor then continue. Score test samples by soft-weighted local likelihood L_c = Σ_{cells} w_{i,z} Σ_k π_ck N(x | μ_ck, Σ_k), where w_{i,z} are the interpolation weights; blend with global-class likelihood with 0.3 weight when effective local support is <200. Compute pseudo-posteriors as P̃_c = π_local(c|cell)·L_c with π_local(c|cell)=0.7*empirical_class_freq(cell,c)+0.3/3, then normalize p_c = P̃_c / Σ_c P̃_c. Emit features p_GALAXY, p_STAR, p_QSO, the three pairwise log-odds margins, top-class probability, confidence gap between top two classes, and posterior entropy; if all likelihoods are numerically invalid, use uniform class probabilities as hard fallback.
- expected_signal: This yields a principled, locality-aware generative representation that captures subtle class overlap structure missed by global color heuristics, so marginal posteriors become more robust for ambiguous and minority-class cases and better aligned with balanced accuracy weighting.
- risk: Sparse bins and multimodal tails can produce unstable local mixtures or over-smoothing, regularization and backoff thresholds become sensitive hyperparameters, and full-covariance GMM fitting across many cells increases compute/memory cost, so implementation must guard against pathological local fits and fallback behavior.

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