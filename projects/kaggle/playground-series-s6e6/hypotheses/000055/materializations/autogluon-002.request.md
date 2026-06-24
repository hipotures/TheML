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
- title: Asinh-flux reliability-weighted continuum shape encoding
- group_name: asinh_jacobian_weighted_shape
- family: flux_uncertainty_geometry
- summary: Map ugriz magnitudes into luptitude-derived flux space and derive a low-SNR-aware five-band SED shape representation that emphasizes continuum geometry (level, tilt, curvature, and controlled-color differences) rather than raw magnitude offsets.
- strategy: Use the SDSS luptitude inverse for each band b in {u,g,r,i,z}: q_b = (1/(2*b_b)) * sinh(-(ln(10)/2.5)*m_b - ln(b_b)), with b_b={1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10}; clip the argument of sinh to [-35,35] for numeric safety. Compute reliability weight w_b = |q_b|/(|q_b|+2*b_b), then n_eff = Σ w_b. If n_eff <= 1e-6 (or any non-finite q_b), replace q_b by deterministic train-derived medians for the matching redshift-bin × spectral_type × galaxy_population cell (if empty, then training global medians); set low_confidence=1 else low_confidence=0. Take band order t = {-2,-1,0,1,2} and fit q_b = s0 + s1*t + s2*t^2 by weighted least squares with weights w_b, and emit coefficients s0,s1,s2 plus weighted residual RMS and weighted MAE from fitted values. Build weighted adjacent contrasts c_ug,c_gr,c_ri,c_iz where c_ab = sqrt(w_a*w_b/(w_a+w_b))*(q_a-q_b), and weighted curvature triplets d1 = sqrt(w_u*w_g/(w_u+w_g))*(q_u-2*q_g+q_r), d2 = sqrt(w_g*w_r/(w_g+w_r))*(q_g-2*q_r+q_i), d3 = sqrt(w_r*w_i/(w_r+w_i))*(q_r-2*q_i+q_z). Add reliability descriptors frac_hi=count(w_b>0.6)/5, frac_mid=count(w_b>0.2)/5, min_w=min(w_b), max_w=max(w_b), and mean_adj_gap = mean(|q_b-q_{b+1}| for adjacent pairs where both w_b,w_{b+1} > 0.1).
- expected_signal: The luptitude-to-flux map preserves ordering at low or negative flux while allowing finite operations in low-SNR tails, and the Jacobian-based reliability weights prevent unstable bands from dominating color geometry; polynomial-plus-contrast shape descriptors should separate quasar UV excess and line-affected slopes from stellar/galaxy continua more cleanly in classes with overlapping raw magnitudes.
- risk: Mis-calibration risk is non-trivial: if the actual photometric release uses different softening constants or implicit zero-points, all derived q_b and derived geometry can be biased; aggressive clipping and fallback imputation can also compress rare but informative outliers, and any fallback statistics must be computed strictly from training-only data to avoid leakage in validation.

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