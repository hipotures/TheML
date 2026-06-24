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
- title: Robust Main-Sequence Parallax Plausibility Residuals
- group_name: main_sequence_parallax_plausibility
- family: photometric_parallax
- summary: Create features that quantify how closely each object's ugriz photometry matches a main-sequence-derived distance model, treating deviations from this physically grounded absolute-magnitude manifold as a direct cue for non-stellar objects.
- strategy: For each row compute x = g - i. Create deterministic edge flags x_below = (x < 0.20), x_above = (x > 4.00), x_turnoff_zone = (x < 0.45) as a turnoff-risk indicator, and define x_c = clip(x, 0.20, 4.00). Compute M_r(x_c) with the calibrated relation M_r = -5.06 + 14.32*x_c - 12.97*x_c^2 + 6.127*x_c^3 - 1.267*x_c^4 + 0.0967*x_c^5, then mu = r - M_r, d_pc = 10^((mu + 5)/5), and log_d = log10(clamp(d_pc, 1, 1e10)). Compute absolute-magnitude counterparts M_u = u - mu, M_g = g - mu, M_i = i - mu, M_z = z - mu and at least these four absolute colors: M_u - M_g, M_g - M_r, M_r - M_i, M_i - M_z (M_r - M_i is included as a compact internal consistency check). From training data only, build 80 equal-frequency bins of x_c, and for each bin store robust statistics (median and MAD_scaled = 1.4826 * median(|x - median(x)|)) for mu, log_d, each absolute color, and each absolute-magnitude delta of interest; require min_bin_n = 500. For scoring, assign each row to a bin; if count < min_bin_n, expand to adjacent bins iteratively (±1, ±2, ±3) and aggregate until min_bin_n is met; if still unavailable, fall back to global training medians/MADs. Then emit standardized residuals z_mu, z_log_d, and z_abscolor_i = (feature - feature_bin_median)/(feature_bin_mad + 1e-6), with all z-scores clipped to [-10, 10], plus within-bin percentile rank of mu. Persist and apply bin edges and statistics from training fold/model fitting only (no refit on validation/test at inference).
- expected_signal: The same observed colors can map to overlapping raw feature values for GALAXY, STAR, and QSO, but stars should remain near a coherent main-sequence absolute-magnitude/distance manifold while galaxies and quasars generally produce implausible implied distances or absolute-color structures, so this residual-based representation provides a stable non-linear separation signal for balanced multiclass classification.
- risk: The mapping is trained on main-sequence assumptions and is less reliable for turnoff stars, giants, unresolved blends, very red/blue extremes, extinction shifts, and low-SNR photometry, so valid-looking stars can be mislabeled as implausible; sparse tail bins can create unstable normalization if fallback logic is weak, and if statistics are learned on full data without proper fold-scoped fitting it can induce subtle selection-shift leakage into validation.

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