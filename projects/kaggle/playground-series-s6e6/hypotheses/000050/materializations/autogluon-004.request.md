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
- title: Fold-Safe Main-Sequence Distance Plausibility
- group_name: main_sequence_parallax_plausibility
- family: photometric_parallax
- summary: Represent each object by how physically plausible its ugriz photometry is under a main-sequence stellar distance interpretation, using deviations from a stellar absolute-magnitude manifold as evidence against or for the STAR class.
- strategy: For each row compute x = g - i, flags x_below = 1[x < 0.20], x_above = 1[x > 4.00], x_blue_turnoff = 1[x < 0.45], and x_c = clip(x, 0.20, 4.00). Evaluate M_r_ms = -5.06 + 14.32*x_c - 12.97*x_c^2 + 6.127*x_c^3 - 1.267*x_c^4 + 0.0967*x_c^5. Define mu_ms = r - M_r_ms, log_d_pc = clip((mu_ms + 5) / 5, 0, 10), and d_pc_clipped = 10^log_d_pc if a tree model can use raw-scale distances safely. Compute implied absolute magnitudes M_u = u - mu_ms, M_g = g - mu_ms, M_i = i - mu_ms, M_z = z - mu_ms, plus residual colors abs_ug = M_u - M_g, abs_gr = M_g - M_r_ms, abs_ri = M_r_ms - M_i, abs_iz = M_i - M_z. Also emit raw consistency deltas between observed color and the main-sequence-implied color, especially delta_gr = abs_gr - (g - r), delta_ri = abs_ri - (r - i), and delta_iz = abs_iz - (i - z), even though some are algebraically simple, because they expose the distance-model residuals to models that do not discover interactions easily. During fitting only, create 80 equal-frequency bins on x_c from the fold's training rows. For each bin, estimate robust class-agnostic medians and scaled MADs for mu_ms, log_d_pc, M_u, M_g, M_r_ms, M_i, M_z, abs_ug, abs_gr, abs_ri, abs_iz, and the listed deltas; use MAD_scaled = max(1.4826 * median_abs_deviation, 1e-4). Require min_bin_n = 500; if a bin is smaller, merge symmetrically with adjacent bins until the threshold is reached, otherwise fall back to global fold-training statistics. At transform time, reuse the stored bin edges and statistics without refitting on validation or test data, then output clipped standardized residuals z_feature = clip((feature - bin_median) / bin_MAD_scaled, -10, 10), plus the within-bin percentile rank of mu_ms computed from stored fold-training empirical quantiles. Keep all clipping thresholds, bin edges, fallback rules, and feature ordering fixed between training and inference.
- expected_signal: True main-sequence stars should produce coherent implied distances, absolute magnitudes, and residual colors within their g-i color neighborhood, while galaxies and quasars often need implausible distances or inconsistent absolute-color structures to satisfy the same stellar model, giving the classifier a physically meaningful separation signal that can help balanced accuracy on minority or overlapping classes.
- risk: The relation is only a main-sequence approximation, so giants, turnoff stars, unresolved binaries, extinction effects, metallicity differences, saturated or noisy photometry, and very blue or red edge cases can appear falsely implausible; the residuals are partly redundant with raw colors and magnitudes, and any bin statistics or percentiles computed outside the training fold would leak distribution information into validation.

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