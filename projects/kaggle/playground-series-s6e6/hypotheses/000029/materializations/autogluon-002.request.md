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
- title: Phase-weighted Lyman dropout geometry in rest-frame ugriz
- group_name: redshifted_lyman_discontinuity
- family: restframe_discontinuity
- summary: Engineer deterministic rest-frame features that encode where the Lyman-limit and Lyman-alpha breaks fall across the ugriz bands and how sharply the continuum changes across that location, so the model can exploit physically grounded dropout geometry rather than only global color trends.
- strategy: For each object, define observed central wavelengths λobs = [3551, 4686, 6165, 7481, 8931] Å and magnitudes m = [u, g, r, i, z]. Use z_t = max(redshift, 0) and compute rest-frame centers λrest = λobs / (1 + z_t). For each break B in {912, 1216} Å: let j = np.searchsorted(λrest, B). If j = 0 or j = 5, set edge_available_B = 0 and emit jump_B = 0, local_B = 0, phase_B = 0, edge_quality_B = 0, and all break-bin flags = 0. Otherwise set edge_available_B = 1, phase_B = (B - λrest[j-1]) / (λrest[j] - λrest[j-1]) clipped to [0, 1], and encode break position with break_bin_B in {0,1,2,3,4,5} where 1=u→g, 2=g→r, 3=r→i, 4=i→z (0 and 5 are outside cases). Add edge confidence edge_quality_B = clip(1 - 2 * abs(phase_B - 0.5), 0, 1) to downweight boundaries (phase near 0 or 1). Compute blue-side magnitude baseline m_blue = mean(m[max(0, j-2):j]) and red-side baseline m_red = mean(m[j:min(5, j+2)]). Define jump_B = m_blue - m_red, local_B = m[j-1] - m[j], and optional local log-wavelength slope slope_B = (m[j] - m[j-1]) / (log(λrest[j]) - log(λrest[j-1])). Return interaction terms jump_B * edge_quality_B and local_B * edge_quality_B (plus optional slope_B * edge_quality_B and one-hot/binary indicators for break_bin_B) so features are deterministic, continuous near boundaries, and muted when redshifted break placement is uncertain.
- expected_signal: By tying break location directly to redshifted passband geometry, this group captures high-redshift UV dropout behavior that is strongly class-discriminative for quasars and galaxies but weak for most stars, while quality-gated phase and bounded averaging make the signal more stable than a single hard color-difference feature and reduce noise when the break is near filter edges or outside the optical window.
- risk: The approach is sensitive to redshift errors and can become unstable when B lies very near a band boundary, and it may remain redundant with other redshift-dependent break/line geometry signals; errors in u-band photometry can also dominate these contrast features for low-SNR objects, so regularization or feature pruning may be required.

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