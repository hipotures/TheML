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
- title: Robust phase-weighted Lyman dropout geometry
- group_name: redshifted_lyman_discontinuity
- family: restframe_discontinuity
- summary: Create rest-frame broadband features that describe whether the Lyman-limit and Lyman-alpha discontinuities fall inside the observed ugriz window and how strongly the SED changes across the corresponding blue and red passbands.
- strategy: Use fixed ugriz effective wavelengths λobs = [3551, 4686, 6165, 7481, 8931] Å and magnitudes m = [u, g, r, i, z]. Set z_eff = max(redshift, 0) to avoid unstable behavior from small negative redshifts, and for each Lyman break B in {912, 1216} Å compute the observed break wavelength λB = B * (1 + z_eff). Let j = searchsorted(λobs, λB), so bands < j are blueward of the break and bands >= j are redward. If j == 0 or j == 5, set edge_available_B = 0, phase_B = 0, edge_quality_B = 0, break_bin_B to the corresponding outside bin, and all contrast/slope features to 0. Otherwise set edge_available_B = 1 and encode the interval bin j in {1:u-g, 2:g-r, 3:r-i, 4:i-z}. Define phase_B = clip((λB - λobs[j-1]) / (λobs[j] - λobs[j-1]), 0, 1), boundary_distance_B = min(phase_B, 1 - phase_B), and edge_quality_B = clip(2 * boundary_distance_B, 0, 1), giving highest confidence when the break is centered between adjacent filters and lowest confidence near a filter center boundary. Compute local_color_B = m[j-1] - m[j]. Compute short-window baselines m_blue_B = mean(m[max(0, j-2):j]) and m_red_B = mean(m[j:min(5, j+2)]), then broad_jump_B = m_blue_B - m_red_B. Also compute flux-ratio contrast using f = 10^(-0.4 * m): flux_jump_B = log10((mean(f[j:min(5, j+2)]) + 1e-12) / (mean(f[max(0, j-2):j]) + 1e-12)). Add local log-wavelength slope slope_B = (m[j] - m[j-1]) / (log(λobs[j]) - log(λobs[j-1])). Return raw contrasts plus quality-weighted versions broad_jump_B * edge_quality_B, local_color_B * edge_quality_B, flux_jump_B * edge_quality_B, and slope_B * edge_quality_B, along with one-hot break interval indicators and outside-window flags for each B.
- expected_signal: The features isolate high-redshift dropout geometry that is expected for quasars and some galaxies but generally absent for stars, giving the classifier class-specific evidence beyond ordinary color indices while the phase and availability terms help tree or linear models treat in-window, edge-adjacent, and outside-window cases differently.
- risk: The group is highly dependent on the provided redshift and may overfit redshift-derived class structure if validation is not representative; the contrasts can be noisy when u or g photometry is weak, and the information may be partly redundant with existing color, redshift, or rest-frame break features, so regularization and feature importance pruning may be needed.

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