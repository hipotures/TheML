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
- title: SDSS Depth-Regime Margin Features
- group_name: survey_depth_limit_margins
- family: selection_depth_geometry
- summary: Encode apparent brightness as proximity to canonical SDSS-style galaxy and quasar spectroscopic depth boundaries so the model can learn class changes at bright galaxy limits, quasar targeting depths, and faint survey edges.
- strategy: Create deterministic features from the observed g, r, and i magnitudes only. For every threshold L, compute signed_margin_{band}_{L} = mag_band - L, where negative means brighter than the limit and positive means fainter. Use galaxy-oriented r thresholds L = 17.77, 19.2, 19.5, and quasar-oriented thresholds i = 15.0, 19.1, 20.2, 20.4, 21.3, 22.45 plus g = 22.0 and r = 21.85. For each ordered interval [a,b], add inside_{band}_{a}_{b} = 1 if a <= mag_band <= b else 0, signed_interval_margin_{band}_{a}_{b} = 0 inside the interval, mag_band - a when brighter than the lower bound, and mag_band - b when fainter than the upper bound, plus abs_interval_margin as its absolute value. Use intervals i:[15.0,19.1], i:[15.0,20.2], i:[20.2,21.3], i:[21.3,22.45], r:[17.77,19.2], r:[19.2,19.5], and r:[17.77,19.5]. Add aggregate descriptors: nearest_qso_abs_margin = minimum absolute margin across all quasar thresholds, nearest_gal_abs_margin = minimum absolute margin across the galaxy r thresholds, nearest_qso_signed_margin and nearest_gal_signed_margin from the same nearest thresholds, qso_minus_gal_boundary_distance = nearest_qso_abs_margin - nearest_gal_abs_margin, and one-hot indicators for which quasar threshold and which galaxy threshold is nearest. Keep all transforms row-local and deterministic; if any future magnitude is missing or non-finite, set features derived from that band to 0 and set that band's interval indicators and nearest-threshold indicators to 0.
- expected_signal: Balanced accuracy may improve because class prevalence is unlikely to vary smoothly only with raw brightness: nearby galaxies cluster around bright r-selected regimes, quasars extend into deeper i/g/r-selected regimes, and stars often overlap quasar colors but can differ in how they occupy these selection-depth neighborhoods.
- risk: These are historical survey-selection proxy features rather than intrinsic labels, so they may be redundant with raw magnitudes and colors or overfit if the dataset was simulated with different depth rules, extinction handling, morphology assumptions, or train/test-specific magnitude artifacts.

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