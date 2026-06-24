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
- title: SDSS Selection-Depth Boundary Margins
- group_name: survey_depth_limit_margins
- family: selection_depth_geometry
- summary: Represent each object by its signed distance to canonical SDSS galaxy and quasar targeting depth landmarks in ugriz space so the learner can capture how class likelihood shifts when an object sits near historical spectroscopic magnitude boundaries.
- strategy: Compute deterministic signed margin features from available magnitudes: for each threshold L in each relevant band, use margin_b_L = mag_b - L (negative means brighter than the cutoff, positive means fainter). Use galaxy-relevant r-band limits L = 17.77, 19.2, 19.5, and use quasar-relevant i-band limits L = 15.0, 19.1, 20.2, 20.4, 21.3, 22.45 plus g=22.0 and r=21.85. Build interval-structure features with inclusive bounds by computing both membership and distance-to-interval-boundary: in_[a,b] = 1{a <= mag_b <= b}; dist_to_interval_[a,b] = 0 if inside interval, else -(a-mag_b) if mag_b < a, else +(mag_b-b) if mag_b > b. Recommended intervals: i in [15.0,19.1], i in [15.0,20.2], i in [20.2,21.3], i in [21.3,22.45], r in [17.77,19.2], and r in [19.2,19.5]. Add compact regime descriptors: d_qso = min absolute margin over all quasar thresholds (per object), d_gal = min absolute margin over {17.77,19.2,19.5} in r, nearest_qso_band = argmin threshold distance among {i-15.0, i-19.1, i-20.2, i-20.4, i-21.3, i-22.45, g-22.0, r-21.85}, nearest_gal_band = analogous over galaxy r thresholds, and boundary_gap = d_gal - d_qso. Optionally include signs of the nearest qso and nearest gal margins to distinguish brighter-side versus fainter-side excursions. All features are pure deterministic transforms; if a magnitude is missing/non-finite, set all derived features from that band to 0 and associated interval flags for that band to 0.
- expected_signal: These features inject prior structure from known survey selection geometry that is often reflected in class composition: bright flux-limited cuts align strongly with nearby galaxies, very faint point-like objects align with quasar-style targeting depth, and star/qso separators are most ambiguous near shared color regimes; explicit boundary-aware features can therefore improve balanced accuracy at the difficult QSO versus STAR and bright/faint GALAXY transition regions.
- risk: The margins are proxy features tied to historical SDSS targeting rules, so if the train/test generation does not follow those specific boundaries (different tile geometry, photometric calibration, or newer selection logic), they can become noisy, redundant with raw magnitudes and color features, and may overfit to release- or chunk-specific artifacts rather than intrinsic class structure.

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