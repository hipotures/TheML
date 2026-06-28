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
- title: Robust eBOSS c1/c3 wedge and bright-pocket margins
- group_name: eboss_ptf_c1_c3_geometry
- family: color_plane_selection_geometry
- summary: Encode the SDSS-style rotated ugri color-plane geometry used for quasar selection as robust boundary-margin and bright-contaminant-pocket proximity features that summarize how far each object lies from quasar-like versus stellar or low-redshift color manifolds.
- strategy: Compute adjacent colors from original magnitudes: c_ug = u - g, c_gr = g - r, c_ri = r - i. Fit all robustness parameters on train only, then reuse for test: replace nonfinite color values with the corresponding train median and clip each derived color to train p0.1-p99.9 bounds, with fallback bounds [-10, 10] if quantiles are unavailable. Compute rotated coordinates c1 = 0.95*c_ug + 0.31*c_gr + 0.11*c_ri and c3 = -0.39*c_ug + 0.79*c_gr + 0.47*c_ri. Define wedge upper boundaries b1 = 1.4 - 0.55*c1 and b2 = 0.3 - 0.1*c1, margins m1 = b1 - c3 and m2 = b2 - c3, and emit core_margin = min(m1, m2), abs_core_margin = abs(core_margin), inside_margin_1 = max(m1, 0), inside_margin_2 = max(m2, 0), violation_1 = max(-m1, 0), violation_2 = max(-m2, 0), max_violation = max(violation_1, violation_2), and normalized_violation = max_violation / (1 + abs(c1)). For the bright rejection pocket, emit pocket_c1_gap = max(0.85 - c1, 0, c1 - 1.35), pocket_c3_gap = max(-0.2 - c3, 0), pocket_distance = max(pocket_c1_gap, pocket_c3_gap), pocket_flag = 1 when r < 20.5, pocket_c1_gap == 0, and pocket_c3_gap == 0, else 0, plus bright_scaled_pocket_distance = pocket_distance / (1 + max(20.5 - r, 0)). Add deterministic discretized companions using train-only edges: core_margin_bin from sign-aware bins {negative, near_zero, positive} where near_zero is bounded by the train 45th-55th percentile interval clipped to include 0, c1_tertile from train tertiles of c1, and max_violation_bin from train tertiles of max_violation; clamp test values outside fitted edges to the nearest bin.
- expected_signal: Balanced accuracy may improve because these features express physically motivated quasar-targeting geometry as signed distances rather than raw colors alone, helping models separate QSO-like objects from STAR and GALAXY contaminants while the explicit bright-pocket terms capture a known region where color similarity can otherwise cause minority-class confusion.
- risk: The c1/c3 constants and pocket thresholds come from SDSS-like target-selection rules and may be miscalibrated for this dataset's photometry or preprocessing, so they can introduce biased margins; train-quantile clipping and bins may suppress rare valid extremes, and the group may be partly redundant with other engineered color-plane or color-ratio features.

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