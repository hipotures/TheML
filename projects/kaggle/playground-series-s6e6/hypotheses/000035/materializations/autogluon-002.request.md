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
- title: Robust eBOSS c1/c3 wedge and bright-pocket geometry
- group_name: eboss_ptf_c1_c3_geometry
- family: color_plane_selection_geometry
- summary: This feature group encodes the rotated ugri color-manifold geometry used in quasar targeting by measuring signed distances to the c1/c3 selection wedge and the proximity of objects to a known bright-object contamination pocket, yielding a compact contamination-aware representation for multiclass separation.
- strategy: Create adjacent colors from clipped magnitudes to reduce outlier sensitivity: c_ug = u-g, c_gr = g-r, c_ri = r-i, with each band first clipped to training-set robust bounds (p0.5 and p99.5, and fallback to [-30, 35] per band if bounds are missing). Compute c1 = 0.95*c_ug + 0.31*c_gr + 0.11*c_ri and c3 = -0.39*c_ug + 0.79*c_gr + 0.47*c_ri. Define wedge responses d1 = (1.4 - 0.55*c1) - c3 and d2 = (0.3 - 0.1*c1) - c3. Emit signed separation: S1 = max(d1, 0), S2 = max(d2, 0), core = min(d1, d2), V1 = max(-d1, 0), V2 = max(-d2, 0), soft_gap = max(V1, V2), and normalized_gap = soft_gap / (1 + abs(c1)) to avoid magnitude dependence at large c1. Emit bright-pocket coordinates with explicit finite handling: pocket_c1 = max(0.85 - c1, 0, c1 - 1.35), pocket_c3 = max(-c3 - 0.2, 0), pocket_flag = 1 if (r < 20.5 and pocket_c1 == 0 and pocket_c3 == 0), else 0; pocket_distance = max(pocket_c1, pocket_c3). Add stable discretized companions using training-time quantiles only: q_core in {below, near, above} from core values (split at two symmetric quantiles around zero, with tie-safe fallback to sign-based bins), c1_bin as train tertiles, and soft_gap_bin from train quantiles [q0.33, q0.67], while clamping all edge/inf values into nearest finite bin.
- expected_signal: The rotated-plane formulation directly reflects quasar-targeting structure, so distance-to-boundary and boundary-violation features distinguish QSO-like points from the dense stellar/galaxy manifold even when raw colors are similar, and the bright-pocket terms add a calibrated penalty for low-r, color-restricted contaminants, likely improving minority-class recall balance in the multiclass setting.
- risk: The linear boundaries and pocket constants are derived from a specific SDSS-like selection context and may be offset if this dataset’s photometric calibration, extinction treatment, or magnitude conventions differ, which can create systematic false positives/negatives; percentile-based clipping and binning can also dampen true-but-rare extremes and add redundancy with other engineered color-ratio features.

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