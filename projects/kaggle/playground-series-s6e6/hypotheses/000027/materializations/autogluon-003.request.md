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
- title: Robust id-order context encoding
- group_name: aide_id_sequence_scan_context
- family: aide_sequence_context
- summary: This feature group models the row identifier as a proxy for acquisition order and batching structure by encoding each object’s global and local position in identifier space to capture systematic non-astronomical effects that may align with class priors.
- strategy: Parse `id` to integer deterministically; set `id_valid` and `id_missing` flags, and when invalid assign neutral numeric values derived from 0 while preserving the missing indicator. Sort rows by numeric id and define zero-based `id_rank` in that ordered index space, then compute `id_rank_norm = id_rank / (N-1)` clipped to [0,1]. Compute global span features from the split-min/max IDs: `id_min`, `id_max`, `id_span = max(1, id_max-id_min)`, `id_rel = (id - id_min) / id_span`, and `log_id = log1p(max(0,id))`; z-score `log_id` using training mean/std. Add block context: `block_1k = floor((id-id_min)/1000)`, `block_100k = floor((id-id_min)/100000)`, and `within_1k = ((id-id_min) mod 1000) / 1000`. Add periodic signatures with bounded trig features for residues: for m in {2,11,97}, `sin(2π*(id mod m)/m)` and `cos(2π*(id mod m)/m)`. Add neighborhood features from id-order using lag/lead: `prev_gap = id - lag(id)` and `next_gap = lead(id) - id`, with boundary rows filled with 0 and boolean `is_first`, `is_last`; clip gaps at training 99th percentile to cap outliers. Add continuity indicators: `id_even = id mod 2`, `noncontiguous = 1 if prev_gap>1 or next_gap>1 else 0` (or next only at edges), and `adjacent_step = 1 if prev_gap==1 and next_gap==1 else 0`. Keep all transformations strictly within identifiers and ordering, with no target or model-output dependence.
- expected_signal: Because train/test IDs are contiguous across split boundaries, id-based sequence context can expose latent survey or ingestion structure (scan strips, batches, chunk boundaries, periodic routing artefacts) that may carry weak but consistent class-dependent shifts not represented by photometric features, improving multiclass balanced accuracy.
- risk: If the ID assignment is random, remapped, or changes across future deployments, these features can become pure noise or encode dataset-specific artifacts; clipped gaps and explicit boundary handling reduce instability, but do not eliminate overfitting risk.

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