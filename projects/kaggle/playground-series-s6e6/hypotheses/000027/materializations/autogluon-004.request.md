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
- title: Train-test aware id sequence context
- group_name: aide_id_sequence_scan_context
- family: aide_sequence_context
- summary: Encode each object's identifier as deterministic position, block, periodic, and local-neighborhood context so the model can test whether row-id order captures acquisition or batching structure related to class prevalence.
- strategy: Parse `id` as integer-like numeric values and create `id_valid` and `id_missing` flags; invalid or missing ids are assigned the training minimum id before numeric transforms while retaining the flags. Fit all scaling, clipping, and normalization constants on training ids only, but compute order-dependent ranks and gaps on the concatenated unlabeled train+test id list when both tables are available so boundary behavior across the contiguous split is represented without using targets. Add monotonic position features: raw numeric id, `id_offset = id - train_id_min`, `id_rel_train = id_offset / max(1, train_id_max - train_id_min)` clipped to a wide range such as [-0.25, 2.0], `id_rank_global`, and `id_rank_global_norm = id_rank_global / max(1, N_global - 1)`. Also add within-table rank features computed separately for train and test, with a `table_is_test_like` indicator omitted during model training if unavailable in validation workflows. Add coarse location features `block_1k = floor(id_offset/1000)`, `block_10k = floor(id_offset/10000)`, `block_100k = floor(id_offset/100000)`, plus normalized within-block positions for 1k and 10k blocks. Add bounded periodic encodings for moduli {2, 11, 97}: both integer residue normalized by modulus and sine/cosine transforms of the residue angle; include `id_even` as a separate binary feature. In sorted global-id order, compute `prev_gap`, `next_gap`, `min_neighbor_gap`, `max_neighbor_gap`, `is_first_global`, and `is_last_global`; fill boundary gaps with 0, cap non-boundary gaps at the training 99th percentile of positive adjacent gaps, and add binary continuity flags for `prev_gap_eq_1`, `next_gap_eq_1`, and `has_noncontiguous_neighbor`. Add `log_id = log1p(max(0,id))` and z-score it using training mean and standard deviation, with std floored at 1e-6. All features must be deterministic functions of ids and unlabeled row membership only, with no class labels, predictions, target encodings, or photometric inputs.
- expected_signal: The train and test ids are contiguous, so identifier order may proxy for survey ingestion order, scan chunks, allocation batches, or synthetic row-generation structure that can shift class priors independently of photometric measurements; balanced accuracy can benefit if this weak context helps minority classes or clarifies boundary regions where QSO, STAR, and GALAXY overlap.
- risk: The signal may be an artifact of this dataset split rather than astronomy, and random cross-validation can overstate its value if nearby ids share labels; raw id extrapolation from train to test can also destabilize some models, so clipped relative scales, bounded periodic encodings, and validation using id-ordered folds are important safeguards.

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