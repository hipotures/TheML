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
- title: Survey Manifold Rarity
- group_name: survey_manifold_rarity
- family: density_rarity
- summary: Estimate how typical or unusual each object is inside the unlabeled survey manifold formed by coarse redshift regime, catalog tags, and broad photometric energy allocation, so the model can use rarity itself as a class cue.
- strategy: For each object, convert ugriz magnitudes to pseudo-fluxes f_b = 10^(-0.4*m_b), compute normalized band shares p_b = f_b / sum(f), then derive three survey-manifold coordinates: blue_balance = (p_u + p_g) - (p_i + p_z), flux_concentration = max(p_u,p_g,p_r,p_i,p_z), and total_brightness = -2.5*log10(sum(f)). Build bins from train.csv only: redshift bins with fixed edges [-inf, 0.002, 0.01, 0.05, 0.2, 0.6, 1.2, 2.5, inf], plus decile bins for blue_balance, flux_concentration, and total_brightness; if quantile edges collapse because of ties, merge duplicate edges and use the resulting ordered bins. Form a joint cell from (redshift_bin, spectral_type, galaxy_population, blue_balance_bin, flux_concentration_bin, total_brightness_bin). From train.csv only, compute Laplace-smoothed counts with alpha = 1 for the full joint cell, for the tag-regime block (redshift_bin, spectral_type, galaxy_population), and for the photometric block (blue_balance_bin, flux_concentration_bin, total_brightness_bin). Output one coherent rarity group consisting of: joint_log_density = log((count_joint + 1) / (N + K_joint)), rarity_score = -joint_log_density, and interaction_surprise = log((count_tag_regime + 1) * (count_photo_block + 1)) - log((count_joint + 1) * N), where N is the train row count and K_joint is the number of observed-or-possible joint cells under the constructed bins. Apply the same train-derived bin edges and smoothed lookup tables to test rows; values outside the train range fall into the nearest end bin, and any unseen categorical level would map to an UNK bucket before lookup.
- expected_signal: QSO examples often occupy sparse or cross-inconsistent regions of the survey manifold, while many STAR and GALAXY objects cluster in denser loci, so explicit rarity and interaction surprise can improve per-class separation and help balanced accuracy rather than only overall majority fit.
- risk: If the joint grid is too sparse, these features can become noisy and effectively memorize idiosyncratic train cells; they may also be partially redundant with existing color- and redshift-based groups and can drift if the test distribution populates different density regions.

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