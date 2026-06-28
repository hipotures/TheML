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
- title: Tie-Aware Ordinal Band Topology
- group_name: band_order_topology
- family: ordinal_sed
- summary: Encode the shape of each object's ugriz brightness ordering and slope topology so the model can use relative spectral-continuum structure without depending on absolute apparent brightness.
- strategy: Let m=[u,g,r,i,z] in wavelength order and use a tolerance τ=0.02 mag, with smaller magnitude meaning brighter. Compute Δ=max(m)-min(m). If Δ<=2τ, set flat_topology=1 and set topology-derived counts and positions to neutral values: extrema positions=0, inversion counts=0, tie_fraction=1, monotone=1, sign_changes=0, first_change=0, second_change=0, topology_bucket=flat. Otherwise set flat_topology=0. For every band pair (a,b), define it as tied if |m_a-m_b|<=τ, otherwise order it by brightness. Build tolerance-aware ranks by sorting bands by ascending magnitude while placing tied bands in the same rank block and breaking only the within-block display order by wavelength u<g<r<i<z. Derive the rank block of each band, brightest block position, faintest block position, brightest_is_edge, faintest_is_edge, brightest_is_interior, faintest_is_interior, number of rank blocks, maximum tie-block size, and tie_fraction=tied_pairs/10. Count pairwise inversions among non-tied pairs relative to the wavelength template [u,g,r,i,z] and relative to the reverse template [z,i,r,g,u], and normalize each by the number of non-tied pairs, using 0 when all pairs are tied. Compute adjacent color slopes d=[g-u,r-g,i-r,z-i] and assign raw signs s_k=+1 if d_k>τ, -1 if d_k< -τ, else 0. Replace zero runs only when both nearest non-zero signs on the left and right exist and agree; if a zero run touches one edge, fill from the single nearest non-zero side; if the bracketing signs disagree, keep the run as 0 to preserve an ambiguous local extremum. From the filled signs derive monotone=1 when all non-zero signs are identical and no ambiguous zero remains, sign_changes as the count of adjacent non-zero sign flips, one_turn=1 when exactly one flip occurs, first_change and second_change as 1-based boundaries between adjacent colors with 0 for absent changes, ambiguous_slope_count as the remaining zeros, and topology_bucket in {strict_monotone, single_turn, multi_turn, ambiguous, flat}. Also record whether the inferred single turn, when present, is peak-like or valley-like based on the sign direction around the first change.
- expected_signal: Balanced accuracy can improve because these features give the classifier class-relevant SED shape cues that raw magnitudes may dilute: stars often have smooth monotone broadband orderings, galaxies often show one dominant turnover or break-like shape, and QSOs are more likely to produce irregular or multi-turn band orderings as emission features move through filters.
- risk: The features intentionally discard amplitude information and may overlap with color features, while the discrete rank and turn states can still flip for low signal-to-noise objects near the τ thresholds; tolerance choices that are too small add noise, and choices that are too large can erase informative class-specific color structure.

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