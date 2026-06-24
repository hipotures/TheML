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
- title: Robust Ordinal Band Topology Signature
- group_name: band_order_topology
- family: ordinal_sed
- summary: Encode each object by the shape of its ugriz brightness ordering and local monotonicity pattern so the model can separate smooth stellar continua, galaxy-like single turnovers, and more irregular quasar band patterns while remaining insensitive to absolute flux scale.
- strategy: Let m=[u,g,r,i,z] and τ=0.02 mag. Compute spread Δ=max(m)-min(m). If Δ<=2τ, mark flat_topology=1 and set rank/turn features to neutral defaults (monotone=1, inversion counts=0, sign_changes=0, first_change=0, peak/valley positions=0). Otherwise set flat_topology=0 and proceed. Form a deterministic ordered-band sequence by sorting indices by ascending magnitude (brighter first), breaking exact equality or |m_a-m_b|<=τ ties by wavelength index u<g<r<i<z. For this sequence derive: (a) position of brightest and faintest bands (1..5), (b) whether each is interior {g,r,i} vs edge {u,z}, (c) total tie-block count, (d) pairwise inversion count versus template [u,g,r,i,z], and (e) pairwise inversion count versus reverse template [z,i,r,g,u], both restricted to non-tied pairs. Compute adjacent differences d1=g-u,d2=r-g,d3=i-r,d4=z-i and initial signs qk=sign(dk) where sign=+1 if dk>τ, -1 if dk< -τ, else 0. Impute zeros deterministically: scan from left to right to fill each interior zero with nearest non-zero sign; if still missing use nearest on the right, and if both neighbors non-zero and conflicting keep 0 (ambiguous inflection). From imputed q derive monotone flag (all non-zero signs equal), number of sign changes over consecutive q entries, a one_turn flag, first_change position in {0..3} (0 when none), and second_change if present. Map the imputed sign pattern into 4 buckets: strictly_monotone, single_turn, multi_turn, and flat/ambiguous. These structural outputs are the hypothesis features.
- expected_signal: By encoding only relative order and local slope signs, the representation is robust to magnitude zero-point offsets and global extinction-like shifts, while still preserving the class-distinctive topology cues (smooth monotonicity for many stars, one interior turnover for many galaxies, and mixed turnovers for many QSOs) that can improve recall balance on minority structure-heavy classes under balanced_accuracy.
- risk: The discrete topology features can be brittle when adjacent bands differ by tiny amounts, so small photometric noise may flip tie/turn states; aggressive tie smoothing may also over-stabilize genuinely informative micro-structure, and several signals may be correlated with explicit color features, reducing marginal gain versus noise.

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