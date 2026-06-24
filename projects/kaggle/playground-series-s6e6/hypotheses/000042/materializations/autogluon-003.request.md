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
- title: Adaptive Deconvolved Locus Offsets in Redshift-Brightness Cells
- group_name: error_deconvolved_locus_tube_residuals
- family: error_aware_locus_geometry
- summary: Model local one-dimensional color-manifold tubes in two adjacent color domains with redshift- and brightness-conditioned, robust noise-corrected orthogonal offsets and directional side information so that high-confidence in-locus objects are centered while outlying quasar/galaxy-like objects receive stable, comparable anomaly scores.
- strategy: Compute colors c1=u-g, c2=g-r, c3=r-i, c4=i-z for every row. Build two 3D color tubes A=(c1,c2,c3) and B=(c2,c3,c4). Bin rows on redshift and i-band with fixed, deterministic edges: Z=[-0.01,0.15,0.30,0.50,0.80,1.20,1.80,2.40,3.00,3.80,4.70,5.60,7.05] and I bins of width 0.5 from 10.0 to 26.0. For each (z,i) cell and each tube, if n<180, iteratively expand to adjacent i bins, then at most ±1 adjacent redshift bin; if still n<120, replace the cell model with the nearest populated cell by L1 distance in (z,i) bin indices, and if absent fall back to the global training median and principal direction. Inside each resolved cell: (i) center by component-wise median m and MAD scale q_j=1.4826*MAD_j, (ii) whiten coordinates x'=(x-m)/q, (iii) fit PCA to x' and extract unit vectors v1 (largest variance axis) and v2 (second axis), (iv) compute orthogonal tube distance d=||x'-(x'·v1)v1||2 for each row, and geometry spread s=1.4826*MAD(d)+1e-6, (v) estimate error floor n_i by 5-nearest-neighbor median absolute distance in the same cell (raw tube units), then define cell floor n_cell=median(n_i) with floor max(n_cell,0.15*s); if 5-NN is impossible, set n_cell=0.15*s. Deconvolved score is t=max(d-n_cell,0)/(s), and signed score is ts=sign(x'·v2)*t. Output per row: A_abs=t_A, A_signed=ts_A, B_abs=t_B, B_signed=ts_B. Add context-aware interaction features only as deterministic gating by redshift: g_2_4_3_0=abs((A_abs+B_abs)/2) * I(2.4<=redshift<3.0), g_3_5_plus=abs((A_abs+B_abs)/2) * I(redshift>=3.5). Clip absolute scores to [0,10] and signed scores to [-10,10], and replace any unresolved missing cell outputs with nearest backfilled cell values from step 1.
- expected_signal: By separating true manifold distance from expected local broadening, this group should preserve weak but stable deviations at bright regimes while preventing spurious separation from faint-noise inflation, which is expected to improve class discrimination for boundary cases where quasar/compact-galaxy objects deviate from the stellar locus.
- risk: Adaptive bin merging and sparse-cell backfilling can smooth away true local structure at extreme redshift or very faint magnitudes, 5-NN error estimates may mix subpopulations within a bin and bias the deconvolution, and the derived signed-distance features may remain redundant with other color-derived signals if they are already present in the model.

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