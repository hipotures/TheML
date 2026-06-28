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
- title: Fold-Safe Deconvolved Color-Locus Tube Offsets
- group_name: error_deconvolved_locus_tube_residuals
- family: error_aware_locus_geometry
- summary: Fit robust local color-locus tubes conditioned on redshift and apparent brightness, then encode noise-corrected orthogonal offsets and side-of-locus information so objects that genuinely depart from the narrow photometric manifold are separated from locally broadened in-locus objects.
- strategy: Compute colors c1=u-g, c2=g-r, c3=r-i, c4=i-z and define two overlapping 3D color spaces A=(c1,c2,c3) and B=(c2,c3,c4). Fit all tube statistics inside the modeling pipeline: during cross-validation, fit only on the training fold and transform the validation fold; for final inference, refit on all training rows while ignoring labels and transform train/test with the same fitted maps. Use fixed bins Z=[-0.02,0.05,0.15,0.30,0.50,0.80,1.20,1.80,2.40,3.00,3.80,4.70,5.60,7.05] and i-band bins of width 0.5 from 10.0 to 28.0, clipping out-of-range values to the nearest edge. For each original (z,i) cell and each color space, resolve a reference sample deterministically: use the cell if n>=200; otherwise expand symmetrically over adjacent i bins up to radius 3, then over adjacent redshift bins up to radius 1, stopping at the first neighborhood with n>=200; if no such neighborhood exists but n>=80, use the largest expanded neighborhood; otherwise backfill from the nearest already fitted cell by L1 distance in bin indices, falling back to a global model if needed. Within each resolved sample, compute component medians m and robust scales q_j=max(1.4826*MAD_j,0.03), standardize x'=(x-m)/q, fit an initial PCA, compute initial orthogonal distances, retain the lowest 85 percent by distance as the robust core, and refit PCA on that core. Orient v1 and v2 deterministically by making the loading with largest absolute value positive. For every row assigned to the cell, compute residual vector r=x'-((x' dot v1)*v1), distance d=sqrt(sum(r^2)), spread s=max(1.4826*MAD(d_core),0.05), and a local broadening floor u from core residuals: for each core row find up to 7 nearest core neighbors in coordinates (x' dot v1, redshift, i), compute the norm between its orthogonal residual and the neighbor median residual, set u=median(jitter)/sqrt(2), then clip u to [0.10*s,0.70*s]; if neighbors are unavailable, set u=0.20*s. Define the deconvolved absolute score t=max(d-u,0)/(s+1e-6) and signed score ts=sign(r dot v2)*t for both A and B. Output A_abs, A_signed, B_abs, and B_signed clipped to [0,12] for absolute scores and [-12,12] for signed scores. Also output two deterministic context interactions using mean_abs=(A_abs+B_abs)/2: mean_abs*I(2.4<=redshift<3.0) and mean_abs*I(redshift>=3.5). Replace any non-finite values with the corresponding fitted-cell median, and if a cell cannot be fitted directly use the nearest-cell or global fallback described above.
- expected_signal: This should help balanced accuracy by adding a calibrated geometric anomaly signal that is less dominated by faint-object color noise than raw color distances, making QSO and galaxy boundary cases more separable from stellar-locus-like objects while preserving directional information for different outlier sides.
- risk: The tube can still represent a mixture of classes rather than a true stellar locus in dense galaxy or quasar regions, sparse-bin fallback may smooth away real high-redshift structure, k-nearest-neighbor broadening estimates can overcorrect in overlapping populations, and validation will be optimistic if the unsupervised fitting step accidentally uses validation or test rows.

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