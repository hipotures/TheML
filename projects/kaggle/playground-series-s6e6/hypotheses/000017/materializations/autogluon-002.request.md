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
- title: Redshift-normalized absolute-luminosity regime features
- group_name: redshift_luminosity_plausibility
- family: absolute_photometry
- summary: Convert observed ugriz magnitudes into redshift-scaled intrinsic-brightness proxies and summarize their cross-band behavior to provide physically grounded class separation cues between nearby stars, intermediate-luminosity galaxies, and very luminous quasars.
- strategy: For every row, compute a deterministic luminosity-distance proxy from redshift under a fixed flat cosmology (H0=70 km/s/Mpc, Ωm=0.3, ΩΛ=0.7) with c=299792.458 km/s, using z_eff = max(redshift, 1e-4) for distance math. Calculate D_L via D_C=(c/H0)*∫_0^{z_eff}(1/sqrt(Ωm(1+z)^3+ΩΛ))dz and D_L=(1+z_eff)*D_C, using the same numerical approximation at train and test; retain flags z_nonpos=(redshift<=0), z_low=(redshift<0.01), and z_hi=(redshift>=3). For each band b in {u,g,r,i,z}, derive M_b = m_b - 5*log10(D_L_Mpc*1e6) + 5 (equivalently m_b - 5*log10(D_L_Mpc) - 25), then clip each M_b to [-35, 10]. Build deterministic aggregates: M_mean, M_median, M_iqr, M_sd, M_min, M_max, M_rg = M_r - M_g, M_ri = M_r - M_i, and pairwise absolute spread (max-min). Add threshold-margin and bin features on M_r: Δ_star = M_r - (-10), Δ_midgal = M_r - (-18), Δ_brightgal = M_r - (-23), Δ_qso = M_r - (-27), plus binary flags for intervals M_r <= -27, -27 < M_r <= -23, -23 < M_r <= -18, -18 < M_r <= -10, and M_r > -10. Add scale-preserving cross terms M_r * z_eff and M_ri * z_eff to reduce instability where redshift dominates and to keep relative structure visible at high z.
- expected_signal: The derived features resolve ambiguity where apparent magnitudes alone are class-confounding, because they encode whether an observed brightness is physically plausible at the implied distance; stars are expected to occupy low-redshift, moderate-faint absolute regimes while galaxies occupy a mid-luminosity band and quasars align with extreme-luminosity regime flags, improving balanced-class separability.
- risk: The absolute-magnitude transform is an astrophysical approximation that omits extinction and k-corrections, and negative/near-zero redshift handling is heuristic, so very low-z noise can create artifacts despite clipping and flags; the constructs may also be partially redundant with raw redshift and photometric features, which can overfit if model capacity is not regularized.

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