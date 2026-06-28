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
- title: Cosmology-scaled luminosity regime plausibility
- group_name: redshift_luminosity_plausibility
- family: absolute_photometry
- summary: Convert apparent ugriz photometry into redshift-scaled intrinsic-brightness plausibility descriptors that expose whether each object’s observed flux is consistent with a stellar, galactic, or quasar-like luminosity regime.
- strategy: For each row, use the reported redshift only as an input to a fixed deterministic distance transform: set z_eff=max(redshift, 1e-4), retain redshift quality flags z_nonpos=(redshift<=0), z_very_low=(redshift<0.01), z_low=(redshift<0.05), and z_high=(redshift>=3), and compute luminosity distance under one flat cosmology with H0=70 km/s/Mpc, Omega_m=0.3, Omega_lambda=0.7, and c=299792.458 km/s. Approximate D_C=(c/H0)*integral_0^z_eff 1/sqrt(Omega_m*(1+z)^3+Omega_lambda) dz using the same fixed quadrature or interpolation scheme for train and test, then D_L_Mpc=(1+z_eff)*D_C and distance_modulus=5*log10(max(D_L_Mpc, 1e-6))+25. For each band b in {u,g,r,i,z}, compute pseudo absolute magnitude M_b=m_b-distance_modulus, then clip M_b to a broad finite range such as [-35, 15]. Summarize absolute luminosity level with M_r, M_i, mean(M_b), median(M_b), min(M_b), max(M_b), and a robust central value 0.5*(M_r+M_i). Summarize cross-band plausibility with sd(M_b), max(M_b)-min(M_b), iqr(M_b), and deviations M_b-mean(M_b) for each band so the model can distinguish globally bright objects from color-driven band outliers. Add regime-margin features for M_r and the robust r/i central magnitude against fixed thresholds -10, -18, -23, and -27, encoded as signed distances to each threshold. Add ordinal regime indicators for M_r and the r/i central magnitude: stellar-like faint (>-10), compact/intermediate (-18 to -10), normal-galaxy-like (-23 to -18), bright-galaxy or AGN-like (-27 to -23), and extreme-quasar-like (<=-27). Add counts over all five bands falling into the same threshold regimes, plus clipped interaction terms distance_modulus*redshift, M_r*z_eff, and central_M_ri*z_eff to let tree or linear models learn when high apparent brightness at high redshift requires unusually large intrinsic luminosity. All thresholds, clipping bounds, cosmology constants, and numerical-distance settings are fixed before fitting and applied identically to train and test.
- expected_signal: Balanced accuracy may improve because the features turn apparent brightness into a class-relevant physical plausibility scale: stars should mostly align with very low-redshift and faint intrinsic regimes, galaxies with intermediate absolute magnitudes, and QSOs with high-redshift extreme-luminosity regimes, reducing confusion among objects with similar observed colors or magnitudes.
- risk: The transform is approximate because it ignores extinction, k-corrections, passband evolution, and redshift measurement uncertainty; near-zero or negative redshifts require heuristic clipping that can create artificial boundaries. The features are also strongly derived from redshift and magnitudes already present, so high-capacity models may overfit redundant thresholds unless validation is stratified and regularization is adequate.

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