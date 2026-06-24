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
- title: Asinh regime topology with censored-band geometry
- group_name: asinh_censoring_regime_geometry
- family: detection_regime_geometry
- summary: Represent each source by a wavelength-ordered topology of SDSS asinh-photometric detection regimes that explicitly separates high-SNR bands, low-SNR asinh bands, and negative/very-faint bands, and summarizes how the pattern of censored measurements reshapes SED shape descriptors used for class separation.
- strategy: For each band b in {u,g,r,i,z}, invert the SDSS asinh system using m_b = -(2.5/ln10)*[asinh((f_b/f0)/(2*b_b)) + ln(b_b)], therefore x_b = f_b/f0 = 2*b_b*sinh(-(ln(10)/2.5)*m_b - ln(b_b)). Use b_b = {u:1.4e-10, g:0.9e-10, r:1.2e-10, i:1.8e-10, z:7.4e-10}. Winsorize x_b to training-only quantiles [q0.1,q99.9] per band before any slope/diversity calculations. Define fixed SDSS depth anchors m10_b={22.12,22.60,22.29,21.85,20.32} and m0_b={24.63,25.11,24.80,24.36,22.83}. Then create three deterministic regime states per band: R_b=2 if m_b<=m10_b (high-SNR/Pogson-like), R_b=1 if m10_b<m_b<=m0_b (low-SNR asinh regime), and R_b=0 if m_b>m0_b (negative/near-or-below noise regime). Add a smooth boundary confidence c_b = sigmoid((m10_b - m_b)/0.10) and a censor-depth ratio d_b = clip((m_b - m10_b)/(m0_b - m10_b), 0, 1) so threshold uncertainty is represented continuously. Build a single topology block in band order u->z from R_b: counts of {R=2, R=1, R=0}, number of late-dropouts conditioned on earlier detections (e.g., u-drop and g-drop cases, and u+g dropout while any of r/i/z are detected), longest contiguous run of R<=1 and longest contiguous run of R==0, first detected band index, last detected band index, and counts of non-detected redder-than-blue bands. For shape features, compute color geometry only on bands with R>0 using x_b: adjacent log-slope s_k=log10((x_{k+1}+ε)/(x_k+ε)), adjacent curvature κ_k=s_{k+1}-s_k, and a mass-centered summary of valid flux allocation where p_b = |x_b| / Σ_{R>0}|x_b| if Σ|x_b|>0 else 0, plus normalized entropy H= -Σ p_b log(p_b) and centroid μ=Σ p_b*position_b; set shape terms to 0 when fewer than two valid adjacent bands exist. Handle edge cases with ε=1e-12 in logs and always use train-only statistics for any winsorization so there is no target leakage.
- expected_signal: This makes low-SNR and dropout behavior a first-order object descriptor rather than an implicit side-effect of raw magnitudes, so the model can better capture blue/UV suppression and wavelength-localized censoring patterns consistent with quasar/galaxy/stellar boundaries at the faint end where class confusion is highest under balanced accuracy.
- risk: Hard-coded SDSS asinh softening and depth anchors can be miscalibrated if preprocessing conventions differ from the source convention, and regime boundaries are inherently noisy near m10 and m0; adding hard one-hot states can duplicate information from existing asinh/flux features and amplify instability unless aggressively regularized and kept to a compact topology block.

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