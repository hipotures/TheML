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
- title: Asinh Censoring Topology and Dropout Geometry
- group_name: asinh_censoring_regime_geometry
- family: detection_regime_geometry
- summary: Describe each source by the wavelength-ordered pattern of reliable, low-signal, and effectively censored asinh-photometric bands, then summarize how that censoring pattern changes the observable SED shape.
- strategy: For each band in wavelength order u,g,r,i,z, convert the asinh magnitude m_b to relative linear flux x_b=f_b/f0 using the SDSS softening constants beta={u:1.4e-10,g:0.9e-10,r:1.2e-10,i:1.8e-10,z:7.4e-10}: x_b=2*beta_b*sinh(-(ln(10)/2.5)*m_b-ln(beta_b)). Use q_b=x_b/(2*beta_b) as the dimensionless asinh argument. Derive regime anchors from the same constants rather than hard-coding them: m0_b=-(2.5/ln(10))*ln(beta_b), where x_b=0, and m10_b=-(2.5/ln(10))*(asinh(5)+ln(beta_b)), where x_b=10*beta_b. Assign a three-state regime R_b=2 for q_b>=5 reliable/Pogson-like detection, R_b=1 for 0<=q_b<5 low-signal positive asinh flux, and R_b=0 for q_b<0 negative or below-zero flux. Keep regime assignment un-winsorized, but winsorize x_b to train-only per-band quantiles, for example [0.001,0.999], before continuous shape calculations. Add per-band features R_b one-hot, signed margins m10_b-m_b and m0_b-m_b, boundary softness sigmoid((m10_b-m_b)/0.10), and censor-depth ratio clip((m_b-m10_b)/(m0_b-m10_b),0,1). Build topology features from the five R_b values: counts of each state, base-3 full-pattern code, one-hot flags for common blue-side dropouts such as u low while redder bands are detected and u+g low while r/i/z are detected, analogous red-side censoring flags, first and last reliable band index, leading and trailing censored run lengths, longest contiguous run with R_b<=1, longest contiguous run with R_b==0, number of regime transitions, and pairwise counts of bluer-censored/redder-detected and bluer-detected/redder-censored inversions. For SED shape, use only positive observed bands with R_b>0 and winsorized x_b: compute adjacent log-flux slopes log10((x_{b2}+1e-12)/(x_{b1}+1e-12)) when both adjacent bands are valid, adjacent slope curvatures when both neighboring slopes are valid, and set invalid terms to 0 with matching validity indicators. Also compute positive-flux allocation summaries after zeroing R_b==0 bands: p_b=x_b/sum_valid(x_b), normalized entropy, wavelength-index centroid, and concentration max(p_b). All fitted cutoffs or quantiles must be estimated on the training fold only and reused unchanged for validation or test rows.
- expected_signal: Balanced accuracy should benefit because faint-end class errors often occur where raw ugriz magnitudes blur together reliable colors, low-SNR colors, and non-detections; this feature block makes blue dropouts, red-side censoring, and censored SED shape explicit so the classifier can separate quasars, galaxies, and stars in boundary regions without relying only on raw magnitudes.
- risk: The thresholds assume the provided magnitudes follow the same SDSS asinh convention and softening constants, so any preprocessing mismatch can make the regimes miscalibrated. Hard regime boundaries can be noisy near m10 and m0, full-pattern indicators may duplicate raw flux/color information, and rare topology patterns can overfit unless regularized or validated with fold-local preprocessing.

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