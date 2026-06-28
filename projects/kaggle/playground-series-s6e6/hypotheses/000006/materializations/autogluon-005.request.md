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
- title: Confidence-weighted bandpass break localization
- group_name: bandpass_break_localization
- family: spectral_break
- summary: Encode the observed ugriz spectrum by the position, direction, concentration, and local context of its strongest adjacent-band discontinuity, distinguishing decisive spectral breaks from diffuse or noisy broadband curvature.
- strategy: Use only the ordered magnitude bands u,g,r,i,z and convert each magnitude to a relative log-flux proxy l_b=-0.4*m_b. Compute adjacent signed slopes d_ug=l_u-l_g, d_gr=l_g-l_r, d_ri=l_r-l_i, and d_iz=l_i-l_z, with absolute amplitudes a_j=abs(d_j). Let A=sum_j a_j and eps=1e-8. If A<1e-6, mark break_present=0, set dominant position to a sentinel category, and set all numeric break descriptors to 0. Otherwise mark break_present=1, choose the dominant break k as the adjacent pair with maximum a_j using bluest-position tie breaking, and emit: signed amplitude d_k, absolute amplitude a_k, one-hot dominant position over ug/gr/ri/iz, direction sign sign(d_k), sharpness a_k/(A+eps), soft position sum_j p_j*idx(j) where p_j=a_j/(A+eps) and idx is 1..4, normalized entropy -sum_j p_j*log(max(p_j,eps))/log(4), signed blue-side mean and red-side mean of d values excluding k with 0 for empty sides, absolute blue-side mean and red-side mean of a values excluding k with 0 for empty sides, signed local contrast d_k minus the average signed side mean, absolute local contrast a_k minus the average absolute side mean, asymmetry as absolute blue-side mean minus absolute red-side mean, and neighbor sign-change count across the dominant break. Fit any optional sharpness or entropy quantile bins on the training fold only, then apply the same cutpoints to validation and test rows to avoid leakage.
- expected_signal: The strongest break location and its confidence should help separate galaxies with broad continuum breaks, QSOs whose redshifted features can create band-dependent jumps, and stars with smoother stellar continua, improving balanced accuracy especially for minority-class boundaries that are not fully resolved by raw colors.
- risk: The descriptors overlap with adjacent color features and the argmax position can be unstable for faint or near-flat spectra, so discrete location and bin features may overfit photometric noise or survey-specific calibration artifacts unless validated with fold-wise preprocessing.

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