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
- title: Bandpass break localization with confidence-aware soft assignment
- group_name: bandpass_break_localization
- family: spectral_break
- summary: Represent each object by the location of its strongest broadband discontinuity across ugriz and how strong, asymmetric, and reliable that discontinuity is, so class-specific spectral-shape signatures become easier to separate.
- strategy: Use only bands u,g,r,i,z. Compute l_u,l_g,l_r,l_i,l_z as log-flux proxies with l_b=-0.4*m_b. Let d_ug=l_u-l_g, d_gr=l_g-l_r, d_ri=l_r-l_i, d_iz=l_i-l_z, and a_uj=|d_j| for j in {ug,gr,ri,iz}. Let A=a_ug+a_gr+a_ri+a_iz and eps=1e-8. If A<1e-6 (near-flat spectrum), emit break_present=0, break_position=-1, and set all break-derived numeric outputs to 0 with a dedicated flag. Otherwise set break_present=1 and choose dominant break index k=argmax_j a_j with tie broken by smallest j (bluest) for deterministic behavior. For all j define soft posterior p_j=a_j/(A+eps), entropy H=−∑ p_j log(max(p_j,eps)), and expected position k_soft=∑ p_j·idx(j) where idx maps ug/gr/ri/iz to 1..4. For chosen k, compute d_k and s_k=sign(d_k); blue_mean=mean(a_ug..a_{prev(k)-1}) (0 if k=1), red_mean=mean(a_{k+1}..a_iz) (0 if k=4), and continuity_break= d_k - (blue_mean+red_mean)/2. Define asymmetry=blue_mean-red_mean, sharpness=|d_k|/(A+eps), and turnover_count = I(k>1 and sign(d_{k-1})!=s_k)+I(k<4 and sign(d_{k+1})!=s_k). Optionally add confidence bins by applying equal-frequency cutpoints on sharpness and H computed on training data (e.g., quartiles) and record low/med/high bins to avoid brittle behavior in near-tie cases.
- expected_signal: The approach gives the model a compact description of where and how abrupt the largest spectral break is, which is physically aligned with galaxy continuum breaks, redshifted quasar features moving across ugriz, and the smoother monotonic trends of stars, so it should improve recall-balanced discrimination where global colors alone are ambiguous.
- risk: Because magnitudes can be noisy at faint flux levels, argmax selection may flip between adjacent bands and inflate discrete break-location features; entropy and soft-weights reduce but do not remove this, and these descriptors are partially redundant with adjacent-color slopes, so careless use can still overfit survey-specific photometric quirks.

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