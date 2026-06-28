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
- title: Fold-safe quasar stellar-locus conflict geometry
- group_name: quasar_stellar_locus_conflict
- family: locus_conflict_geometry
- summary: Measure how strongly each object departs from the stellar color locus while adding redshift-aware quasar corridor support and dampening support in compact stellar-mimic regions where quasar and star colors are intrinsically ambiguous.
- strategy: Compute colors c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z and use only finite rows after deterministic clipping of magnitudes/colors to train-derived 0.1%-99.9% bounds. Fit all statistics inside each training fold, and for final test inference refit on the full labeled train set; never use test rows to estimate geometry. Estimate the stellar locus from train rows with class==STAR only, using v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). For each cube, use component medians as mu and a 5% winsorized covariance; take the first principal direction p, compute orthogonal residual r=(v-mu)-p*(p^T(v-mu)), and regularize residual covariance as C=(0.85*Cov(r)+0.15*diag(diag(Cov(r)))+0.03^2*I+1e-6*I). Compute stellar-locus distances d1=sqrt(r1^T C1^-1 r1) and d2 likewise, clipped to [0,30]. Normalize distances using STAR-fold medians and MADs: z1=(d1-med_star_d1)/(1.4826*MAD_star_d1+1e-6), z2 similarly. Convert outlier evidence with o1=clip((d1-p95_star_d1)/(p99_star_d1-p95_star_d1+1e-6),0,1) and o2 likewise, so the score is calibrated to the upper tail of the stellar locus rather than arbitrary absolute cutoffs. Add redshift-conditioned quasar corridor features: blue_uv=clip(sigmoid((-c_ug-0.05)/0.20)*sigmoid((0.45-c_gr)/0.18)*sigmoid((20.5-i)/0.9),0,1); mid_bridge=clip(exp(-0.5*((c_ug-0.90)^2/0.22^2+(c_gr-0.18)^2/0.14^2))*sigmoid((redshift-0.4)/0.35)*sigmoid((1.9-redshift)/0.7),0,1); high_z=clip(exp(-0.5*((c_ug-1.60)^2/0.35^2+(c_ri-0.25)^2/0.25^2))*sigmoid((redshift-2.1)/0.4)*sigmoid((4.2-redshift)/1.1)*sigmoid((21.0-i)/1.0),0,1). Add compact stellar-mimic suppressors centered on A/F-like and white-dwarf-like strips using q_af=exp(-0.5*((c_ug-0.55)^2/0.14^2+(c_gr-0.10)^2/0.09^2+(c_ri-0.05)^2/0.09^2)) and q_wd=exp(-0.5*((c_ug-1.05)^2/0.18^2+(c_gr-0.30)^2/0.12^2+(c_ri-0.12)^2/0.11^2)); set mimic_penalty=clip((q_af+q_wd)*sigmoid((0.35-abs(redshift-2.7))/0.35)*sigmoid((20.2-i)/0.7),0,1). Emit bounded features d1,d2,z1,z2,o1,o2,blue_uv,mid_bridge,high_z,mimic_penalty plus combined geom_signal=clip(0.42*o1+0.38*o2+0.10*blue_uv+0.10*high_z,0,1) and quasar_conflict=clip(geom_signal+0.20*mid_bridge-0.55*mimic_penalty,-1,1). Replace NaN or inf with 0.0 and clamp every generated bounded score to its stated range.
- expected_signal: Using STAR-only fold-safe locus fitting makes the orthogonal distance genuinely measure departure from the stellar manifold, while the redshift corridor and suppressor terms expose quasar-support versus stellar-mimic conflict that a tree or boosting model can use to improve QSO recall without paying as many STAR false positives, directly supporting balanced accuracy.
- risk: Because the corridor centers and contaminant strips are heuristic, they may be miscalibrated for this synthetic or transformed dataset; STAR-only fitting can also become unstable in small validation folds if the STAR class is sparse, and these features may overlap with other color-locus or redshift interaction groups, limiting incremental gain.

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