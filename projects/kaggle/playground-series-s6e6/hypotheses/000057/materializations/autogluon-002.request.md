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
- title: Redshift-aware quasar outlier geometry with calibrated contamination suppression
- group_name: quasar_stellar_locus_conflict
- family: locus_conflict_geometry
- summary: Create a compact quasar-support signal from orthogonal color-manifold outlier distance and redshift-conditioned color corridors, then explicitly dampen that signal in narrow stellar-mimic strips to improve separation in known confusion regions.
- strategy: Fit all geometry statistics on train only. Compute colors c_ug=u-g, c_gr=g-r, c_ri=r-i, c_iz=i-z. Form two vectors v1=(c_ug,c_gr,c_ri) and v2=(c_gr,c_ri,c_iz). For each vector set, estimate mu as median per component and covariance S from 5% winsorized train values; extract first principal axis p1,p2. For each row, residuals are r1=v1-mu1 - p1*(p1^T(v1-mu1)) and r2=v2-mu2 - p2*(p2^T(v2-mu2)). Build orthogonal covariance Co1,Co2 from train residuals with ridge and shrinkage: C* = (1-0.1)Cov(r)+0.1*diag(diag(Cov(r))) + 0.03^2*I, then add eps=1e-6*I before inversion. Compute Mahalanobis orthogonal distances d1=sqrt(r1^T C*1^{-1} r1), d2 likewise, then clip to [0,30]. Normalize with train medians and MAD: z1=(d1-p50_1)/(1.4826*MAD_1+1e-6), z2=(d2-p50_2)/(1.4826*MAD_2+1e-6). Define outlier soft features o1=clip((d1-t1)/4,0,1), o2=clip((d2-t2)/4,0,1), with t1,t2=p95 of d1,d2 from train. Define corridor gates with explicit bins: blue low-z gate b = sigmoid((−c_ug−0.05)/0.20)*sigmoid((0.45-c_gr)/0.18)*sigmoid((20.5-i)/0.9); mid-z bridge m = exp(−0.5*((c_ug-0.85)^2/0.18^2 + (c_gr-0.20)^2/0.12^2))*sigmoid((1.8 - redshift)/0.8)*sigmoid((redshift-0.4)/0.35); high-z gate h = exp(−0.5*((c_ug-1.55)^2/0.30^2 + (c_ri-0.25)^2/0.22^2))*sigmoid((redshift-2.2)/0.4)*sigmoid((3.9-redshift)/1.0)*sigmoid((21.0-i)/1.0). Build white-dwarf/A-star suppressors: q1=exp(−0.5*((c_ug-0.55)^2/0.12^2 + (c_gr-0.10)^2/0.08^2 + (c_ri-0.05)^2/0.08^2)), q2=exp(−0.5*((c_ug-1.05)^2/0.16^2 + (c_gr-0.30)^2/0.10^2 + (c_ri-0.12)^2/0.10^2)); penalty q = (q1+q2)*sigmoid((0.2-abs(redshift-2.7))/0.4)*sigmoid((20.0-i)/0.6). Keep generated features bounded: geom_signal=clip(0.45*o1+0.35*o2+0.1*b+0.1*h,0,1), mid_bridge=clip(m,0,1), conflict=clip(geom_signal-0.6*q+0.2*m,-1,1). Replace any NaN/inf with 0 and keep all intermediate calculations deterministic.
- expected_signal: This revision preserves the locus-conflict core idea while adding train-safe fitting, stable regularization, and explicit redshift-conditioned quasar corridors, which should better separate true quasar outliers from the stellar manifold and reduce false positives from narrow contaminant strips, improving balanced accuracy through stronger minority-class discrimination.
- risk: Thresholds and strip centers are still heuristic and can be miscalibrated if this dataset’s photometric zero-points differ from SDSS-style behavior, and the added geometry + corridor signals can be redundant with other locus-based groups, so gains may be modest without careful orthogonal integration.

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