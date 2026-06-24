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
- title: Sign-stable redshift-local color-tube residual encoding
- group_name: redshift_adaptive_color_tube_residuals
- family: color_manifold_geometry
- summary: This feature group models the stellar color locus as redshift-local thin tubes in adjacent color cubes and converts each object into standardized along-locus and off-locus coordinates, with robust scaling, deterministic tube orientation, and explicit emphasis in the known quasar-star overlap regime.
- strategy: Compute c1=u-g, c2=g-r, c3=r-i, c4=i-z and z=max(redshift,0). On the training data, create initial equal-frequency redshift bins from quantiles (e.g., 30 bins) and then merge adjacent bins until every retained bin has at least 5000 rows, storing final sorted bin edges and per-bin medians of z; use these same edges for both train and test and clamp out-of-range test z to the nearest edge bin so mapping is deterministic. For each final bin and each 3D color cube Ugri=[c1,c2,c3] and Griz=[c2,c3,c4], compute robust center and scale per feature as median and MAD, then standardize x'=(x-mu)/s with s_j=1.4826*MAD(x_j). Fit PCA on standardized colors per bin and cube, obtaining v1,v2,v3. Resolve eigenvector orientation deterministically: enforce sign(v1·[1,1,1])>0 in every bin, then for b>1 set vk_b=-vk_b if dot(vk_b,vk_{b-1})<0 for k in {1,2,3} to preserve continuity across adjacent bins. For each row in bin b, compute t=dot(v1,x'), q2=dot(v2,x'), q3=dot(v3,x'), d=sqrt(q2^2+q3^2) separately for Ugri and Griz. Standardize these by corresponding bin-wise scales s_t, s_q2, s_q3, s_d (same 1e-6 floor) and output t_hat=t/(s_t+eps), q2_hat=q2/(s_q2+eps), q3_hat=q3/(s_q3+eps), d_hat=d/(s_d+eps). Clip each standardized value to bin-local abs 99.5th-percentile thresholds from training and also keep unclipped variants for capacity tests. Add signed side flags sign2=sign(q2), sign3=sign(q3), and ratio features t_hat/(|q2_hat|+1e-4) and t_hat/(|q3_hat|+1e-4) for both cubes. Add overlap-aware emphasis g(z)=exp(-0.5*((z-2.7)/0.45)^2) and emit g*d_hat and g*|q2_hat| for Ugri and Griz to upweight the documented degeneracy window. If a bin has unstable scales (MAD<=0) or was merged from too few points, fallback to the nearest valid bin by bin-centre redshift and copy its PCA and scaling parameters plus a fallback flag.
- expected_signal: Because SDSS-based quasar selection is explicitly treated as identifying objects away from the stellar locus and notes the stellar/quasar locus crossing near z≈2.7, these engineered tube coordinates provide the model with geometry-aware, redshift-conditioned separation cues that are not directly available in raw magnitudes and should improve class discrimination where colors alone are most ambiguous.
- risk: Residual tube features can still be noisy in very sparse high-redshift tails even after merging, and sign/ratio features may be unstable for near-locus points; furthermore, the large set of derived residual summaries is strongly correlated with base colors and can increase overfitting risk if downstream regularization is weak.

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