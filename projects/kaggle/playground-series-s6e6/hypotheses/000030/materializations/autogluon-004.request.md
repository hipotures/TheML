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
- title: Fold-safe SDSS quasar targeting surface
- group_name: sdss_quasar_targeting_surface
- family: astrophysical_selection_geometry
- summary: Encode SDSS-inspired quasar target selection as smooth, fold-fitted distances from robust stellar-locus tubes in ugriz color space plus soft inclusion and rejection surfaces for the redshift and magnitude regimes where quasars overlap stellar contaminants.
- strategy: Use only predictors and fit all learned statistics on the training fold when cross-validating, then on full train for test. Domain constants follow the SDSS legacy quasar target-selection description at https://www.sdss4.org/dr15/algorithms/legacy_target_selection/. Compute colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. For geometry only, winsorize each color to train-fold 0.5 and 99.5 percentiles, storing the bounds and applying the same bounds to validation/test. Build cube A=(c1,c2,c3), indexed by c3, and cube B=(c2,c3,c4), indexed by c2. Use bin centers every 0.08 mag over the clipped train-fold range, but evaluate each object with a triangular kernel K=max(0,1-abs(v-center)/0.12) over nearby centers so distances change continuously across bin edges. For each cube/bin, compute weighted median center, per-axis MAD scale with floor 0.015, then form a high-density core by retaining rows with standardized radius below the weighted 80th percentile or 3.0, whichever keeps more rows. Recompute weighted center, scale, and covariance on that core, take the first principal component as the local tube direction, and orient it consistently with the global cube direction. If effective bin size is below 80 or covariance is ill-conditioned, shrink parameters toward global cube parameters with weight w=clamp((n_eff-30)/100,0,1). Also compute stratum-specific versions by spectral_type and galaxy_population when n_eff>=120, otherwise use the same shrinkage to the global tube. For each object and cube, compute standardized residual z=(x-mu)/s, projection p=z dot t, orthogonal residue q=z-p*t, distance d=norm(q), clipped to [0,25], then kernel-average distances across active bins. Convert distances to soft outlier scores O_A=sigmoid((d_A-4.0)/0.45), O_B=sigmoid((d_B-4.0)/0.45), and keep global, stratum-specific, max, and difference versions. Define soft box gate box(v,L,U,tau)=sigmoid((v-L)/tau)*sigmoid((U-v)/tau), low_mag=sigmoid((i-15.0)/0.25)*sigmoid((19.1-i)/0.60), high_mag=sigmoid((i-15.0)/0.25)*sigmoid((20.2-i)/0.60), low_z=sigmoid((2.25-redshift)/0.35), mid_z_redshift=sigmoid((redshift-2.45)/0.30)*sigmoid((3.05-redshift)/0.30), and high_z_redshift=sigmoid((redshift-3.0)/0.35). Inclusion surfaces are UVX=low_mag*low_z*sigmoid((0.60-c1)/0.05), MIDZ=low_mag*mid_z_redshift*box(c1,0.65,1.50,0.07)*box(c2,0.00,0.20,0.06), and HIZ=high_mag*high_z_redshift*sigmoid((c1-1.50)/0.08). Exclusion surfaces are WD=box(c2,-0.8,-0.2,0.10)*box(c3,-0.6,-0.2,0.10)*box(c4,-1.0,0.0,0.10), MWD=box(c2,0.0,1.6,0.10)*box(c3,0.6,2.0,0.10), A_REJ=box(c1,0.9,1.5,0.06)*box(c2,-0.35,0.0,0.06), BLUE_REJ=sigmoid((0.90-c1)/0.08)*sigmoid((0.80-c2)/0.08)*sigmoid((i-19.0)/1.0), and BRIGHT_REJ=sigmoid((15.0-i)/0.25). Output the raw outlier, inclusion, rejection, and magnitude-gate terms plus composite soft-union scores: LOW_QSO=low_mag*(1-(1-O_A)*(1-UVX)*(1-MIDZ))*(1-WD)*(1-MWD)*(1-A_REJ)*(1-BRIGHT_REJ), HIGH_QSO=high_mag*(1-(1-O_B)*(1-HIZ))*(1-BLUE_REJ)*(1-WD)*(1-MWD)*(1-A_REJ)*(1-BRIGHT_REJ), and ANY_QSO=1-(1-LOW_QSO)*(1-HIGH_QSO).
- expected_signal: This preserves the original SDSS selection geometry while making it fold-safe, smoother across bin boundaries, and less contaminated by non-locus outliers; it should give the model calibrated evidence for QSO versus STAR in the stellar-locus crossing regions and high-redshift color regimes, which is valuable for balanced accuracy because QSO recall errors are otherwise costly.
- risk: The SDSS constants assume PSF, extinction-corrected legacy photometry and may not align exactly with this dataset's synthetic or calibrated magnitudes; redshift, spectral_type, and galaxy_population conditioning may be redundant with strong raw predictors, sparse stratum tubes can still be unstable in tails, and the many correlated smooth surfaces can overfit unless validation is fold-fitted and the downstream model is regularized.

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