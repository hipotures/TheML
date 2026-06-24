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
- title: Tag-Color Concordance with Redshift-Calibrated Residuals
- group_name: catalog_tag_color_concordance
- family: tag_photometry_consistency
- summary: Generate deterministic color-consistency descriptors that score how well an object’s observed ugriz color profile aligns with the physical expectation encoded by its spectral-type and galaxy-population tags, while explicitly accounting for redshift-driven color shifts so that agreement and contradiction signals remain interpretable across classes.
- strategy: For each object, compute adjacent color components c1=u-g, c2=g-r, c3=r-i, c4=i-z. Convert each to a robustly scaled value in [0,1] using training-set quantiles: rj = clip((cj - q0.02_j) / (q0.98_j - q0.02_j), 0, 1), where q0.02_j and q0.98_j are the 2nd and 98th percentiles of color cj in the training data and are fixed after fitting. Build an observed redness scalar R = 0.40*r1 + 0.30*r2 + 0.20*r3 + 0.10*r4. Map spectral_type to te ∈ {O/B→0.00, A/F→0.33, G/K→0.67, M→1.00} and galaxy_population to ge ∈ {Blue_Cloud→0.00, Red_Sequence→1.00}; any unseen category maps to 0.50. Compute scalar agreement and mismatch terms: At = 1 - |R-te|, Ag = 1 - |R-ge|, M_t = |R-te|, M_g = |R-ge|, and T = te - ge with signed and absolute versions. Also compute per-color residual vectors dj_t = rj - te and dj_g = rj - ge; derive their mean, median, L1 sum, L2 sum (sum of squares), and max absolute values separately for tag and for each tag pair. Define a redshift trust gate w_z = 1 - clip((redshift - 0.45)/1.25, 0, 1); multiply all mismatch and contradiction-derived features by w_z and produce both raw and weighted versions. Create contradiction binaries: htc = 1 if te <= 0.34 and R >= 0.67 else 0; ctc = 1 if te >= 0.67 and R <= 0.33 else 0; rbc = 1 if ge >= 0.67 and R <= 0.33 else 0; brc = 1 if ge <= 0.34 and R >= 0.67 else 0; dual_contra = 1 if M_t >= 0.35 and M_g >= 0.35 else 0. Include clipped color/feature handling for safety: any non-finite R, M_t, M_g, or residual becomes 0.50 for score terms and 0 for indicator terms.
- expected_signal: The hypothesis preserves the same catalog-tag consistency intuition but improves precision by replacing fixed color constants with train-calibrated normalization, adding signed and magnitude mismatch geometry at both scalar and per-color levels, and de-weighting those disagreements when redshift is high where color tracks shift for real astrophysical reasons, which should help separate noisy or contradictory tag/color combinations and improve class separation for minority classes under balanced accuracy.
- risk: Quantile scaling and fixed thresholds can still be brittle when photometric distributions drift from the training sample, and redshift-based down-weighting may mask useful signal for genuinely contradictory high-z objects or amplify noise when redshift is uncertain or extreme.

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