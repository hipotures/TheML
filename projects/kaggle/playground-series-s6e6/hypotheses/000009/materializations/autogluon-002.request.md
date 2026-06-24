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
- title: SDSS Locus Coordinates with Magnitude-Validated Axis Masks
- group_name: canonical_locus_coordinates
- family: locus_projection
- summary: Transform each object into SDSS-inspired principal-color geometry relative to the canonical stellar and red-galaxy manifolds, then summarize proximity-to-locus and outlier behavior as compact, bounded manifold-consistency signals that are directly interpretable for STAR/QSO/GALAXY separation.
- strategy: Compute ug = u - g, gr = g - r, ri = r - i, and iz = i - z. Compute principal coordinates and along-locus coordinates with fixed SDSS coefficients: s = -0.249*u + 0.794*g - 0.555*r + 0.234, w = -0.227*g + 0.792*r - 0.567*i + 0.050, x = 0.707*g - 0.707*r - 0.988, y = -0.270*r + 0.800*i - 0.534*z + 0.054; p1_s = 0.910*u - 0.495*g - 0.415*r - 1.280, p1_w = 0.928*g - 0.556*r - 0.372*i - 0.425, p1_x = r - i, p1_y = 0.895*r - 0.448*i - 0.447*z - 0.600. Standardize each principal coordinate to intrinsic SDSS locus units using ns = clip(s/0.031, -12, 12), nw = clip(w/0.025, -12, 12), nx = clip(x/0.042, -12, 12), ny = clip(y/0.023, -12, 12). Define axis-validity masks with Ivezic-style support windows: ms = (r <= 19.0) & (-0.2 <= p1_s <= 0.8), mw = (r <= 20.0) & (-0.2 <= p1_w <= 0.6), mx = (r <= 19.0) & (0.8 <= p1_x <= 1.6), my = (r <= 19.5) & (0.1 <= p1_y <= 1.2). Let active_count = ms + mw + mx + my. For feature aggregation, use only active axes when active_count > 0; if active_count = 0, keep active_count = 0 and aggregate over all four standardized axes to preserve a deterministic fallback. Compute aggregate lens/manifold features: locus_active_axes = active_count, locus_active_ratio = active_count/4.0, locus_min_abs = min(|nz|) over selected axes, locus_mean_abs = mean(|nz|), locus_max_abs = max(|nz|), locus_std_abs = std(|nz|), locus_signed_sum = sum(nz), locus_within1 = count(|nz| <= 1), locus_within2 = count(|nz| <= 2), locus_between2and4 = count(2 < |nz| <= 4), locus_beyond4 = count(|nz| > 4), where nz is ns/nw/nx/ny selected per active-mask logic and all counts are computed over the effective axis set (fallback set when no active axis). Keep raw values ns,nw,nx,ny and axis masks ms,mw,mx,my as outputs. Also compute red-galaxy-track coordinates with fixed SDSS-like formulas: c_perp = ri - gr/4 - 0.18, c_par = 0.7*gr + 1.2*(ri - 0.18), and expose c_perp, |c_perp|, c_par, and sign(c_perp) as output features.
- expected_signal: Stars are expected to show small standardized distances on the validated axis subset, whereas quasars are expected to show concentrated large perpendicular residuals in one or more principal directions, and galaxies are expected to sit closer to the red-sequence manifold than outlier quasars but less tightly than normal stars; this makes class-relevant separation more explicit for balanced recall under multiclass balancing while preserving direct astrophysical semantics.
- risk: The principal-color constants and validity windows assume SDSS-style calibration and observing-system behavior, so any unmodelled photometric systematics (reddening treatment differences, zero-point drift, survey depth mismatches, or color-dependent noise at fainter magnitudes) can shift residuals and weaken transferability; because these are derived from linear combinations of the original magnitudes they remain correlated with base colors and may not generalize uniformly to atypical spectral-energy distributions (e.g., unusual blue galaxies or dust-reddened AGN), so aggressive model reliance on them can overfit to this locus prior.

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