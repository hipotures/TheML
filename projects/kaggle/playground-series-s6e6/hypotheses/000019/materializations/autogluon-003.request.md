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
- title: Observed SED Continuum Moments with Fold-Safe Normalization
- group_name: observed_sed_continuum_moments
- family: spectral_shape
- summary: Represent the five observed ugriz magnitudes as a brightness-normalized broadband continuum shape that captures smooth spectral tilt, broad curvature, and localized band departures from a low-order spectral trend.
- strategy: For each training fold, learn only from that fold the per-band median and 0.1%/99.9% clipping limits for u,g,r,i,z, and reuse those fitted values for validation or test rows; when fitting on all training data for final prediction, learn the same limits from the full train set. Replace non-finite magnitudes with the learned band median, then clamp to the learned limits, with fallback limits [-1,35] if a learned value is unavailable. Convert each cleaned magnitude to log relative flux f_b=-0.4*m_b, subtract the row mean across the five bands to remove absolute brightness, and use fixed SDSS effective wavelengths λ=[3543,4770,6231,7625,9134] Angstrom with t_b=log10(λ_b)-mean(log10(λ)). Fit an equal-weight quadratic continuum f'_b=β0+β1*t_b+β2*q_b+e_b, where q_b=t_b^2-mean(t^2), using a precomputed pseudoinverse for the fixed five-band design. Emit β1 as continuum tilt, β2 as broad curvature, residual RMS sqrt(mean(e_b^2)), mean absolute residual, max absolute residual, signed residuals e_u,e_g,e_r,e_i,e_z, blue slope (f'_g-f'_u)/(t_g-t_u), red slope (f'_z-f'_r)/(t_z-t_r), and red-minus-blue slope contrast. If any emitted value is non-finite after preprocessing and fitting, replace only that value with zero.
- expected_signal: Balanced accuracy may improve because class separation is strongly encoded in broadband SED morphology: stars tend to follow smooth temperature-driven color continua, galaxies often introduce broader curvature from population breaks, and QSOs can have power-law-like continua with distinctive band residuals, so normalized shape moments can expose class structure without relying on absolute flux scale.
- risk: These features are correlated with ordinary colors and raw magnitudes, so gains may be small for strong tree models; residuals may also capture survey calibration artifacts or extinction patterns rather than astrophysical class signal, and fold-unsafe preprocessing would leak validation distribution information if clipping statistics are not fit strictly inside each training split.

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