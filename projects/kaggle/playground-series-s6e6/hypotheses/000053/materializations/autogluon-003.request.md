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
- title: Soft-Assigned Dual Rest-Frame Break Contrast
- group_name: dual_restframe_break_alignment
- family: break_competition_geometry
- summary: Encode whether each object’s redshift places a strong continuum discontinuity at the observed ugriz location expected for either the Lyman 1216 Å break or the 4000 Å break, then compare the two signals to distinguish quasar-like dropout structure, galaxy-like Balmer/4000 Å structure, and smoother stellar spectra.
- strategy: Use ugriz magnitudes m=[u,g,r,i,z], SDSS effective wavelengths λ=[3551,4670,6170,7480,8930] Å, log wavelengths x=ln(λ), and clipped redshift zc=clip(redshift,0,7). For each break λb in {1216,4000}, compute observed position xb=ln(λb*(1+zc)). If xb is outside [x0,x4], set miss_b=1 and all jump/residual features for that break to 0. Otherwise compute the four adjacent observed color jumps J_k=m_k-m_{k+1} for k=0..3 and assign soft interval weights w_k from triangular distance to interval midpoints c_k=(x_k+x_{k+1})/2: w_k=max(0,1-|xb-c_k|/h_k), where h_k is half the interval width plus half the nearest neighboring interval width; normalize weights to sum to 1. This preserves a primary aligned interval while reducing hard flips near filter boundaries. For each interval k with nonzero weight, estimate a smooth-continuum expected jump E_k from a least-squares line in magnitude versus log wavelength using all bands except the two bands straddling the candidate break when at least two outside bands exist; otherwise use the nearest two same-side bands. Set E_k=slope*(x_k-x_{k+1}). Define residual R_k=(J_k-E_k)/(median(|m0-m1|,|m1-m2|,|m2-m3|,|m3-m4|)+0.05), clipping each residual to [-8,8]. For each break output weighted raw jump J_b=sum(w_k*J_k), weighted expected jump E_b=sum(w_k*E_k), weighted residual R_b=sum(w_k*R_k), absolute weighted residual A_b=sum(w_k*abs(R_k)), alignment sharpness C_b=max(w_k), edge flag edge_b=1 if C_b<0.45 else 0, dominant interval index or one-hot interval flags, and regime flags lya_regime=(1.8<=zc<=6.6), balmer_regime=(0<=zc<=1.35). Build competition features only from valid breaks: lya_vs_balmer=R_1216*C_1216-R_4000*C_4000, abs_balance=A_1216*C_1216-A_4000*C_4000, signed_ratio=(R_1216*C_1216)/(abs(R_4000*C_4000)+1), and valid_pair=(1-miss_1216)*(1-miss_4000). Set invalid break terms to 0 while retaining miss and regime flags.
- expected_signal: Balanced accuracy may improve because the features emphasize class-relevant spectral geometry rather than absolute brightness: QSOs should often have stronger redshift-aligned Lyman-break residuals, galaxies should more often have stronger 4000 Å residuals at low redshift, and stars should tend to have weaker or less redshift-consistent break competition.
- risk: The signal depends strongly on the supplied redshift, so redshift errors or class-correlated redshift artifacts could dominate; broad ugriz filters make true break positions coarse, emission lines can mimic jumps, and these features may be partly redundant with raw colors and redshift unless regularized by the downstream model.

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