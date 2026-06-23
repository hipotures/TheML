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

# External Data Description for /home/xai/DEV/aideml/workspaces/2-liberal-celadon-panther/input/star_classification.csv

Original SDSS17 Stellar Classification Dataset.

This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.

Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.

Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.

Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.

Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

# Hypothesis
- title: Redshift-anchored 4000A break curvature
- group_name: redshifted_4000a_break_curvature
- family: restframe_break_geometry
- summary: Project the known 4000 Angstrom continuum break into the observed ugriz frame using redshift and encode how strongly each object bends at that physically expected transition, which is especially diagnostic of galaxy stellar populations versus quasar and star-like continua.
- strategy: Define SDSS effective band centers as lambda_u=3562, lambda_g=4686, lambda_r=6165, lambda_i=7481, lambda_z=8931. Compute lambda_break = 4000*(1+redshift). Also compute color triplets c1=u-g, c2=g-r, c3=r-i, c4=i-z. Assign each row to one active break regime by lambda_break:
- G regime: 3562 <= lambda_break < 4686 (redshift < 0.17) or equivalently u-g region for near-zero z,
- GR regime: 4686 <= lambda_break < 6165 (approx 0.17 <= z < 0.54),
- RI regime: 6165 <= lambda_break < 7481 (approx 0.54 <= z < 0.87),
- IZ regime: 7481 <= lambda_break < 8931 (approx 0.87 <= z < 1.23).
Use these as one-hot flags plus out_of_band = 1 when z < 0 or z >= 1.23, with break outputs set to 0 there. For each active regime, compute position p = clip((lambda_break-lambda_left)/(lambda_right-lambda_left),0,1) and a boundary confidence weight w = 4*p*(1-p) so estimates are suppressed near filter edges.
- G regime: break_excess = (u-g) - 0.5*((g-r)+(u-g? if unavailable due edge, use 0)) equivalent to local jump relative to adjacent slope; also compute asym = ((u-g)-(g-r)) - ((g-r)-(r-i)).
- GR regime: break_excess = (g-r) - 0.5*((u-g)+(r-i)); asym = ((g-r)-(u-g)) - ((r-i)-(g-r)).
- RI regime: break_excess = (r-i) - 0.5*((g-r)+(i-z)); asym = ((r-i)-(g-r)) - ((i-z)-(r-i)).
- IZ regime: break_excess = (i-z) - (r-i); asym = (i-z)-(r-i).
Emit weighted outputs w*break_excess and w*asym, plus raw regime position p and selected regime flags so a model can learn smooth, physically localized transitions.
- expected_signal: Because the 4000A break shifts from g to r to i across 0.17<=z<1.23, this captures a redshift-dependent shape cue that is tightly linked to stellar-population-dominated galaxy continua but much less consistent for quasars and main-sequence stars, reducing confusion in mixed-color redshift regions.
- risk: Signal weakens when redshift noise, dust effects, or emission features distort broad-band colors, and for high-z objects where the break is beyond z-band the feature provides little information; there is moderate redundancy with other spectral-break shape groups and potential instability on very noisy faint photometry.

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