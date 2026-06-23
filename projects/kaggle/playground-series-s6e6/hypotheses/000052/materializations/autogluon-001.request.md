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
- title: SEGUE-inspired stellar subtype plane margins
- group_name: segue_stellar_subtype_margins
- family: stellar_color_geometry
- summary: Translate legacy SDSS-SEGUE color-selection boundaries for specific stellar subtypes into signed distance features so each object carries how strongly it conforms to or departs from known stellar loci in calibrated color index space as a direct complement to generic broad-band shape features.
- strategy: Compute base colors ug=u-g, gr=g-r, ri=r-i, iz=i-z, gi=g-i and linear principal colors l_color=-0.436*u+1.129*g-0.119*r-0.574*i+0.1984, s_color=-0.249*u+0.794*g-0.555*r+0.234, p1=0.91*(u-g)+0.415*(g-r)-1.280, and v=0.283*(u-g)-0.354*(g-r)+0.455*(r-i)+0.766*(i-z). Define a helper margin function for an interval [a,b]: I(x)=0 if a<=x<=b else min(|x-a|,|x-b|), and for one-sided cut x<=k/x>=k use the positive excess beyond the bound. For each SEGUE-style prototype define a normalized score as the weighted sum of clause margins across its color inequalities (weights equal or scaled by clause width): white_dwarf: gr in [-1,-0.2], ug in [-1,0.7], ug+2*gr<-0.1; cool_white_dwarf: 14.5<r<20.5 and gi in [-2,1.7] with branch adjustment; A/BHB: 0.8<ug<1.5 and -0.5<gr<0.2 plus v around 0 (e.g., |v-0|/1.0); K_giants: 0.35<gr<0.8, 0.15<ri<0.6, l_color>0.07; low_metallicity: -0.5<gr<0.75, 0.6<ug<3.0, l_color>0.135; M_subdwarf: ri<0.9, ri<0.787*gr-0.356, 1.8<gi<2.4; main_sequence_white_dwarf_pairs: ug<2.25, -0.2<gr<1.2, 0.5<ri<2.0, gr>-19.78*ri+11.13, gr<0.95*ri+0.5, plus iz side conditions by ri branch. Also produce aggregate features: best_star_margin=min(all prototype margins), second_best_margin, margin_gap=second_best-best, and count_inside_prototypes (non-negative margins). Normalize each clause by a robust MAD/scale from training and clip margins to a finite range (for stability) before aggregation.
- expected_signal: This adds explicit, survey-proven stellar-template geometry that is complementary to smooth manifold or flux-shape features and should raise separation in regions where stars mimic galaxies or quasars, because many non-stellar classes remain farther from these subtype envelopes while true stars frequently lie close to one of them.
- risk: These are legacy targeting cuts, not universal physical laws; they can be shifted by calibration, extinction treatment, or sample-selection bias and may add redundancy with existing color-geometry families, causing misclassification of exotic quasars/galaxies with stellar-like colors or reduced performance outside SDSS-like photometric conditions.

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