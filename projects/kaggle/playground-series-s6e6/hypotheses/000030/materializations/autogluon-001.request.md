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
- title: SDSS quasar target-surface geometry
- group_name: sdss_quasar_targeting_surface
- family: astrophysical_selection_geometry
- summary: Reconstruct the legacy SDSS quasar-selection decision geometry in ugriz color space by encoding both locus-outlier behavior in the ugri and griz photometric cubes and the known inclusion/exclusion region logic that controls how quasars were accepted near the stellar-locus crossings, so class boundaries are represented as geometry against documented selection surfaces.
- strategy: Create colors c1=u-g, c2=g-r, c3=r-i, c4=i-z. In cube A=(c1,c2,c3), bin objects by c3 (for example 0.08-mag bins between global 5th and 95th percentiles, with endpoints clamped), and in cube B=(c2,c3,c4), bin by c2 with the same style. For each occupied bin compute robust centroid μ and robust scale s (MAD, clamped minimum 0.01), then compute a local 1D ridge direction t via principal component on standardized colors. For each object define orthogonal residual distances d_A and d_B as Mahalanobis-like distances to the ridge in each cube, d = sqrt(sum((r/s)^2)), where r is the perpendicular residual after removing projection onto t; clip d to [0,20] for stability. Build candidate-surface scores: S_A=clip((d_A-4)/2,0,1) and S_B=clip((d_B-4)/2,0,1), reflecting the historical 4σ locus-outlier logic. Add inclusion-surface terms from documented legacy cuts: MZ=1 if 0.65<=c1<=1.5 and 0<=c2<=0.2 (mid-z bridge support), UVX=1 if c1<=0.6 (optional with i-band fading weight w=clip((20.0-i)/1.0,0,1)), HIZ=1 if redshift>3 and i<20 and c1>1.5, and BLUE_REJECT=1 if c1<0.9 and c2<0.8 and i>19 (negative evidence). Add exclusion-surface penalties from known color-rejection boxes: WD_REJ for approximately -0.8<g-r<-0.2, -0.6<r-i<-0.2, -1.0<i-z<0; MWD_REJ for 0.0<g-r<1.6 and 0.6<r-i<2.0; A_REJ for 0.9<u-g<1.5 and -0.35<g-r<0. These are signed margin features so the group remains continuous near boundaries. Combine scores with redshift gating (e.g., weight MZ for 2.3<=redshift<=3.2 and HIZ for redshift>=3) without using any target labels.
- expected_signal: Quasars are historically selected as broad outliers from the stellar locus with specific redshift-dependent exception surfaces, so these geometry features directly encode decision boundaries and ambiguities (especially the 2.7-2.8 crossing and high-redshift color regions), providing separability signals where raw color features alone are less stable and helping reduce confusion with stars and compact galaxies.
- risk: The constants come from SDSS historical selection logic and may be mildly misaligned with this dataset’s exact photometric calibration and redshift uncertainties, especially for faint objects, and hard cut-derived surfaces can be brittle in out-of-distribution color tails; if the underlying survey systematics differ, these features may overemphasize legacy-selection artifacts and reduce robustness.

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