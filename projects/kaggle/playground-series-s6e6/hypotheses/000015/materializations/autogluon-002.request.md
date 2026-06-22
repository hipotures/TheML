Fix one failed hypothesis materialization.

Return a corrected version of the same feature-group module. Do not invent a
new hypothesis or change the feature family. Keep the fix narrow: address the
observed failure while preserving the intended preprocessing behavior.

You are repairing Python preprocessing code for one stored hypothesis. The
existing file is a feature-group module imported by a fixed runtime wrapper.
Generate only the corrected module.

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

# Task

## Goal
Predict the stellar class for each test-set object.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the true class. Submission file must contain `id,class` with one label per test row, where `class` is one of GALAXY, STAR, or QSO.

## Data description
`train.csv` contains 577347 rows with 10 feature columns plus `id`, `galaxy_population`, `spectral_type`, and the target `class`. `test.csv` contains the same predictors without the target across 247435 rows. `sample_submission.csv` shows the required submission columns `id` and `class`. The target is a 3-class stellar classification problem with labels GALAXY, QSO, and STAR.

# Project Target
- ID column: id
- Target column: class
- Problem type: multiclass
- Evaluation metric: balanced_accuracy

# Failed Materialization

- Mode: autogluon
- Hypothesis ID: 000015
- Source file: autogluon-001.py
- Failed node: 20260622T143300-b5af9439-18
- Run: 20260622T034843-eca160d1

# Execution Error

```text
failed.yaml error:
Process exited with 1

stderr.log:
 by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
	Stage 1 Generators:
		Fitting AsTypeFeatureGenerator...
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Warning: dtype boolean is not recognized as a valid dtype by numpy! AutoGluon may incorrectly handle this feature...
Cannot interpret 'BooleanDtype' as a data type
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T143300-b5af9439-18/02-code.py", line 638, in main
    predictor.fit(**fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/common/utils/decorators.py", line 34, in _call
    return f(*gargs, **gkwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/predictor/predictor.py", line 1412, in fit
    self._fit(ag_fit_kwargs=ag_fit_kwargs, ag_post_fit_kwargs=ag_post_fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/predictor/predictor.py", line 1418, in _fit
    self._learner.fit(**ag_fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/abstract_learner.py", line 159, in fit
    return self._fit(X=X, X_val=X_val, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/default_learner.py", line 95, in _fit
    X, y, X_val, y_val, X_test, y_test, X_unlabeled, holdout_frac, num_bag_folds, groups = self.general_data_processing(
                                                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/default_learner.py", line 284, in general_data_processing
    X_super = self.fit_transform_features(X_super, y_super, problem_type=self.label_cleaner.problem_type_transform, eval_metric=self.eval_metric)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/abstract_learner.py", line 483, in fit_transform_features
    X = feature_generator.fit_transform(X, y, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/pipeline.py", line 70, in fit_transform
    X_out = super().fit_transform(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/abstract.py", line 302, in fit_transform
    X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/pipeline.py", line 77, in _fit_transform
    X_out, type_group_map_special = super()._fit_transform(X=X, y=y, **kwargs)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/bulk.py", line 169, in _fit_transform
    X, self.generators[i], feature_metadata = self._fit_transform_stage(
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/bulk.py", line 203, in _fit_transform_stage
    feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata_in, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/abstract.py", line 302, in fit_transform
    X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/astype.py", line 116, in _fit_transform
    feature_bool_val = get_bool_true_val(uniques=uniques)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/common/features/infer_types.py", line 188, in get_bool_true_val
    if is_nan:
       ^^^^^^
  File "pandas/_libs/missing.pyx", line 392, in pandas._libs.missing.NAType.__bool__
TypeError: boolean value of NA is ambiguous
Traceback (most recent call last):
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T143300-b5af9439-18/02-code.py", line 709, in <module>
    main()
  File "/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/runs/20260622T034843-eca160d1/artifacts/20260622T143300-b5af9439-18/02-code.py", line 638, in main
    predictor.fit(**fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/common/utils/decorators.py", line 34, in _call
    return f(*gargs, **gkwargs)
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/predictor/predictor.py", line 1412, in fit
    self._fit(ag_fit_kwargs=ag_fit_kwargs, ag_post_fit_kwargs=ag_post_fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/predictor/predictor.py", line 1418, in _fit
    self._learner.fit(**ag_fit_kwargs)
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/abstract_learner.py", line 159, in fit
    return self._fit(X=X, X_val=X_val, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/default_learner.py", line 95, in _fit
    X, y, X_val, y_val, X_test, y_test, X_unlabeled, holdout_frac, num_bag_folds, groups = self.general_data_processing(
                                                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/default_learner.py", line 284, in general_data_processing
    X_super = self.fit_transform_features(X_super, y_super, problem_type=self.label_cleaner.problem_type_transform, eval_metric=self.eval_metric)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/tabular/learner/abstract_learner.py", line 483, in fit_transform_features
    X = feature_generator.fit_transform(X, y, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/pipeline.py", line 70, in fit_transform
    X_out = super().fit_transform(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/abstract.py", line 302, in fit_transform
    X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/pipeline.py", line 77, in _fit_transform
    X_out, type_group_map_special = super()._fit_transform(X=X, y=y, **kwargs)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/bulk.py", line 169, in _fit_transform
    X, self.generators[i], feature_metadata = self._fit_transform_stage(
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/bulk.py", line 203, in _fit_transform_stage
    feature_df_list.append(generator.fit_transform(X, feature_metadata_in=feature_metadata_in, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/abstract.py", line 302, in fit_transform
    X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/features/generators/astype.py", line 116, in _fit_transform
    feature_bool_val = get_bool_true_val(uniques=uniques)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xai/DEV/TheML/.venv/lib/python3.12/site-packages/autogluon/common/features/infer_types.py", line 188, in get_bool_true_val
    if is_nan:
       ^^^^^^
  File "pandas/_libs/missing.pyx", line 392, in pandas._libs.missing.NAType.__bool__
TypeError: boolean value of NA is ambiguous

stdout.log:
AutoGluon materialization: starting feature groups
TheML feature group: start name=redshift_template_domain_margins
TheML feature group: ok name=redshift_template_domain_margins elapsed_s=1.006 rows=824782 cols=30
AutoGluon materialization: finished feature groups rows=824782 cols=41 elapsed=1.174s
AutoGluon materialization: ignored_columns=['id']
AutoGluon materialization: holdout validation rows=115470 train_rows=461877
TML_RUNTIME|event=start|stage=autogluon_fit
```

# Previous Code

```python
import numpy as np
import pandas as pd

def add_redshift_template_domain_margins(raw, deps, aux):
    redshift = pd.to_numeric(raw["redshift"], errors="coerce")
    index = redshift.index
    z = redshift.to_numpy(dtype=float)
    finite = np.isfinite(z)

    velocity_width = 1200.0 / 299792.458
    galaxy_low, galaxy_high = -0.01, 1.0
    qso_low, qso_high = 0.0333, 7.0
    regime_boundaries = (0.0, 0.0333, 1.0, 2.2, 3.0, 3.5, 4.5, 5.0)
    regime_names = ("0_0", "0_0333", "1_0", "2_2", "3_0", "3_5", "4_5", "5_0")

    features = pd.DataFrame(index=index)

    abs_z = np.abs(z)
    features["redshift_abs"] = abs_z
    features["redshift_sign"] = np.sign(z)

    stellar_margin = np.where(
        finite,
        np.where(abs_z > velocity_width, abs_z - velocity_width, 0.0),
        np.nan,
    )
    features["redshift_stellar_velocity_margin"] = stellar_margin

    features["redshift_stellar_velocity_signed_margin"] = np.where(
        finite,
        np.where(
            abs_z > velocity_width,
            (abs_z - velocity_width) * np.sign(z),
            0.0,
        ),
        np.nan,
    )
    features["redshift_in_stellar_velocity_domain"] = pd.Series(
        np.where(finite, abs_z <= velocity_width, pd.NA),
        index=index,
        dtype="boolean",
    )

    galaxy_lower_margin = np.where(finite, z - galaxy_low, np.nan)
    galaxy_upper_margin = np.where(finite, z - galaxy_high, np.nan)
    galaxy_in_domain = finite & (z >= galaxy_low) & (z <= galaxy_high)
    galaxy_distance = np.where(
        finite,
        np.where(
            galaxy_in_domain,
            0.0,
            np.where(
                z < galaxy_low,
                galaxy_low - z,
                z - galaxy_high,
            ),
        ),
        np.nan,
    )
    features["redshift_galaxy_lower_margin"] = galaxy_lower_margin
    features["redshift_galaxy_upper_margin"] = galaxy_upper_margin
    features["redshift_in_galaxy_domain"] = pd.Series(
        np.where(galaxy_in_domain, True, pd.NA),
        index=index,
        dtype="boolean",
    )
    features["redshift_galaxy_distance_to_domain"] = galaxy_distance

    qso_lower_margin = np.where(finite, z - qso_low, np.nan)
    qso_upper_margin = np.where(finite, z - qso_high, np.nan)
    qso_in_domain = finite & (z >= qso_low) & (z <= qso_high)
    qso_distance = np.where(
        finite,
        np.where(
            qso_in_domain,
            0.0,
            np.where(
                z < qso_low,
                qso_low - z,
                z - qso_high,
            ),
        ),
        np.nan,
    )
    features["redshift_qso_lower_margin"] = qso_lower_margin
    features["redshift_qso_upper_margin"] = qso_upper_margin
    features["redshift_in_qso_domain"] = pd.Series(
        np.where(qso_in_domain, True, pd.NA),
        index=index,
        dtype="boolean",
    )
    features["redshift_qso_distance_to_domain"] = qso_distance

    regime_bin = np.full(z.shape[0], -1, dtype=np.int16)
    finite_vals = z[finite]
    if finite_vals.size > 0:
        regime_bin[finite] = np.searchsorted(
            np.array(regime_boundaries, dtype=float),
            finite_vals,
            side="right",
        )
    features["redshift_regime_bin"] = regime_bin

    for boundary, name in zip(regime_boundaries, regime_names):
        safe_label = name.replace(".", "p")
        features[f"redshift_minus_bnd_{safe_label}"] = np.where(finite, z - boundary, np.nan)
        features[f"redshift_ge_bnd_{safe_label}"] = pd.Series(
            np.where(finite, z >= boundary, pd.NA),
            index=index,
            dtype="boolean",
        )

    return features

FEATURE_GROUPS = [
    {
        "name": "redshift_template_domain_margins",
        "fn": add_redshift_template_domain_margins,
        "depends_on": [],
        "description": "Builds redshift-domain geometry features for stellar, galaxy, and quasar template feasibility with signed margins, in-domain flags, and ordered boundary regime indicators.",
    },
]
```

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

Mode-specific boundary:
- Do not train AutoGluon.
- Do not import or instantiate `TabularPredictor`.
- Do not call `.fit()`, `.predict()`, `.predict_proba()`, or `.leaderboard()`.
- Do not define `main()`.
- Do not read project data files.
- The fixed wrapper handles all non-preprocessing work.