# Local/Public Alignment Notes

Date: 2026-06-29

## Evidence

Known scored submissions in `tml.db`:

- 73 rows have `public_score`.
- 63 of those are from the quick profile `ag-s6e6-boost-gpu-ens-v1` and run in <=600 seconds.
- Quick profile local `WeightedEnsemble_L2` vs public is negatively correlated:
  - Pearson: -0.2219
  - Spearman: -0.1339
  - local range: 0.967929..0.968423
  - public range: 0.96762..0.96856

The strongest counterexample is `B000206`: quick local 0.968423, public 0.96764. This branch is top local and near-bottom public.

Known good public feature sets:

- `B000159`: quick public 0.96856; 30m ds0-stack public 0.96885.
- `B000180`: quick public 0.96851; 2h stack public 0.96873; 30m ds0-stack public 0.96836.
- `B000448`: quick public 0.96853.

## Root Cause Hypothesis

The quick profile uses one fixed stratified holdout. Hundreds of feature-group branches were selected against that same holdout, so local score is now a feature-selection target rather than an unbiased estimate.

This is visible in component effects:

- `000016` Galactic and SDSS Sky Geometry: local delta positive, public delta negative.
- `000056` sky-context locus residuals: local delta positive, public delta negative.
- `000060` metadata class-prior encoding: local delta positive, public delta negative.
- `000028` calibrated dust-vector color decomposition: local delta positive, public delta negative.
- `000002` redshift-regime catalog consistency: public delta strongly negative in known quick submissions.

The relation is therefore not a simple leaderboard/public anti-correlation. It is mostly local holdout overfitting plus a few feature families that exploit quirks of the local split.

## Recommended 10m Scoring Profile

Use `profiles/autogluon/ag-s6e6-boost-gpu-xgb-cv3-10m-v1.yaml`.

Key settings:

- `included_model_types: [XGB]`
- `time_limit: 600`
- `validation_strategy: autogluon`
- `num_bag_folds: 3`
- `num_stack_levels: 0`
- `auto_stack: false`
- `class_balance: balanced`
- GPU XGBoost enabled

Rationale:

- 3-fold bagging avoids reusing the overfit fixed holdout.
- XGBoost is the only model family that reliably fits within the 10-minute budget on this dataset.
- Historical `cv3` on `B000180` showed XGBoost_BAG_L1 fit in about 145s, while GBM/L2/stacking pushed total runtime above 10 minutes.

## Feature-Gating Rule

For quick feature search, do not accept a branch only because quick holdout `WeightedEnsemble_L2` improves.

Use this gate:

1. Score candidate features with `ag-s6e6-boost-gpu-xgb-cv3-10m-v1`.
2. Prefer candidates whose 3-fold XGB score improves by at least 0.00015 over the current anchor.
3. Reject candidates adding any of `000002`, `000016`, `000056`, `000060`, `000040`, `000032`, `000015`, or `000008` unless the 3-fold XGB gain is at least 0.00025.
4. Treat deltas below 0.00010 as noise; do not spend Kaggle submits on them.
5. Keep feature count near known-good branches (`B000159`, `B000180`, `B000448`), not near the widest local winners.

Current confidence:

- Old quick holdout ensemble score: low confidence; observed rank signal is negative.
- Quick CatBoost holdout subscore: medium diagnostic value only; Spearman vs public is about 0.31, but it is not the final model.
- Proposed 3-fold XGB score: best available 10-minute local proxy, but it still needs a Kaggle calibration submit before being treated as final.

## Submit Plan

Use Kaggle submits only after a branch passes the 3-fold XGB gate.

First calibration target should be one known-good branch (`B000159` or `B000180`) trained with the new 10m profile. If its public score lands in the expected public band for that branch family, use remaining submits to compare new candidates against that calibrated anchor.
