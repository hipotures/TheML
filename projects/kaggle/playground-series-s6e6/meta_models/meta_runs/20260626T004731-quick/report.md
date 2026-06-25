# AutoGluon Experiment Meta-Model Report

Created: 2026-06-26T00:49:03
Output: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick`

## Dataset

- Records: 477
- CV labels: 406
- Public labels: 40
- Dataset CSV: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick/dataset/meta_dataset.csv`
- Feature CSV: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick/dataset/meta_features.csv`
- Feature count: 151

Leakage policy: Targets are excluded from feature_columns. Parent and group-history features are computed only from earlier records sorted by created_at, step, and node_id.

## Targets

### cv_score

- Status: complete
- Examples: 406
- Low confidence: false
- mae_mean: 0.000231
- rmse_mean: 0.000488
- median_absolute_error_mean: 0.000143
- r2_mean: 0.914616
- spearman_mean: 0.856380
- pearson_mean: 0.957649
- top10_precision_mean: 0.363636
- top10_recall_mean: 0.363636
- group_split_mae: 0.000452
- group_split_spearman: 0.727705
- high_similarity_row_rate: 0.738162
- p90_nearest_group_jaccard: 1.000000
- Feature importance: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick/targets/cv_score/feature_importance.csv`

### public_score

- Status: complete
- Examples: 40
- Low confidence: true
- mae_mean: 0.000212
- rmse_mean: 0.000271
- median_absolute_error_mean: 0.000172
- r2_mean: -1.801055
- spearman_mean: 0.438384
- pearson_mean: 0.519514
- top10_precision_mean: 0.000000
- top10_recall_mean: 0.000000
- high_similarity_row_rate: 1.000000
- p90_nearest_group_jaccard: 1.000000
- Feature importance: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick/targets/public_score/feature_importance.csv`

### public_gap

- Status: complete
- Examples: 40
- Low confidence: true
- mae_mean: 0.000204
- rmse_mean: 0.000264
- median_absolute_error_mean: 0.000170
- r2_mean: 0.014659
- spearman_mean: 0.721212
- pearson_mean: 0.741150
- top10_precision_mean: 0.333333
- top10_recall_mean: 0.333333
- high_similarity_row_rate: 1.000000
- p90_nearest_group_jaccard: 1.000000
- Feature importance: `/home/xai/DEV/TheML/projects/kaggle/playground-series-s6e6/meta_models/meta_runs/20260626T004731-quick/targets/public_gap/feature_importance.csv`

## Required Questions

- Czy `cv_score` da sie przewidywac? CV ma sygnal predykcyjny.
- Jaki jest realny blad predykcji? MAE=0.000231 score units.
- Czy model dobrze szereguje kandydatow? Spearman=0.8564, top10 precision=0.363636, top10 recall=0.363636.
- Czy public score daje sygnal? Public score jest low-confidence: n=40, MAE=0.000212, Spearman=0.438384.
- Czy model radzi sobie z nowymi grupami? Unseen-group MAE=0.000603 vs known-group MAE=0.000194.
- Czy walidacja moze byc zawyzona przez podobne node'y? Group split MAE=0.000452; porownaj z random split przed decyzja.
- Czy warto integrowac advisory/pruning? Wartosc praktyczna: ostroznie jako advisory, nie jako automatyczne pruning. Ten modul powinien pozostac raportem/advisory do czasu mocniejszej walidacji.

This report is diagnostic only. No workflow pruning or node selection logic was changed.
