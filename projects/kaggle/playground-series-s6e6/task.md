# playground-series-s6e6

## Goal
Predict the stellar class for each test-set record.

## Evaluation
Submissions are evaluated using balanced accuracy between predicted class labels and the observed target. Submission file must contain `id,class` with one predicted label per row, using class labels such as GALAXY, STAR, or QSO.

## Data description
`train.csv` contains the training data with `class` as the target. `test.csv` contains the test features for which `class` must be predicted. `sample_submission.csv` shows the required submission format.
