from __future__ import annotations

import json

from .client import AiRequest, AiResponse


class MockAiClient:
    def call(self, request: AiRequest) -> AiResponse:
        if request.role == "hypothesis":
            payload = {
                "title": "Baseline robust tabular preprocessing",
                "summary": "Create a small leakage-safe first ROOT hypothesis for tabular data.",
                "feature_family": "baseline_preprocessing",
                "feature_strategy": "Add bounded, target-free preprocessing and preserve row order.",
                "materialization_hint": "Autogluon mode should expose preprocess(df); legacy mode may run a full script.",
                "expected_signal": "Useful as a smoke-tested baseline before richer ROOT hypotheses.",
                "risk": "This mock hypothesis is intentionally generic and should be replaced by a real model backend.",
                "sources": [],
            }
        elif request.role == "code":
            payload = {"code": legacy_code() if "legacy" in request.prompt else autogluon_code()}
        elif request.role == "metadata":
            payload = project_metadata(request.prompt)
        else:
            payload = {"summary": "Mock response", "ok": True}
        return AiResponse(text=json.dumps(payload, indent=2), metadata={"backend": "mock"})


def autogluon_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import pandas as pd\n\n\n"
        "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = df.copy()\n"
        "    return out\n"
    )


def legacy_code() -> str:
    return (
        "from __future__ import annotations\n\n"
        "import json\n\n"
        "result = {'status': 'ok', 'metric': None, 'maximize': True}\n"
        "print('TML_RESULT_JSON:' + json.dumps(result, sort_keys=True))\n"
    )


def project_metadata(prompt: str) -> dict[str, object]:
    lowered = prompt.lower()
    target_column = "class" if "class" in lowered else None
    metric = None
    sklearn_metric = None
    problem_type = None
    submission_kind = None
    if "balanced accuracy" in lowered:
        metric = "sklearn.metrics.balanced_accuracy_score"
        sklearn_metric = metric
        problem_type = "multiclass" if any(label in prompt for label in ("GALAXY", "STAR", "QSO")) else None
        submission_kind = "labels"
    return {
        "goal": "Predict the stellar class." if "stellar class" in lowered else "Solve the Kaggle prediction task.",
        "evaluation": (
            "Submissions are evaluated using balanced accuracy. "
            "Predictions must be class labels."
            if metric
            else "Use the competition evaluation metric described by Kaggle."
        ),
        "data_description": (
            "train.csv contains the target column class; test.csv requires predictions for class; "
            "sample_submission.csv defines the required submission format."
            if target_column == "class"
            else "Use train.csv, test.csv, and sample_submission.csv."
        ),
        "target": {
            "id_column": "id",
            "target_column": target_column,
            "problem_type": problem_type,
            "metric": metric,
            "metric_source": "sklearn" if sklearn_metric else "unknown",
            "sklearn_metric": sklearn_metric,
            "metric_description": "Balanced accuracy between predicted and observed class." if metric else None,
            "maximize": True,
            "submission_kind": submission_kind,
        },
    }
