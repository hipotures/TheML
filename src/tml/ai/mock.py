from __future__ import annotations

import json
import re

from .client import AiRequest, AiResponse
from .invocation import ModelInvocation, ProviderResult


class MockAiClient:
    def invoke(self, invocation: ModelInvocation, spec: object) -> ProviderResult:
        response = self.call(AiRequest(role=invocation.role, model=invocation.model, prompt=invocation.prompt))
        return ProviderResult(
            text=response.text,
            metadata=response.metadata,
            raw={
                "provider": "mock",
                "model": invocation.model,
                "role": invocation.role,
                "simulated": True,
            },
        )

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
    id_column, target_column = _sample_submission_columns(prompt)
    target_column = target_column or _target_column_from_text(prompt)
    metric = _metric_from_text(lowered)
    problem_type = _problem_type_from_text(lowered, metric)
    submission_kind = _submission_kind_from_text(lowered, problem_type)
    goal = _goal_from_text(prompt)
    metric_description = _metric_description(metric)
    return {
        "goal": goal or "Solve the Kaggle prediction task.",
        "evaluation": _evaluation_summary(metric, submission_kind),
        "data_description": _data_description_summary(target_column),
        "target": {
            "id_column": id_column or "id",
            "target_column": target_column,
            "problem_type": problem_type,
            "metric": metric,
            "metric_source": "sklearn" if metric else "unknown",
            "sklearn_metric": metric,
            "metric_description": metric_description,
            "maximize": True,
            "submission_kind": submission_kind,
        },
    }


def _sample_submission_columns(prompt: str) -> tuple[str | None, str | None]:
    match = re.search(r"Sample submission header:\s*\[([^\]]*)\]", prompt)
    if not match:
        return None, None
    columns = [part.strip().strip("'\"") for part in match.group(1).split(",")]
    columns = [column for column in columns if column]
    first = columns[0] if columns else None
    second = columns[1] if len(columns) > 1 else None
    return first, second


def _target_column_from_text(prompt: str) -> str | None:
    patterns = (
        r"with `([^`]+)` as target",
        r"predict `([^`]+)`",
        r"predict the ([A-Za-z_][A-Za-z0-9_]*)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _goal_from_text(prompt: str) -> str | None:
    match = re.search(r"\*\*Your Goal:\*\*\s*([^\n]+)", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _metric_from_text(lowered: str) -> str | None:
    if "balanced accuracy" in lowered:
        return "sklearn.metrics.balanced_accuracy_score"
    if "root mean squared error" in lowered or "rmse" in lowered:
        return "sklearn.metrics.root_mean_squared_error"
    if "mean absolute error" in lowered or " mae" in lowered:
        return "sklearn.metrics.mean_absolute_error"
    if "roc auc" in lowered or "area under the roc" in lowered:
        return "sklearn.metrics.roc_auc_score"
    if "log loss" in lowered:
        return "sklearn.metrics.log_loss"
    if "accuracy" in lowered:
        return "sklearn.metrics.accuracy_score"
    return None


def _problem_type_from_text(lowered: str, metric: str | None) -> str | None:
    if metric in {
        "sklearn.metrics.balanced_accuracy_score",
        "sklearn.metrics.accuracy_score",
        "sklearn.metrics.log_loss",
    }:
        return "multiclass"
    if metric == "sklearn.metrics.roc_auc_score":
        return "binary"
    if metric in {
        "sklearn.metrics.root_mean_squared_error",
        "sklearn.metrics.mean_absolute_error",
    }:
        return "regression"
    if "numeric" in lowered or "continuous" in lowered or metric in {
        "sklearn.metrics.root_mean_squared_error",
        "sklearn.metrics.mean_absolute_error",
    }:
        return "regression"
    if "class label" in lowered or "multiclass" in lowered or "balanced accuracy" in lowered:
        return "multiclass"
    if "probability" in lowered or "binary" in lowered or metric == "sklearn.metrics.roc_auc_score":
        return "binary"
    return None


def _submission_kind_from_text(lowered: str, problem_type: str | None) -> str | None:
    if problem_type == "regression":
        return "numeric"
    if problem_type in {"binary", "multiclass"}:
        return "labels"
    if "numeric" in lowered:
        return "numeric"
    if "probability" in lowered:
        return "probabilities"
    if "class label" in lowered:
        return "labels"
    return None


def _metric_description(metric: str | None) -> str | None:
    descriptions = {
        "sklearn.metrics.balanced_accuracy_score": "Balanced accuracy between predicted and observed labels.",
        "sklearn.metrics.root_mean_squared_error": "Root mean squared error between predicted and observed numeric targets.",
        "sklearn.metrics.mean_absolute_error": "Mean absolute error between predicted and observed numeric targets.",
        "sklearn.metrics.roc_auc_score": "ROC AUC between predicted probabilities and observed binary labels.",
        "sklearn.metrics.log_loss": "Log loss between predicted probabilities and observed labels.",
        "sklearn.metrics.accuracy_score": "Accuracy between predicted and observed labels.",
    }
    return descriptions.get(metric)


def _evaluation_summary(metric: str | None, submission_kind: str | None) -> str:
    if metric is None:
        return "Use the competition evaluation metric described by Kaggle."
    text = f"Submissions are evaluated using {_metric_label(metric)}."
    if submission_kind:
        text += f" Predictions must be {submission_kind}."
    return text


def _metric_label(metric: str) -> str:
    labels = {
        "sklearn.metrics.balanced_accuracy_score": "balanced accuracy",
        "sklearn.metrics.root_mean_squared_error": "root mean squared error",
        "sklearn.metrics.mean_absolute_error": "mean absolute error",
        "sklearn.metrics.roc_auc_score": "ROC AUC",
        "sklearn.metrics.log_loss": "log loss",
        "sklearn.metrics.accuracy_score": "accuracy",
    }
    return labels.get(metric, metric)


def _data_description_summary(target_column: str | None) -> str:
    if target_column:
        return (
            f"train.csv contains the target column {target_column}; "
            f"test.csv requires predictions for {target_column}; "
            "sample_submission.csv defines the required submission format."
        )
    return "Use train.csv, test.csv, and sample_submission.csv."
