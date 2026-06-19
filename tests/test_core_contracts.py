from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

from tml.ai import AiRequest
from tml.ai.models import resolve_role_model
from tml.ai.mock import MockAiClient
from tml.core.ids import node_id, run_id
from tml.core.kaggle import download_competition_data
from tml.core.metadata import normalize_metric
from tml.core.errors import TmlError
from tml.db.reindex import classify_node
from tml.execution.autogluon_wrapper import (
    CLASS_WEIGHT_COL,
    _fit_kwargs_from_profile,
    _predictor_kwargs_from_profile,
    _training_plan_from_profile,
    run_autogluon_materialization,
)


def test_run_and_node_ids_are_timestamp_random_and_step_based():
    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", run_id())
    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}-7", node_id(7))


def test_models_mapping_resolves_model_spec_and_role_options():
    model, options = resolve_role_model(
        {
            "metadata": {
                "provider": "codex",
                "model": "gpt-5.4",
                "effort": "low",
                "timeout_seconds": 30,
            }
        },
        "metadata",
    )

    assert model == "codex:gpt-5.4:low"
    assert options == {"timeout_seconds": 30}


def test_classify_node_uses_phase_files(tmp_path: Path):
    node = tmp_path / "node"
    node.mkdir()
    assert classify_node(node) == "missing_hypothesis"

    (node / "01-hypothesis.yaml").write_text("hypothesis_id: '000001'\n", encoding="utf-8")
    assert classify_node(node) == "missing_code"

    (node / "02-code.py").write_text("print('x')\n", encoding="utf-8")
    assert classify_node(node) == "aborted"

    attempt = node / "03-execute" / "attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "started.yaml").write_text("created_at: now\n", encoding="utf-8")
    assert classify_node(node) == "execution_interrupted"

    (node / "artifact-manifest.yaml").write_text("node_id: node\n", encoding="utf-8")
    assert classify_node(node) == "ready_to_finalize"

    (node / "node.done.yaml").write_text("status: complete\n", encoding="utf-8")
    assert classify_node(node) == "complete"


def test_autogluon_wrapper_fails_clearly_when_required_data_is_missing(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    code = tmp_path / "autogluon.py"
    code.write_text("def preprocess(df):\n    return df\n", encoding="utf-8")

    result = run_autogluon_materialization(
        code_path=code,
        project_dir=project,
        work_dir=tmp_path / "work",
    )

    assert result.status == "failed"
    assert result.returncode == 2
    assert "Missing AutoGluon input files" in str(result.error)


def test_autogluon_profile_is_mapped_to_fit_kwargs():
    fit_kwargs = _fit_kwargs_from_profile(
        {
            "schema_version": 1,
            "profile_id": "ag-test-v1",
            "mode": "autogluon",
            "time_limit": 600,
            "presets": "medium_quality",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "hyperparameters": {"XGB": [{"device": "cuda"}]},
            "use_gpu": True,
            "validation_strategy": "holdout",
            "validation_fraction": 0.2,
            "class_balance": "balanced",
            "memory_limit": 12,
            "calibrate_decision_threshold": False,
            "future_autogluon_kwarg": "kept",
            "fit_args": {"save_space": True, "fit_weighted_ensemble": False},
        }
    )

    assert fit_kwargs["time_limit"] == 600
    assert fit_kwargs["presets"] == "medium_quality"
    assert fit_kwargs["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert fit_kwargs["hyperparameters"] == {"XGB": [{"device": "cuda"}]}
    assert fit_kwargs["num_gpus"] == 1
    assert fit_kwargs["memory_limit"] == 12
    assert fit_kwargs["calibrate_decision_threshold"] is False
    assert fit_kwargs["future_autogluon_kwarg"] == "kept"
    assert fit_kwargs["save_space"] is True
    assert fit_kwargs["fit_weighted_ensemble"] is False
    for reserved in ("schema_version", "profile_id", "mode", "validation_strategy", "validation_fraction", "class_balance", "use_gpu"):
        assert reserved not in fit_kwargs


def test_autogluon_profile_handles_class_balance_holdout_and_bagging():
    train_model = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
            "target": [0, 0, 0, 1, 1, 1],
        }
    )
    holdout_profile = {
        "class_balance": "balanced",
        "validation_strategy": "holdout",
        "validation_fraction": 0.33,
        "seed": 7,
        "use_gpu": False,
    }

    predictor_kwargs = _predictor_kwargs_from_profile(
        label="target",
        eval_metric="accuracy",
        model_path=Path("model"),
        profile=holdout_profile,
    )
    plan = _training_plan_from_profile(train_model, "target", holdout_profile)
    fit_kwargs = _fit_kwargs_from_profile(holdout_profile, train_data=plan.train_data, valid_data=plan.valid_data)

    assert predictor_kwargs["sample_weight"] == CLASS_WEIGHT_COL
    assert predictor_kwargs["weight_evaluation"] is False
    assert CLASS_WEIGHT_COL in plan.train_data.columns
    assert plan.valid_data is not None
    assert CLASS_WEIGHT_COL in plan.valid_data.columns
    assert fit_kwargs["train_data"] is plan.train_data
    assert fit_kwargs["tuning_data"] is plan.valid_data
    assert fit_kwargs["num_gpus"] == 0

    bagged_profile = {
        "validation_strategy": "holdout",
        "fit_args": {"save_space": True, "num_bag_folds": 3, "auto_stack": False},
    }
    bagged_plan = _training_plan_from_profile(train_model, "target", bagged_profile)
    bagged_fit_kwargs = _fit_kwargs_from_profile(
        bagged_profile,
        train_data=bagged_plan.train_data,
        valid_data=bagged_plan.valid_data,
        fit_args=bagged_plan.fit_args,
    )

    assert bagged_plan.valid_data is None
    assert bagged_plan.defer_save_space is True
    assert "tuning_data" not in bagged_fit_kwargs
    assert "save_space" not in bagged_fit_kwargs
    assert bagged_fit_kwargs["num_bag_folds"] == 3


def test_kaggle_download_fails_clearly_when_cli_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", "")

    try:
        download_competition_data("playground-series-s6e6", tmp_path / "data")
    except TmlError as exc:
        assert "Kaggle CLI is not installed" in str(exc)
    else:
        raise AssertionError("Expected TmlError when Kaggle CLI is missing")


def test_kaggle_download_fails_clearly_when_cli_script_is_broken(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kaggle = bin_dir / "kaggle"
    kaggle.write_text(
        "#!/bin/sh\n"
        "echo 'Traceback (most recent call last):' >&2\n"
        "echo 'ModuleNotFoundError: No module named '\\''kaggle'\\''' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    kaggle.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    try:
        download_competition_data("playground-series-s6e6", tmp_path / "data")
    except TmlError as exc:
        message = str(exc)
        assert "Kaggle CLI is installed but its Python package is not importable" in message
        assert "Traceback" not in message
    else:
        raise AssertionError("Expected TmlError when Kaggle CLI is broken")


def test_kaggle_download_gzips_extracted_files(tmp_path: Path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kaggle = bin_dir / "kaggle"
    kaggle.write_text(
        """#!{python}
from pathlib import Path
import sys
import zipfile

out = Path(sys.argv[sys.argv.index("-p") + 1])
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out / "playground-series-s6e6.zip", "w") as archive:
    archive.writestr("train.csv", "id,target\\n1,0\\n")
    archive.writestr("test.csv", "id\\n2\\n")
    archive.writestr("sample_submission.csv", "id,target\\n2,0\\n")
""".format(python=sys.executable),
        encoding="utf-8",
    )
    kaggle.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))

    download_competition_data("playground-series-s6e6", tmp_path / "data")

    assert (tmp_path / "data" / "playground-series-s6e6.zip").exists()
    assert not (tmp_path / "data" / "train.csv").exists()
    assert not (tmp_path / "data" / "test.csv").exists()
    assert not (tmp_path / "data" / "sample_submission.csv").exists()
    assert (tmp_path / "data" / "train.csv.gz").exists()
    assert (tmp_path / "data" / "test.csv.gz").exists()
    assert (tmp_path / "data" / "sample_submission.csv.gz").exists()
    with gzip.open(tmp_path / "data" / "sample_submission.csv.gz", "rt", encoding="utf-8") as handle:
        assert handle.readline().strip() == "id,target"


def test_project_metadata_metric_normalization_supports_sklearn_and_custom():
    sklearn_metric = normalize_metric(
        {
            "sklearn_metric": "sklearn.metrics.balanced_accuracy_score",
            "metric_description": "Balanced accuracy between predicted and observed class.",
            "maximize": True,
        }
    )
    assert sklearn_metric["autogluon_metric"] == "balanced_accuracy"
    assert sklearn_metric["sklearn_metric"] == "sklearn.metrics.balanced_accuracy_score"
    assert sklearn_metric["metric_description"] == "Balanced accuracy between predicted and observed class."

    custom_metric = normalize_metric(
        {
            "autogluon_metric": "custom",
            "metric_description": "Competition-specific grouped concordance metric.",
            "maximize": False,
        }
    )
    assert custom_metric["autogluon_metric"] == "custom"
    assert custom_metric["sklearn_metric"] is None
    assert custom_metric["metric_description"] == "Competition-specific grouped concordance metric."


def test_mock_metadata_parser_is_not_tied_to_one_competition():
    prompt = """
Competition slug: tabular-price
Sample submission header: ['id', 'price']

## Kaggle page: abstract
**Your Goal:** Predict the sale price.

## Kaggle page: Evaluation
Submissions are evaluated using Root Mean Squared Error between the predicted price and observed target.
For each id in the test set, you must predict a numeric price value.

## Kaggle page: data-description
**train.csv** - the training set, with `price` as target
**test.csv** - the test set, used to predict `price`
"""
    response = MockAiClient().call(AiRequest(role="metadata", model="mock", prompt=prompt))
    payload = json.loads(response.text)

    assert payload["goal"] == "Predict the sale price."
    assert payload["target"]["target_column"] == "price"
    assert payload["target"]["problem_type"] == "regression"
    assert payload["target"]["autogluon_metric"] == "root_mean_squared_error"
    assert payload["target"]["sklearn_metric"] == "sklearn.metrics.root_mean_squared_error"
    assert payload["target"]["submission_kind"] == "numeric"
