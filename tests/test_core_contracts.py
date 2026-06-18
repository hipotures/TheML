from __future__ import annotations

import re
from pathlib import Path

from tml.core.ids import node_id, run_id
from tml.core.kaggle import download_competition_data
from tml.core.errors import TmlError
from tml.db.reindex import classify_node
from tml.execution.autogluon_wrapper import run_autogluon_materialization


def test_run_and_node_ids_are_timestamp_random_and_step_based():
    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", run_id())
    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}-7", node_id(7))


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


def test_kaggle_download_fails_clearly_when_cli_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PATH", "")

    try:
        download_competition_data("playground-series-s6e6", tmp_path / "data")
    except TmlError as exc:
        assert "Kaggle CLI is not installed" in str(exc)
    else:
        raise AssertionError("Expected TmlError when Kaggle CLI is missing")
