from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

from tml.core.ids import node_id, run_id
from tml.core.kaggle import download_competition_data
from tml.core.errors import TmlError
from tml.db.reindex import classify_node
from tml.execution.autogluon_wrapper import _fit_kwargs_from_profile, run_autogluon_materialization


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


def test_autogluon_profile_is_mapped_to_fit_kwargs():
    fit_kwargs = _fit_kwargs_from_profile(
        {
            "time_limit": 600,
            "presets": "medium_quality",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "hyperparameters": {"XGB": [{"device": "cuda"}]},
            "validation_strategy": "holdout",
            "validation_fraction": 0.2,
            "fit_args": {"save_space": True, "fit_weighted_ensemble": False},
        }
    )

    assert fit_kwargs["time_limit"] == 600
    assert fit_kwargs["presets"] == "medium_quality"
    assert fit_kwargs["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert fit_kwargs["hyperparameters"] == {"XGB": [{"device": "cuda"}]}
    assert fit_kwargs["holdout_frac"] == 0.2
    assert fit_kwargs["save_space"] is True
    assert fit_kwargs["fit_weighted_ensemble"] is False


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
