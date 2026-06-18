from __future__ import annotations

import importlib.util
import traceback
from pathlib import Path

from tml.core.config import active_profile_id, load_project_config
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import write_yaml

from .result import ExecutionResult


def run_autogluon_materialization(
    *,
    code_path: Path,
    project_dir: Path,
    work_dir: Path,
) -> ExecutionResult:
    data_dir = project_dir / "data"
    required = [
        _data_file(data_dir, "train.csv"),
        _data_file(data_dir, "test.csv"),
        _data_file(data_dir, "sample_submission.csv"),
    ]
    missing = [path.relative_to(project_dir).as_posix() for path in required if not path.exists()]
    if missing:
        return ExecutionResult(
            status="failed",
            returncode=2,
            stdout="",
            stderr="",
            error="Missing AutoGluon input files: " + ", ".join(missing),
        )
    if importlib.util.find_spec("autogluon.tabular") is None:
        return ExecutionResult(
            status="failed",
            returncode=3,
            stdout="",
            stderr="",
            error="AutoGluon is not installed in the active uv environment.",
        )
    marker = work_dir / "tml-autogluon-workdir" / ".tml-autogluon-workdir"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("TheML AutoGluon workdir\n", encoding="utf-8")
    source = code_path.read_text(encoding="utf-8")
    if "def preprocess" not in source:
        return ExecutionResult(
            status="failed",
            returncode=4,
            stdout="",
            stderr="",
            error="AutoGluon materialization must define preprocess(df).",
        )
    try:
        metric = _run_tabular(code_path=code_path, project_dir=project_dir, work_dir=work_dir)
    except Exception as exc:
        atomic_write_text(work_dir / "autogluon-error.txt", traceback.format_exc())
        return ExecutionResult(
            status="failed",
            returncode=5,
            stdout="",
            stderr=traceback.format_exc(),
            error=str(exc),
        )
    return ExecutionResult(
        status="ok",
        returncode=0,
        stdout="AutoGluon training completed\n",
        stderr="",
        metric=metric,
        maximize=True,
    )


def _run_tabular(*, code_path: Path, project_dir: Path, work_dir: Path) -> float | None:
    import importlib.util

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    config = load_project_config(project_dir)
    target = config.get("target", {}) if isinstance(config.get("target"), dict) else {}
    target_col = str(target.get("target_column") or "target")
    id_col = str(target.get("id_column") or "id")
    metric = str(target.get("metric") or "balanced_accuracy")
    profile_id = active_profile_id(config, "autogluon")
    profile = _load_profile(project_dir, profile_id)

    data_dir = project_dir / "data"
    train = pd.read_csv(_data_file(data_dir, "train.csv"))
    test = pd.read_csv(_data_file(data_dir, "test.csv"))
    sample = pd.read_csv(_data_file(data_dir, "sample_submission.csv"))
    if target_col not in train.columns:
        raise ValueError(f"Target column {target_col!r} not found in train.csv")

    module_spec = importlib.util.spec_from_file_location("tml_materialization", code_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot import materialization: {_project_path(project_dir, code_path)}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    preprocess = getattr(module, "preprocess", None)
    if not callable(preprocess):
        raise ValueError("AutoGluon materialization must define callable preprocess(df)")

    train_features = train.drop(columns=[target_col])
    combined = pd.concat([train_features, test], ignore_index=True, sort=False)
    transformed = preprocess(combined.copy())
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError("preprocess(df) must return a pandas DataFrame")
    if len(transformed) != len(combined):
        raise ValueError("preprocess(df) must preserve row count")

    train_out = transformed.iloc[: len(train)].reset_index(drop=True)
    test_out = transformed.iloc[len(train) :].reset_index(drop=True)
    train_out[target_col] = train[target_col].reset_index(drop=True)

    predictor = TabularPredictor(
        label=target_col,
        eval_metric=metric,
        path=str(work_dir / "tml-autogluon-workdir" / "AutoGluonModels"),
    )
    fit_kwargs = {
        "time_limit": int(profile.get("time_limit", 60)),
        "presets": profile.get("presets", "medium_quality"),
    }
    predictor.fit(train_out, **fit_kwargs)

    predictions = predictor.predict(test_out)
    submission = sample.copy()
    prediction_cols = [col for col in submission.columns if col != id_col]
    if not prediction_cols:
        prediction_cols = [submission.columns[-1]]
    submission[prediction_cols[0]] = predictions.values
    artifacts = work_dir.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    submission.to_csv(artifacts / "submission.csv", index=False)

    leaderboard = predictor.leaderboard(silent=True)
    if "score_val" in leaderboard.columns and not leaderboard.empty:
        value = leaderboard.iloc[0]["score_val"]
        return float(value) if pd.notna(value) else None
    return None


def _load_profile(project_dir: Path, profile_id: str) -> dict[str, object]:
    from tml.utils.yaml_io import read_yaml

    return read_yaml(project_dir / "profiles" / "root" / f"{profile_id}.yaml")


def _project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.name


def _data_file(data_dir: Path, name: str) -> Path:
    plain = data_dir / name
    if plain.exists():
        return plain
    return plain.with_name(plain.name + ".gz")
