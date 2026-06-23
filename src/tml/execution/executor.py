from __future__ import annotations

import contextlib
import importlib.util
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from tml.core.config import load_project_config
from tml.features.groups import has_feature_groups, run_feature_groups
from tml.features.validation import validate_group_code_source
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import write_yaml

from .result import ExecutionResult


RESULT_PREFIX = "TML_RESULT_JSON:"


def run_legacy_group_materialization(
    *,
    code_path: Path,
    project_dir: Path,
    work_dir: Path,
    timeout_seconds: int = 900,
) -> ExecutionResult:
    _ = timeout_seconds
    work_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    try:
        metric, maximize, run_stats = _run_legacy_group_baseline(
            code_path=code_path,
            project_dir=project_dir,
            work_dir=work_dir,
        )
    except Exception as exc:
        return ExecutionResult(
            status="failed",
            returncode=5,
            stdout="",
            stderr="",
            error=str(exc),
        )
    total_elapsed = time.perf_counter() - started_at
    run_stats["execution_time"] = float(total_elapsed)
    print(f"TML_STAGE|legacy execution_time_s={total_elapsed:.3f}", flush=True)
    return ExecutionResult(
        status="ok",
        returncode=0,
        stdout="Legacy group baseline completed\n",
        stderr="",
        metric=metric,
        maximize=maximize,
        payload={
            "is_bug": False,
            "summary": "Legacy group baseline completed.",
            "metric": metric,
            "maximize": maximize,
            "run_stats": run_stats,
        },
    )


def run_python_script(
    script: Path,
    work_dir: Path,
    timeout_seconds: int = 900,
    *,
    cwd: Path | None = None,
) -> ExecutionResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    process_cwd = cwd or work_dir
    started_at = time.monotonic()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def _reader(stream, name: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((name, line))
        finally:
            output_queue.put((name, None))

    print(
        f"TML_EXEC|event=start|script={script}|work_dir={work_dir}|cwd={process_cwd}|timeout_s={timeout_seconds}",
        flush=True,
    )
    try:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=process_cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
    except Exception as exc:
        return ExecutionResult(
            status="failed",
            returncode=1,
            stdout="",
            stderr="",
            error=str(exc),
        )

    assert process.stdout is not None
    assert process.stderr is not None
    threads = [
        threading.Thread(target=_reader, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=_reader, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    closed = set()
    timed_out = False
    while len(closed) < 2:
        if process.poll() is None and time.monotonic() - started_at > timeout_seconds:
            timed_out = True
            process.kill()
        try:
            name, text = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if text is None:
            closed.add(name)
            continue
        if name == "stdout":
            stdout_parts.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            stderr_parts.append(text)
            sys.stderr.write(text)
            sys.stderr.flush()

    returncode = process.wait()
    for thread in threads:
        thread.join(timeout=1)

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        print(f"TML_EXEC|event=timeout|timeout_s={timeout_seconds}", flush=True)
        return ExecutionResult(
            status="failed",
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            error=f"Timed out after {timeout_seconds} seconds",
        )

    metric = None
    maximize = None
    result_payload = None
    for line in stdout.splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        result_payload = payload
        raw_metric = payload.get("metric")
        metric = float(raw_metric) if isinstance(raw_metric, (int, float)) else None
        raw_maximize = payload.get("maximize")
        maximize = bool(raw_maximize) if isinstance(raw_maximize, bool) else None
    status = "ok" if returncode == 0 else "failed"
    print(f"TML_EXEC|event=end|returncode={returncode}|status={status}", flush=True)
    return ExecutionResult(
        status=status,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        metric=metric,
        maximize=maximize,
        error=None if returncode == 0 else f"Process exited with {returncode}",
        payload=result_payload,
    )


def _run_legacy_group_baseline(*, code_path: Path, project_dir: Path, work_dir: Path) -> tuple[float | None, bool, dict[str, object]]:
    import pandas as pd

    stages: list[dict[str, object]] = []
    config = load_project_config(project_dir)
    target = config.get("target", {}) if isinstance(config.get("target"), dict) else {}
    target_col = str(target.get("target_column") or "target")
    id_col = str(target.get("id_column") or "id")
    metric_name = str(target.get("sklearn_metric") or target.get("autogluon_metric") or "balanced_accuracy")
    maximize = bool(target.get("maximize", True))

    with _stage("load_data_stage", stages):
        source = code_path.read_text(encoding="utf-8")
        validate_group_code_source(source)
        module_spec = importlib.util.spec_from_file_location("tml_legacy_group_materialization", code_path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Cannot import materialization: {code_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        if not has_feature_groups(module):
            raise ValueError("Legacy group materialization must define non-empty FEATURE_GROUPS")

        data_dir = project_dir / str(config.get("data_dir") or "data")
        train = pd.read_csv(_data_file(data_dir, "train.csv"))
        test = pd.read_csv(_data_file(data_dir, "test.csv"))
        sample = pd.read_csv(_data_file(data_dir, "sample_submission.csv"))
        if target_col not in train.columns:
            raise ValueError(f"Target column {target_col!r} not found in train data")

    with _stage("build_features_stage", stages):
        train_features = train.drop(columns=[target_col])
        combined = pd.concat([train_features, test], ignore_index=True, sort=False)
        transformed = run_feature_groups(
            combined,
            getattr(module, "FEATURE_GROUPS"),
            log_path=work_dir / "feature-groups.jsonl",
        )
        train_x = transformed.iloc[: len(train)].reset_index(drop=True)
        test_x = transformed.iloc[len(train) :].reset_index(drop=True)
        if id_col in train_x.columns:
            train_x = train_x.drop(columns=[id_col])
            test_x = test_x.drop(columns=[id_col], errors="ignore")

    y = train[target_col].reset_index(drop=True)
    metric, model_stats = _fit_score_and_write_submission(
        train_x=train_x,
        y=y,
        test_x=test_x,
        sample=sample,
        target_col=target_col,
        id_col=id_col,
        metric_name=metric_name,
        work_dir=work_dir,
        stages=stages,
    )
    run_stats: dict[str, object] = {
        "feature_count": int(len(train_x.columns)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "metric_name": metric_name,
        "stages": stages,
        **model_stats,
    }
    return metric, maximize, run_stats


def _fit_score_and_write_submission(
    *,
    train_x,
    y,
    test_x,
    sample,
    target_col: str,
    id_col: str,
    metric_name: str,
    work_dir: Path,
    stages: list[dict[str, object]],
) -> tuple[float | None, dict[str, object]]:
    import pandas as pd

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    is_regression = pd.api.types.is_numeric_dtype(y) and "class" not in metric_name and "accuracy" not in metric_name and "auc" not in metric_name
    stratify = None if is_regression else y
    with _stage("make_folds_stage", stages):
        train_part, valid_part, y_train, y_valid = train_test_split(
            train_x,
            y,
            test_size=0.2,
            random_state=1,
            stratify=stratify if y.nunique(dropna=True) > 1 else None,
        )
        print(
            f"TML_STAGE|legacy holdout rows train={len(train_part)} valid={len(valid_part)}",
            flush=True,
        )

    with _stage("fit_predict_stage", stages):
        numeric_columns = train_x.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_columns = [column for column in train_x.columns if column not in numeric_columns]
        preprocessor = ColumnTransformer(
            [
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
            ],
            remainder="drop",
        )
        estimator = Ridge() if is_regression else LogisticRegression(max_iter=300)
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        print(f"TML_STAGE|fitting legacy estimator={type(estimator).__name__}", flush=True)
        model.fit(train_part, y_train)
        valid_pred = model.predict(valid_part)
        test_pred = model.predict(test_x)

    with _stage("score_stage", stages):
        metric = _score_metric(metric_name, y_valid, valid_pred, model=model, valid_part=valid_part, is_regression=is_regression)
        print(f"TML_STAGE|legacy {metric_name}={metric:.6f}", flush=True)

    with _stage("write_outputs_stage", stages):
        submission = sample.copy()
        prediction_cols = [column for column in submission.columns if column != id_col]
        target_output_col = prediction_cols[0] if prediction_cols else target_col
        submission[target_output_col] = test_pred
        artifacts = work_dir.parent / "artifacts"
        artifacts.mkdir(exist_ok=True)
        submission_path = artifacts / "submission.csv.gz"
        submission.to_csv(submission_path, index=False, compression="gzip")
        print(f"TML_STAGE|legacy submission saved to {submission_path}", flush=True)

    return metric, {
        "estimator": type(estimator).__name__,
        "is_regression": bool(is_regression),
        "valid_rows": int(len(valid_part)),
    }


def _score_metric(metric_name: str, y_true, y_pred, *, model, valid_part, is_regression: bool) -> float | None:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    name = metric_name.rsplit(".", 1)[-1]
    if is_regression:
        if name in {"mean_absolute_error", "mae"}:
            return float(mean_absolute_error(y_true, y_pred))
        if name in {"r2_score", "r2"}:
            return float(r2_score(y_true, y_pred))
        return float(mean_squared_error(y_true, y_pred) ** 0.5)
    if name in {"accuracy_score", "accuracy"}:
        return float(accuracy_score(y_true, y_pred))
    if name in {"f1_score", "f1"}:
        return float(f1_score(y_true, y_pred, average="macro"))
    if name in {"roc_auc_score", "roc_auc"} and hasattr(model, "predict_proba"):
        proba = model.predict_proba(valid_part)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
    return float(balanced_accuracy_score(y_true, y_pred))


def _data_file(data_dir: Path, name: str) -> Path:
    plain = data_dir / name
    if plain.exists():
        return plain
    return plain.with_name(plain.name + ".gz")


@contextlib.contextmanager
def _stage(name: str, stages: list[dict[str, object]]):
    print(f"TML_STAGE|event=start|stage={name}", flush=True)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started_at
        stages.append({"stage": name, "elapsed_s": round(elapsed, 6)})
        print(f"TML_STAGE|event=end|stage={name}|elapsed_s={elapsed:.3f}", flush=True)


def write_attempt_result(attempt_dir: Path, result: ExecutionResult) -> None:
    atomic_write_text(attempt_dir / "stdout.log", result.stdout)
    atomic_write_text(attempt_dir / "stderr.log", result.stderr)
    payload = {
        "status": result.status,
        "returncode": result.returncode,
        "metric": result.metric,
        "maximize": result.maximize,
        "error": result.error,
    }
    if result.payload is not None:
        payload["result_payload"] = result.payload
        run_stats = result.payload.get("run_stats")
        if isinstance(run_stats, dict):
            payload["run_stats"] = run_stats
    if result.status == "ok":
        write_yaml(attempt_dir / "result.yaml", payload)
    else:
        write_yaml(attempt_dir / "failed.yaml", payload)
