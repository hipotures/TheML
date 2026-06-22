from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.config import repo_root_for_project


def build_wrapped_materialization_source(
    mode: str,
    group_code: str,
    project_dir: Path,
    profile_overrides: dict[str, object] | None = None,
) -> str:
    if mode == "legacy":
        return _legacy_wrapper_source(group_code, project_dir, profile_overrides=profile_overrides)
    if mode == "autogluon":
        return _autogluon_wrapper_source(group_code, project_dir, profile_overrides=profile_overrides)
    raise ValueError(f"Unsupported materialization mode: {mode}")


def _autogluon_wrapper_source(
    group_code: str,
    project_dir: Path,
    *,
    profile_overrides: dict[str, object] | None,
) -> str:
    project_literal = repr(_project_relative_path(project_dir))
    overrides_literal = repr(_normalized_profile_overrides(profile_overrides or {}))
    return f'''# Generated AutoGluon materialization.
# Feature-group code is followed by the fixed executable AutoGluon wrapper.

{group_code.rstrip()}


def main():
    import contextlib
    import importlib.util
    import json
    import signal
    import time
    import traceback
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    from tml.core.config import active_profile_id, load_project_config
    from tml.core.profiles import load_profile
    from tml.features.groups import run_feature_groups

    def _resolve_project_dir(relative_path):
        for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
            if (parent / "project.yaml").exists():
                return parent
            candidate = parent / relative_path
            if (candidate / "project.yaml").exists():
                return candidate
        for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            candidate = parent / relative_path
            if (candidate / "project.yaml").exists():
                return candidate
        import tml

        package_root = Path(tml.__file__).resolve().parents[2]
        candidate = package_root / relative_path
        if (candidate / "project.yaml").exists():
            return candidate
        raise FileNotFoundError(f"Cannot resolve project directory for {{relative_path}}")

    project_relative_path = Path({project_literal})
    project_dir = _resolve_project_dir(project_relative_path)
    profile_overrides = {overrides_literal}
    data_dir = project_dir / "data"
    work_dir = Path.cwd()
    artifacts_dir = work_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_log = artifacts_dir / "autogluon_stdout.log"
    reserved_profile_keys = {{
        "schema_version",
        "profile_id",
        "source_profile",
        "mode",
        "preprocess_timeout",
        "validation_strategy",
        "validation_fraction",
        "seed",
        "use_gpu",
        "class_balance",
        "fit_args",
        "predictor_args",
    }}

    class _TeeWriter:
        def __init__(self, primary, log_file):
            self.primary = primary
            self.log_file = log_file

        def write(self, text):
            self.primary.write(text)
            self.log_file.write(text)

        def flush(self):
            self.primary.flush()
            self.log_file.flush()

    @contextlib.contextmanager
    def _preprocess_timeout(seconds):
        if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
            yield
            return
        previous = signal.getsignal(signal.SIGALRM)

        def _raise_timeout(_signum, _frame):
            raise TimeoutError(f"AutoGluon preprocess exceeded {{seconds}} seconds")

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)

    def _data_file(stem):
        gz = data_dir / f"{{stem}}.csv.gz"
        if gz.exists():
            return gz
        return data_dir / f"{{stem}}.csv"

    def _fit_kwargs_from_profile(profile, train_data):
        fit_kwargs = {{"train_data": train_data}}
        for key, value in profile.items():
            if key in reserved_profile_keys or value is None:
                continue
            fit_kwargs[key] = value
        if profile.get("use_gpu") is not None:
            fit_kwargs["num_gpus"] = 1 if profile.get("use_gpu") else 0
        raw_fit_args = profile.get("fit_args")
        if isinstance(raw_fit_args, dict):
            fit_kwargs.update(raw_fit_args)
        return fit_kwargs

    try:
        with target_log.open("a", encoding="utf-8", buffering=1) as log_file:
            stdout = _TeeWriter(__import__("sys").stdout, log_file)
            stderr = _TeeWriter(__import__("sys").stderr, log_file)
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                config = load_project_config(project_dir)
                target = config.get("target", {{}}) if isinstance(config.get("target"), dict) else {{}}
                target_col = str(target.get("target_column") or "target")
                id_col = str(target.get("id_column") or "id")
                metric = str(target.get("autogluon_metric") or target.get("metric") or "balanced_accuracy")
                profile_id = active_profile_id(config, "autogluon")
                profile = load_profile(project_dir, "autogluon", profile_id)
                profile.update(profile_overrides)

                train = pd.read_csv(_data_file("train"))
                test = pd.read_csv(_data_file("test"))
                sample = pd.read_csv(_data_file("sample_submission"))
                train_features = train.drop(columns=[target_col])
                combined = pd.concat([train_features, test], ignore_index=True, sort=False)

                print("AutoGluon materialization: starting feature groups", flush=True)
                preprocess_started_at = time.time()
                with _preprocess_timeout(int(profile.get("preprocess_timeout", 180))):
                    transformed = run_feature_groups(
                        combined.copy(),
                        FEATURE_GROUPS,
                        log_path=work_dir / "feature-groups.jsonl",
                    )
                preprocess_time = time.time() - preprocess_started_at
                print(
                    f"AutoGluon materialization: finished feature groups rows={{len(transformed)}} cols={{len(transformed.columns)}} elapsed={{preprocess_time:.3f}}s",
                    flush=True,
                )
                if len(transformed) != len(combined):
                    raise ValueError("feature-group preprocessing must preserve row count")

                train_out = transformed.iloc[: len(train)].reset_index(drop=True)
                test_out = transformed.iloc[len(train) :].reset_index(drop=True)
                if id_col in train_out.columns:
                    train_out = train_out.drop(columns=[id_col])
                    test_out = test_out.drop(columns=[id_col], errors="ignore")
                train_out[target_col] = train[target_col].reset_index(drop=True)

                predictor_args = dict(profile.get("predictor_args", {{}}) or {{}})
                predictor_args.setdefault("label", target_col)
                predictor_args.setdefault("eval_metric", metric)
                predictor_args.setdefault("path", work_dir / "AutoGluonModels")
                predictor = TabularPredictor(**predictor_args)

                fit_kwargs = _fit_kwargs_from_profile(profile, train_out)

                print("AutoGluon materialization: starting fit", flush=True)
                fit_started_at = time.time()
                predictor.fit(**fit_kwargs)
                training_time = time.time() - fit_started_at
                print(f"AutoGluon materialization: finished fit elapsed={{training_time:.3f}}s", flush=True)

                predictions = predictor.predict(test_out)
                submission = sample.copy()
                prediction_cols = [col for col in submission.columns if col != id_col]
                if not prediction_cols:
                    prediction_cols = [submission.columns[-1]]
                submission[prediction_cols[0]] = predictions.values
                submission_path = artifacts_dir / "submission.csv"
                submission.to_csv(submission_path, index=False)
                result = {{
                    "is_bug": False,
                    "summary": "AutoGluon materialization completed.",
                    "metric": None,
                    "maximize": True,
                    "lower_is_better": False,
                    "run_stats": {{
                        "feature_count": int(len(transformed.columns)),
                        "preprocess_time": float(preprocess_time),
                        "training_time": float(training_time),
                    }},
                }}
                print("TML_RESULT_JSON: " + json.dumps(result, sort_keys=True), flush=True)
    except Exception:
        traceback.print_exc()
        raise


main()
'''


def _legacy_wrapper_source(
    group_code: str,
    project_dir: Path,
    *,
    profile_overrides: dict[str, object] | None,
) -> str:
    _ = profile_overrides
    project_literal = repr(_project_relative_path(project_dir))
    return f'''# Generated legacy materialization.
# Feature-group code is followed by the fixed executable legacy wrapper.

{group_code.rstrip()}


def main():
    import json
    from dataclasses import asdict
    from pathlib import Path

    from tml.execution.executor import run_legacy_group_materialization

    project_relative_path = Path({project_literal})
    project_dir = _resolve_project_dir(project_relative_path)
    work_dir = Path.cwd()
    result = run_legacy_group_materialization(
        code_path=Path(__file__),
        project_dir=project_dir,
        work_dir=work_dir,
    )
    print("TML_RESULT_JSON: " + json.dumps(asdict(result), default=str, sort_keys=True))
    raise SystemExit(result.returncode)


def _resolve_project_dir(relative_path):
    for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (parent / "project.yaml").exists():
            return parent
        candidate = parent / relative_path
        if (candidate / "project.yaml").exists():
            return candidate
    for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        candidate = parent / relative_path
        if (candidate / "project.yaml").exists():
            return candidate
    import tml

    package_root = Path(tml.__file__).resolve().parents[2]
    candidate = package_root / relative_path
    if (candidate / "project.yaml").exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve project directory for {{relative_path}}")


main()
'''


def _project_relative_path(project_dir: Path) -> str:
    resolved = project_dir.resolve()
    root = repo_root_for_project(resolved).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def _normalized_profile_overrides(overrides: dict[str, object]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in {"mode", "id", "hypothesis"}:
            continue
        if key == "preset":
            normalized["presets"] = value
        elif key == "time":
            normalized["time_limit"] = value
        else:
            normalized[key] = value
    return normalized
