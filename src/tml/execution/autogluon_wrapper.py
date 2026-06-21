from __future__ import annotations

import contextlib
import importlib.util
import signal
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tml.core.config import active_profile_id, load_project_config
from tml.core.profiles import load_profile
from tml.features.groups import has_feature_groups, run_feature_groups
from tml.features.validation import validate_group_code_source
from tml.utils.atomic import atomic_write_text

from .result import ExecutionResult


CLASS_WEIGHT_COL = "__tml_sample_weight"

RESERVED_PROFILE_KEYS = {
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
}


@dataclass(frozen=True)
class TrainingPlan:
    train_data: Any
    valid_data: Any | None
    fit_args: dict[str, object]
    defer_save_space: bool


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
    if "FEATURE_GROUPS" not in source and "def preprocess" not in source:
        return ExecutionResult(
            status="failed",
            returncode=4,
            stdout="",
            stderr="",
            error="AutoGluon materialization must define FEATURE_GROUPS or preprocess(df).",
        )
    if "FEATURE_GROUPS" in source:
        validate_group_code_source(source)
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
    metric = str(target.get("autogluon_metric") or target.get("metric") or "balanced_accuracy")
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

    train_features = train.drop(columns=[target_col])
    combined = pd.concat([train_features, test], ignore_index=True, sort=False)
    with _preprocess_timeout(int(profile.get("preprocess_timeout", 180))):
        if has_feature_groups(module):
            transformed = run_feature_groups(
                combined.copy(),
                getattr(module, "FEATURE_GROUPS"),
                log_path=work_dir / "feature-groups.jsonl",
            )
        else:
            preprocess = getattr(module, "preprocess", None)
            if not callable(preprocess):
                raise ValueError("AutoGluon materialization must define FEATURE_GROUPS or callable preprocess(df)")
            transformed = preprocess(combined.copy())
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError("preprocessing must return a pandas DataFrame")
    if len(transformed) != len(combined):
        raise ValueError("preprocessing must preserve row count")

    train_out = transformed.iloc[: len(train)].reset_index(drop=True)
    test_out = transformed.iloc[len(train) :].reset_index(drop=True)
    train_out[target_col] = train[target_col].reset_index(drop=True)

    training_plan = _training_plan_from_profile(train_out, target_col, profile)
    predictor = TabularPredictor(
        **_predictor_kwargs_from_profile(
            label=target_col,
            eval_metric=metric,
            model_path=work_dir / "tml-autogluon-workdir" / "AutoGluonModels",
            profile=profile,
        )
    )
    fit_kwargs = _fit_kwargs_from_profile(
        profile,
        train_data=training_plan.train_data,
        valid_data=training_plan.valid_data,
        fit_args=training_plan.fit_args,
    )
    ignored_columns = list(fit_kwargs.pop("ignored_columns", []) or [])
    if id_col in train_out.columns and id_col not in ignored_columns:
        ignored_columns.append(id_col)
    if ignored_columns:
        fit_kwargs["ignored_columns"] = ignored_columns
    predictor.fit(**fit_kwargs)

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
    if training_plan.defer_save_space:
        try:
            predictor.save_space(remove_data=True, remove_fit_stack=True)
        except Exception:
            pass
    if "score_val" in leaderboard.columns and not leaderboard.empty:
        value = leaderboard.iloc[0]["score_val"]
        return float(value) if pd.notna(value) else None
    return None


def _predictor_kwargs_from_profile(
    *,
    label: str,
    eval_metric: str,
    model_path: Path,
    profile: dict[str, object],
) -> dict[str, object]:
    predictor_kwargs: dict[str, object] = {
        "label": label,
        "eval_metric": eval_metric,
        "path": str(model_path),
    }
    predictor_args = profile.get("predictor_args")
    if isinstance(predictor_args, dict):
        predictor_kwargs.update(predictor_args)
    if profile.get("class_balance") == "balanced":
        predictor_kwargs["sample_weight"] = CLASS_WEIGHT_COL
        predictor_kwargs["weight_evaluation"] = False
    return predictor_kwargs


def _load_profile(project_dir: Path, profile_id: str) -> dict[str, object]:
    return load_profile(project_dir, "autogluon", profile_id)


def _training_plan_from_profile(train_model, target_col: str, profile: dict[str, object]) -> TrainingPlan:
    train_model = train_model.copy()
    if profile.get("class_balance") == "balanced":
        train_model[CLASS_WEIGHT_COL] = _balanced_sample_weight(train_model[target_col])

    fit_args = dict(profile.get("fit_args") or {}) if isinstance(profile.get("fit_args"), dict) else {}
    bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
    defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
    if bagged_mode:
        return TrainingPlan(train_data=train_model, valid_data=None, fit_args=fit_args, defer_save_space=defer_save_space)

    if profile.get("validation_strategy") == "holdout":
        from sklearn.model_selection import train_test_split

        stratify = train_model[target_col] if _should_stratify_holdout(train_model[target_col]) else None
        train_data, valid_data = train_test_split(
            train_model,
            test_size=float(profile.get("validation_fraction", 0.2)),
            random_state=int(profile.get("seed", 42)),
            stratify=stratify,
        )
        return TrainingPlan(
            train_data=train_data,
            valid_data=valid_data,
            fit_args=fit_args,
            defer_save_space=defer_save_space,
        )

    return TrainingPlan(train_data=train_model, valid_data=None, fit_args=fit_args, defer_save_space=defer_save_space)


def _fit_kwargs_from_profile(
    profile: dict[str, object],
    *,
    train_data: Any | None = None,
    valid_data: Any | None = None,
    fit_args: dict[str, object] | None = None,
) -> dict[str, object]:
    fit_kwargs: dict[str, object] = {}
    if train_data is not None:
        fit_kwargs["train_data"] = train_data
    if valid_data is not None:
        fit_kwargs["tuning_data"] = valid_data

    for key, value in profile.items():
        if key in RESERVED_PROFILE_KEYS or value is None:
            continue
        fit_kwargs[key] = value

    if profile.get("use_gpu") is not None:
        fit_kwargs["num_gpus"] = 1 if profile.get("use_gpu") else 0
    if fit_args is None:
        raw_fit_args = profile.get("fit_args")
        fit_args = dict(raw_fit_args) if isinstance(raw_fit_args, dict) else {}
    fit_kwargs.update(fit_args)
    return fit_kwargs


def _balanced_sample_weight(labels):
    import pandas as pd

    labels = pd.Series(labels).reset_index(drop=True)
    counts = labels.value_counts(dropna=False)
    if counts.empty:
        raise ValueError("Cannot compute class weights for empty labels")
    weights_by_class = len(labels) / (len(counts) * counts.astype(float))
    return labels.map(weights_by_class).astype(float).to_numpy()


def _should_stratify_holdout(target) -> bool:
    import pandas as pd

    unique_count = target.nunique(dropna=True)
    if unique_count == 2:
        return True
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return True
    return False


@contextlib.contextmanager
def _preprocess_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_preprocess_timeout(_signum, _frame):
        raise TimeoutError(
            "AutoGluon preprocess exceeded the dedicated timeout of "
            f"{seconds} seconds. This timeout is separate from AutoGluon training time_limit."
        )

    previous = signal.signal(signal.SIGALRM, _raise_preprocess_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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
