from __future__ import annotations

import contextlib
import importlib.util
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tml.core.config import active_profile_id, load_project_config, project_preprocess_timeout, repo_root_for_project
from tml.core.paths import context_path
from tml.core.profiles import load_profile
from tml.features.groups import has_feature_groups, run_feature_groups
from tml.features.validation import validate_group_code_source
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import read_yaml

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
    "ignored_columns",
    "aux_file",
    "auxiliary_file",
    "selection_score",
}


@dataclass(frozen=True)
class TrainingPlan:
    train_data: Any
    valid_data: Any | None
    audit_data: Any | None
    fit_args: dict[str, object]
    defer_save_space: bool
    split_stats: dict[str, object]


def run_autogluon_materialization(
    *,
    code_path: Path,
    project_dir: Path,
    work_dir: Path,
    profile_id: str | None = None,
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
        metric = _run_tabular(code_path=code_path, project_dir=project_dir, work_dir=work_dir, profile_id=profile_id)
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


def _run_tabular(*, code_path: Path, project_dir: Path, work_dir: Path, profile_id: str | None = None) -> float | None:
    import importlib.util

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    _patch_xgboost_cpu_inference()

    config = load_project_config(project_dir)
    target = config.get("target", {}) if isinstance(config.get("target"), dict) else {}
    target_col = str(target.get("target_column") or "target")
    id_col = str(target.get("id_column") or "id")
    metric = str(target.get("autogluon_metric") or target.get("metric") or "balanced_accuracy")
    resolved_profile_id = profile_id or active_profile_id(config, "autogluon")
    profile = _load_profile(project_dir, resolved_profile_id)
    preprocess_timeout = project_preprocess_timeout(config)
    runtime_options = _autogluon_runtime_options(project_dir)
    _print_autogluon_runtime_options(runtime_options)

    data_dir = project_dir / "data"
    train = pd.read_csv(_data_file(data_dir, "train.csv"))
    test = pd.read_csv(_data_file(data_dir, "test.csv"))
    sample = pd.read_csv(_data_file(data_dir, "sample_submission.csv"))
    aux_df = _read_aux_csv(config, profile, project_dir=project_dir, data_dir=data_dir)
    if target_col not in train.columns:
        raise ValueError(f"Target column {target_col!r} not found in train.csv")

    module_spec = importlib.util.spec_from_file_location("tml_materialization", code_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot import materialization: {_project_path(project_dir, code_path)}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    train_features = train.drop(columns=[target_col])
    combined = pd.concat([train_features, test], ignore_index=True, sort=False)
    with _preprocess_timeout(preprocess_timeout):
        if has_feature_groups(module):
            transformed = run_feature_groups(
                combined.copy(),
                getattr(module, "FEATURE_GROUPS"),
                aux=aux_df,
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
    ignored_columns = _ignored_columns_from_profile(profile, id_col=id_col, columns=train_out.columns)
    train_out[target_col] = train[target_col].reset_index(drop=True)

    training_plan = _training_plan_from_profile(
        train_out,
        target_col,
        profile,
        audit_config=runtime_options["audit_score"],
    )
    predictor = TabularPredictor(
        **_predictor_kwargs_from_profile(
            label=target_col,
            eval_metric=metric,
            model_path=work_dir / "tml-autogluon-workdir" / "AutoGluonModels",
            profile=profile,
            ignored_columns=ignored_columns,
        )
    )
    fit_kwargs = _fit_kwargs_from_profile(
        profile,
        train_data=training_plan.train_data,
        valid_data=training_plan.valid_data,
        fit_args=training_plan.fit_args,
    )
    predictor.fit(**fit_kwargs)

    leaderboard = predictor.leaderboard(data=training_plan.audit_data, silent=True)
    score_columns = ("score_test",) if training_plan.audit_data is not None else ("score_val", "score_test")
    selected_model = _best_model_from_leaderboard(
        leaderboard,
        score_column="score_test" if training_plan.audit_data is not None else "score_val",
    )
    predictions = predictor.predict(test_out, model=selected_model)
    submission = sample.copy()
    prediction_cols = [col for col in submission.columns if col != id_col]
    if not prediction_cols:
        prediction_cols = [submission.columns[-1]]
    submission[prediction_cols[0]] = predictions.values
    artifacts = work_dir.parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    submission.to_csv(artifacts / "submission.csv.gz", index=False, compression="gzip")

    leaderboard.to_csv(artifacts / "leaderboard.csv.gz", index=False, compression="gzip")
    if training_plan.audit_data is not None:
        audit_features = training_plan.audit_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore")
        audit_predictions = predictor.predict(audit_features, model=selected_model)
        audit_frame = pd.DataFrame(
            {
                "row": range(len(training_plan.audit_data)),
                "target": training_plan.audit_data[target_col].reset_index(drop=True),
                "prediction": pd.Series(audit_predictions).reset_index(drop=True),
            }
        )
        audit_frame.to_csv(artifacts / "audit_predictions.csv.gz", index=False, compression="gzip")
    _maybe_write_feature_importance(
        predictor,
        audit_data=training_plan.audit_data,
        valid_data=training_plan.valid_data,
        group_features=[column for column in transformed.columns if str(column).startswith("G")],
        config=runtime_options["feature_importance"],
        artifacts_dir=artifacts,
    )
    if training_plan.defer_save_space:
        try:
            predictor.save_space(remove_data=True, remove_fit_stack=True)
        except Exception:
            pass
    value = _metric_from_leaderboard(leaderboard, score_columns=score_columns)
    if training_plan.audit_data is not None and value is None:
        raise ValueError("Audit score is enabled but AutoGluon leaderboard did not produce score_test")
    if value is not None:
        return value
    return None


def _patch_xgboost_cpu_inference() -> None:
    try:
        from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
    except Exception:
        return

    if getattr(XGBoostModel, "_tml_cpu_inference_patch", False):
        return

    original_predict_proba = XGBoostModel._predict_proba

    def _predict_proba_with_cpu_device(self, X, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is not None:
            try:
                if model.get_params().get("device") != "cpu":
                    model.set_params(device="cpu")
                model.get_booster().set_param({"device": "cpu"})
            except Exception:
                pass
        return original_predict_proba(self, X, *args, **kwargs)

    XGBoostModel._predict_proba = _predict_proba_with_cpu_device
    XGBoostModel._tml_cpu_inference_patch = True


def _predictor_kwargs_from_profile(
    *,
    label: str,
    eval_metric: str,
    model_path: Path,
    profile: dict[str, object],
    ignored_columns: list[str] | None = None,
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
    if ignored_columns:
        learner_kwargs = dict(predictor_kwargs.get("learner_kwargs") or {})
        learner_kwargs["ignored_columns"] = list(ignored_columns)
        predictor_kwargs["learner_kwargs"] = learner_kwargs
    return predictor_kwargs


def _load_profile(project_dir: Path, profile_id: str) -> dict[str, object]:
    return load_profile(project_dir, "autogluon", profile_id)


def _read_aux_csv(
    config: dict[str, Any],
    profile: dict[str, object],
    *,
    project_dir: Path,
    data_dir: Path,
) -> Any | None:
    aux_file = None
    if _external_enabled(project_dir):
        aux_file = _project_external_file(config)
    aux_file = aux_file or profile.get("aux_file") or profile.get("auxiliary_file")
    if not aux_file:
        return None
    path = Path(str(aux_file))
    if not path.is_absolute():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Configured aux file not found: {path}")
    import pandas as pd

    return pd.read_csv(path)


def _external_enabled(project_dir: Path) -> bool:
    try:
        root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    except Exception:
        root_config = {}
    external = root_config.get("external") if isinstance(root_config.get("external"), dict) else {}
    return bool(external.get("enabled", False))


def _project_external_file(config: dict[str, Any]) -> object | None:
    external = config.get("external") if isinstance(config.get("external"), dict) else {}
    return external.get("file") or external.get("path") or external.get("aux")


def _autogluon_runtime_options(project_dir: Path) -> dict[str, dict[str, object]]:
    try:
        root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    except Exception:
        root_config = {}
    autogluon = root_config.get("autogluon") if isinstance(root_config.get("autogluon"), dict) else {}
    audit = autogluon.get("audit_score") if isinstance(autogluon.get("audit_score"), dict) else {}
    feature_importance = (
        autogluon.get("feature_importance")
        if isinstance(autogluon.get("feature_importance"), dict)
        else {}
    )
    return {
        "audit_score": {
            "enabled": _bool_option(audit.get("enabled"), default=False),
            "fraction": _fraction_option(audit.get("fraction", 0.1), "autogluon.audit_score.fraction"),
        },
        "feature_importance": {
            "enabled": _bool_option(feature_importance.get("enabled"), default=False),
            "subsample_size": _positive_float_option(
                feature_importance.get("subsample_size", 0.1),
                "autogluon.feature_importance.subsample_size",
            ),
            "num_shuffle_sets": _positive_int_option(
                feature_importance.get("num_shuffle_sets", 10),
                "autogluon.feature_importance.num_shuffle_sets",
            ),
            "include_confidence_band": _bool_option(
                feature_importance.get("include_confidence_band"),
                default=True,
            ),
        },
    }


def _print_autogluon_runtime_options(options: dict[str, dict[str, object]]) -> None:
    audit = options["audit_score"]
    feature_importance = options["feature_importance"]
    print(
        "TML_RUNTIME|autogluon_options"
        f"|audit_score_enabled={bool(audit.get('enabled'))}"
        f"|audit_fraction={audit.get('fraction')}"
        f"|feature_importance_enabled={bool(feature_importance.get('enabled'))}"
        f"|fi_subsample_size={feature_importance.get('subsample_size')}"
        f"|fi_num_shuffle_sets={feature_importance.get('num_shuffle_sets')}"
        f"|fi_include_confidence_band={bool(feature_importance.get('include_confidence_band'))}",
        flush=True,
    )


def _bool_option(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean option value: {value!r}")


def _fraction_option(value: object, name: str) -> float:
    fraction = _positive_float_option(value, name)
    if fraction >= 1:
        raise ValueError(f"{name} must be greater than 0 and less than 1")
    return fraction


def _positive_float_option(value: object, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return numeric


def _positive_int_option(value: object, name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return numeric


def _training_plan_from_profile(
    train_model,
    target_col: str,
    profile: dict[str, object],
    *,
    audit_config: dict[str, object],
) -> TrainingPlan:
    train_model = train_model.copy()
    if profile.get("class_balance") == "balanced":
        train_model[CLASS_WEIGHT_COL] = _balanced_sample_weight(train_model[target_col])

    fit_args = dict(profile.get("fit_args") or {}) if isinstance(profile.get("fit_args"), dict) else {}
    bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
    defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
    seed = int(profile.get("seed", 42))
    audit_enabled = bool(audit_config.get("enabled"))
    audit_fraction = float(audit_config.get("fraction", 0.1)) if audit_enabled else 0.0
    audit_data = None
    split_stats: dict[str, object] = {
        "audit_enabled": audit_enabled,
        "audit_fraction": audit_fraction if audit_enabled else None,
        "validation_strategy": profile.get("validation_strategy"),
        "bagged_mode": bagged_mode,
    }
    if audit_enabled:
        train_model, audit_data = _split_frame(
            train_model,
            target_col,
            test_size=audit_fraction,
            seed=seed,
        )
        split_stats["audit_rows"] = int(len(audit_data))

    if bagged_mode:
        split_stats["train_rows"] = int(len(train_model))
        return TrainingPlan(
            train_data=train_model,
            valid_data=None,
            audit_data=audit_data,
            fit_args=fit_args,
            defer_save_space=defer_save_space,
            split_stats=split_stats,
        )

    if profile.get("validation_strategy") == "holdout":
        validation_fraction = float(
            profile.get("validation_fraction", 0.1 if audit_enabled else 0.2)
        )
        if audit_enabled and audit_fraction + validation_fraction >= 1:
            raise ValueError("audit_score.fraction + validation_fraction must be < 1")
        holdout_test_size = validation_fraction / (1.0 - audit_fraction) if audit_enabled else validation_fraction
        train_data, valid_data = _split_frame(
            train_model,
            target_col,
            test_size=holdout_test_size,
            seed=seed,
        )
        split_stats.update(
            {
                "validation_fraction": validation_fraction,
                "validation_rows": int(len(valid_data)),
                "train_rows": int(len(train_data)),
            }
        )
        return TrainingPlan(
            train_data=train_data,
            valid_data=valid_data,
            audit_data=audit_data,
            fit_args=fit_args,
            defer_save_space=defer_save_space,
            split_stats=split_stats,
        )

    split_stats["train_rows"] = int(len(train_model))
    return TrainingPlan(
        train_data=train_model,
        valid_data=None,
        audit_data=audit_data,
        fit_args=fit_args,
        defer_save_space=defer_save_space,
        split_stats=split_stats,
    )


def _split_frame(frame, target_col: str, *, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    if not 0 < float(test_size) < 1:
        raise ValueError(f"split fraction must be between 0 and 1, got {test_size!r}")
    stratify = frame[target_col] if _should_stratify_holdout(frame[target_col]) else None
    train_data, test_data = train_test_split(
        frame,
        test_size=float(test_size),
        random_state=seed,
        stratify=stratify,
    )
    return train_data, test_data


def _metric_from_leaderboard(leaderboard, *, score_columns: tuple[str, ...]) -> float | None:
    for column in score_columns:
        if column in leaderboard.columns and not leaderboard.empty:
            values = leaderboard[column].dropna()
            if not values.empty:
                return float(values.max())
    return None


def _best_model_from_leaderboard(leaderboard, *, score_column: str) -> str | None:
    if score_column not in leaderboard.columns or "model" not in leaderboard.columns or leaderboard.empty:
        return None
    scored = leaderboard.dropna(subset=[score_column])
    if scored.empty:
        return None
    return str(scored.sort_values(score_column, ascending=False).iloc[0]["model"])


def _maybe_write_feature_importance(
    predictor,
    *,
    audit_data,
    valid_data,
    group_features: list[object],
    config: dict[str, object],
    artifacts_dir: Path,
) -> None:
    if not bool(config.get("enabled")):
        return
    source_data = audit_data if audit_data is not None else valid_data
    if source_data is None:
        return
    available_features = set(predictor.feature_metadata_in.get_features())
    fi_group_features = [feature for feature in group_features if feature in available_features]
    if not fi_group_features:
        return
    subsample_size = _feature_importance_subsample_size(config.get("subsample_size", 0.1), len(source_data))
    num_shuffle_sets = int(config.get("num_shuffle_sets", 10))
    include_confidence_band = bool(config.get("include_confidence_band", True))
    print(
        "TML_RUNTIME|event=start|stage=feature_importance"
        f"|source={'audit' if audit_data is not None else 'validation'}"
        f"|features={len(fi_group_features)}"
        f"|subsample_size={subsample_size}"
        f"|configured_subsample_size={config.get('subsample_size', 0.1)}"
        f"|num_shuffle_sets={num_shuffle_sets}"
        f"|include_confidence_band={include_confidence_band}",
        flush=True,
    )
    started_at = time.perf_counter()
    fi = predictor.feature_importance(
        data=source_data,
        features=[("TML_GROUP_FEATURES_ALL", fi_group_features), *fi_group_features],
        subsample_size=subsample_size,
        num_shuffle_sets=num_shuffle_sets,
        include_confidence_band=include_confidence_band,
    )
    elapsed = time.perf_counter() - started_at
    fi["elapsed_s"] = elapsed
    fi.to_csv(artifacts_dir / "feature_importance.csv.gz", index=True, compression="gzip")
    print(f"TML_RUNTIME|event=end|stage=feature_importance|elapsed_s={elapsed:.3f}", flush=True)


def _feature_importance_subsample_size(value: object, row_count: int) -> int:
    numeric = _positive_float_option(value, "autogluon.feature_importance.subsample_size")
    if numeric <= 1:
        return max(1, int(round(row_count * numeric)))
    return int(numeric)


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


def _ignored_columns_from_profile(profile: dict[str, object], *, id_col: str, columns) -> list[str]:
    raw_ignored = profile.get("ignored_columns")
    if raw_ignored is None:
        ignored = [id_col]
    elif isinstance(raw_ignored, str):
        ignored = [raw_ignored]
    else:
        ignored = list(raw_ignored)
    available = set(columns)
    return [str(column) for column in ignored if str(column) in available]


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
