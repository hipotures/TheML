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
    import logging
    import signal
    import shutil
    import sys
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    from tml.core.config import active_profile_id, load_project_config
    from tml.core.profiles import load_profile
    from tml.features.groups import run_feature_groups

    class_weight_col = "__tml_sample_weight"

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
        "aux_file",
        "auxiliary_file",
    }}

    class _TeeWriter:
        def __init__(self, primary, log_file):
            self.primary = primary
            self.log_file = log_file

        def write(self, text):
            written = self.primary.write(text)
            self.log_file.write(text)
            self.flush()
            return written

        def flush(self):
            if hasattr(self.primary, "flush"):
                self.primary.flush()
            self.log_file.flush()

        def fileno(self):
            if hasattr(self.primary, "fileno"):
                return self.primary.fileno()
            fallback = getattr(sys, "__stderr__", None) or getattr(sys, "__stdout__", None)
            if fallback is not None and hasattr(fallback, "fileno"):
                return fallback.fileno()
            return self.log_file.fileno()

        def isatty(self):
            return bool(self.primary.isatty()) if hasattr(self.primary, "isatty") else False

        @property
        def encoding(self):
            return getattr(self.primary, "encoding", "utf-8")

    def _data_file(stem):
        gz = data_dir / f"{{stem}}.csv.gz"
        if gz.exists():
            return gz
        return data_dir / f"{{stem}}.csv"

    def _read_aux_csv(profile):
        aux_file = profile.get("aux_file") or profile.get("auxiliary_file")
        if not aux_file:
            return None
        path = data_dir / str(aux_file)
        if not path.exists():
            raise FileNotFoundError(f"Configured aux file not found: {{path}}")
        return pd.read_csv(path)

    def _force_autogluon_cpu_resources(profile):
        if profile.get("use_gpu") is not False:
            return
        try:
            from autogluon.common.utils.resource_utils import ResourceManager
        except Exception:
            return
        ResourceManager.get_gpu_count = staticmethod(lambda: 0)
        ResourceManager.get_gpu_count_torch = staticmethod(lambda cuda_only=False: 0)

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

    @contextlib.contextmanager
    def _tee_model_output():
        with target_log.open("a", encoding="utf-8", buffering=1) as log_file:
            stdout_writer = _TeeWriter(sys.stdout, log_file)
            stderr_writer = _TeeWriter(sys.stderr, log_file)
            log_handler = logging.StreamHandler(stderr_writer)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            loggers = [logging.getLogger(name) for name in ["", "autogluon"]]
            previous_states = [
                (logger.level, logger.disabled, list(logger.handlers), logger.propagate)
                for logger in loggers
            ]
            for logger in loggers:
                logger.disabled = False
                logger.setLevel(logging.INFO)
                logger.handlers = [log_handler]
                logger.propagate = False
            try:
                with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                    yield
            finally:
                for logger, (level, disabled, handlers, propagate) in zip(loggers, previous_states):
                    logger.setLevel(level)
                    logger.disabled = disabled
                    logger.handlers = handlers
                    logger.propagate = propagate

    def _json_safe_scalar(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, str | int | float | bool):
            return value
        return str(value)

    def _leaderboard_records(predictor):
        keep_columns = [
            "model",
            "score_val",
            "score_test",
            "eval_metric",
            "fit_time",
            "fit_time_marginal",
            "pred_time_val",
            "pred_time_val_marginal",
            "stack_level",
            "can_infer",
            "fit_order",
        ]
        try:
            leaderboard = predictor.leaderboard(silent=True)
        except Exception as exc:
            return [{{"error": f"leaderboard unavailable: {{type(exc).__name__}}: {{exc}}"}}]
        return [
            {{column: _json_safe_scalar(row.get(column)) for column in keep_columns if column in row}}
            for row in leaderboard.to_dict(orient="records")
        ]

    def _metric_from_leaderboard(predictor):
        try:
            leaderboard = predictor.leaderboard(silent=True)
        except Exception:
            return None
        for column in ("score_val", "score_test"):
            if column in leaderboard.columns and not leaderboard.empty:
                value = leaderboard[column].max()
                return float(value) if pd.notna(value) else None
        return None

    def _prediction_from_proba(proba):
        if isinstance(proba, pd.Series):
            return proba.reset_index(drop=True)
        return proba.idxmax(axis=1).reset_index(drop=True)

    def _positive_probability_from_proba(proba):
        if isinstance(proba, pd.Series):
            return proba.reset_index(drop=True)
        for positive_class in (1, 1.0, "1", "1.0", True):
            if positive_class in proba.columns:
                return proba[positive_class].reset_index(drop=True)
        return proba.iloc[:, -1].reset_index(drop=True)

    def _values_from_proba(proba, eval_metric):
        if eval_metric == "roc_auc":
            return _positive_probability_from_proba(proba)
        return _prediction_from_proba(proba)

    def _predict_values(predictor, data, eval_metric, model=None):
        if eval_metric == "roc_auc":
            proba = predictor.predict_proba(data, model=model)
            return _positive_probability_from_proba(proba)
        return pd.Series(predictor.predict(data, model=model)).reset_index(drop=True)

    def _safe_prediction_name(name):
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
        return safe or "model"

    def _save_prediction_artifact(frame, filename):
        if not filename.endswith(".gz"):
            filename = f"{{filename}}.gz"
        path = artifacts_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, compression="gzip")
        return path

    def _save_autogluon_prediction_artifacts(predictor, train_target, test_model, test_ids, test_pred, eval_metric, id_col, target_col, valid_data, valid_pred):
        artifacts = {{}}
        test_predictions = pd.DataFrame({{
            id_col: pd.Series(test_ids).reset_index(drop=True),
            target_col: pd.Series(test_pred).reset_index(drop=True),
        }})
        test_path = _save_prediction_artifact(test_predictions, "test_predictions.csv")
        artifacts["test_predictions"] = str(test_path)

        try:
            per_model = []
            for model_name in predictor.model_names():
                try:
                    model_oof_proba = predictor.predict_proba_oof(
                        model=model_name,
                        transformed=False,
                        as_multiclass=True,
                    )
                    model_oof_pred = _values_from_proba(model_oof_proba, eval_metric)
                    if len(model_oof_pred) != len(train_target):
                        raise ValueError(f"OOF row count {{len(model_oof_pred)}} != train rows {{len(train_target)}}")
                    safe_name = _safe_prediction_name(model_name)
                    model_oof_frame = pd.DataFrame({{
                        "row": np.arange(len(train_target)),
                        "target": pd.Series(train_target).reset_index(drop=True),
                        "prediction": model_oof_pred.reset_index(drop=True),
                        "model": model_name,
                    }})
                    model_oof_path = _save_prediction_artifact(
                        model_oof_frame,
                        f"model_predictions/{{safe_name}}-oof.csv",
                    )
                    model_test_pred = _predict_values(
                        predictor,
                        test_model,
                        eval_metric,
                        model=model_name,
                    )
                    model_test_frame = pd.DataFrame({{
                        id_col: pd.Series(test_ids).reset_index(drop=True),
                        target_col: pd.Series(model_test_pred).reset_index(drop=True),
                        "model": model_name,
                    }})
                    model_test_path = _save_prediction_artifact(
                        model_test_frame,
                        f"model_predictions/{{safe_name}}-test.csv",
                    )
                    per_model.append({{
                        "model": model_name,
                        "oof_predictions": str(model_oof_path),
                        "test_predictions": str(model_test_path),
                        "rows": int(len(model_oof_frame)),
                        "test_rows": int(len(model_test_frame)),
                    }})
                except Exception as exc:
                    per_model.append({{"model": model_name, "error": f"{{type(exc).__name__}}: {{exc}}"}})
            artifacts["model_predictions"] = per_model
            artifacts["model_predictions_ok"] = sum(1 for item in per_model if "error" not in item)

            oof_proba = predictor.predict_proba_oof(transformed=False, as_multiclass=True)
            oof_pred = _values_from_proba(oof_proba, eval_metric)
            if len(oof_pred) != len(train_target):
                raise ValueError(f"OOF row count {{len(oof_pred)}} != train rows {{len(train_target)}}")
            oof_frame = pd.DataFrame({{
                "row": np.arange(len(train_target)),
                "target": pd.Series(train_target).reset_index(drop=True),
                "prediction": oof_pred.reset_index(drop=True),
            }})
            oof_path = _save_prediction_artifact(oof_frame, "oof_predictions.csv")
            artifacts["oof_predictions"] = str(oof_path)
            artifacts["oof_rows"] = int(len(oof_frame))
        except Exception as exc:
            artifacts["oof_error"] = f"{{type(exc).__name__}}: {{exc}}"

        if valid_data is not None and valid_pred is not None:
            validation_frame = pd.DataFrame({{
                "row": np.arange(len(valid_data)),
                "target": valid_data[target_col].reset_index(drop=True),
                "prediction": pd.Series(valid_pred).reset_index(drop=True),
            }})
            validation_path = _save_prediction_artifact(validation_frame, "validation_predictions.csv")
            artifacts["validation_predictions"] = str(validation_path)
            artifacts["validation_rows"] = int(len(validation_frame))
        return artifacts

    def _make_submission(sample, test_ids, test_pred, id_col, target_col):
        prediction_frame = pd.DataFrame({{
            id_col: pd.Series(test_ids).reset_index(drop=True),
            target_col: pd.Series(test_pred).reset_index(drop=True),
        }})
        if prediction_frame[id_col].duplicated().any():
            raise ValueError(f"test data contains duplicate {{id_col}} values")
        submission = sample.copy()
        if target_col in submission.columns:
            mapped = submission[[id_col]].merge(
                prediction_frame,
                on=id_col,
                how="left",
                validate="one_to_one",
            )[target_col]
            if mapped.isna().any():
                raise ValueError(f"missing predictions for {{int(mapped.isna().sum())}} sample_submission ids")
            submission[target_col] = mapped.to_numpy()
            return submission.sort_values(id_col, kind="mergesort").reset_index(drop=True)
        prediction_cols = [column for column in submission.columns if column != id_col]
        if not prediction_cols:
            prediction_cols = [submission.columns[-1]]
        submission[prediction_cols[0]] = pd.Series(test_pred).reset_index(drop=True).to_numpy()
        return submission

    def _should_stratify_holdout(target):
        unique_count = target.nunique(dropna=True)
        if unique_count == 2:
            return True
        if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
            return True
        return False

    def _balanced_sample_weight(labels):
        labels = pd.Series(labels).reset_index(drop=True)
        counts = labels.value_counts(dropna=False)
        if counts.empty:
            raise ValueError("Cannot compute class weights for empty labels")
        weights_by_class = len(labels) / (len(counts) * counts.astype(float))
        return labels.map(weights_by_class).astype(float).to_numpy()

    def _training_plan_from_profile(train_model, target_col, profile):
        train_model = train_model.copy()
        if profile.get("class_balance") == "balanced":
            train_model[class_weight_col] = _balanced_sample_weight(train_model[target_col])

        fit_args = dict(profile.get("fit_args") or {{}}) if isinstance(profile.get("fit_args"), dict) else {{}}
        bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
        defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
        if bagged_mode:
            print("AutoGluon materialization: bagged mode detected; using internal OOF validation without tuning_data", flush=True)
            return train_model, None, fit_args, defer_save_space

        if profile.get("validation_strategy") == "holdout":
            from sklearn.model_selection import train_test_split

            stratify = train_model[target_col] if _should_stratify_holdout(train_model[target_col]) else None
            train_data, valid_data = train_test_split(
                train_model,
                test_size=float(profile.get("validation_fraction", 0.2)),
                random_state=int(profile.get("seed", 42)),
                stratify=stratify,
            )
            print(
                f"AutoGluon materialization: holdout validation rows={{len(valid_data)}} train_rows={{len(train_data)}}",
                flush=True,
            )
            return train_data, valid_data, fit_args, defer_save_space

        return train_model, None, fit_args, defer_save_space

    def _fit_kwargs_from_profile(profile, train_data, valid_data, fit_args):
        fit_kwargs = {{"train_data": train_data}}
        if valid_data is not None:
            fit_kwargs["tuning_data"] = valid_data
        for key, value in profile.items():
            if key in reserved_profile_keys or value is None:
                continue
            fit_kwargs[key] = value
        if profile.get("use_gpu") is not None:
            fit_kwargs["num_gpus"] = 1 if profile.get("use_gpu") else 0
        fit_kwargs.update(fit_args)
        return fit_kwargs

    try:
        with _tee_model_output():
                config = load_project_config(project_dir)
                target = config.get("target", {{}}) if isinstance(config.get("target"), dict) else {{}}
                target_col = str(target.get("target_column") or "target")
                id_col = str(target.get("id_column") or "id")
                metric = str(target.get("autogluon_metric") or target.get("metric") or "balanced_accuracy")
                profile_id = active_profile_id(config, "autogluon")
                profile = load_profile(project_dir, "autogluon", profile_id)
                profile.update(profile_overrides)
                _force_autogluon_cpu_resources(profile)

                train = pd.read_csv(_data_file("train"))
                test = pd.read_csv(_data_file("test"))
                sample = pd.read_csv(_data_file("sample_submission"))
                aux_df = _read_aux_csv(profile)
                if aux_df is not None:
                    aux_file = profile.get("aux_file") or profile.get("auxiliary_file")
                    print(
                        f"AutoGluon materialization: loaded aux file {{aux_file}} rows={{len(aux_df)}} cols={{len(aux_df.columns)}} passed_to_groups=True",
                        flush=True,
                    )
                train_features = train.drop(columns=[target_col])
                combined = pd.concat([train_features, test], ignore_index=True, sort=False)

                print("AutoGluon materialization: starting feature groups", flush=True)
                preprocess_started_at = time.time()
                with _preprocess_timeout(int(profile.get("preprocess_timeout", 180))):
                    transformed = run_feature_groups(
                        combined.copy(),
                        FEATURE_GROUPS,
                        aux=aux_df,
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
                train_target = train[target_col].reset_index(drop=True)
                train_data, valid_data, fit_args, defer_save_space = _training_plan_from_profile(
                    train_out,
                    target_col,
                    profile,
                )

                predictor_args = dict(profile.get("predictor_args", {{}}) or {{}})
                predictor_args.setdefault("label", target_col)
                predictor_args.setdefault("eval_metric", metric)
                predictor_args.setdefault("path", str(work_dir / "AutoGluonModels"))
                predictor_args.setdefault("verbosity", 2)
                if profile.get("class_balance") == "balanced":
                    predictor_args["sample_weight"] = class_weight_col
                    predictor_args["weight_evaluation"] = False
                shutil.rmtree(Path(predictor_args["path"]), ignore_errors=True)
                predictor = TabularPredictor(**predictor_args)

                fit_kwargs = _fit_kwargs_from_profile(profile, train_data, valid_data, fit_args)

                print("AutoGluon materialization: starting fit", flush=True)
                fit_started_at = time.time()
                predictor.fit(**fit_kwargs)
                training_time = time.time() - fit_started_at
                print(f"AutoGluon materialization: finished fit elapsed={{training_time:.3f}}s", flush=True)

                print("AutoGluon materialization: starting validation and prediction", flush=True)
                valid_pred = None
                metric_value = _metric_from_leaderboard(predictor)
                lower_is_better = False
                if valid_data is not None:
                    valid_features = valid_data.drop(columns=[target_col, class_weight_col], errors="ignore")
                    valid_pred = _predict_values(predictor, valid_features, metric)
                    scores = predictor.evaluate(valid_data, silent=True)
                    if metric in scores:
                        metric_value = float(scores[metric])
                predictions = _predict_values(predictor, test_out, metric)
                prediction_artifacts = _save_autogluon_prediction_artifacts(
                    predictor,
                    train_target,
                    test_out,
                    test[id_col],
                    predictions,
                    metric,
                    id_col,
                    target_col,
                    valid_data,
                    valid_pred,
                )
                if defer_save_space:
                    try:
                        predictor.save_space(remove_data=True, remove_fit_stack=True)
                        prediction_artifacts["save_space_after_artifacts"] = True
                    except Exception as exc:
                        prediction_artifacts["save_space_error"] = f"{{type(exc).__name__}}: {{exc}}"
                submission = _make_submission(sample, test[id_col], predictions, id_col, target_col)
                submission_path = artifacts_dir / "submission.csv"
                submission.to_csv(submission_path, index=False)
                print("AutoGluon materialization: finished validation and prediction", flush=True)
                print(f"AutoGluon materialization: submission saved to {{submission_path}}", flush=True)
                if prediction_artifacts.get("oof_predictions"):
                    print(f"AutoGluon materialization: OOF predictions saved to {{prediction_artifacts['oof_predictions']}}", flush=True)
                elif prediction_artifacts.get("oof_error"):
                    print(f"AutoGluon materialization: OOF predictions unavailable: {{prediction_artifacts['oof_error']}}", flush=True)
                if prediction_artifacts.get("validation_predictions"):
                    print(f"AutoGluon materialization: validation predictions saved to {{prediction_artifacts['validation_predictions']}}", flush=True)
                print(f"AutoGluon materialization: test predictions saved to {{prediction_artifacts['test_predictions']}}", flush=True)
                if metric_value is not None:
                    print(f"Validation {{metric}}: {{metric_value:.6f}}", flush=True)
                print("Submission saved successfully.", flush=True)
                result = {{
                    "is_bug": False,
                    "summary": "AutoGluon materialization completed.",
                    "metric": metric_value,
                    "eval_metric": str(getattr(predictor.eval_metric, "name", predictor.eval_metric)),
                    "maximize": True,
                    "lower_is_better": lower_is_better,
                    "run_stats": {{
                        "feature_count": int(len(transformed.columns)),
                        "preprocess_time": float(preprocess_time),
                        "training_time": float(training_time),
                        "eval_metric": str(getattr(predictor.eval_metric, "name", predictor.eval_metric)),
                        "models": _leaderboard_records(predictor),
                        "prediction_artifacts": prediction_artifacts,
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
