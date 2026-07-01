from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.config import repo_root_for_project


def build_wrapped_materialization_source(
    mode: str,
    group_code: str,
    project_dir: Path,
    profile_overrides: dict[str, object] | None = None,
    profile_id: str | None = None,
) -> str:
    if mode == "legacy":
        return _legacy_wrapper_source(group_code, project_dir, profile_overrides=profile_overrides)
    if mode == "autogluon":
        return _autogluon_wrapper_source(group_code, project_dir, profile_overrides=profile_overrides, profile_id=profile_id)
    raise ValueError(f"Unsupported materialization mode: {mode}")


def _autogluon_wrapper_source(
    group_code: str,
    project_dir: Path,
    *,
    profile_overrides: dict[str, object] | None,
    profile_id: str | None,
) -> str:
    project_literal = repr(_project_relative_path(project_dir))
    overrides_literal = repr(_normalized_profile_overrides(profile_overrides or {}))
    profile_id_literal = repr(profile_id)
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

    from tml.core.config import active_profile_id, load_project_config, repo_root_for_project
    from tml.core.paths import context_path
    from tml.core.profiles import load_profile
    from tml.features.groups import run_feature_groups
    from tml.utils.yaml_io import read_yaml

    class_weight_col = "__tml_sample_weight"

    def _patch_xgboost_cpu_inference():
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
                    model.get_booster().set_param({{"device": "cpu"}})
                except Exception:
                    pass
            return original_predict_proba(self, X, *args, **kwargs)

        XGBoostModel._predict_proba = _predict_proba_with_cpu_device
        XGBoostModel._tml_cpu_inference_patch = True

    _patch_xgboost_cpu_inference()

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
    explicit_profile_id = {profile_id_literal}
    data_dir = project_dir / "data"
    runtime_dir = Path(__file__).resolve().parent
    work_dir = runtime_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = runtime_dir / "artifacts"
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
        "ignored_columns",
        "aux_file",
        "auxiliary_file",
        "selection_score",
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

    def _root_config():
        try:
            return read_yaml(context_path(repo_root_for_project(project_dir)))
        except Exception:
            return {{}}

    def _root_external_enabled():
        root_config = _root_config()
        external = root_config.get("external") if isinstance(root_config.get("external"), dict) else {{}}
        return bool(external.get("enabled", False))

    def _bool_option(value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {{"1", "true", "yes", "on"}}:
                return True
            if normalized in {{"0", "false", "no", "off"}}:
                return False
        raise ValueError(f"Invalid boolean option value: {{value!r}}")

    def _positive_float_option(value, name):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{{name}} must be numeric") from exc
        if numeric <= 0:
            raise ValueError(f"{{name}} must be greater than 0")
        return numeric

    def _fraction_option(value, name):
        fraction = _positive_float_option(value, name)
        if fraction >= 1:
            raise ValueError(f"{{name}} must be greater than 0 and less than 1")
        return fraction

    def _positive_int_option(value, name):
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{{name}} must be an integer") from exc
        if numeric <= 0:
            raise ValueError(f"{{name}} must be greater than 0")
        return numeric

    def _autogluon_runtime_options():
        root_config = _root_config()
        autogluon = root_config.get("autogluon") if isinstance(root_config.get("autogluon"), dict) else {{}}
        audit = autogluon.get("audit_score") if isinstance(autogluon.get("audit_score"), dict) else {{}}
        feature_importance = (
            autogluon.get("feature_importance")
            if isinstance(autogluon.get("feature_importance"), dict)
            else {{}}
        )
        return {{
            "audit_score": {{
                "enabled": _bool_option(audit.get("enabled"), default=False),
                "fraction": _fraction_option(audit.get("fraction", 0.1), "autogluon.audit_score.fraction"),
            }},
            "feature_importance": {{
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
            }},
        }}

    def _print_autogluon_runtime_options(options):
        audit = options["audit_score"]
        feature_importance = options["feature_importance"]
        print(
            "TML_RUNTIME|autogluon_options"
            f"|audit_score_enabled={{bool(audit.get('enabled'))}}"
            f"|audit_fraction={{audit.get('fraction')}}"
            f"|feature_importance_enabled={{bool(feature_importance.get('enabled'))}}"
            f"|fi_subsample_size={{feature_importance.get('subsample_size')}}"
            f"|fi_num_shuffle_sets={{feature_importance.get('num_shuffle_sets')}}"
            f"|fi_include_confidence_band={{bool(feature_importance.get('include_confidence_band'))}}",
            flush=True,
        )

    def _project_external_file(config):
        external = config.get("external") if isinstance(config.get("external"), dict) else {{}}
        return external.get("file") or external.get("path") or external.get("aux")

    def _resolve_external_path(value):
        path = Path(str(value))
        if not path.is_absolute():
            path = data_dir / path
        return path

    def _read_aux_csv(config, profile):
        aux_file = None
        if _root_external_enabled():
            aux_file = _project_external_file(config)
        aux_file = aux_file or profile.get("aux_file") or profile.get("auxiliary_file")
        if not aux_file:
            return None, None
        path = _resolve_external_path(aux_file)
        if not path.exists():
            raise FileNotFoundError(f"Configured aux file not found: {{path}}")
        return pd.read_csv(path), aux_file

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

    def _leaderboard_records(predictor, leaderboard=None):
        keep_columns = [
            "model",
            "score_val",
            "score_test",
            "eval_metric",
            "fit_time",
            "fit_time_marginal",
            "pred_time_val",
            "pred_time_val_marginal",
            "pred_time_test",
            "pred_time_test_marginal",
            "stack_level",
            "can_infer",
            "fit_order",
        ]
        if leaderboard is None:
            try:
                leaderboard = predictor.leaderboard(silent=True)
            except Exception as exc:
                return [{{"error": f"leaderboard unavailable: {{type(exc).__name__}}: {{exc}}"}}]
        return [
            {{column: _json_safe_scalar(row.get(column)) for column in keep_columns if column in row}}
            for row in leaderboard.to_dict(orient="records")
        ]

    def _metric_from_leaderboard(leaderboard, score_columns):
        for column in score_columns:
            if column in leaderboard.columns and not leaderboard.empty:
                values = leaderboard[column].dropna()
                if not values.empty:
                    return float(values.max())
        return None

    def _best_model_from_leaderboard(leaderboard, score_column):
        if score_column not in leaderboard.columns or "model" not in leaderboard.columns or leaderboard.empty:
            return None
        scored = leaderboard.dropna(subset=[score_column])
        if scored.empty:
            return None
        return str(scored.sort_values(score_column, ascending=False).iloc[0]["model"])

    def _print_leaderboard(leaderboard):
        display_columns = [
            "model",
            "score_val",
            "score_test",
            "eval_metric",
            "fit_time",
            "pred_time_val",
            "pred_time_test",
            "stack_level",
            "fit_order",
        ]
        available = [column for column in display_columns if column in leaderboard.columns]
        print("TML_RUNTIME|leaderboard", flush=True)
        if available and not leaderboard.empty:
            print(leaderboard[available].to_string(index=False), flush=True)
        else:
            print("TML_RUNTIME|leaderboard|status=empty", flush=True)

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

    def _save_autogluon_prediction_artifacts(
        predictor,
        train_target,
        test_model,
        test_ids,
        test_pred,
        eval_metric,
        id_col,
        target_col,
        valid_data,
        valid_pred,
        audit_data,
        audit_pred,
    ):
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
        if audit_data is not None and audit_pred is not None:
            audit_frame = pd.DataFrame({{
                "row": np.arange(len(audit_data)),
                "target": audit_data[target_col].reset_index(drop=True),
                "prediction": pd.Series(audit_pred).reset_index(drop=True),
            }})
            audit_path = _save_prediction_artifact(audit_frame, "audit_predictions.csv")
            artifacts["audit_predictions"] = str(audit_path)
            artifacts["audit_rows"] = int(len(audit_frame))
        return artifacts

    def _feature_importance_subsample_size(value, row_count):
        numeric = _positive_float_option(value, "autogluon.feature_importance.subsample_size")
        if numeric <= 1:
            return max(1, int(round(row_count * numeric)))
        return int(numeric)

    def _compute_feature_importance(predictor, audit_data, valid_data, group_features, config):
        stats = {{
            "enabled": bool(config.get("enabled")),
            "status": "disabled",
        }}
        if not stats["enabled"]:
            return stats
        source_name = None
        source_data = None
        if audit_data is not None:
            source_name = "audit"
            source_data = audit_data
        elif valid_data is not None:
            source_name = "validation"
            source_data = valid_data
        if source_data is None:
            stats.update({{"status": "skipped", "reason": "no_audit_or_validation_data"}})
            print("TML_RUNTIME|feature_importance|status=skipped|reason=no_audit_or_validation_data", flush=True)
            return stats
        available_features = set(predictor.feature_metadata_in.get_features())
        fi_group_features = [feature for feature in group_features if feature in available_features]
        if not fi_group_features:
            stats.update({{"status": "skipped", "reason": "no_group_features_available"}})
            print("TML_RUNTIME|feature_importance|status=skipped|reason=no_group_features_available", flush=True)
            return stats
        subsample_size = _feature_importance_subsample_size(config.get("subsample_size", 0.1), len(source_data))
        num_shuffle_sets = int(config.get("num_shuffle_sets", 10))
        include_confidence_band = bool(config.get("include_confidence_band", True))
        feature_spec = [("TML_GROUP_FEATURES_ALL", fi_group_features), *fi_group_features]
        print(
            f"TML_RUNTIME|event=start|stage=feature_importance|source={{source_name}}|features={{len(fi_group_features)}}|subsample_size={{subsample_size}}|configured_subsample_size={{config.get('subsample_size', 0.1)}}|num_shuffle_sets={{num_shuffle_sets}}|include_confidence_band={{include_confidence_band}}",
            flush=True,
        )
        started_at = time.time()
        fi = predictor.feature_importance(
            data=source_data,
            features=feature_spec,
            subsample_size=subsample_size,
            num_shuffle_sets=num_shuffle_sets,
            include_confidence_band=include_confidence_band,
        )
        elapsed = time.time() - started_at
        fi_path = artifacts_dir / "feature_importance.csv.gz"
        fi.to_csv(fi_path, index=True, compression="gzip")
        negative_count = int((fi["importance"] < 0).sum()) if "importance" in fi.columns else None
        high_columns = [column for column in fi.columns if str(column).endswith("_high")]
        confident_negative_count = None
        if high_columns:
            confident_negative_count = int((fi[high_columns[0]] < 0).sum())
        stats.update({{
            "status": "ok",
            "source": source_name,
            "path": str(fi_path),
            "source_rows": int(len(source_data)),
            "subsample_size": int(subsample_size),
            "configured_subsample_size": _json_safe_scalar(config.get("subsample_size", 0.1)),
            "group_feature_count": int(len(fi_group_features)),
            "reported_rows": int(len(fi)),
            "num_shuffle_sets": num_shuffle_sets,
            "include_confidence_band": include_confidence_band,
            "elapsed_s": float(elapsed),
            "negative_count": negative_count,
            "confident_negative_count": confident_negative_count,
        }})
        print(f"TML_RUNTIME|event=end|stage=feature_importance|elapsed_s={{elapsed:.3f}}", flush=True)
        print(
            f"TML_RUNTIME|feature_importance|status=ok|source={{source_name}}|features={{len(fi_group_features)}}|subsample_size={{subsample_size}}|configured_subsample_size={{config.get('subsample_size', 0.1)}}|num_shuffle_sets={{num_shuffle_sets}}|include_confidence_band={{include_confidence_band}}|elapsed_s={{elapsed:.3f}}|negative={{negative_count}}|confident_negative={{confident_negative_count}}",
            flush=True,
        )
        print(f"TML_RUNTIME|artifact=feature_importance|path={{fi_path}}", flush=True)
        return stats

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

    def _split_frame(frame, target_col, test_size, seed):
        from sklearn.model_selection import train_test_split

        if not 0 < float(test_size) < 1:
            raise ValueError(f"split fraction must be between 0 and 1, got {{test_size!r}}")
        stratify = frame[target_col] if _should_stratify_holdout(frame[target_col]) else None
        return train_test_split(
            frame,
            test_size=float(test_size),
            random_state=seed,
            stratify=stratify,
        )

    def _training_plan_from_profile(train_model, target_col, profile, audit_config):
        train_model = train_model.copy()
        if profile.get("class_balance") == "balanced":
            train_model[class_weight_col] = _balanced_sample_weight(train_model[target_col])

        fit_args = dict(profile.get("fit_args") or {{}}) if isinstance(profile.get("fit_args"), dict) else {{}}
        bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
        defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
        seed = int(profile.get("seed", 42))
        audit_enabled = bool(audit_config.get("enabled"))
        audit_fraction = float(audit_config.get("fraction", 0.1)) if audit_enabled else 0.0
        audit_data = None
        split_stats = {{
            "audit_enabled": audit_enabled,
            "audit_fraction": audit_fraction if audit_enabled else None,
            "validation_strategy": profile.get("validation_strategy"),
            "bagged_mode": bagged_mode,
        }}
        if audit_enabled:
            train_model, audit_data = _split_frame(
                train_model,
                target_col,
                audit_fraction,
                seed,
            )
            split_stats["audit_rows"] = int(len(audit_data))
            print(
                f"AutoGluon materialization: audit rows={{len(audit_data)}} remaining_rows={{len(train_model)}} fraction={{audit_fraction}}",
                flush=True,
            )
        if bagged_mode:
            print("AutoGluon materialization: bagged mode detected; using internal OOF validation without tuning_data", flush=True)
            split_stats["train_rows"] = int(len(train_model))
            return train_model, None, audit_data, fit_args, defer_save_space, split_stats

        if profile.get("validation_strategy") == "holdout":
            validation_fraction = float(profile.get("validation_fraction", 0.1 if audit_enabled else 0.2))
            if audit_enabled and audit_fraction + validation_fraction >= 1:
                raise ValueError("audit_score.fraction + validation_fraction must be < 1")
            holdout_test_size = validation_fraction / (1.0 - audit_fraction) if audit_enabled else validation_fraction
            train_data, valid_data = _split_frame(
                train_model,
                target_col,
                holdout_test_size,
                seed,
            )
            split_stats.update({{
                "validation_fraction": validation_fraction,
                "validation_rows": int(len(valid_data)),
                "train_rows": int(len(train_data)),
            }})
            print(
                f"AutoGluon materialization: holdout validation rows={{len(valid_data)}} train_rows={{len(train_data)}}",
                flush=True,
            )
            return train_data, valid_data, audit_data, fit_args, defer_save_space, split_stats

        split_stats["train_rows"] = int(len(train_model))
        return train_model, None, audit_data, fit_args, defer_save_space, split_stats

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
                runtime_options = _autogluon_runtime_options()
                _print_autogluon_runtime_options(runtime_options)
                profile_id = explicit_profile_id or active_profile_id(config, "autogluon")
                profile = load_profile(project_dir, "autogluon", profile_id)
                profile.update(profile_overrides)
                _force_autogluon_cpu_resources(profile)

                train = pd.read_csv(_data_file("train"))
                test = pd.read_csv(_data_file("test"))
                sample = pd.read_csv(_data_file("sample_submission"))
                aux_df, aux_file = _read_aux_csv(config, profile)
                if aux_df is not None:
                    print(
                        f"AutoGluon materialization: loaded aux file {{aux_file}} rows={{len(aux_df)}} cols={{len(aux_df.columns)}} passed_to_groups=True",
                        flush=True,
                    )
                train_features = train.drop(columns=[target_col])
                combined = pd.concat([train_features, test], ignore_index=True, sort=False)

                print("AutoGluon materialization: starting feature groups", flush=True)
                preprocess_started_at = time.time()
                with _preprocess_timeout(int(profile.get("preprocess_timeout", 900))):
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
                raw_ignored_columns = profile.get("ignored_columns")
                if raw_ignored_columns is None:
                    ignored_columns = [id_col] if id_col in train_out.columns else []
                elif isinstance(raw_ignored_columns, str):
                    ignored_columns = [raw_ignored_columns]
                else:
                    ignored_columns = list(raw_ignored_columns)
                ignored_columns = [column for column in ignored_columns if column in train_out.columns]
                if ignored_columns:
                    print(f"AutoGluon materialization: ignored_columns={{ignored_columns}}", flush=True)
                train_out[target_col] = train[target_col].reset_index(drop=True)
                group_features = [column for column in transformed.columns if str(column).startswith("G")]
                train_data, valid_data, audit_data, fit_args, defer_save_space, split_stats = _training_plan_from_profile(
                    train_out,
                    target_col,
                    profile,
                    runtime_options["audit_score"],
                )
                train_target = train_data[target_col].reset_index(drop=True)

                predictor_args = dict(profile.get("predictor_args", {{}}) or {{}})
                predictor_args.setdefault("label", target_col)
                predictor_args.setdefault("eval_metric", metric)
                predictor_args.setdefault("path", str(work_dir / "AutoGluonModels"))
                predictor_args.setdefault("verbosity", 2)
                if profile.get("class_balance") == "balanced":
                    predictor_args["sample_weight"] = class_weight_col
                    predictor_args["weight_evaluation"] = False
                learner_kwargs = dict(predictor_args.get("learner_kwargs", {{}}) or {{}})
                if ignored_columns:
                    learner_kwargs["ignored_columns"] = ignored_columns
                    predictor_args["learner_kwargs"] = learner_kwargs
                shutil.rmtree(Path(predictor_args["path"]), ignore_errors=True)
                predictor = TabularPredictor(**predictor_args)

                fit_kwargs = _fit_kwargs_from_profile(profile, train_data, valid_data, fit_args)

                print("TML_RUNTIME|event=start|stage=autogluon_fit", flush=True)
                fit_started_at = time.time()
                predictor.fit(**fit_kwargs)
                training_time = time.time() - fit_started_at
                print(f"TML_RUNTIME|event=end|stage=autogluon_fit|elapsed_s={{training_time:.3f}}", flush=True)

                print("TML_RUNTIME|event=start|stage=autogluon_predict", flush=True)
                valid_pred = None
                audit_pred = None
                leaderboard = predictor.leaderboard(data=audit_data, silent=True)
                leaderboard_path = _save_prediction_artifact(leaderboard, "leaderboard.csv")
                _print_leaderboard(leaderboard)
                score_columns = ("score_test",) if audit_data is not None else ("score_val", "score_test")
                metric_value = _metric_from_leaderboard(leaderboard, score_columns)
                if audit_data is not None and metric_value is None:
                    raise ValueError("Audit score is enabled but AutoGluon leaderboard did not produce score_test")
                selected_model = _best_model_from_leaderboard(
                    leaderboard,
                    "score_test" if audit_data is not None else "score_val",
                )
                if selected_model:
                    print(
                        f"TML_RUNTIME|selected_model={{selected_model}}|score_source={{'audit_score_test' if audit_data is not None else 'autogluon_validation'}}",
                        flush=True,
                    )
                lower_is_better = False
                if valid_data is not None:
                    valid_features = valid_data.drop(columns=[target_col, class_weight_col], errors="ignore")
                    valid_pred = _predict_values(predictor, valid_features, metric, model=selected_model)
                    if audit_data is None and metric_value is None:
                        scores = predictor.evaluate(valid_data, silent=True)
                        if metric in scores:
                            metric_value = float(scores[metric])
                if audit_data is not None:
                    audit_features = audit_data.drop(columns=[target_col, class_weight_col], errors="ignore")
                    audit_pred = _predict_values(predictor, audit_features, metric, model=selected_model)
                predictions = _predict_values(predictor, test_out, metric, model=selected_model)
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
                    audit_data,
                    audit_pred,
                )
                prediction_artifacts["leaderboard"] = str(leaderboard_path)
                feature_importance_stats = _compute_feature_importance(
                    predictor,
                    audit_data,
                    valid_data,
                    group_features,
                    runtime_options["feature_importance"],
                )
                if defer_save_space:
                    try:
                        predictor.save_space(remove_data=True, remove_fit_stack=True)
                        prediction_artifacts["save_space_after_artifacts"] = True
                    except Exception as exc:
                        prediction_artifacts["save_space_error"] = f"{{type(exc).__name__}}: {{exc}}"
                submission = _make_submission(sample, test[id_col], predictions, id_col, target_col)
                submission_path = artifacts_dir / "submission.csv.gz"
                submission.to_csv(submission_path, index=False, compression="gzip")
                print("TML_RUNTIME|event=end|stage=autogluon_predict", flush=True)
                print(f"TML_RUNTIME|artifact=submission|path={{submission_path}}", flush=True)
                if prediction_artifacts.get("oof_predictions"):
                    print(f"TML_RUNTIME|artifact=oof_predictions|path={{prediction_artifacts['oof_predictions']}}", flush=True)
                elif prediction_artifacts.get("oof_error"):
                    print(f"TML_RUNTIME|artifact=oof_predictions|status=unavailable|reason={{prediction_artifacts['oof_error']}}", flush=True)
                if prediction_artifacts.get("validation_predictions"):
                    print(f"TML_RUNTIME|artifact=validation_predictions|path={{prediction_artifacts['validation_predictions']}}", flush=True)
                if prediction_artifacts.get("audit_predictions"):
                    print(f"TML_RUNTIME|artifact=audit_predictions|path={{prediction_artifacts['audit_predictions']}}", flush=True)
                print(f"TML_RUNTIME|artifact=leaderboard|path={{leaderboard_path}}", flush=True)
                print(f"TML_RUNTIME|artifact=test_predictions|path={{prediction_artifacts['test_predictions']}}", flush=True)
                if metric_value is not None:
                    print(f"TML_RUNTIME|metric={{metric}}|value={{metric_value:.6f}}", flush=True)
                print("TML_RUNTIME|event=complete|status=ok", flush=True)
                result = {{
                    "is_bug": False,
                    "summary": "AutoGluon materialization completed.",
                    "metric": metric_value,
                    "eval_metric": str(getattr(predictor.eval_metric, "name", predictor.eval_metric)),
                    "maximize": True,
                    "lower_is_better": lower_is_better,
                    "run_stats": {{
                        "feature_count": int(len(transformed.columns)),
                        "ignored_columns": ignored_columns,
                        "preprocess_time": float(preprocess_time),
                        "training_time": float(training_time),
                        "eval_metric": str(getattr(predictor.eval_metric, "name", predictor.eval_metric)),
                        "score_source": "audit_score_test" if audit_data is not None else "autogluon_validation",
                        "selected_model": selected_model,
                        "split": split_stats,
                        "models": _leaderboard_records(predictor, leaderboard),
                        "prediction_artifacts": prediction_artifacts,
                        "feature_importance": feature_importance_stats,
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
