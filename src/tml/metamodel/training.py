from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from tml.metamodel.features import FeatureBuildResult, write_feature_artifacts
from tml.metamodel.importer import DatasetBuildResult, build_meta_dataset, load_records_from_csv
from tml.metamodel.metrics import aggregate_metric_dicts, evaluate_predictions, similarity_diagnostic
from tml.metamodel.records import MetaRecord


@dataclass(frozen=True)
class MetaTrainingProfile:
    name: str
    split_count: int
    holdout_fraction: float
    split_time_limit: int
    final_time_limit: int
    presets: str
    random_seed: int
    min_cv_examples: int
    min_public_examples: int
    include_group_split: bool


@dataclass(frozen=True)
class TargetTrainingResult:
    target: str
    status: str
    output_dir: Path
    model_dir: Path | None
    validation_predictions_csv: Path | None
    metrics_json: Path
    leaderboard_csv: Path | None
    feature_importance_csv: Path | None
    training_config_json: Path
    n_examples: int
    low_confidence: bool
    metrics: dict[str, Any]


@dataclass(frozen=True)
class MetaRunResult:
    output_dir: Path
    dataset_result: DatasetBuildResult
    feature_result: FeatureBuildResult
    target_results: list[TargetTrainingResult]
    summary_json: Path


PROFILES = {
    "quick": MetaTrainingProfile(
        name="quick",
        split_count=3,
        holdout_fraction=0.25,
        split_time_limit=20,
        final_time_limit=30,
        presets="medium_quality",
        random_seed=42,
        min_cv_examples=30,
        min_public_examples=30,
        include_group_split=True,
    ),
    "best": MetaTrainingProfile(
        name="best",
        split_count=0,
        holdout_fraction=0.0,
        split_time_limit=0,
        final_time_limit=7200,
        presets="best_quality",
        random_seed=42,
        min_cv_examples=30,
        min_public_examples=30,
        include_group_split=False,
    ),
}


def default_run_dir(project_dir: Path, profile_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return project_dir / "meta_models" / "meta_runs" / f"{stamp}-{profile_name}"


def run_meta_modeling(
    project_dir: Path,
    *,
    output_dir: Path | None = None,
    profile_name: str = "quick",
    train_public: bool = True,
) -> MetaRunResult:
    profile = _profile(profile_name)
    output_dir = output_dir or default_run_dir(project_dir, profile.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    records, dataset_result = build_meta_dataset(project_dir, dataset_dir)
    feature_frame, feature_result = write_feature_artifacts(records, dataset_dir)
    target_results = [
        train_target(
            project_dir,
            records=records,
            feature_frame=feature_frame,
            feature_columns=feature_result.feature_columns,
            target="cv_score",
            output_dir=output_dir / "targets" / "cv_score",
            profile=profile,
            min_examples=profile.min_cv_examples,
            low_confidence=False,
        )
    ]
    if train_public:
        target_results.append(
            train_target(
                project_dir,
                records=records,
                feature_frame=feature_frame,
                feature_columns=feature_result.feature_columns,
                target="public_score",
                output_dir=output_dir / "targets" / "public_score",
                profile=profile,
                min_examples=profile.min_public_examples,
                low_confidence=True,
            )
        )
        target_results.append(
            train_target(
                project_dir,
                records=records,
                feature_frame=feature_frame,
                feature_columns=feature_result.feature_columns,
                target="public_gap",
                output_dir=output_dir / "targets" / "public_gap",
                profile=profile,
                min_examples=profile.min_public_examples,
                low_confidence=True,
            )
        )
    summary = {
        "schema_version": 1,
        "kind": "tml_meta_model_run",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": profile.__dict__,
        "output_dir": str(output_dir),
        "dataset": dataset_result.__dict__ | {"project_dir": str(dataset_result.project_dir), "output_dir": str(dataset_result.output_dir), "dataset_csv": str(dataset_result.dataset_csv), "dataset_parquet": str(dataset_result.dataset_parquet) if dataset_result.dataset_parquet else None, "metadata_json": str(dataset_result.metadata_json)},
        "features": feature_result.__dict__ | {"feature_csv": str(feature_result.feature_csv), "feature_spec_json": str(feature_result.feature_spec_json)},
        "targets": [_target_result_json(result) for result in target_results],
    }
    summary_json = output_dir / "meta_run.json"
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MetaRunResult(
        output_dir=output_dir,
        dataset_result=dataset_result,
        feature_result=feature_result,
        target_results=target_results,
        summary_json=summary_json,
    )


def train_target_from_dataset(
    project_dir: Path,
    *,
    dataset_csv: Path,
    output_dir: Path,
    target: str,
    profile_name: str = "quick",
) -> TargetTrainingResult:
    profile = _profile(profile_name)
    records = load_records_from_csv(dataset_csv)
    feature_frame, feature_result = write_feature_artifacts(records, output_dir / "dataset")
    min_examples = profile.min_cv_examples if target == "cv_score" else profile.min_public_examples
    return train_target(
        project_dir,
        records=records,
        feature_frame=feature_frame,
        feature_columns=feature_result.feature_columns,
        target=target,
        output_dir=output_dir / "targets" / target,
        profile=profile,
        min_examples=min_examples,
        low_confidence=target != "cv_score",
    )


def train_target(
    project_dir: Path,
    *,
    records: list[MetaRecord],
    feature_frame,
    feature_columns: list[str],
    target: str,
    output_dir: Path,
    profile: MetaTrainingProfile,
    min_examples: int,
    low_confidence: bool,
) -> TargetTrainingResult:
    pd = _require_pandas()
    np = _require_numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_frame = feature_frame[feature_frame[target].notna()].copy()
    n_examples = int(len(target_frame))
    metrics_json = output_dir / "metrics.json"
    training_config_json = output_dir / "training_config.json"
    config = {
        "schema_version": 1,
        "target": target,
        "profile": profile.__dict__,
        "n_examples": n_examples,
        "feature_columns": feature_columns,
        "low_confidence": low_confidence,
        "autogluon_problem_type": "regression",
        "autogluon_eval_metric": "mean_absolute_error",
    }
    training_config_json.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if n_examples < min_examples:
        metrics = {
            "status": "skipped",
            "reason": f"Only {n_examples} labeled examples; minimum is {min_examples}.",
            "n_examples": n_examples,
            "low_confidence": low_confidence,
        }
        metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return TargetTrainingResult(
            target=target,
            status="skipped",
            output_dir=output_dir,
            model_dir=None,
            validation_predictions_csv=None,
            metrics_json=metrics_json,
            leaderboard_csv=None,
            feature_importance_csv=None,
            training_config_json=training_config_json,
            n_examples=n_examples,
            low_confidence=low_confidence,
            metrics=metrics,
        )

    from sklearn.model_selection import train_test_split

    validation_rows: list[dict[str, Any]] = []
    split_metrics: list[dict[str, Any]] = []
    split_advisory: list[list[dict[str, Any]]] = []
    split_dir = output_dir / "split_models"
    split_dir.mkdir(parents=True, exist_ok=True)
    X = target_frame[feature_columns].copy()
    y = target_frame[target].astype(float).copy()
    node_ids = target_frame["node_id"].astype(str).tolist()
    has_unseen = target_frame.get("has_unseen_group")

    for split_index in range(profile.split_count):
        train_idx, valid_idx = train_test_split(
            np.arange(n_examples),
            test_size=profile.holdout_fraction,
            random_state=profile.random_seed + split_index,
            shuffle=True,
        )
        result = _fit_split(
            train_frame=_labeled_frame(X.iloc[train_idx], y.iloc[train_idx], target),
            valid_features=X.iloc[valid_idx],
            valid_target=y.iloc[valid_idx],
            valid_node_ids=[node_ids[index] for index in valid_idx],
            valid_has_unseen=has_unseen.iloc[valid_idx] if has_unseen is not None else None,
            train_node_ids=[node_ids[index] for index in train_idx],
            label=target,
            model_path=split_dir / f"random_{split_index + 1:02d}" / "AutoGluonModels",
            time_limit=profile.split_time_limit,
            presets=profile.presets,
            split_name=f"random_{split_index + 1:02d}",
        )
        validation_rows.extend(result["rows"])
        split_metrics.append(result["metrics"])
        split_advisory.append(result["advisory"])

    group_split_metrics: dict[str, Any] | None = None
    if profile.include_group_split:
        group_split_metrics = _try_group_split(
            X=X,
            y=y,
            target_frame=target_frame,
            node_ids=node_ids,
            has_unseen=has_unseen,
            label=target,
            model_path=split_dir / "group_root" / "AutoGluonModels",
            time_limit=profile.split_time_limit,
            presets=profile.presets,
        )
        if group_split_metrics is not None:
            validation_rows.extend(group_split_metrics.pop("rows"))

    final_model_dir = output_dir / "final" / "AutoGluonModels"
    final_predictor = _fit_predictor(
        _labeled_frame(X, y, target),
        label=target,
        model_path=final_model_dir,
        time_limit=profile.final_time_limit,
        presets=profile.presets,
        log_path=output_dir / "final_training.log",
    )
    leaderboard_csv = _write_leaderboard(final_predictor, output_dir / "leaderboard.csv")
    feature_importance_csv = _write_feature_importance(
        final_predictor,
        _labeled_frame(X, y, target),
        output_dir / "feature_importance.csv",
    )
    predictions_csv = output_dir / "validation_predictions.csv"
    pd.DataFrame(validation_rows).to_csv(predictions_csv, index=False)
    metrics = {
        "status": "complete",
        "target": target,
        "n_examples": n_examples,
        "low_confidence": low_confidence,
        "random_splits": {
            "split_count": profile.split_count,
            "metrics_by_split": split_metrics,
            "aggregate": aggregate_metric_dicts(split_metrics),
        },
        "group_split": group_split_metrics,
        "advisory_by_split": split_advisory,
        "similarity_diagnostic": similarity_diagnostic(records, validation_rows),
        "uncertainty": _uncertainty_summary(validation_rows),
    }
    metrics_json.write_text(json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TargetTrainingResult(
        target=target,
        status="complete",
        output_dir=output_dir,
        model_dir=final_model_dir,
        validation_predictions_csv=predictions_csv,
        metrics_json=metrics_json,
        leaderboard_csv=leaderboard_csv,
        feature_importance_csv=feature_importance_csv,
        training_config_json=training_config_json,
        n_examples=n_examples,
        low_confidence=low_confidence,
        metrics=metrics,
    )


def load_predictor(model_dir: Path):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError("AutoGluon Tabular is required for prediction. Install the autogluon extra.") from exc
    return TabularPredictor.load(str(model_dir))


def predict_with_uncertainty(model_dir: Path, frame) -> dict[str, Any]:
    np = _require_numpy()
    predictor = load_predictor(model_dir)
    final_pred = np.asarray(predictor.predict(frame), dtype=float)
    split_root = model_dir.parents[1] / "split_models"
    split_predictions = []
    if split_root.exists():
        for split_model in sorted(split_root.glob("*/AutoGluonModels")):
            try:
                split_predictor = load_predictor(split_model)
                split_predictions.append(np.asarray(split_predictor.predict(frame), dtype=float))
            except Exception:
                continue
    if split_predictions:
        matrix = np.vstack(split_predictions)
        return {
            "prediction": final_pred.tolist(),
            "split_prediction_mean": np.mean(matrix, axis=0).tolist(),
            "split_prediction_std": np.std(matrix, axis=0).tolist(),
            "split_prediction_p10": np.quantile(matrix, 0.1, axis=0).tolist(),
            "split_prediction_p90": np.quantile(matrix, 0.9, axis=0).tolist(),
            "split_model_count": int(matrix.shape[0]),
        }
    return {
        "prediction": final_pred.tolist(),
        "split_prediction_mean": None,
        "split_prediction_std": None,
        "split_prediction_p10": None,
        "split_prediction_p90": None,
        "split_model_count": 0,
    }


def _fit_split(
    *,
    train_frame,
    valid_features,
    valid_target,
    valid_node_ids: list[str],
    valid_has_unseen,
    train_node_ids: list[str],
    label: str,
    model_path: Path,
    time_limit: int,
    presets: str,
    split_name: str,
) -> dict[str, Any]:
    predictor = _fit_predictor(
        train_frame,
        label=label,
        model_path=model_path,
        time_limit=time_limit,
        presets=presets,
        log_path=model_path.parent / "training.log",
    )
    predictions = predictor.predict(valid_features)
    summary = evaluate_predictions(valid_target, predictions, has_unseen_group=valid_has_unseen)
    rows = []
    for node_id, true_value, pred_value in zip(valid_node_ids, valid_target, predictions, strict=False):
        rows.append(
            {
                "split": split_name,
                "node_id": node_id,
                "target": label,
                "y_true": float(true_value),
                "y_pred": float(pred_value),
                "abs_error": abs(float(pred_value) - float(true_value)),
                "train_node_ids": train_node_ids,
            }
        )
    return {
        "rows": rows,
        "metrics": {**summary.metrics, **summary.cold_start_metrics},
        "advisory": summary.advisory,
    }


def _try_group_split(
    *,
    X,
    y,
    target_frame,
    node_ids: list[str],
    has_unseen,
    label: str,
    model_path: Path,
    time_limit: int,
    presets: str,
) -> dict[str, Any] | None:
    from sklearn.model_selection import GroupShuffleSplit

    groups = target_frame["root_id"].fillna(target_frame["branch_id"]).fillna(target_frame["node_id"]).astype(str)
    if int(groups.nunique()) < 3:
        return None
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=917)
    try:
        train_idx, valid_idx = next(splitter.split(X, y, groups=groups))
    except ValueError:
        return None
    if len(valid_idx) < 5 or len(train_idx) < 10:
        return None
    result = _fit_split(
        train_frame=_labeled_frame(X.iloc[train_idx], y.iloc[train_idx], label),
        valid_features=X.iloc[valid_idx],
        valid_target=y.iloc[valid_idx],
        valid_node_ids=[node_ids[index] for index in valid_idx],
        valid_has_unseen=has_unseen.iloc[valid_idx] if has_unseen is not None else None,
        train_node_ids=[node_ids[index] for index in train_idx],
        label=label,
        model_path=model_path,
        time_limit=time_limit,
        presets=presets,
        split_name="group_root",
    )
    return {
        "group_column": "root_id",
        "train_groups": int(groups.iloc[train_idx].nunique()),
        "valid_groups": int(groups.iloc[valid_idx].nunique()),
        "metrics": result["metrics"],
        "advisory": result["advisory"],
        "rows": result["rows"],
    }


def _fit_predictor(train_data, *, label: str, model_path: Path, time_limit: int, presets: str, log_path: Path):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError("AutoGluon Tabular is required for meta-model training. Install the autogluon extra.") from exc
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle):
        handle.write(f"Training target={label} path={model_path} time_limit={time_limit} presets={presets}\n")
        handle.flush()
        predictor = TabularPredictor(
            label=label,
            problem_type="regression",
            eval_metric="mean_absolute_error",
            path=str(model_path),
            verbosity=2,
            log_to_file=True,
            log_file_path=str(log_path),
        )
        predictor.fit(train_data=train_data, time_limit=time_limit, presets=presets)
    return predictor


def _write_leaderboard(predictor, path: Path) -> Path | None:
    try:
        leaderboard = predictor.leaderboard(silent=True, extra_info=True)
        leaderboard.to_csv(path, index=False)
    except Exception:
        return None
    return path


def _write_feature_importance(predictor, labeled_frame, path: Path) -> Path | None:
    try:
        importance = predictor.feature_importance(labeled_frame)
        importance.to_csv(path)
    except Exception:
        return None
    return path


def _uncertainty_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    np = _require_numpy()
    by_node: dict[str, list[float]] = {}
    for row in rows:
        node_id = str(row.get("node_id") or "")
        value = row.get("y_pred")
        if isinstance(value, int | float):
            by_node.setdefault(node_id, []).append(float(value))
    spreads = [float(np.std(values)) for values in by_node.values() if len(values) >= 2]
    if not spreads:
        return {
            "method": "repeated random split prediction spread",
            "available": False,
            "message": "No node received predictions from at least two validation splits.",
        }
    return {
        "method": "repeated random split prediction spread",
        "available": True,
        "mean_prediction_std": float(np.mean(spreads)),
        "p90_prediction_std": float(np.quantile(spreads, 0.9)),
        "max_prediction_std": float(np.max(spreads)),
    }


def _labeled_frame(features, target_values, label: str):
    frame = features.copy()
    frame[label] = target_values.values
    return frame


def _profile(name: str) -> MetaTrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown meta-model profile {name!r}. Allowed: {allowed}") from exc


def _target_result_json(result: TargetTrainingResult) -> dict[str, Any]:
    return {
        "target": result.target,
        "status": result.status,
        "output_dir": str(result.output_dir),
        "model_dir": str(result.model_dir) if result.model_dir else None,
        "validation_predictions_csv": str(result.validation_predictions_csv) if result.validation_predictions_csv else None,
        "metrics_json": str(result.metrics_json),
        "leaderboard_csv": str(result.leaderboard_csv) if result.leaderboard_csv else None,
        "feature_importance_csv": str(result.feature_importance_csv) if result.feature_importance_csv else None,
        "training_config_json": str(result.training_config_json),
        "n_examples": result.n_examples,
        "low_confidence": result.low_confidence,
        "metrics": result.metrics,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Meta-model training requires pandas. Install project requirements with uv pip.") from exc
    return pd


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Meta-model training requires numpy. Install project requirements with uv pip.") from exc
    return np
