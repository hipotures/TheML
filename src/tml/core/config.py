from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.paths import context_path
from tml.utils.yaml_io import read_yaml

DEFAULT_PREPROCESS_TIMEOUT_SECONDS = 900


def load_project_config(project_dir: Path) -> dict[str, Any]:
    config = read_yaml(project_dir / "project.yaml")
    if not config:
        raise FileNotFoundError(f"Missing project config: {project_dir / 'project.yaml'}")
    config["_project_dir"] = str(project_dir)
    return config


def active_mode(config: dict[str, Any]) -> str:
    root = config.get("root") if isinstance(config.get("root"), dict) else {}
    return str(root.get("active_mode") or "autogluon")


def active_profile_id(config: dict[str, Any], mode: str | None = None) -> str:
    mode = mode or active_mode(config)
    root = config.get("root") if isinstance(config.get("root"), dict) else {}
    profiles = root.get("active_profiles") if isinstance(root.get("active_profiles"), dict) else {}
    global_profile = _global_active_profile_id(config, mode)
    return str(profiles.get(mode) or global_profile or f"{mode}-root-start-v1")


def active_branch_algorithm_id(config: dict[str, Any]) -> str:
    project_branch = config.get("branch") if isinstance(config.get("branch"), dict) else {}
    global_algorithm = _global_active_branch_algorithm_id(config)
    return str(project_branch.get("active_algorithm") or global_algorithm or "default")


def project_preprocess_timeout(config: dict[str, Any], overrides: dict[str, object] | None = None) -> int:
    override_value = (overrides or {}).get("preprocess_timeout")
    if override_value is not None:
        return _int_or_default(override_value, DEFAULT_PREPROCESS_TIMEOUT_SECONDS)
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    return _int_or_default(runtime.get("preprocess_timeout"), DEFAULT_PREPROCESS_TIMEOUT_SECONDS)


def rerun_profile_id(project_dir: Path, mode: str) -> str:
    root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    rerun = root_config.get("rerun") if isinstance(root_config.get("rerun"), dict) else {}
    value = rerun.get(mode)
    if not value:
        raise ValueError(f"Missing rerun profile configuration: rerun.{mode} in tml.yaml")
    return str(value)


def _global_active_profile_id(config: dict[str, Any], mode: str) -> str | None:
    project_dir_value = config.get("_project_dir")
    if not isinstance(project_dir_value, str):
        return None
    try:
        root_config = read_yaml(context_path(repo_root_for_project(Path(project_dir_value))))
    except Exception:
        return None
    root = root_config.get("root") if isinstance(root_config.get("root"), dict) else {}
    profiles = root.get("active_profiles") if isinstance(root.get("active_profiles"), dict) else {}
    value = profiles.get(mode)
    return str(value) if value else None


def _global_active_branch_algorithm_id(config: dict[str, Any]) -> str | None:
    project_dir_value = config.get("_project_dir")
    if not isinstance(project_dir_value, str):
        return None
    try:
        root_config = read_yaml(context_path(repo_root_for_project(Path(project_dir_value))))
    except Exception:
        return None
    branch = root_config.get("branch") if isinstance(root_config.get("branch"), dict) else {}
    value = branch.get("active_algorithm")
    return str(value) if value else None


def repo_root_for_project(project_dir: Path) -> Path:
    parts = project_dir.parts
    if len(parts) >= 3 and parts[-3] == "projects":
        return project_dir.parents[2]
    return Path.cwd()


def repo_providers_for_project(project_dir: Path) -> dict[str, object]:
    try:
        root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    except Exception:
        return {}
    providers = root_config.get("providers")
    return providers if isinstance(providers, dict) else {}


def repo_models_for_project(project_dir: Path) -> dict[str, object]:
    try:
        root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    except Exception:
        return {}
    models = root_config.get("models")
    return models if isinstance(models, dict) else {}


def _int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
