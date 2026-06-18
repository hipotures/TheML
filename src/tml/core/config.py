from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.paths import context_path
from tml.utils.yaml_io import read_yaml


def load_project_config(project_dir: Path) -> dict[str, Any]:
    config = read_yaml(project_dir / "project.yaml")
    if not config:
        raise FileNotFoundError(f"Missing project config: {project_dir / 'project.yaml'}")
    return config


def active_mode(config: dict[str, Any]) -> str:
    root = config.get("root") if isinstance(config.get("root"), dict) else {}
    return str(root.get("active_mode") or "autogluon")


def active_profile_id(config: dict[str, Any], mode: str | None = None) -> str:
    mode = mode or active_mode(config)
    root = config.get("root") if isinstance(config.get("root"), dict) else {}
    profiles = root.get("active_profiles") if isinstance(root.get("active_profiles"), dict) else {}
    return str(profiles.get(mode) or f"{mode}-root-start-v1")


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
