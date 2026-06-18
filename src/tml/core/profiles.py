from __future__ import annotations

from importlib import resources
from pathlib import Path

from tml.utils.yaml_io import read_yaml


DEFAULT_AUTOGLUON_PROFILE_ID = "ag-medium-10m-v1"
LEGACY_PROFILE_ID = "legacy-root-start-v1"


def ensure_profile_dirs(profiles_dir: Path) -> None:
    (profiles_dir / "autogluon").mkdir(parents=True, exist_ok=True)
    (profiles_dir / "legacy").mkdir(parents=True, exist_ok=True)


def profile_path(project_dir: Path, mode: str, profile_id: str) -> Path:
    group = "autogluon" if mode == "autogluon" else "legacy"
    return project_dir / "profiles" / group / f"{profile_id}.yaml"


def load_profile(project_dir: Path, mode: str, profile_id: str) -> dict[str, object]:
    project_profile = profile_path(project_dir, mode, profile_id)
    if project_profile.exists():
        return read_yaml(project_profile)
    group = "autogluon" if mode == "autogluon" else "legacy"
    resource = resources.files("tml.profiles.default").joinpath(group, f"{profile_id}.yaml")
    if resource.is_file():
        value = read_yaml_from_text(resource.read_text(encoding="utf-8"))
        return value
    raise FileNotFoundError(f"Missing profile {profile_id!r} for mode {mode!r}")


def profile_hash(project_dir: Path, mode: str, profile_id: str) -> str:
    from tml.utils.hashing import sha256_file, sha256_text

    project_profile = profile_path(project_dir, mode, profile_id)
    if project_profile.exists():
        return sha256_file(project_profile)
    group = "autogluon" if mode == "autogluon" else "legacy"
    resource = resources.files("tml.profiles.default").joinpath(group, f"{profile_id}.yaml")
    if resource.is_file():
        return sha256_text(resource.read_text(encoding="utf-8"))
    return "missing"


def read_yaml_from_text(text: str) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(text)
    return value if isinstance(value, dict) else {}
