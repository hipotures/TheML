from __future__ import annotations

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
    root_profile = _repo_root(project_dir) / "profiles" / group / f"{profile_id}.yaml"
    if root_profile.exists():
        return read_yaml(root_profile)
    raise FileNotFoundError(f"Missing profile {profile_id!r} for mode {mode!r}")


def profile_hash(project_dir: Path, mode: str, profile_id: str) -> str:
    from tml.utils.hashing import sha256_file

    project_profile = profile_path(project_dir, mode, profile_id)
    if project_profile.exists():
        return sha256_file(project_profile)
    group = "autogluon" if mode == "autogluon" else "legacy"
    root_profile = _repo_root(project_dir) / "profiles" / group / f"{profile_id}.yaml"
    if root_profile.exists():
        return sha256_file(root_profile)
    return "missing"


def _repo_root(project_dir: Path) -> Path:
    try:
        return project_dir.parents[2]
    except IndexError:
        return Path.cwd()
