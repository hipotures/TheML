from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from tml.core.paths import ProjectRef, context_path, find_project
from tml.core.metadata import detect_project_metadata, metadata_task_markdown
from tml.core.profiles import DEFAULT_AUTOGLUON_PROFILE_ID, LEGACY_PROFILE_ID
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import read_yaml, write_yaml


ROOT_CONFIG_DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "defaults": {
        "project_kind": "kaggle",
        "download_data": True,
        "root_mode": "autogluon",
        "prompt_output": "tmp",
        "probe_output": "prompt-lab",
    },
    "active_project": None,
    "active_run": None,
    "models": {
        "hypothesis": "mock",
        "code": "mock",
        "review": "mock",
        "bugfix": "mock",
    },
}


GITIGNORE_TEXT = """# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
.env

# TheML local DB/cache
tml.db
tml.db-wal
tml.db-shm
*.tmp
*.partial

# Runtime workdirs / heavy ML products
projects/**/workspaces/
projects/**/runs/**/artifacts/**/work/
**/AutoGluonModels/
*.csv.gz
*.parquet
*.pkl
*.pickle
*.joblib
*.model

# Large generated prediction artifacts
projects/**/runs/**/artifacts/**/artifacts/oof_predictions*
projects/**/runs/**/artifacts/**/artifacts/test_predictions*
"""


def ensure_gitignore(root: Path) -> None:
    path = root / ".gitignore"
    if not path.exists():
        atomic_write_text(path, GITIGNORE_TEXT)
        return
    text = path.read_text(encoding="utf-8")
    missing = [line for line in GITIGNORE_TEXT.splitlines() if line and line not in text]
    if missing:
        atomic_write_text(path, text.rstrip() + "\n" + "\n".join(missing) + "\n")


def ensure_root_config(root: Path) -> dict[str, Any]:
    path = context_path(root)
    existing = read_yaml(path)
    merged = _merge_root_config(existing)
    if existing != merged:
        write_yaml(path, merged)
    return merged


def default_project_kind(root: Path) -> str:
    config = ensure_root_config(root)
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    return str(defaults.get("project_kind") or "kaggle")


def default_download_data(root: Path) -> bool:
    config = ensure_root_config(root)
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    value = defaults.get("download_data")
    return value if isinstance(value, bool) else True


def init_project(
    root: Path,
    slug: str,
    kind: str | None = None,
    *,
    download: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ProjectRef:
    ensure_gitignore(root)
    root_config = ensure_root_config(root)
    kind = kind or default_project_kind(root)
    ref = ProjectRef(root=root, kind=kind, slug=slug)
    project_dir = ref.path
    _progress(progress, f"Preparing project directory: {project_dir.relative_to(root).as_posix()}")
    for rel in (
        "data",
        "hypotheses",
        "runs",
        "prompt-lab",
        "submissions",
        "logs",
        "docs",
    ):
        (project_dir / rel).mkdir(parents=True, exist_ok=True)
    if download:
        from tml.core.kaggle import download_competition_data

        download_competition_data(slug, project_dir / "data", progress=progress)
    inferred_target = _infer_target(project_dir)
    models = root_config.get("models") if isinstance(root_config.get("models"), dict) else {}
    metadata = None
    if kind == "kaggle":
        metadata = detect_project_metadata(
            project_dir,
            slug=slug,
            model=str(models.get("metadata") or models.get("hypothesis") or "mock"),
            sample_submission_header=[
                str(inferred_target["id_column"]),
                *([str(inferred_target["target_column"])] if inferred_target.get("target_column") else []),
            ],
            progress=progress,
        )
    if not (project_dir / "task.md").exists():
        if metadata is not None:
            atomic_write_text(project_dir / "task.md", metadata_task_markdown(metadata, slug))
        else:
            atomic_write_text(project_dir / "task.md", f"# {slug}\n\nDescribe the ML task here.\n")
    if not (project_dir / "project.yaml").exists():
        target = _merge_target_metadata(inferred_target, metadata)
        write_yaml(
            project_dir / "project.yaml",
            {
                "schema_version": 1,
                "project_id": slug,
                "kind": kind,
                "kaggle_slug": slug if kind == "kaggle" else None,
                "task_file": "task.md",
                "data_dir": "data",
                "target": target,
                "root": {
                    "target_count": 20,
                    "active_mode": str(root_config.get("defaults", {}).get("root_mode") or "autogluon"),
                    "active_profiles": {
                        "autogluon": DEFAULT_AUTOGLUON_PROFILE_ID,
                        "legacy": LEGACY_PROFILE_ID,
                    },
                },
                "models": {
                    "hypothesis": str(models.get("hypothesis") or "mock"),
                    "code": str(models.get("code") or "mock"),
                    "review": str(models.get("review") or "mock"),
                    "bugfix": str(models.get("bugfix") or "mock"),
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    else:
        _ensure_project_profile_defaults(project_dir / "project.yaml")
    return ref


def _progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def use_project(root: Path, slug: str) -> ProjectRef:
    config = ensure_root_config(root)
    ref = find_project(root, slug)
    config["active_project"] = {"kind": ref.kind, "slug": ref.slug}
    config.setdefault("active_run", None)
    write_yaml(
        context_path(root),
        config,
    )
    return ref


def _merge_root_config(existing: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "schema_version": existing.get("schema_version", ROOT_CONFIG_DEFAULTS["schema_version"]),
        "defaults": {
            **ROOT_CONFIG_DEFAULTS["defaults"],
            **(existing.get("defaults") if isinstance(existing.get("defaults"), dict) else {}),
        },
        "active_project": existing.get("active_project") if "active_project" in existing else None,
        "active_run": existing.get("active_run") if "active_run" in existing else None,
        "models": {
            **ROOT_CONFIG_DEFAULTS["models"],
            **(existing.get("models") if isinstance(existing.get("models"), dict) else {}),
        },
    }
    return merged


def _infer_target(project_dir: Path) -> dict[str, Any]:
    target = {
        "id_column": "id",
        "target_column": None,
        "problem_type": None,
        "metric": None,
        "maximize": True,
        "submission_kind": None,
    }
    sample = _data_file(project_dir, "sample_submission.csv")
    if sample.exists():
        import csv
        import gzip

        opener = gzip.open if sample.suffix == ".gz" else Path.open
        with opener(sample, mode="rt", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        if header:
            target["id_column"] = header[0]
        if len(header) >= 2:
            target["target_column"] = header[1]
            target["submission_kind"] = "labels"
    return target


def _data_file(project_dir: Path, name: str) -> Path:
    plain = project_dir / "data" / name
    if plain.exists():
        return plain
    return plain.with_name(plain.name + ".gz")


def _merge_target_metadata(target: dict[str, Any], metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return target
    meta_target = metadata.get("target") if isinstance(metadata.get("target"), dict) else {}
    merged = dict(target)
    for key in (
        "id_column",
        "target_column",
        "problem_type",
        "metric",
        "metric_source",
        "sklearn_metric",
        "metric_description",
        "submission_kind",
    ):
        if merged.get(key) in {None, ""} and meta_target.get(key) not in {None, ""}:
            merged[key] = meta_target[key]
    if isinstance(meta_target.get("maximize"), bool):
        merged["maximize"] = meta_target["maximize"]
    return merged


def _ensure_project_profile_defaults(project_yaml: Path) -> None:
    config = read_yaml(project_yaml)
    root = config.setdefault("root", {})
    if not isinstance(root, dict):
        root = {}
        config["root"] = root
    active_profiles = root.setdefault("active_profiles", {})
    if not isinstance(active_profiles, dict):
        active_profiles = {}
        root["active_profiles"] = active_profiles
    if active_profiles.get("autogluon") is None:
        active_profiles["autogluon"] = DEFAULT_AUTOGLUON_PROFILE_ID
    if active_profiles.get("legacy") in {None, "legacy-root-start-v1"}:
        active_profiles["legacy"] = LEGACY_PROFILE_ID
    write_yaml(project_yaml, config)
