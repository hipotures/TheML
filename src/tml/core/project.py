from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tml.core.paths import ProjectRef, context_path, find_project
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import write_yaml


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


def init_project(root: Path, slug: str, kind: str) -> ProjectRef:
    ensure_gitignore(root)
    ref = ProjectRef(root=root, kind=kind, slug=slug)
    project_dir = ref.path
    for rel in (
        "data",
        "profiles/root",
        "hypotheses",
        "runs",
        "prompt-lab",
        "submissions",
        "logs",
        "docs",
    ):
        (project_dir / rel).mkdir(parents=True, exist_ok=True)
    if not (project_dir / "task.md").exists():
        atomic_write_text(project_dir / "task.md", f"# {slug}\n\nDescribe the ML task here.\n")
    if not (project_dir / "project.yaml").exists():
        write_yaml(
            project_dir / "project.yaml",
            {
                "schema_version": 1,
                "project_id": slug,
                "kind": kind,
                "task_file": "task.md",
                "data_dir": "data",
                "target": {
                    "id_column": "id",
                    "target_column": "target",
                    "problem_type": "unknown",
                    "metric": "balanced_accuracy",
                    "maximize": True,
                    "submission_kind": "labels",
                },
                "root": {
                    "target_count": 20,
                    "active_mode": "autogluon",
                    "active_profiles": {
                        "autogluon": "autogluon-root-start-v1",
                        "legacy": "legacy-root-start-v1",
                    },
                },
                "models": {
                    "hypothesis": "mock",
                    "code": "mock",
                    "review": "mock",
                    "bugfix": "mock",
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    _write_default_profiles(project_dir)
    return ref


def use_project(root: Path, slug: str) -> ProjectRef:
    ref = find_project(root, slug)
    write_yaml(
        context_path(root),
        {"schema_version": 1, "active_project": {"kind": ref.kind, "slug": ref.slug}},
    )
    return ref


def _write_default_profiles(project_dir: Path) -> None:
    profiles = {
        "autogluon-root-start-v1.yaml": {
            "schema_version": 1,
            "profile_id": "autogluon-root-start-v1",
            "mode": "autogluon",
            "time_limit": 60,
            "presets": "medium_quality",
            "validation_fraction": 0.2,
            "seed": 1,
        },
        "legacy-root-start-v1.yaml": {
            "schema_version": 1,
            "profile_id": "legacy-root-start-v1",
            "mode": "legacy",
            "timeout_seconds": 900,
            "seed": 1,
        },
    }
    for filename, payload in profiles.items():
        path = project_dir / "profiles" / "root" / filename
        if not path.exists():
            write_yaml(path, payload)
