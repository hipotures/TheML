from __future__ import annotations

from pathlib import Path

from tml.utils.atomic import atomic_write_text


ROOT_GITIGNORE_TEXT = """# Python
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


def ensure_root_gitignore(root: Path) -> None:
    _ensure_gitignore_lines(root / ".gitignore", ROOT_GITIGNORE_TEXT)


def ensure_project_gitignore(project_dir: Path) -> None:
    _ensure_gitignore_lines(project_dir / ".gitignore", _project_gitignore_template(project_dir))


def _ensure_gitignore_lines(path: Path, template: str) -> None:
    if not path.exists():
        atomic_write_text(path, template)
        return
    text = path.read_text(encoding="utf-8")
    missing = [line for line in template.splitlines() if line and line not in text]
    if missing:
        atomic_write_text(path, text.rstrip() + "\n" + "\n".join(missing) + "\n")


def _project_gitignore_template(project_dir: Path) -> str:
    template = project_dir.parents[2] / "control" / "templates" / "project.gitignore"
    if template.exists():
        return template.read_text(encoding="utf-8")
    return "data/\nlogs/\nsubmissions/\ntml.db\n"
