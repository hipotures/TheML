from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from tml.core.errors import ContextError
from tml.utils.yaml_io import read_yaml


def workspace_root() -> Path:
    return Path(os.environ.get("TML_CWD", os.getcwd())).resolve()


@dataclass(frozen=True)
class ProjectRef:
    root: Path
    kind: str
    slug: str

    @property
    def path(self) -> Path:
        return self.root / "projects" / self.kind / self.slug

    @property
    def db_path(self) -> Path:
        return self.path / "tml.db"


def context_path(root: Path) -> Path:
    return root / "tml.yaml"


def active_project_ref(root: Path | None = None) -> ProjectRef:
    root = root or workspace_root()
    context = read_yaml(context_path(root))
    active = context.get("active_project")
    if not isinstance(active, dict):
        raise ContextError("No active project. Example: tml project use demo_project")
    return ProjectRef(
        root=root,
        kind=str(active.get("kind") or "kaggle"),
        slug=str(active.get("slug") or ""),
    )


def find_project(root: Path, slug: str) -> ProjectRef:
    candidates = sorted((root / "projects").glob(f"*/{slug}"))
    if not candidates:
        raise ContextError(f"Project {slug!r} not found. Example: tml init project {slug}")
    kaggle = [path for path in candidates if path.parent.name == "kaggle"]
    if len(kaggle) == 1:
        return ProjectRef(root=root, kind="kaggle", slug=slug)
    if len(candidates) > 1:
        kinds = ", ".join(path.parent.name for path in candidates)
        raise ContextError(f"Project {slug!r} is ambiguous across kinds: {kinds}")
    return ProjectRef(root=root, kind=candidates[0].parent.name, slug=slug)
