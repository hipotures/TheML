from __future__ import annotations

from pathlib import Path


def template_text(project_dir: Path, template_id: str) -> tuple[str, str]:
    rel = _template_relpath(template_id)
    override = project_dir / "prompts" / rel
    if override.exists():
        return override.read_text(encoding="utf-8"), str(override.relative_to(project_dir))
    root_template = _repo_root(project_dir) / "prompts" / rel
    if root_template.exists():
        return root_template.read_text(encoding="utf-8"), f"prompts/{rel}"
    raise FileNotFoundError(f"Missing prompt template {template_id!r}: prompts/{rel}")


def _template_relpath(template_id: str) -> str:
    mapping = {
        "root.hypothesis": "root/hypothesis.md.j2",
        "root.materialize-autogluon": "root/materialize-autogluon.md.j2",
        "root.materialize-legacy": "root/materialize-legacy.md.j2",
        "branch.add-existing-root-group": "branch/add-existing-root-group.md.j2",
        "branch.modify-existing-group": "branch/modify-existing-group.md.j2",
        "review.execution": "review/execution-review.md.j2",
        "project.metadata": "project/metadata.md.j2",
    }
    if template_id not in mapping:
        raise ValueError(f"Unknown template id {template_id!r}")
    return mapping[template_id]


def _repo_root(project_dir: Path) -> Path:
    try:
        return project_dir.parents[2]
    except IndexError:
        return Path.cwd()
