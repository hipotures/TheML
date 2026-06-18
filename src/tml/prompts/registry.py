from __future__ import annotations

from importlib import resources
from pathlib import Path


def template_text(project_dir: Path, template_id: str) -> tuple[str, str]:
    rel = _template_relpath(template_id)
    override = project_dir / "prompts" / rel
    if override.exists():
        return override.read_text(encoding="utf-8"), str(override)
    package = resources.files("tml.prompts.default").joinpath(rel)
    return package.read_text(encoding="utf-8"), str(package)


def _template_relpath(template_id: str) -> str:
    mapping = {
        "root.hypothesis": "root/hypothesis.md.j2",
        "root.materialize-autogluon": "root/materialize-autogluon.md.j2",
        "root.materialize-legacy": "root/materialize-legacy.md.j2",
        "review.execution": "review/execution-review.md.j2",
    }
    if template_id not in mapping:
        raise ValueError(f"Unknown template id {template_id!r}")
    return mapping[template_id]
