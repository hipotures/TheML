from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from tml.utils.hashing import sha256_text

from .registry import template_text


def render_template(project_dir: Path, template_id: str, context: dict[str, Any]) -> dict[str, str]:
    source, source_path = template_text(project_dir, template_id)
    env = Environment(undefined=StrictUndefined, autoescape=False)
    rendered = env.from_string(source).render(**context)
    return {
        "template_id": template_id,
        "template_path": source_path,
        "template_hash": sha256_text(source),
        "rendered": rendered,
        "rendered_hash": sha256_text(rendered),
    }
