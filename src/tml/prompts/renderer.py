from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.utils.hashing import sha256_text

from .registry import template_text


def render_template(project_dir: Path, template_id: str, context: dict[str, Any]) -> dict[str, str]:
    source, source_path = template_text(project_dir, template_id)
    prompt_source = _strip_front_matter(source)
    rendered = _render(prompt_source, context)
    return {
        "template_id": template_id,
        "template_path": source_path,
        "template_hash": sha256_text(source),
        "rendered": rendered,
        "rendered_hash": sha256_text(rendered),
    }


def _strip_front_matter(source: str) -> str:
    if not source.startswith("---\n"):
        return source
    end = source.find("\n---\n", 4)
    if end == -1:
        return source
    front_matter = source[4:end]
    if "template_id:" not in front_matter:
        return source
    return source[end + len("\n---\n") :]


def _render(source: str, context: dict[str, Any]) -> str:
    try:
        from jinja2 import Environment, StrictUndefined
    except ModuleNotFoundError:
        return _render_simple(source, context)
    env = Environment(undefined=StrictUndefined, autoescape=False)
    return env.from_string(source).render(**context)


def _render_simple(source: str, context: dict[str, Any]) -> str:
    rendered = source
    for expression in _template_expressions(source):
        value = _resolve_expression(expression, context)
        rendered = rendered.replace("{{ " + expression + " }}", str(value))
        rendered = rendered.replace("{{" + expression + "}}", str(value))
    return rendered


def _template_expressions(source: str) -> list[str]:
    import re

    return [match.strip() for match in re.findall(r"{{\s*([^}]+?)\s*}}", source)]


def _resolve_expression(expression: str, context: dict[str, Any]) -> Any:
    value: Any = context
    for part in expression.split("."):
        part = part.strip()
        if isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    return value
