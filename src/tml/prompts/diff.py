from __future__ import annotations

import difflib
from pathlib import Path

from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.yaml_io import read_yaml


def diff_prompt(project_dir: Path, target: str, stage: str) -> tuple[str, str]:
    node_dir = _resolve_node(project_dir, target)
    if stage not in {"code", "hypothesis"}:
        raise ValueError("stage must be one of: hypothesis, code")
    saved_path = node_dir / ("02-code.request.md" if stage == "code" else "01-hypothesis.request.md")
    if not saved_path.exists():
        return "saved request differs", f"Missing saved request: {saved_path}"
    start = read_yaml(node_dir / "node.start.yaml")
    hypothesis = read_yaml(node_dir / "01-hypothesis.yaml")
    mode = str(start.get("mode") or "autogluon")
    template_id = "root.hypothesis" if stage == "hypothesis" else f"root.materialize-{mode}"
    rendered = render_template(
        project_dir,
        template_id,
        project_prompt_context(project_dir, hypothesis=hypothesis, count=1),
    )["rendered"]
    saved = saved_path.read_text(encoding="utf-8")
    if saved == rendered:
        return "saved request matches", ""
    diff = "\n".join(
        difflib.unified_diff(
            saved.splitlines(),
            rendered.splitlines(),
            fromfile=str(saved_path),
            tofile="current-template",
            lineterm="",
        )
    )
    return "saved request differs", diff


def _resolve_node(project_dir: Path, target: str) -> Path:
    candidates: list[Path] = []
    if target.isdigit():
        step = int(target)
        for start_path in sorted((project_dir / "runs").glob("*/artifacts/*/node.start.yaml")):
            if read_yaml(start_path).get("step") == step:
                candidates.append(start_path.parent)
    else:
        candidates = sorted((project_dir / "runs").glob(f"*/artifacts/{target}"))
    if not candidates:
        raise ValueError(f"No node found for {target!r}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Step {target!r} is ambiguous. Specify node_id: {names}")
    return candidates[0]
