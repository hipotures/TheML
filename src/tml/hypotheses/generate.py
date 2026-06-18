from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.core.config import load_project_config, repo_providers_for_project, repo_root_for_project
from tml.core.ids import hypothesis_id
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.yaml_io import write_yaml

from .model import hypothesis_dirs


def generate_missing_root_hypotheses(project_dir: Path, count: int | None = None) -> int:
    config = load_project_config(project_dir)
    target = count or int(config.get("root", {}).get("target_count", 20))
    existing = len(hypothesis_dirs(project_dir))
    created = 0
    model = str(config.get("models", {}).get("hypothesis", "mock"))
    providers = repo_providers_for_project(project_dir)
    for number in range(existing + 1, target + 1):
        hid = hypothesis_id(number)
        hdir = project_dir / "hypotheses" / hid
        hdir.mkdir(parents=True, exist_ok=True)
        rendered = render_template(
            project_dir,
            "root.hypothesis",
            project_prompt_context(project_dir, count=1),
        )
        response = run_model_invocation(
            ModelInvocation(
                role="hypothesis",
                model=model,
                prompt=rendered["rendered"],
                template_id=rendered["template_id"],
                template_path=rendered["template_path"],
                template_hash=rendered["template_hash"],
                rendered_prompt_hash=rendered["rendered_hash"],
                cwd=repo_root_for_project(project_dir),
                sandbox="read_only",
            ),
            artifact_dir=hdir,
            providers=providers,
            response_prefix="01-hypothesis",
        )
        payload = _parse_hypothesis(response.text)
        payload.update(
            {
                "schema_version": 1,
                "hypothesis_id": hid,
                "enabled": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_yaml(hdir / "hypothesis.yaml", payload)
        created += 1
    return created


def _parse_hypothesis(text: str) -> dict[str, object]:
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
        first = parsed["hypotheses"][0]
        return first if isinstance(first, dict) else {}
    return parsed if isinstance(parsed, dict) else {}
