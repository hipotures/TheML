from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_role_model
from tml.core.config import load_project_config, repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.db.state import next_hypothesis_number, upsert_hypothesis, upsert_project
from tml.core.ids import hypothesis_id
from tml.features.validation import validate_root_hypothesis
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.yaml_io import write_yaml


@dataclass(frozen=True)
class GeneratedHypothesis:
    hypothesis_id: str
    path: Path


def generate_missing_root_hypotheses(
    project_dir: Path,
    count: int | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[GeneratedHypothesis]:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    target = count or int(config.get("root", {}).get("target_count", 20))
    created: list[GeneratedHypothesis] = []
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "hypothesis")
    providers = repo_providers_for_project(project_dir)
    for number in range(next_hypothesis_number(project_dir), target + 1):
        hid = hypothesis_id(number)
        hdir = project_dir / "hypotheses" / hid
        hdir.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress(f"Generating ROOT hypothesis {hid} with {model}...")
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
                progress=progress,
            ),
            artifact_dir=hdir,
            providers=providers,
            role_options=role_options,
            response_prefix="01-hypothesis",
        )
        payload = _parse_hypothesis(response.text)
        validate_root_hypothesis(payload)
        payload.update(
            {
                "schema_version": 1,
                "hypothesis_id": hid,
                "enabled": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_yaml(hdir / "hypothesis.yaml", payload)
        upsert_hypothesis(project_dir, hdir)
        created.append(GeneratedHypothesis(hypothesis_id=hid, path=hdir))
    return created


def _parse_hypothesis(text: str) -> dict[str, object]:
    parsed = json.loads(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
        first = parsed["hypotheses"][0]
        return first if isinstance(first, dict) else {}
    return parsed if isinstance(parsed, dict) else {}
