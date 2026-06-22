from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
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


@dataclass(frozen=True)
class RootGenerationPlan:
    target: int
    next_number: int
    iteration_count: int
    hypothesis_ids: list[str]
    role: str
    model: str
    provider: str
    provider_kind: object
    resolved_model: str | None
    reasoning_effort: str | None
    timeout_seconds: object
    sandbox: str
    web_search_enabled: bool


def root_generation_plan(project_dir: Path, count: int | None = None) -> RootGenerationPlan:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    target = count or int(config.get("root", {}).get("target_count", 20))
    next_number = next_hypothesis_number(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "hypothesis")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    web_search_enabled = _web_search_enabled(provider_config.get("web_search"))
    hypothesis_ids = [hypothesis_id(number) for number in range(next_number, target + 1)]
    return RootGenerationPlan(
        target=target,
        next_number=next_number,
        iteration_count=len(hypothesis_ids),
        hypothesis_ids=hypothesis_ids,
        role="hypothesis",
        model=model,
        provider=spec.provider,
        provider_kind=provider_config.get("kind"),
        resolved_model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        timeout_seconds=provider_config.get("timeout_seconds"),
        sandbox="read_only",
        web_search_enabled=web_search_enabled,
    )


def generate_missing_root_hypotheses(
    project_dir: Path,
    count: int | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> list[GeneratedHypothesis]:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    target = count or int(config.get("root", {}).get("target_count", 20))
    created: list[GeneratedHypothesis] = []
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "hypothesis")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    web_search_enabled = _web_search_enabled(provider_config.get("web_search"))
    numbers = list(range(next_hypothesis_number(project_dir), target + 1))
    total = len(numbers)
    for index, number in enumerate(numbers, start=1):
        if stop_requested is not None and stop_requested():
            break
        hid = hypothesis_id(number)
        hdir = project_dir / "hypotheses" / hid
        hdir.mkdir(parents=True, exist_ok=True)
        progress_prefix = f"ROOT hypothesis {hid} ({index}/{total})"
        if progress is not None:
            progress(f"Generating {progress_prefix} with {model}...")

        def invocation_progress(message: str, *, prefix: str = progress_prefix) -> None:
            if progress is not None:
                progress(f"{prefix}: {message}")

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
                metadata={"web_search_enabled": web_search_enabled},
                progress=invocation_progress if progress is not None else None,
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


def _web_search_enabled(value: object) -> bool:
    if str(value or "").strip().lower() in {"live", "cached"}:
        return True
    return _bool(value)


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
