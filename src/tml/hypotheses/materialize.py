from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_role_model
from tml.core.config import repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.features.validation import validate_group_code_source
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml

from .model import hypothesis_dirs


def materialize_missing(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
    progress: Callable[[str, int | None], None] | None = None,
) -> int:
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    created = 0
    target_hypothesis_id = hypothesis_id.zfill(6) if hypothesis_id else None
    for hdir in hypothesis_dirs(project_dir):
        if target_hypothesis_id and hdir.name != target_hypothesis_id:
            continue
        mat_dir = hdir / "materializations"
        mat_dir.mkdir(parents=True, exist_ok=True)
        target = mat_dir / f"{mode}-001.py"
        if target.exists():
            continue
        timeout_seconds = int(role_options.get("timeout_seconds") or 900)
        if progress is not None:
            progress(f"Materializing {hdir.name} ({mode}) with {model}...", timeout_seconds)
        hypothesis = read_yaml(hdir / "hypothesis.yaml")
        template_id = f"root.materialize-{mode}"
        rendered = render_template(
            project_dir,
            template_id,
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
        prompt = f"{mode}\n\n{rendered['rendered']}"
        response = run_model_invocation(
            ModelInvocation(
                role="materializations",
                model=model,
                prompt=prompt,
                template_id=rendered["template_id"],
                template_path=rendered["template_path"],
                template_hash=rendered["template_hash"],
                rendered_prompt_hash=rendered["rendered_hash"],
                cwd=repo_root_for_project(project_dir),
                sandbox="read_only",
                metadata={"mode": mode},
                progress=(lambda message: progress(message, timeout_seconds)) if progress is not None else None,
            ),
            artifact_dir=mat_dir,
            providers=providers,
            role_options=role_options,
            response_prefix=f"{mode}-001",
        )
        code = _parse_code(response.text)
        validate_group_code_source(code)
        atomic_write_text(target, code)
        _update_manifest(hdir, mode, target, hypothesis)
        created += 1
    return created


def _parse_code(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict) and isinstance(parsed.get("code"), str):
        return parsed["code"]
    return text


def _update_manifest(hdir: Path, mode: str, path: Path, hypothesis: dict[str, object]) -> None:
    manifest_path = hdir / "manifest.yaml"
    manifest = read_yaml(manifest_path)
    versions = manifest.setdefault("materializations", {})
    if not isinstance(versions, dict):
        versions = {}
        manifest["materializations"] = versions
    versions[mode] = {
        "active": path.name,
        "sha256": sha256_file(path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    group_name = str(hypothesis.get("group_name") or hdir.name)
    manifest["feature_group"] = {
        "logical_name": group_name,
        "version_id": f"{group_name}@{hdir.name}",
        "source_hypothesis_id": str(hypothesis.get("hypothesis_id") or hdir.name),
        "operation": "create_new_root_group",
        "depends_on": list(hypothesis.get("depends_on") or []),
        "code_artifact": path.name,
    }
    write_yaml(manifest_path, manifest)
