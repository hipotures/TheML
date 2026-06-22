from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
from tml.core.config import repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.db.state import materialization_candidates, upsert_materialization, upsert_project
from tml.features.validation import validate_group_code_source
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml

from .baseline import BASELINE_HYPOTHESIS_ID, BASELINE_MODE, ensure_root_baseline


@dataclass(frozen=True)
class RootMaterializationPlan:
    mode: str
    role: str
    model: str
    provider: str
    provider_kind: object
    resolved_model: str | None
    reasoning_effort: str | None
    timeout_seconds: object
    sandbox: str
    candidate_count: int
    iteration_count: int
    hypothesis_ids: list[str]


def root_materialization_plan(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
) -> RootMaterializationPlan:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    candidates = materialization_candidates(project_dir, hypothesis_id=hypothesis_id)
    pending_ids: list[str] = []
    for record in candidates:
        if str(record["hypothesis_id"]) == BASELINE_HYPOTHESIS_ID and mode != BASELINE_MODE:
            continue
        hdir = _candidate_hypothesis_dir(project_dir, record)
        target = _materialization_target(hdir, mode)
        if target.exists():
            continue
        pending_ids.append(str(record["hypothesis_id"]))
    return RootMaterializationPlan(
        mode=mode,
        role="materializations",
        model=model,
        provider=spec.provider,
        provider_kind=provider_config.get("kind"),
        resolved_model=spec.model,
        reasoning_effort=spec.reasoning_effort,
        timeout_seconds=provider_config.get("timeout_seconds"),
        sandbox="read_only",
        candidate_count=len(candidates),
        iteration_count=len(pending_ids),
        hypothesis_ids=pending_ids,
    )


def materialize_missing(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
    progress: Callable[[str, int | None], None] | None = None,
) -> int:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    created = 0
    candidates = materialization_candidates(project_dir, hypothesis_id=hypothesis_id)
    pending_total = sum(
        1
        for record in candidates
        if str(record["hypothesis_id"]) != BASELINE_HYPOTHESIS_ID
        and not _materialization_target(_candidate_hypothesis_dir(project_dir, record), mode).exists()
    )
    pending_index = 0
    for record in candidates:
        if str(record["hypothesis_id"]) == BASELINE_HYPOTHESIS_ID and mode != BASELINE_MODE:
            continue
        hdir = _candidate_hypothesis_dir(project_dir, record)
        mat_dir = hdir / "materializations"
        mat_dir.mkdir(parents=True, exist_ok=True)
        target = _materialization_target(hdir, mode)
        if target.exists():
            active_file = _manifest_active_file(hdir, mode)
            is_active_target = active_file is None or active_file == target.name
            if _is_runtime_wrapper(target):
                hypothesis = read_yaml(hdir / "hypothesis.yaml")
                if _rewrite_group_only_from_response(hdir, mat_dir, mode, target, hypothesis):
                    upsert_materialization(project_dir, hdir, mode, target, active=is_active_target)
                    created += 1
            else:
                upsert_materialization(project_dir, hdir, mode, target, active=is_active_target)
            continue
        timeout_seconds = int(role_options.get("timeout_seconds") or 900)
        pending_index += 1
        progress_prefix = f"ROOT materialization {hdir.name} {mode} ({pending_index}/{pending_total})"
        if progress is not None:
            progress(f"Materializing {progress_prefix} with {model}...", timeout_seconds)

        def invocation_progress(message: str, *, prefix: str = progress_prefix) -> None:
            if progress is not None:
                progress(f"{prefix}: {message}", timeout_seconds)

        hypothesis = read_yaml(hdir / "hypothesis.yaml")
        template_id = f"root.materialize-{mode}"
        rendered = render_template(
            project_dir,
            template_id,
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
        _validate_materialization_prompt(rendered["rendered"])
        response = run_model_invocation(
            ModelInvocation(
                role="materializations",
                model=model,
                prompt=rendered["rendered"],
                template_id=rendered["template_id"],
                template_path=rendered["template_path"],
                template_hash=rendered["template_hash"],
                rendered_prompt_hash=rendered["rendered_hash"],
                cwd=repo_root_for_project(project_dir),
                sandbox="read_only",
                metadata={"mode": mode},
                progress=invocation_progress if progress is not None else None,
            ),
            artifact_dir=mat_dir,
            providers=providers,
            role_options=role_options,
            response_prefix=f"{mode}-001",
        )
        group_code = _parse_code(response.text)
        try:
            validate_group_code_source(group_code)
        except ValueError as exc:
            response_path = mat_dir / f"{mode}-001.response.md"
            raise ValueError(
                f"Invalid materialization for hypothesis {hdir.name} ({mode}); "
                f"response={response_path}: {exc}"
            ) from exc
        atomic_write_text(target, group_code)
        _update_manifest(hdir, mode, target, hypothesis)
        upsert_materialization(project_dir, hdir, mode, target)
        created += 1
    return created


def _candidate_hypothesis_dir(project_dir: Path, record: dict[str, object]) -> Path:
    return project_dir / str(record["path"]).rsplit("/", 1)[0]


def _materialization_target(hdir: Path, mode: str) -> Path:
    return hdir / "materializations" / f"{mode}-001.py"


def _is_runtime_wrapper(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "# Generated AutoGluon materialization." in source or "# Generated legacy materialization." in source


def _rewrite_group_only_from_response(
    hdir: Path,
    mat_dir: Path,
    mode: str,
    target: Path,
    hypothesis: dict[str, object],
) -> bool:
    response_path = mat_dir / f"{mode}-001.response.md"
    if not response_path.exists():
        return False
    group_code = _parse_code(response_path.read_text(encoding="utf-8"))
    try:
        validate_group_code_source(group_code)
    except ValueError as exc:
        raise ValueError(
            f"Invalid materialization for hypothesis {hdir.name} ({mode}); "
            f"response={response_path}: {exc}"
        ) from exc
    atomic_write_text(target, group_code)
    _update_manifest(hdir, mode, target, hypothesis)
    return True


def _validate_materialization_prompt(prompt: str) -> None:
    banned = [
        "- hypothesis_id:",
        "# Project Target",
        "sample_submission",
        "def add_group_name(raw, deps, aux, ctx)",
        "- depends_on: []",
    ]
    found = [item for item in banned if item in prompt]
    if found:
        raise ValueError(f"Materialization prompt contains stale/banned content: {found}")


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


def _manifest_active_file(hdir: Path, mode: str) -> str | None:
    manifest = read_yaml(hdir / "manifest.yaml")
    versions = manifest.get("materializations")
    if not isinstance(versions, dict):
        return None
    entry = versions.get(mode)
    if not isinstance(entry, dict):
        return None
    active = entry.get("active")
    return active if isinstance(active, str) else None
