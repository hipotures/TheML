from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
from tml.core.config import active_mode, load_project_config, repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.core.errors import TmlError
from tml.core.ids import run_id
from tml.core.metadata import render_project_metadata_prompt
from tml.features.validation import validate_group_code_source
from tml.hypotheses.materialize import _parse_code
from tml.hypotheses.revisions import latest_revision_record, migrate_hypothesis_dir, normalize_hypothesis_id, revision_records
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_json, atomic_write_text


def render_prompt(
    project_dir: Path,
    *,
    target: str | None = None,
    stage: str | None = None,
    hypothesis_id: str | None = None,
    output_path: Path | None = None,
    tmp_root: Path | None = None,
) -> Path:
    rendered = _render_for_target(project_dir, target=target, stage=stage, hypothesis_id=hypothesis_id)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output_path, rendered["rendered"])
        return output_path
    out_dir = _tmp_dir(project_dir, tmp_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "request.md"
    atomic_write_text(path, rendered["rendered"])
    atomic_write_json(out_dir / "request.json", _request_json(project_dir, rendered, "render"))
    return path


def probe_prompt(
    project_dir: Path,
    *,
    tmp: bool,
    target: str | None = None,
    stage: str | None = None,
    model_override: str | None = None,
    profile_overrides: dict[str, object] | None = None,
    tmp_root: Path | None = None,
    progress: Callable[[str, int | None], None] | None = None,
) -> Path:
    _ = tmp
    out_dir = _tmp_dir(project_dir, tmp_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    role = _role_for_target(target, stage)
    models = repo_models_for_project(project_dir)
    providers = repo_providers_for_project(project_dir)
    model, role_options = resolve_role_model(
        models,
        role,
        fallback_role=_fallback_role_for_role(role),
        model_override=model_override,
    )
    timeout_seconds = int(role_options.get("timeout_seconds") or 1)
    rendered = _render_for_target(project_dir, target=target, stage=stage)
    if progress is not None:
        progress(f"Output dir: {out_dir}", timeout_seconds)
        progress(f"Request will be written to {out_dir / 'request.md'}", timeout_seconds)
        progress(f"Sending {role} prompt to {model}...", timeout_seconds)
    invocation_metadata: dict[str, object] = {"kind": "probe"}
    if rendered.get("hypothesis_id"):
        invocation_metadata["hypothesis_id"] = rendered["hypothesis_id"]
    if rendered.get("hypothesis_path"):
        invocation_metadata["hypothesis_path"] = rendered["hypothesis_path"]
    response = run_model_invocation(
        ModelInvocation(
            role=role,
            model=model,
            prompt=rendered["rendered"],
            template_id=rendered["template_id"],
            template_path=rendered["template_path"],
            template_hash=rendered["template_hash"],
            rendered_prompt_hash=rendered["rendered_hash"],
            cwd=repo_root_for_project(project_dir),
            sandbox="read_only",
            metadata=invocation_metadata,
            progress=(lambda message: progress(message, timeout_seconds)) if progress is not None else None,
        ),
        artifact_dir=out_dir,
        providers=providers,
        role_options=role_options,
    )
    if target and stage == "code":
        mode = active_mode(load_project_config(project_dir))
        code = _parse_code(response.text)
        validate_group_code_source(code)
        atomic_write_text(out_dir / f"{mode}-001.py", code)
        atomic_write_text(
            out_dir / f"{mode}-001-runtime.py",
            build_wrapped_materialization_source(
                mode,
                code,
                project_dir,
                profile_overrides=profile_overrides,
            ),
        )
    return out_dir


def _render_for_target(
    project_dir: Path,
    *,
    target: str | None,
    stage: str | None,
    hypothesis_id: str | None = None,
) -> dict[str, str]:
    config = load_project_config(project_dir)
    if target is None and stage is None:
        return render_template(project_dir, "root.hypothesis", project_prompt_context(project_dir, count=1))
    if target in {"revise", "root.revise-hypothesis"} and stage is None:
        if not hypothesis_id:
            raise TmlError("Missing required parameter: id=<hypothesis>.")
        hid = normalize_hypothesis_id(hypothesis_id)
        hdir = project_dir / "hypotheses" / hid
        records = revision_records(hdir)
        if not records:
            raise FileNotFoundError(f"No canonical ROOT revisions in {hdir}")
        models = repo_models_for_project(project_dir)
        providers = repo_providers_for_project(project_dir)
        model, role_options = resolve_role_model(models, "hypothesis")
        spec = resolve_model_spec(model, providers)
        provider_config = {**(spec.provider_config or {}), **role_options}
        rendered = render_template(
            project_dir,
            "root.revise-hypothesis",
            project_prompt_context(
                project_dir,
                hypothesis_id=hid,
                previous_revisions=[record.payload for record in records],
                web_search_enabled=_web_search_enabled(provider_config.get("web_search")),
            ),
        )
        rendered["hypothesis_id"] = hid
        return rendered
    if target == "project" and stage == "metadata":
        return render_project_metadata_prompt(
            project_dir,
            slug=str(config.get("kaggle_slug") or config.get("project_id") or project_dir.name),
        )
    if target and stage == "code":
        mode = active_mode(config)
        hypothesis, hypothesis_path = _hypothesis_for_target(project_dir, target)
        rendered = render_template(
            project_dir,
            f"root.materialize-{mode}",
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
        rendered["hypothesis_id"] = str(hypothesis.get("hypothesis_id") or target)
        rendered["hypothesis_path"] = _relative_path(project_dir, hypothesis_path) if hypothesis_path else ""
        return rendered
    raise TmlError(_unknown_prompt_target_message(target, stage))


def _role_for_target(target: str | None, stage: str | None) -> str:
    if target is None and stage is None:
        return "hypothesis"
    if target == "project" and stage == "metadata":
        return "metadata"
    if target and stage == "code":
        return "materializations"
    raise TmlError(_unknown_prompt_target_message(target, stage))


def _fallback_role_for_role(role: str) -> str | None:
    if role == "materializations":
        return "code"
    if role in {"hypothesis", "metadata"}:
        return None
    return "hypothesis"


def _unknown_prompt_target_message(target: str | None, stage: str | None) -> str:
    supplied = " ".join(part for part in (target, stage) if part)
    return (
        f"Unknown prompt target: {supplied or '<empty>'}. "
        "Use one of: tml prompt render, tml prompt render project metadata, "
        "tml prompt render <hypothesis_id> code, "
        "tml prompt render revise id=<hypothesis>."
    )


def _web_search_enabled(value: object) -> bool:
    if str(value or "").strip().lower() in {"live", "cached"}:
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _hypothesis_for_target(project_dir: Path, target: str | None) -> tuple[dict[str, object], Path | None]:
    if target and target.isdigit():
        hdir = project_dir / "hypotheses" / target.zfill(6)
        migrate_hypothesis_dir(project_dir, hdir)
        if list(hdir.glob("??-hypothesis.yaml")):
            from tml.utils.yaml_io import read_yaml

            record = latest_revision_record(hdir)
            return read_yaml(record.path), record.path
    return {"hypothesis_id": target or "next"}, None


def _request_json(project_dir: Path, rendered: dict[str, str], kind: str, model: str = "mock") -> dict[str, object]:
    return {
        "kind": kind,
        "model": model,
        "provider": model,
        "messages": [{"role": "user", "content": rendered["rendered"]}],
        "template_id": rendered["template_id"],
        "template_path": rendered["template_path"],
        "template_hash": rendered["template_hash"],
        "context_hash": rendered["rendered_hash"],
        "rendered_prompt_hash": rendered["rendered_hash"],
        "project_dir": ".",
        "hypothesis_id": rendered.get("hypothesis_id"),
        "hypothesis_path": rendered.get("hypothesis_path"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _tmp_dir(project_dir: Path, tmp_root: Path | None) -> Path:
    root = tmp_root or Path("/tmp")
    return root / "tml" / project_dir.name / run_id()


def _relative_path(project_dir: Path, path: Path) -> str:
    root = repo_root_for_project(project_dir)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
