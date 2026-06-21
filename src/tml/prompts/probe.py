from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_role_model
from tml.core.config import active_mode, load_project_config, repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.core.errors import TmlError
from tml.core.ids import run_id
from tml.core.metadata import render_project_metadata_prompt
from tml.features.validation import validate_group_code_source
from tml.hypotheses.materialize import _parse_code
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_json, atomic_write_text


def render_prompt(
    project_dir: Path,
    *,
    target: str | None = None,
    stage: str | None = None,
    tmp_root: Path | None = None,
) -> Path:
    out_dir = _tmp_dir(project_dir, tmp_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = _render_for_target(project_dir, target=target, stage=stage)
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
            metadata={"kind": "probe"},
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
        atomic_write_text(out_dir / f"{mode}-001.py", _probe_wrapper_source(mode, code, project_dir))
    return out_dir


def _render_for_target(project_dir: Path, *, target: str | None, stage: str | None) -> dict[str, str]:
    config = load_project_config(project_dir)
    if target is None and stage is None:
        return render_template(project_dir, "root.hypothesis", project_prompt_context(project_dir, count=1))
    if target == "project" and stage == "metadata":
        return render_project_metadata_prompt(
            project_dir,
            slug=str(config.get("kaggle_slug") or config.get("project_id") or project_dir.name),
        )
    if target and stage == "code":
        mode = active_mode(config)
        hypothesis = _hypothesis_for_target(project_dir, target)
        return render_template(
            project_dir,
            f"root.materialize-{mode}",
            project_prompt_context(project_dir, hypothesis=hypothesis),
        )
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
        "tml prompt render <hypothesis_id> code."
    )


def _hypothesis_for_target(project_dir: Path, target: str | None) -> dict[str, object]:
    if target and target.isdigit():
        path = project_dir / "hypotheses" / target.zfill(6) / "hypothesis.yaml"
        if path.exists():
            from tml.utils.yaml_io import read_yaml

            return read_yaml(path)
    return {"hypothesis_id": target or "next"}


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
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _tmp_dir(project_dir: Path, tmp_root: Path | None) -> Path:
    root = tmp_root or Path("/tmp")
    return root / "tml" / project_dir.name / run_id()


def _probe_wrapper_source(mode: str, group_code: str, project_dir: Path) -> str:
    project_literal = repr(str(project_dir.resolve()))
    if mode == "legacy":
        runner_import = "from tml.execution.executor import run_legacy_group_materialization"
        runner_call = (
            "run_legacy_group_materialization("
            "code_path=Path(__file__), project_dir=project_dir, work_dir=work_dir)"
        )
    else:
        runner_import = "from tml.execution.autogluon_wrapper import run_autogluon_materialization"
        runner_call = (
            "run_autogluon_materialization("
            "code_path=Path(__file__), project_dir=project_dir, work_dir=work_dir)"
        )
    return (
        "# TheML probe artifact.\n"
        "# Generated feature-group code is below.\n"
        f"# Fixed wrapper implementation: src/tml/execution/{'executor.py' if mode == 'legacy' else 'autogluon_wrapper.py'}.\n"
        "# The __main__ block is a local preview wrapper so this file can be\n"
        "# inspected or run manually without changing the real materialization.\n\n"
        f"{group_code.rstrip()}\n\n\n"
        "if __name__ == \"__main__\":\n"
        "    import json\n"
        "    from dataclasses import asdict\n"
        "    from pathlib import Path\n\n"
        f"    {runner_import}\n\n"
        f"    project_dir = Path({project_literal})\n"
        f"    work_dir = Path(__file__).with_suffix(\"\") / \"{mode}-probe-workdir\"\n"
        f"    result = {runner_call}\n"
        "    print(\"TML_PROBE_RESULT_JSON: \" + json.dumps(asdict(result), default=str, sort_keys=True))\n"
        "    raise SystemExit(result.returncode)\n"
    )
