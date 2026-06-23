from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
from tml.core.config import repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.db.state import (
    bugfix_candidates,
    materialization_rows,
    upsert_failed_materialization,
    upsert_materialization,
    upsert_project,
)
from tml.features.validation import validate_group_code_source
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import read_yaml

from .baseline import ensure_root_baseline
from .materialize import _parse_code, _update_manifest
from .revisions import load_revision, materialization_revision


@dataclass(frozen=True)
class RootBugfixPlan:
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
    source_files: list[str]
    target_files: list[str]


def root_bugfix_plan(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
) -> RootBugfixPlan:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    candidates = bugfix_candidates(project_dir, mode, hypothesis_id=hypothesis_id)
    source_files: list[str] = []
    target_files: list[str] = []
    hypothesis_ids: list[str] = []
    for record in candidates:
        hdir = _candidate_hypothesis_dir(project_dir, record)
        source = hdir / "materializations" / str(record["file"])
        source_files.append(source.name)
        target_files.append(_next_materialization_path(project_dir, source.parent, mode, hdir.name).name)
        hypothesis_ids.append(str(record["hypothesis_id"]))
    return RootBugfixPlan(
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
        iteration_count=len(candidates),
        hypothesis_ids=hypothesis_ids,
        source_files=source_files,
        target_files=target_files,
    )


def bugfix_failed_materializations(
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
    timeout_seconds = int(role_options.get("timeout_seconds") or 900)
    candidates = bugfix_candidates(project_dir, mode, hypothesis_id=hypothesis_id)
    total = len(candidates)
    created = 0
    for index, record in enumerate(candidates, start=1):
        hid = str(record["hypothesis_id"])
        hdir = _candidate_hypothesis_dir(project_dir, record)
        mat_dir = hdir / "materializations"
        source = mat_dir / str(record["file"])
        target = _next_materialization_path(project_dir, mat_dir, mode, hid)
        progress_prefix = f"ROOT bugfix {hid} {mode} ({index}/{total})"
        if progress is not None:
            progress(f"Fixing {progress_prefix} with {model}...", timeout_seconds)

        def invocation_progress(message: str, *, prefix: str = progress_prefix) -> None:
            if progress is not None:
                progress(f"{prefix}: {message}", timeout_seconds)

        bugfix_context = {
            "mode": mode,
            "hypothesis_id": hid,
            "source_file": source.name,
            "target_file": target.name,
            "source_code": _source_materialization_code(mat_dir, source.name),
            "node_id": str(record.get("node_id") or ""),
            "run_id": str(record.get("run_id") or ""),
            "error_text": _failure_context(project_dir, record),
        }
        rendered = render_template(
            project_dir,
            "root.bugfix",
            project_prompt_context(project_dir, bugfix=bugfix_context),
        )
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
                metadata={
                    "mode": mode,
                    "hypothesis_id": hid,
                    "bugfix_source_file": source.name,
                    "bugfix_source_node_id": bugfix_context["node_id"],
                },
                progress=invocation_progress if progress is not None else None,
            ),
            artifact_dir=mat_dir,
            providers=providers,
            role_options=role_options,
            response_prefix=target.stem,
        )
        group_code = _parse_code(response.text)
        try:
            validate_group_code_source(group_code)
        except (SyntaxError, ValueError) as exc:
            response_path = mat_dir / f"{target.stem}.response.md"
            error_text = (
                f"Invalid bugfix for hypothesis {hid} ({mode}); "
                f"response={_project_path_text(project_dir, response_path)}: {exc}"
            )
            atomic_write_text(mat_dir / f"{target.stem}.error.txt", error_text + "\n")
            upsert_failed_materialization(project_dir, hdir, mode, target.name, code_text=group_code)
            if progress is not None:
                progress(f"{progress_prefix}: failed validation: {exc}", None)
            continue
        atomic_write_text(target, group_code)
        source_revision = materialization_revision(hdir, mode, source.name)
        hypothesis = load_revision(hdir, source_revision).payload
        _update_manifest(hdir, mode, target, hypothesis, revision=source_revision)
        upsert_materialization(
            project_dir,
            hdir,
            mode,
            target,
            status="fixed",
            active=True,
            source_node_id=bugfix_context["node_id"],
            fixed_from_file=source.name,
            fixed_from_code_hash=str(record["code_hash"]),
            hypothesis_revision=source_revision,
        )
        created += 1
    return created


def _candidate_hypothesis_dir(project_dir: Path, record: dict[str, object]) -> Path:
    return project_dir / str(record["path"]).rsplit("/", 1)[0]


def _project_path_text(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.name


def _next_materialization_path(project_dir: Path, mat_dir: Path, mode: str, hypothesis_id: str) -> Path:
    highest = 0
    for path in mat_dir.glob(f"{mode}-*.py"):
        try:
            number = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        highest = max(highest, number)
    for row in materialization_rows(project_dir, mode=mode, hypothesis_id=hypothesis_id):
        try:
            number = int(Path(str(row.get("file") or "")).stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        highest = max(highest, number)
    return mat_dir / f"{mode}-{highest + 1:03d}.py"


def _source_materialization_code(mat_dir: Path, file_name: str) -> str:
    source = mat_dir / file_name
    if source.exists():
        return source.read_text(encoding="utf-8")
    response = mat_dir / f"{Path(file_name).stem}.response.md"
    if response.exists():
        return _parse_code(response.read_text(encoding="utf-8"))
    raise ValueError(f"Cannot find source materialization code or response text for {file_name}")


def _failure_context(project_dir: Path, record: dict[str, object]) -> str:
    node_path = record.get("node_path")
    if not node_path:
        hdir = _candidate_hypothesis_dir(project_dir, record)
        stem = Path(str(record.get("file") or "")).stem
        error_path = hdir / "materializations" / f"{stem}.error.txt"
        if error_path.exists():
            return _sanitize_error_text(project_dir, error_path.read_text(encoding="utf-8", errors="replace").strip())
        return "No failed node artifact path was recorded. The materialization failed during validation before execution."
    node_dir = project_dir / str(node_path)
    parts: list[str] = []
    failed = read_yaml(node_dir / "failed.yaml")
    if failed.get("error"):
        parts.append(f"failed.yaml error:\n{failed['error']}")
    result = read_yaml(node_dir / "result.yaml")
    if result.get("error") and result.get("error") != failed.get("error"):
        parts.append(f"result.yaml error:\n{result['error']}")
    for name in ("stderr.log", "stdout.log"):
        path = node_dir / name
        if path.exists():
            text = _truncate(path.read_text(encoding="utf-8", errors="replace"), 12000)
            if text.strip():
                parts.append(f"{name}:\n{text}")
    return "\n\n".join(parts) if parts else "No error text was found in the failed node artifacts."


def _sanitize_error_text(project_dir: Path, text: str) -> str:
    return text.replace(str(project_dir) + "/", "")


def _truncate(text: str, limit: int) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[-limit:]
