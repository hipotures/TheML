from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.ai import ModelInvocation, run_model_invocation
from tml.ai.models import resolve_model_spec, resolve_role_model
from tml.core.config import repo_models_for_project, repo_providers_for_project, repo_root_for_project
from tml.db.state import (
    materialization_candidates,
    materialization_rows,
    materialization_status,
    upsert_failed_materialization,
    upsert_materialization,
    upsert_project,
)
from tml.features.validation import validate_group_code_source
from tml.prompts.context import project_prompt_context
from tml.prompts.renderer import render_template
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml

from .baseline import BASELINE_HYPOTHESIS_ID, BASELINE_MODE, ensure_root_baseline
from .revisions import (
    active_materialization_file,
    append_materialization,
    load_revision,
    materialization_entries,
    migrate_root_revisions,
    normalize_hypothesis_id,
    revision_records,
)


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
    existing_count: int
    iteration_count: int
    hypothesis_ids: list[str]
    revisions: list[int]
    target_files: list[str]
    version: str | None


def root_materialization_plan(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
    version: str | None = None,
    revision: int | None = None,
) -> RootMaterializationPlan:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    migrate_root_revisions(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    spec = resolve_model_spec(model, providers)
    provider_config = {**(spec.provider_config or {}), **role_options}
    candidates = _materialization_revision_candidates(project_dir, mode, hypothesis_id=hypothesis_id, revision=revision)
    pending_ids: list[str] = []
    revisions: list[int] = []
    target_files: list[str] = []
    existing_count = 0
    for record in candidates:
        if str(record["hypothesis_id"]) == BASELINE_HYPOTHESIS_ID and mode != BASELINE_MODE:
            continue
        hdir = project_dir / "hypotheses" / str(record["hypothesis_id"])
        target = _materialization_target(project_dir, hdir, mode, version=version)
        if _revision_has_materialization(hdir, mode, int(record["revision"])):
            existing_count += 1
            continue
        pending_ids.append(str(record["hypothesis_id"]))
        revisions.append(int(record["revision"]))
        target_files.append(target.name)
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
        existing_count=existing_count,
        iteration_count=len(pending_ids),
        hypothesis_ids=pending_ids,
        revisions=revisions,
        target_files=target_files,
        version=version,
    )


def materialize_missing(
    project_dir: Path,
    mode: str,
    hypothesis_id: str | None = None,
    version: str | None = None,
    revision: int | None = None,
    progress: Callable[[str, int | None], None] | None = None,
) -> int:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    migrate_root_revisions(project_dir)
    models = repo_models_for_project(project_dir)
    model, role_options = resolve_role_model(models, "materializations", fallback_role="code")
    providers = repo_providers_for_project(project_dir)
    created = 0
    candidates = _materialization_revision_candidates(project_dir, mode, hypothesis_id=hypothesis_id, revision=revision)
    pending_total = sum(1 for record in candidates if not _revision_has_materialization(project_dir / "hypotheses" / str(record["hypothesis_id"]), mode, int(record["revision"])))
    pending_index = 0
    for record in candidates:
        if str(record["hypothesis_id"]) == BASELINE_HYPOTHESIS_ID and mode != BASELINE_MODE:
            continue
        hdir = project_dir / "hypotheses" / str(record["hypothesis_id"])
        selected_revision = int(record["revision"])
        mat_dir = hdir / "materializations"
        mat_dir.mkdir(parents=True, exist_ok=True)
        target = _materialization_target(project_dir, hdir, mode, version=version)
        if _revision_has_materialization(hdir, mode, selected_revision):
            continue
        if target.exists():
            active_file = active_materialization_file(hdir, mode)
            is_active_target = active_file is None or active_file == target.name
            if _is_runtime_wrapper(target):
                hypothesis = load_revision(hdir, selected_revision).payload
                if _rewrite_group_only_from_response(hdir, mat_dir, mode, target, hypothesis):
                    append_materialization(hdir, mode, target, revision=selected_revision, active=is_active_target)
                    upsert_materialization(project_dir, hdir, mode, target, active=is_active_target, hypothesis_revision=selected_revision)
                    created += 1
            else:
                append_materialization(hdir, mode, target, revision=selected_revision, active=is_active_target)
                upsert_materialization(project_dir, hdir, mode, target, active=is_active_target, hypothesis_revision=selected_revision)
            continue
        timeout_seconds = int(role_options.get("timeout_seconds") or 900)
        pending_index += 1
        progress_prefix = f"ROOT materialization {hdir.name} rev {selected_revision} {mode} ({pending_index}/{pending_total})"
        if progress is not None:
            progress(f"Materializing {progress_prefix} with {model}...", timeout_seconds)

        def invocation_progress(message: str, *, prefix: str = progress_prefix) -> None:
            if progress is not None:
                progress(f"{prefix}: {message}", timeout_seconds)

        hypothesis = load_revision(hdir, selected_revision).payload
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
                metadata={"mode": mode, "hypothesis_id": hdir.name, "hypothesis_revision": selected_revision},
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
                f"Invalid materialization for hypothesis {hdir.name} ({mode}); "
                f"response={_project_path_text(project_dir, response_path)}: {exc}"
            )
            atomic_write_text(mat_dir / f"{target.stem}.error.txt", error_text + "\n")
            upsert_failed_materialization(project_dir, hdir, mode, target.name, code_text=group_code)
            if progress is not None:
                progress(f"{progress_prefix}: failed validation: {exc}", None)
            continue
        atomic_write_text(target, group_code)
        _update_manifest(hdir, mode, target, hypothesis, revision=selected_revision)
        is_active_target = active_materialization_file(hdir, mode) == target.name
        upsert_materialization(project_dir, hdir, mode, target, active=is_active_target, hypothesis_revision=selected_revision)
        created += 1
    return created


def _materialization_revision_candidates(
    project_dir: Path,
    mode: str,
    *,
    hypothesis_id: str | None,
    revision: int | None,
) -> list[dict[str, object]]:
    _ = mode
    target_id = normalize_hypothesis_id(hypothesis_id) if hypothesis_id else None
    candidates: list[dict[str, object]] = []
    for hdir in sorted((project_dir / "hypotheses").glob("*")):
        if not hdir.is_dir():
            continue
        if target_id and hdir.name != target_id:
            continue
        for record in revision_records(hdir):
            if revision is not None and record.revision != revision:
                continue
            candidates.append({"hypothesis_id": hdir.name, "revision": record.revision})
    return candidates


def _revision_has_materialization(hdir: Path, mode: str, revision: int) -> bool:
    return any(int(entry.get("revision") or 0) == revision for entry in materialization_entries(hdir, mode))


def _candidate_hypothesis_dir(project_dir: Path, record: dict[str, object]) -> Path:
    return project_dir / str(record["path"]).rsplit("/", 1)[0]


def _project_path_text(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.name


def _has_materialization_record(
    project_dir: Path,
    record: dict[str, object],
    mode: str,
    target: Path,
    *,
    version: str | None,
) -> bool:
    hypothesis_id = str(record["hypothesis_id"])
    status = materialization_status(
        project_dir,
        hypothesis_id=hypothesis_id,
        mode=mode,
        file_name=target.name,
    )
    if status in {"failed", "bug", "superseded"}:
        return True
    if version is None:
        return any(bool(row.get("active")) for row in materialization_rows(project_dir, mode=mode, hypothesis_id=hypothesis_id))
    return False


def _needs_materialization(
    project_dir: Path,
    record: dict[str, object],
    mode: str,
    *,
    version: str | None = None,
) -> bool:
    hdir = _candidate_hypothesis_dir(project_dir, record)
    target = _materialization_target(project_dir, hdir, mode, version=version)
    return not target.exists() and not _has_materialization_record(
        project_dir,
        record,
        mode,
        target,
        version=version,
    )


def _materialization_target(project_dir: Path, hdir: Path, mode: str, *, version: str | None = None) -> Path:
    if version is not None:
        text = str(version).strip()
        if text == "new":
            return _new_materialization_target(project_dir, hdir, mode)
        if text.endswith(".py"):
            return hdir / "materializations" / text
        if text.isdigit():
            return hdir / "materializations" / f"{mode}-{int(text):03d}.py"
        if text.startswith(f"{mode}-") and text.removeprefix(f"{mode}-").isdigit():
            return hdir / "materializations" / f"{text}.py"
        raise ValueError("version must be 'new', a number like 2, or a file like autogluon-002.py")
    return hdir / "materializations" / f"{mode}-001.py"


def _new_materialization_target(project_dir: Path, hdir: Path, mode: str) -> Path:
    mat_dir = hdir / "materializations"
    max_index = 0
    for path in mat_dir.glob(f"{mode}-*.py"):
        suffix = path.stem.removeprefix(f"{mode}-")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    for row in materialization_rows(project_dir, mode=mode, hypothesis_id=hdir.name):
        file_name = str(row.get("file") or "")
        suffix = Path(file_name).stem.removeprefix(f"{mode}-")
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return mat_dir / f"{mode}-{max_index + 1:03d}.py"


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
            f"response={_project_path_text(hdir.parents[1], response_path)}: {exc}"
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


def _update_manifest(hdir: Path, mode: str, path: Path, hypothesis: dict[str, object], *, revision: int | None = None) -> None:
    append_materialization(
        hdir,
        mode,
        path,
        revision=int(revision or hypothesis.get("revision") or 1),
        active=active_materialization_file(hdir, mode) is None,
    )


def _manifest_active_file(hdir: Path, mode: str) -> str | None:
    return active_materialization_file(hdir, mode)
