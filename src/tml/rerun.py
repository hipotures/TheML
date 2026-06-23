from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.core.config import rerun_profile_id, repo_root_for_project
from tml.core.ids import node_id
from tml.core.paths import context_path
from tml.core.profiles import load_profile, profile_hash
from tml.db.state import (
    active_or_create_run,
    branch_by_id,
    latest_run_id,
    next_node_step,
    node_record,
    root_materialization_by_code_hash,
    submission_by_sha_prefix,
    upsert_node_result,
    upsert_node_start,
    upsert_project,
)
from tml.db.submissions import find_submission_file
from tml.execution.executor import run_python_script, write_attempt_result
from tml.features.validation import validate_group_code_source
from tml.hypotheses.run import _execution_timeout_seconds
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml


@dataclass(frozen=True)
class RerunSource:
    source_kind: str
    source_id: str
    metadata_path: Path
    metadata_artifact: str
    materialization_path: Path
    code_hash: str


@dataclass(frozen=True)
class RerunPlan:
    source_submission_sha256: str
    source_local_score: float | None
    source_cv_rank: int | None
    source_public_score: float | None
    source_public_rank: int | None
    source_run_id: str | None
    source_node_id: str
    source_step: int | None
    source_profile_id: str | None
    source_kind: str
    source_id: str
    mode: str
    profile_id: str
    profile_hash: str
    profile_time_limit_seconds: int | None
    profile_preset: str | None
    profile_use_gpu: bool | None
    execution_timeout_seconds: int
    aux_enabled: bool
    run_id: str | None
    run_path: str
    next_node_step: int
    materialization_path: str
    code_hash: str


def rerun_plan(project_dir: Path, *, sha_prefix: str) -> RerunPlan:
    upsert_project(project_dir)
    source_row = submission_by_sha_prefix(project_dir, sha_prefix)
    ranked_source_row = _ranked_source_row(project_dir, source_row)
    source_node = node_record(project_dir, str(source_row["node_id"]))
    mode = str(source_row.get("mode") or source_node.get("mode") or "")
    if not mode:
        raise ValueError(f"Source submission {sha_prefix} does not record a materialization mode.")
    profile_id = rerun_profile_id(project_dir, mode)
    profile = load_profile(project_dir, mode, profile_id)
    source = _source_from_submission(project_dir, source_row, source_node, mode=mode)
    run_id_value = latest_run_id(project_dir)
    source_local_score = _optional_float(ranked_source_row.get("local_score"))
    source_public_score = _optional_float(ranked_source_row.get("public_score"))
    return RerunPlan(
        source_submission_sha256=str(source_row["submission_sha256"]),
        source_local_score=source_local_score,
        source_cv_rank=_optional_int(ranked_source_row.get("cv_rank")) if source_local_score is not None else None,
        source_public_score=source_public_score,
        source_public_rank=_optional_int(ranked_source_row.get("public_rank") or ranked_source_row.get("computed_public_rank"))
        if source_public_score is not None
        else None,
        source_run_id=_optional_str(source_row.get("run_id")),
        source_node_id=str(source_row["node_id"]),
        source_step=_optional_int(source_row.get("step")),
        source_profile_id=_optional_str(source_row.get("profile_id")),
        source_kind=source.source_kind,
        source_id=source.source_id,
        mode=mode,
        profile_id=profile_id,
        profile_hash=profile_hash(project_dir, mode, profile_id),
        profile_time_limit_seconds=_optional_int(profile.get("time_limit") or profile.get("time")),
        profile_preset=_optional_str(profile.get("presets") or profile.get("preset")),
        profile_use_gpu=_optional_bool(profile.get("use_gpu")),
        execution_timeout_seconds=_execution_timeout_seconds(profile),
        aux_enabled=_aux_enabled(project_dir, profile),
        run_id=run_id_value,
        run_path=f"runs/{run_id_value}" if run_id_value else "new run on start",
        next_node_step=next_node_step(project_dir, run_id_value) if run_id_value else 1,
        materialization_path=_project_relative(project_dir, source.materialization_path),
        code_hash=source.code_hash,
    )


def rerun_submission(
    project_dir: Path,
    *,
    sha_prefix: str,
    progress: Callable[[str], None] | None = None,
) -> str:
    plan = rerun_plan(project_dir, sha_prefix=sha_prefix)
    source_row = submission_by_sha_prefix(project_dir, sha_prefix)
    source_node = node_record(project_dir, str(source_row["node_id"]))
    source = _source_from_submission(project_dir, source_row, source_node, mode=plan.mode)
    run = active_or_create_run(project_dir)
    step = next_node_step(project_dir, run.name)
    nid = node_id(step)
    node_dir = run / "artifacts" / nid
    node_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    start_payload = {
        "schema_version": 1,
        "node_id": nid,
        "run_id": run.name,
        "step": int(nid.rsplit("-", 1)[-1]),
        "kind": "rerun",
        "source_kind": source.source_kind,
        "source_id": source.source_id,
        "source_submission_sha256": plan.source_submission_sha256,
        "source_run_id": plan.source_run_id,
        "source_node_id": plan.source_node_id,
        "source_step": plan.source_step,
        "source_profile_id": plan.source_profile_id,
        "mode": plan.mode,
        "profile_id": plan.profile_id,
        "hypothesis_id": source.source_id if source.source_kind == "root" else None,
        "branch_id": source.source_id if source.source_kind == "branch" else None,
        "created_at": created_at,
    }
    write_yaml(node_dir / "node.start.yaml", start_payload)
    upsert_node_start(project_dir, node_dir, start_payload)
    shutil.copy2(source.metadata_path, node_dir / source.metadata_artifact)
    group_code = source.materialization_path.read_text(encoding="utf-8")
    validate_group_code_source(group_code)
    atomic_write_text(
        node_dir / "02-code.py",
        build_wrapped_materialization_source(
            plan.mode,
            group_code,
            project_dir,
            profile_id=plan.profile_id,
        ),
    )
    write_yaml(node_dir / "started.yaml", {"created_at": datetime.now().isoformat(timespec="seconds")})
    if progress is not None:
        progress(f"RERUN {source.source_kind} {source.source_id} {plan.mode}: executing node {nid}")
    result = run_python_script(
        node_dir / "02-code.py",
        node_dir / "work",
        timeout_seconds=plan.execution_timeout_seconds,
        cwd=repo_root_for_project(project_dir) if plan.mode == "autogluon" else None,
    )
    write_attempt_result(node_dir, result)
    finished_at = datetime.now().isoformat(timespec="seconds")
    status = "complete" if result.status == "ok" else "failed"
    submit_status = "not_submitted"
    duplicate_source_hash = False
    submission_path = find_submission_file(node_dir)
    if result.status == "ok" and submission_path is not None:
        submission_hash = sha256_file(submission_path)
        duplicate_source_hash = submission_hash == plan.source_submission_sha256
        if duplicate_source_hash:
            status = "suspicious"
            submit_status = "blocked_duplicate_source"
    _write_rerun_manifest(
        node_dir,
        plan=plan,
        source=source,
        result=result,
        status=status,
        submit_status=submit_status,
        duplicate_source_hash=duplicate_source_hash,
    )
    upsert_node_result(
        project_dir,
        node_dir,
        status=status,
        metric=result.metric,
        code_hash=source.code_hash,
        finished_at=finished_at,
        error=result.error,
    )
    if progress is not None:
        progress(f"RERUN {source.source_kind} {source.source_id} {plan.mode}: {status}")
    return nid


def _source_from_submission(
    project_dir: Path,
    source_row: dict[str, object],
    source_node: dict[str, object],
    *,
    mode: str,
) -> RerunSource:
    branch_id = _optional_str(source_node.get("branch_id") or source_row.get("branch_id"))
    if branch_id:
        branch = branch_by_id(project_dir, branch_id)
        if branch is None:
            raise ValueError(f"Source branch {branch_id} no longer exists.")
        branch_yaml = project_dir / str(branch["path"])
        materialization_path = branch_yaml.parent / "materializations" / str(branch["materialization_file"])
        return RerunSource(
            source_kind="branch",
            source_id=branch_id,
            metadata_path=branch_yaml,
            metadata_artifact="01-branch.yaml",
            materialization_path=materialization_path,
            code_hash=str(branch["code_hash"]),
        )

    hypothesis_id = _optional_str(source_node.get("hypothesis_id") or source_row.get("hypothesis_id"))
    code_hash = _optional_str(source_row.get("code_hash"))
    if not hypothesis_id or not code_hash:
        raise ValueError(f"Source submission {source_row.get('submission_sha256')} is missing root source metadata.")
    materialization = root_materialization_by_code_hash(
        project_dir,
        hypothesis_id=hypothesis_id,
        mode=mode,
        code_hash=code_hash,
    )
    hypothesis_yaml = project_dir / str(materialization["hypothesis_path"])
    materialization_path = hypothesis_yaml.parent / "materializations" / str(materialization["file"])
    return RerunSource(
        source_kind="root",
        source_id=hypothesis_id,
        metadata_path=hypothesis_yaml,
        metadata_artifact="01-hypothesis.yaml",
        materialization_path=materialization_path,
        code_hash=str(materialization["code_hash"]),
    )


def _ranked_source_row(project_dir: Path, source_row: dict[str, object]) -> dict[str, object]:
    from tml.db.state import submission_rows

    source_node_id = str(source_row.get("node_id") or "")
    source_path = str(source_row.get("submission_path") or "")
    for row in submission_rows(project_dir):
        if str(row.get("node_id") or "") == source_node_id and str(row.get("submission_path") or "") == source_path:
            return row
    return source_row


def _write_rerun_manifest(
    node_dir: Path,
    *,
    plan: RerunPlan,
    source: RerunSource,
    result,
    status: str,
    submit_status: str,
    duplicate_source_hash: bool,
) -> None:
    manifest = {
        "schema_version": 1,
        "node_id": node_dir.name,
        "kind": "rerun",
        "source_kind": source.source_kind,
        "source_id": source.source_id,
        "source_submission_sha256": plan.source_submission_sha256,
        "source_run_id": plan.source_run_id,
        "source_node_id": plan.source_node_id,
        "source_step": plan.source_step,
        "source_profile_id": plan.source_profile_id,
        "hypothesis_id": source.source_id if source.source_kind == "root" else None,
        "branch_id": source.source_id if source.source_kind == "branch" else None,
        "mode": plan.mode,
        "profile_id": plan.profile_id,
        "code_hash": source.code_hash,
        "metric": result.metric,
        "submit_status": submit_status,
        "duplicate_source_submission_sha256": duplicate_source_hash,
        "files": {
            source.source_kind: source.metadata_artifact,
            "code": "02-code.py",
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if result.error:
        manifest["error"] = result.error
    if result.payload is not None:
        manifest["result_payload"] = result.payload
        run_stats = result.payload.get("run_stats")
        if isinstance(run_stats, dict):
            manifest["run_stats"] = run_stats
    write_yaml(node_dir / "artifact-manifest.yaml", manifest)
    if status == "complete" or status == "suspicious":
        write_yaml(node_dir / "node.done.yaml", {**manifest, "status": status})
    else:
        write_yaml(node_dir / "failed.yaml", {**manifest, "status": status})


def _aux_enabled(project_dir: Path, profile: dict[str, object]) -> bool:
    try:
        root_config = read_yaml(context_path(repo_root_for_project(project_dir)))
    except Exception:
        root_config = {}
    external = root_config.get("external") if isinstance(root_config.get("external"), dict) else {}
    if bool(external.get("enabled", False)):
        return True
    return bool(profile.get("aux_file") or profile.get("auxiliary_file"))


def _project_relative(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
