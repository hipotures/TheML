from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tml.core.config import active_mode, active_profile_id, load_project_config, repo_root_for_project
from tml.core.ids import node_id
from tml.core.profiles import profile_hash
from tml.branches.runtime_state import clear_branch_runtime_state, write_branch_runtime_state
from tml.db.state import (
    active_or_create_run,
    branch_already_evaluated,
    branch_run_candidates,
    node_record,
    evaluation_record,
    next_node_step,
    pending_branch_run_candidates,
    upsert_node_result,
    upsert_node_start,
    upsert_project,
)
from tml.execution.executor import run_python_script, write_attempt_result
from tml.features.validation import validate_group_code_source
from tml.hypotheses.run import _execution_timeout_seconds
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import read_yaml, write_yaml


@dataclass(frozen=True)
class BranchRunPlan:
    mode: str
    profile_id: str
    profile_hash: str
    execution_timeout_seconds: int
    force: bool
    run_id: str | None
    run_path: str
    next_node_step: int
    candidate_count: int
    already_evaluated_count: int
    iteration_count: int
    branch_ids: list[str]


@dataclass(frozen=True)
class BranchRunItem:
    branch_id: str
    run_status: str
    metric: float | None
    node_id: str
    run_seconds: int | None
    decision_score: float | None = None


def branch_run_plan(
    project_dir: Path,
    mode: str | None = None,
    *,
    branch_id: str | None = None,
    profile_overrides: dict[str, object] | None = None,
    force: bool = False,
) -> BranchRunPlan:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    from tml.db.state import latest_run_id

    run_id_value = latest_run_id(project_dir)
    pending_ids: list[str] = []
    already_done = 0
    candidates = branch_run_candidates(project_dir, active_run_mode, branch_id=branch_id)
    for record in candidates:
        bid = str(record["branch_id"])
        code_hash = str(record["code_hash"])
        if not force and branch_already_evaluated(project_dir, branch_id=bid, mode=active_run_mode, profile_id=profile_id, code_hash=code_hash):
            already_done += 1
            continue
        pending_ids.append(bid)
    return BranchRunPlan(
        mode=active_run_mode,
        profile_id=profile_id,
        profile_hash=profile_hash(project_dir, active_run_mode, profile_id),
        execution_timeout_seconds=_execution_timeout_seconds(config, profile_overrides),
        force=force,
        run_id=run_id_value,
        run_path=f"runs/{run_id_value}" if run_id_value else "new run on start",
        next_node_step=next_node_step(project_dir, run_id_value) if run_id_value else 1,
        candidate_count=len(candidates),
        already_evaluated_count=already_done,
        iteration_count=len(pending_ids),
        branch_ids=pending_ids,
    )


def next_pending_branch(
    project_dir: Path,
    *,
    mode: str,
    profile_id: str,
) -> dict[str, Any] | None:
    records = pending_branch_run_candidates(project_dir, mode=mode, profile_id=profile_id)
    return records[0] if records else None


def run_one_branch(
    project_dir: Path,
    mode: str | None = None,
    *,
    branch_id: str,
    profile_overrides: dict[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BranchRunItem | None:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    profile_id = active_profile_id(config, mode)
    run = active_or_create_run(project_dir)
    records = pending_branch_run_candidates(project_dir, mode=mode, profile_id=profile_id, branch_id=branch_id)
    if not records:
        return None
    next_step = next_node_step(project_dir, run.name)
    return _run_branch_record(
        project_dir,
        config=config,
        mode=mode,
        profile_id=profile_id,
        run=run,
        next_step=next_step,
        record=records[0],
        pending_index=1,
        pending_total=1,
        profile_overrides=profile_overrides,
        progress=progress,
    )


def run_missing_branches(
    project_dir: Path,
    mode: str | None = None,
    *,
    branch_id: str | None = None,
    profile_overrides: dict[str, object] | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    profile_id = active_profile_id(config, mode)
    run = active_or_create_run(project_dir)
    ran: list[str] = []
    next_step = next_node_step(project_dir, run.name)
    pending_records = (
        branch_run_candidates(project_dir, mode, branch_id=branch_id)
        if force
        else pending_branch_run_candidates(project_dir, mode=mode, profile_id=profile_id, branch_id=branch_id)
    )
    pending_total = len(pending_records)
    for pending_index, record in enumerate(pending_records, start=1):
        item = _run_branch_record(
            project_dir,
            config=config,
            mode=mode,
            profile_id=profile_id,
            run=run,
            next_step=next_step,
            record=record,
            pending_index=pending_index,
            pending_total=pending_total,
            profile_overrides=profile_overrides,
            progress=progress,
        )
        next_step += 1
        ran.append(item.branch_id)
    return ran


def _run_branch_record(
    project_dir: Path,
    *,
    config: dict[str, object],
    mode: str,
    profile_id: str,
    run: Path,
    next_step: int,
    record: dict[str, Any],
    pending_index: int,
    pending_total: int,
    profile_overrides: dict[str, object] | None,
    progress: Callable[[str], None] | None,
) -> BranchRunItem:
    bid = str(record["branch_id"])
    branch_dir = project_dir / str(record["path"]).rsplit("/", 1)[0]
    branch_payload = read_yaml(branch_dir / "branch.yaml")
    component_version_notices = _branch_component_version_notices(project_dir, branch_payload)
    materialization = branch_dir / "materializations" / str(record["file"])
    code_hash = str(record["code_hash"])
    nid = node_id(next_step)
    node_dir = run / "artifacts" / nid
    node_dir.mkdir(parents=True, exist_ok=True)
    start_payload = {
        "schema_version": 1,
        "node_id": nid,
        "run_id": run.name,
        "step": int(nid.rsplit("-", 1)[-1]),
        "kind": "branch",
        "branch_id": bid,
        "mode": mode,
        "profile_id": profile_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if component_version_notices:
        start_payload["component_version_notices"] = component_version_notices
    write_yaml(node_dir / "node.start.yaml", start_payload)
    upsert_node_start(project_dir, node_dir, start_payload)
    shutil.copy2(branch_dir / "branch.yaml", node_dir / "01-branch.yaml")
    if component_version_notices:
        write_yaml(
            node_dir / "component-version-notices.yaml",
            {
                "schema_version": 1,
                "node_id": nid,
                "branch_id": bid,
                "mode": mode,
                "notices": component_version_notices,
            },
        )
    group_code = materialization.read_text(encoding="utf-8")
    validate_group_code_source(group_code)
    atomic_write_text(
        node_dir / "02-code.py",
        build_wrapped_materialization_source(
            mode,
            group_code,
            project_dir,
            profile_overrides=profile_overrides,
        ),
    )
    write_yaml(node_dir / "started.yaml", {"created_at": datetime.now().isoformat(timespec="seconds")})
    if progress is not None:
        progress(f"BRANCH run {bid} {mode} ({pending_index}/{pending_total}): executing node {nid}")
    write_branch_runtime_state(
        project_dir,
        branch_id=bid,
        node_id=nid,
        run_id=run.name,
        mode=mode,
        profile_id=profile_id,
    )
    try:
        result = run_python_script(
            node_dir / "02-code.py",
            node_dir / "work",
            timeout_seconds=_execution_timeout_seconds(config, profile_overrides),
            cwd=repo_root_for_project(project_dir) if mode == "autogluon" else None,
        )
        write_attempt_result(node_dir, result)
        finished_at = datetime.now().isoformat(timespec="seconds")
        run_status = "complete" if result.status == "ok" else "failed"
        if result.status == "ok":
            _write_success_markers(node_dir, nid, bid, mode, profile_id, code_hash, result)
            upsert_node_result(project_dir, node_dir, status=run_status, metric=result.metric, code_hash=code_hash, finished_at=finished_at)
        else:
            write_yaml(
                node_dir / "failed.yaml",
                {
                    "schema_version": 1,
                    "node_id": nid,
                    "branch_id": bid,
                    "mode": mode,
                    "profile_id": profile_id,
                    "code_hash": code_hash,
                    "status": run_status,
                    "error": result.error,
                    "created_at": finished_at,
                },
            )
            upsert_node_result(
                project_dir,
                node_dir,
                status=run_status,
                metric=result.metric,
                code_hash=code_hash,
                finished_at=finished_at,
                error=result.error,
            )
        if progress is not None:
            progress(f"BRANCH run {bid} {mode} ({pending_index}/{pending_total}): {result.status}")
    finally:
        clear_branch_runtime_state()
    node = node_record(project_dir, nid)
    evaluation = evaluation_record(project_dir, nid) or {}
    run_seconds_value = node.get("run_seconds")
    return BranchRunItem(
        branch_id=bid,
        run_status=run_status,
        metric=result.metric,
        node_id=nid,
        run_seconds=int(run_seconds_value) if isinstance(run_seconds_value, int) else None,
        decision_score=_optional_float(evaluation.get("decision_score")),
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _branch_component_version_notices(project_dir: Path, branch_payload: dict[str, Any]) -> list[dict[str, Any]]:
    components = branch_payload.get("components")
    if not isinstance(components, list):
        return []

    notices: list[dict[str, Any]] = []
    frozen_versions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for component in components:
        if not isinstance(component, dict):
            continue
        if str(component.get("source_type") or "") != "hypothesis":
            continue
        source_id = str(component.get("source_id") or "")
        mode = str(component.get("mode") or "")
        if not source_id or not mode:
            continue

        normalized_source_id = source_id.zfill(6)
        frozen_versions.setdefault((normalized_source_id, mode), []).append(component)
        manifest_path = project_dir / "hypotheses" / normalized_source_id / "manifest.yaml"
        if not manifest_path.exists():
            continue

        manifest = read_yaml(manifest_path)
        mats = manifest.get("materializations") if isinstance(manifest.get("materializations"), dict) else {}
        mat = mats.get(mode) if isinstance(mats.get(mode), dict) else {}
        active_file = str(mat.get("active") or "")
        active_hash = _manifest_file_sha(mat, active_file)
        frozen_file = str(component.get("file") or "")
        frozen_hash = str(component.get("code_hash") or "")
        if active_file and (active_file != frozen_file or (active_hash and active_hash != frozen_hash)):
            notices.append(
                {
                    "kind": "new_active_materialization_available",
                    "source_type": "hypothesis",
                    "source_id": normalized_source_id,
                    "mode": mode,
                    "role": component.get("role"),
                    "frozen_file": frozen_file,
                    "frozen_code_hash": frozen_hash,
                    "active_file": active_file,
                    "active_code_hash": active_hash,
                    "message": (
                        "This branch node uses the frozen component version; "
                        "the hypothesis manifest has a newer active materialization."
                    ),
                }
            )

    for (source_id, mode), version_components in frozen_versions.items():
        version_keys = {
            (str(component.get("file") or ""), str(component.get("code_hash") or ""))
            for component in version_components
        }
        if len(version_keys) <= 1:
            continue
        notices.append(
            {
                "kind": "multiple_frozen_materializations_in_branch",
                "source_type": "hypothesis",
                "source_id": source_id,
                "mode": mode,
                "versions": [
                    {
                        "role": component.get("role"),
                        "file": component.get("file"),
                        "code_hash": component.get("code_hash"),
                    }
                    for component in version_components
                ],
                "message": "This branch already contains multiple frozen materializations for the same hypothesis.",
            }
        )

    return notices


def _manifest_file_sha(mode_manifest: dict[str, Any], file_name: str) -> str:
    files = mode_manifest.get("files") if isinstance(mode_manifest.get("files"), list) else []
    for item in files:
        if isinstance(item, dict) and item.get("file") == file_name:
            return str(item.get("sha256") or "")
    return ""


def _write_success_markers(node_dir: Path, node_id_value: str, branch_id: str, mode: str, profile_id: str, code_hash: str, result) -> None:
    manifest = {
        "schema_version": 1,
        "node_id": node_id_value,
        "kind": "branch",
        "branch_id": branch_id,
        "mode": mode,
        "profile_id": profile_id,
        "code_hash": code_hash,
        "metric": result.metric,
        "files": {
            "branch": "01-branch.yaml",
            "code": "02-code.py",
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if result.payload is not None:
        manifest["result_payload"] = result.payload
        run_stats = result.payload.get("run_stats")
        if isinstance(run_stats, dict):
            manifest["run_stats"] = run_stats
    write_yaml(node_dir / "artifact-manifest.yaml", manifest)
    write_yaml(node_dir / "node.done.yaml", {**manifest, "status": "complete"})
