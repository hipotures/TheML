from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.core.config import active_mode, active_profile_id, load_project_config, repo_root_for_project
from tml.core.ids import node_id
from tml.core.profiles import profile_hash
from tml.branches.runtime_state import clear_branch_runtime_state, write_branch_runtime_state
from tml.db.state import (
    active_or_create_run,
    branch_already_evaluated,
    branch_run_candidates,
    next_node_step,
    upsert_node_result,
    upsert_node_start,
    upsert_project,
)
from tml.execution.executor import run_python_script, write_attempt_result
from tml.features.validation import validate_group_code_source
from tml.hypotheses.run import _execution_timeout_seconds
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.utils.atomic import atomic_write_text
from tml.utils.yaml_io import write_yaml


@dataclass(frozen=True)
class BranchRunPlan:
    mode: str
    profile_id: str
    profile_hash: str
    execution_timeout_seconds: int
    run_id: str | None
    run_path: str
    next_node_step: int
    candidate_count: int
    already_evaluated_count: int
    iteration_count: int
    branch_ids: list[str]


def branch_run_plan(
    project_dir: Path,
    mode: str | None = None,
    *,
    branch_id: str | None = None,
    profile_overrides: dict[str, object] | None = None,
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
        if branch_already_evaluated(project_dir, branch_id=bid, mode=active_run_mode, profile_id=profile_id, code_hash=code_hash):
            already_done += 1
            continue
        pending_ids.append(bid)
    return BranchRunPlan(
        mode=active_run_mode,
        profile_id=profile_id,
        profile_hash=profile_hash(project_dir, active_run_mode, profile_id),
        execution_timeout_seconds=_execution_timeout_seconds(profile_overrides),
        run_id=run_id_value,
        run_path=f"runs/{run_id_value}" if run_id_value else "new run on start",
        next_node_step=next_node_step(project_dir, run_id_value) if run_id_value else 1,
        candidate_count=len(candidates),
        already_evaluated_count=already_done,
        iteration_count=len(pending_ids),
        branch_ids=pending_ids,
    )


def run_missing_branches(
    project_dir: Path,
    mode: str | None = None,
    *,
    branch_id: str | None = None,
    profile_overrides: dict[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    upsert_project(project_dir)
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    profile_id = active_profile_id(config, mode)
    run = active_or_create_run(project_dir)
    ran: list[str] = []
    next_step = next_node_step(project_dir, run.name)
    records = branch_run_candidates(project_dir, mode, branch_id=branch_id)
    pending_records = [
        record
        for record in records
        if not branch_already_evaluated(
            project_dir,
            branch_id=str(record["branch_id"]),
            mode=mode,
            profile_id=profile_id,
            code_hash=str(record["code_hash"]),
        )
    ]
    pending_total = len(pending_records)
    for pending_index, record in enumerate(pending_records, start=1):
        bid = str(record["branch_id"])
        branch_dir = project_dir / str(record["path"]).rsplit("/", 1)[0]
        materialization = branch_dir / "materializations" / str(record["file"])
        code_hash = str(record["code_hash"])
        nid = node_id(next_step)
        next_step += 1
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
        write_yaml(node_dir / "node.start.yaml", start_payload)
        upsert_node_start(project_dir, node_dir, start_payload)
        shutil.copy2(branch_dir / "branch.yaml", node_dir / "01-branch.yaml")
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
                timeout_seconds=_execution_timeout_seconds(profile_overrides),
                cwd=repo_root_for_project(project_dir) if mode == "autogluon" else None,
            )
            write_attempt_result(node_dir, result)
            finished_at = datetime.now().isoformat(timespec="seconds")
            if result.status == "ok":
                _write_success_markers(node_dir, nid, bid, mode, profile_id, code_hash, result)
                upsert_node_result(project_dir, node_dir, status="complete", metric=result.metric, code_hash=code_hash, finished_at=finished_at)
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
                        "status": "failed",
                        "error": result.error,
                        "created_at": finished_at,
                    },
                )
                upsert_node_result(
                    project_dir,
                    node_dir,
                    status="failed",
                    metric=result.metric,
                    code_hash=code_hash,
                    finished_at=finished_at,
                    error=result.error,
                )
            if progress is not None:
                progress(f"BRANCH run {bid} {mode} ({pending_index}/{pending_total}): {result.status}")
            ran.append(bid)
        finally:
            clear_branch_runtime_state()
    return ran


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
