from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from tml.utils.atomic import atomic_write_text


BRANCH_RUNTIME_STATE_PATH = Path("/tmp/tml-branch-run-current.json")


@dataclass(frozen=True)
class BranchRuntimeState:
    project_dir: Path
    branch_id: str
    node_id: str
    run_id: str
    mode: str
    profile_id: str
    pid: int
    created_at: str
    is_running: bool


def write_branch_runtime_state(
    project_dir: Path,
    *,
    branch_id: str,
    node_id: str,
    run_id: str,
    mode: str,
    profile_id: str,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "branch_run_current",
        "project_dir": str(project_dir.resolve()),
        "branch_id": branch_id,
        "node_id": node_id,
        "run_id": run_id,
        "mode": mode,
        "profile_id": profile_id,
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_text(BRANCH_RUNTIME_STATE_PATH, json.dumps(payload, sort_keys=True) + "\n")


def clear_branch_runtime_state() -> None:
    try:
        BRANCH_RUNTIME_STATE_PATH.unlink()
    except FileNotFoundError:
        return


def read_branch_runtime_state(project_dir: Path, *, mode: str, profile_id: str) -> BranchRuntimeState | None:
    try:
        payload = json.loads(BRANCH_RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "branch_run_current":
        return None
    if Path(str(payload.get("project_dir") or "")).resolve() != project_dir.resolve():
        return None
    if str(payload.get("mode") or "") != mode or str(payload.get("profile_id") or "") != profile_id:
        return None
    pid = _int_or_zero(payload.get("pid"))
    return BranchRuntimeState(
        project_dir=project_dir.resolve(),
        branch_id=str(payload.get("branch_id") or ""),
        node_id=str(payload.get("node_id") or ""),
        run_id=str(payload.get("run_id") or ""),
        mode=mode,
        profile_id=profile_id,
        pid=pid,
        created_at=str(payload.get("created_at") or ""),
        is_running=_pid_exists(pid),
    )


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
