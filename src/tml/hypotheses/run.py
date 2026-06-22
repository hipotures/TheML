from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from tml.core.config import active_mode, active_profile_id, load_project_config
from tml.core.ids import node_id, run_id
from tml.execution.executor import run_python_script, write_attempt_result
from tml.features.validation import validate_group_code_source
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml, write_yaml

from .model import hypothesis_dirs


def run_missing(
    project_dir: Path,
    mode: str | None = None,
    *,
    profile_overrides: dict[str, object] | None = None,
) -> int:
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    profile_id = active_profile_id(config, mode)
    run = _active_or_create_run(project_dir)
    ran = 0
    next_step = _next_step(run)
    for hdir in hypothesis_dirs(project_dir):
        hid = hdir.name
        materialization = hdir / "materializations" / f"{mode}-001.py"
        if not materialization.exists():
            continue
        if _already_evaluated(project_dir, hid, mode, profile_id, materialization):
            continue
        nid = node_id(next_step)
        next_step += 1
        node_dir = run / "artifacts" / nid
        node_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(
            node_dir / "node.start.yaml",
            {
                "schema_version": 1,
                "node_id": nid,
                "run_id": run.name,
                "step": int(nid.rsplit("-", 1)[-1]),
                "kind": "root",
                "hypothesis_id": hid,
                "mode": mode,
                "profile_id": profile_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        shutil.copy2(hdir / "hypothesis.yaml", node_dir / "01-hypothesis.yaml")
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
        attempt_dir = node_dir / "03-execute" / "attempt-001"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(attempt_dir / "started.yaml", {"created_at": datetime.now().isoformat(timespec="seconds")})
        result = run_python_script(
            node_dir / "02-code.py",
            node_dir / "work",
            timeout_seconds=_execution_timeout_seconds(profile_overrides),
        )
        write_attempt_result(attempt_dir, result)
        if result.status == "ok":
            _write_success_markers(node_dir, nid, hid, mode, profile_id, materialization, result)
        else:
            write_yaml(
                node_dir / "failed.yaml",
                {
                    "schema_version": 1,
                    "node_id": nid,
                    "hypothesis_id": hid,
                    "mode": mode,
                    "profile_id": profile_id,
                    "code_hash": sha256_file(materialization),
                    "status": "failed",
                    "error": result.error,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
        ran += 1
    return ran


def _active_or_create_run(project_dir: Path) -> Path:
    runs = sorted(path for path in (project_dir / "runs").glob("*") if path.is_dir())
    if runs:
        return runs[-1]
    rid = run_id()
    run_dir = project_dir / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "run.yaml", {"schema_version": 1, "run_id": rid, "created_at": datetime.now().isoformat(timespec="seconds")})
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir


def _execution_timeout_seconds(profile_overrides: dict[str, object] | None) -> int:
    overrides = profile_overrides or {}
    raw_time = overrides.get("time_limit", overrides.get("time"))
    raw_preprocess = overrides.get("preprocess_timeout", 180)
    try:
        time_limit = int(raw_time) if raw_time is not None else 900
    except (TypeError, ValueError):
        time_limit = 900
    try:
        preprocess_timeout = int(raw_preprocess)
    except (TypeError, ValueError):
        preprocess_timeout = 180
    return max(900, time_limit + preprocess_timeout + 300)


def _next_step(run_dir: Path) -> int:
    starts = list((run_dir / "artifacts").glob("*/node.start.yaml"))
    return len(starts) + 1


def _already_evaluated(
    project_dir: Path,
    hypothesis_id: str,
    mode: str,
    profile_id: str,
    materialization: Path,
) -> bool:
    code_hash = sha256_file(materialization)
    for done in (project_dir / "runs").glob("*/artifacts/*/node.done.yaml"):
        payload = read_yaml(done)
        if (
            payload.get("hypothesis_id") == hypothesis_id
            and payload.get("mode") == mode
            and payload.get("profile_id") == profile_id
            and payload.get("code_hash") == code_hash
        ):
            return True
    for failed in (project_dir / "runs").glob("*/artifacts/*/failed.yaml"):
        payload = read_yaml(failed)
        if (
            payload.get("hypothesis_id") == hypothesis_id
            and payload.get("mode") == mode
            and payload.get("profile_id") == profile_id
            and payload.get("code_hash") == code_hash
        ):
            return True
    return False


def _write_success_markers(
    node_dir: Path,
    node_id_value: str,
    hypothesis_id: str,
    mode: str,
    profile_id: str,
    materialization: Path,
    result,
) -> None:
    code_hash = sha256_file(materialization)
    manifest = {
        "schema_version": 1,
        "node_id": node_id_value,
        "hypothesis_id": hypothesis_id,
        "mode": mode,
        "profile_id": profile_id,
        "code_hash": code_hash,
        "metric": result.metric,
        "files": {
            "hypothesis": "01-hypothesis.yaml",
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
