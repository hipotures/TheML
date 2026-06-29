from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tml.core.config import active_mode, active_profile_id, load_project_config, repo_root_for_project
from tml.core.ids import node_id
from tml.core.profiles import profile_hash
from tml.db.state import (
    active_or_create_run,
    already_evaluated,
    create_run,
    latest_run_id,
    next_node_step,
    run_candidates,
    upsert_node_result,
    upsert_node_start,
    upsert_project,
)
from tml.execution.executor import run_python_script, write_attempt_result
from tml.features.validation import validate_group_code_source
from tml.hypotheses.wrapper_source import build_wrapped_materialization_source
from tml.utils.atomic import atomic_write_text
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import write_yaml

from .baseline import ensure_root_baseline
from .revisions import (
    load_revision,
    migrate_root_revisions,
    set_active_materialization,
)


@dataclass(frozen=True)
class RootRunPlan:
    mode: str
    profile_id: str
    profile_hash: str
    execution_timeout_seconds: int
    force: bool
    all_revisions: bool
    run_id: str | None
    run_path: str
    next_node_step: int
    candidate_count: int
    already_evaluated_count: int
    iteration_count: int
    hypothesis_ids: list[str]
    files: list[str]


def root_run_plan(
    project_dir: Path,
    mode: str | None = None,
    *,
    hypothesis_id: str | None = None,
    revision: int | None = None,
    profile_overrides: dict[str, object] | None = None,
    force: bool = False,
    new_run: bool = False,
    all_revisions: bool = False,
) -> RootRunPlan:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    migrate_root_revisions(project_dir)
    config = load_project_config(project_dir)
    active_run_mode = mode or active_mode(config)
    profile_id = active_profile_id(config, active_run_mode)
    run_id_value = None if new_run else latest_run_id(project_dir)
    pending_ids: list[str] = []
    pending_files: list[str] = []
    already_done = 0
    candidates = run_candidates(
        project_dir,
        active_run_mode,
        hypothesis_id=hypothesis_id,
        revision=revision,
        active_only=not all_revisions,
    )
    for record in candidates:
        hid = str(record["hypothesis_id"])
        code_hash = str(record["code_hash"])
        if not force and already_evaluated(
            project_dir,
            hypothesis_id=hid,
            mode=active_run_mode,
            profile_id=profile_id,
            code_hash=code_hash,
        ):
            already_done += 1
            continue
        pending_ids.append(f"{hid}:{record.get('hypothesis_revision') or 1}")
        pending_files.append(str(record["file"]))
    return RootRunPlan(
        mode=active_run_mode,
        profile_id=profile_id,
        profile_hash=profile_hash(project_dir, active_run_mode, profile_id),
        execution_timeout_seconds=_execution_timeout_seconds(profile_overrides),
        force=force,
        all_revisions=all_revisions,
        run_id=run_id_value,
        run_path=f"runs/{run_id_value}" if run_id_value else "new run on start",
        next_node_step=next_node_step(project_dir, run_id_value) if run_id_value else 1,
        candidate_count=len(candidates),
        already_evaluated_count=already_done,
        iteration_count=len(pending_ids),
        hypothesis_ids=pending_ids,
        files=pending_files,
    )


def run_missing(
    project_dir: Path,
    mode: str | None = None,
    *,
    hypothesis_id: str | None = None,
    revision: int | None = None,
    profile_overrides: dict[str, object] | None = None,
    force: bool = False,
    new_run: bool = False,
    all_revisions: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    upsert_project(project_dir)
    ensure_root_baseline(project_dir)
    migrate_root_revisions(project_dir)
    config = load_project_config(project_dir)
    mode = mode or active_mode(config)
    profile_id = active_profile_id(config, mode)
    run = create_run(project_dir) if new_run else active_or_create_run(project_dir)
    ran: list[str] = []
    next_step = next_node_step(project_dir, run.name)
    records = run_candidates(
        project_dir,
        mode,
        hypothesis_id=hypothesis_id,
        revision=revision,
        active_only=not all_revisions,
    )
    pending_records = [
        record
        for record in records
        if force
        or not already_evaluated(
            project_dir,
            hypothesis_id=str(record["hypothesis_id"]),
            mode=mode,
            profile_id=profile_id,
            code_hash=str(record["code_hash"]),
        )
    ]
    pending_total = len(pending_records)
    for pending_index, record in enumerate(pending_records, start=1):
        hid = str(record["hypothesis_id"])
        hdir = project_dir / str(record["path"]).rsplit("/", 1)[0]
        materialization = hdir / "materializations" / str(record["file"])
        selected_revision = int(record.get("hypothesis_revision") or 1)
        materialization_file = materialization.name
        code_hash = str(record["code_hash"])
        run_label = f"ROOT run {hid} rev={selected_revision} file={materialization_file} {mode} ({pending_index}/{pending_total})"
        nid = node_id(next_step)
        next_step += 1
        node_dir = run / "artifacts" / nid
        node_dir.mkdir(parents=True, exist_ok=True)
        start_payload = {
            "schema_version": 1,
            "node_id": nid,
            "run_id": run.name,
            "step": int(nid.rsplit("-", 1)[-1]),
            "kind": "root",
            "hypothesis_id": hid,
            "hypothesis_revision": selected_revision,
            "materialization_file": materialization_file,
            "mode": mode,
            "profile_id": profile_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_yaml(
            node_dir / "node.start.yaml",
            start_payload,
        )
        upsert_node_start(project_dir, node_dir, start_payload)
        shutil.copy2(load_revision(hdir, selected_revision).path, node_dir / "01-hypothesis.yaml")
        group_code = materialization.read_text(encoding="utf-8")
        try:
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
        except (SyntaxError, ValueError) as exc:
            finished_at = datetime.now().isoformat(timespec="seconds")
            error = str(exc)
            write_yaml(
                node_dir / "failed.yaml",
                {
                    "schema_version": 1,
                    "node_id": nid,
                    "hypothesis_id": hid,
                    "hypothesis_revision": selected_revision,
                    "materialization_file": materialization_file,
                    "mode": mode,
                    "profile_id": profile_id,
                    "code_hash": code_hash,
                    "status": "failed",
                    "error": error,
                    "created_at": finished_at,
                },
            )
            upsert_node_result(
                project_dir,
                node_dir,
                status="failed",
                metric=None,
                code_hash=code_hash,
                finished_at=finished_at,
                error=error,
            )
            if progress is not None:
                progress(f"{run_label}: failed")
            ran.append(f"{hid}:{selected_revision}")
            continue
        write_yaml(node_dir / "started.yaml", {"created_at": datetime.now().isoformat(timespec="seconds")})
        if progress is not None:
            progress(f"{run_label}: executing node {nid}")
        result = run_python_script(
            node_dir / "02-code.py",
            node_dir / "work",
            timeout_seconds=_execution_timeout_seconds(profile_overrides),
            cwd=repo_root_for_project(project_dir) if mode == "autogluon" else None,
        )
        write_attempt_result(node_dir, result)
        if result.status == "ok":
            should_promote = _should_promote(project_dir, hid, mode, profile_id, result.metric, _maximize(project_dir, result.maximize))
            _write_success_markers(node_dir, nid, hid, selected_revision, mode, profile_id, materialization, result)
            upsert_node_result(
                project_dir,
                node_dir,
                status="complete",
                metric=result.metric,
                code_hash=code_hash,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
            if should_promote:
                set_active_materialization(hdir, mode, materialization_file)
                from tml.db.state import upsert_materialization

                upsert_materialization(project_dir, hdir, mode, materialization, active=True, hypothesis_revision=selected_revision)
        else:
            finished_at = datetime.now().isoformat(timespec="seconds")
            write_yaml(
                node_dir / "failed.yaml",
                {
                    "schema_version": 1,
                    "node_id": nid,
                    "hypothesis_id": hid,
                    "hypothesis_revision": selected_revision,
                    "materialization_file": materialization_file,
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
            progress(f"{run_label}: {result.status}")
        ran.append(f"{hid}:{selected_revision}")
    return ran


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

def _write_success_markers(
    node_dir: Path,
    node_id_value: str,
    hypothesis_id: str,
    hypothesis_revision: int,
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
        "hypothesis_revision": hypothesis_revision,
        "materialization_file": materialization.name,
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


def _should_promote(
    project_dir: Path,
    hypothesis_id: str,
    mode: str,
    profile_id: str,
    metric: float | None,
    maximize: bool,
) -> bool:
    if metric is None:
        return False
    from tml.db.connect import connect
    from tml.db.state import ensure_project_db

    direction = "DESC" if maximize else "ASC"
    with connect(ensure_project_db(project_dir)) as conn:
        row = conn.execute(
            f"""
            SELECT e.metric
            FROM evaluations e
            WHERE e.kind='root'
              AND e.hypothesis_id=?
              AND e.mode=?
              AND e.profile_id=?
              AND e.status='complete'
              AND e.metric IS NOT NULL
            ORDER BY e.metric {direction}
            LIMIT 1
            """,
            (hypothesis_id, mode, profile_id),
        ).fetchone()
    if row is None:
        return True
    best = float(row["metric"])
    if not maximize:
        return float(metric) < best
    return float(metric) > best


def _maximize(project_dir: Path, value: bool | None) -> bool:
    if isinstance(value, bool):
        return value
    config = load_project_config(project_dir)
    target = config.get("target") if isinstance(config.get("target"), dict) else {}
    maximize = target.get("maximize")
    return maximize if isinstance(maximize, bool) else True
