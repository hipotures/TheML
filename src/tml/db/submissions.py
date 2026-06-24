from __future__ import annotations

from pathlib import Path
from typing import Any

from tml.core.config import repo_root_for_project
from tml.utils.hashing import sha256_file


def find_submission_file(node_dir: Path) -> Path | None:
    for relative_path in (
        "artifacts/submission.csv.gz",
        "artifacts/submission.csv",
        "submission.csv.gz",
        "submission.csv",
    ):
        path = node_dir / relative_path
        if path.is_file():
            return path
    return None


def build_submission_row(
    *,
    project_dir: Path,
    node_dir: Path,
    start: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
    metric: float | None,
    code_hash: str | None,
    finished_at: str | None,
    run_seconds: int | None,
) -> dict[str, Any] | None:
    submission_path = find_submission_file(node_dir)
    if submission_path is None:
        return None

    result_payload = manifest.get("result_payload") if isinstance(manifest.get("result_payload"), dict) else {}
    run_stats = manifest.get("run_stats") if isinstance(manifest.get("run_stats"), dict) else {}
    payload_stats = result_payload.get("run_stats") if isinstance(result_payload.get("run_stats"), dict) else {}
    stat = submission_path.stat()

    return {
        "node_id": node_dir.name,
        "submission_path": _project_path(project_dir, submission_path),
        "submission_sha256": sha256_file(submission_path),
        "submission_size": stat.st_size,
        "submission_mtime_ns": stat.st_mtime_ns,
        "run_id": start.get("run_id"),
        "step": start.get("step"),
        "hypothesis_id": start.get("hypothesis_id") or manifest.get("hypothesis_id"),
        "hypothesis_revision": start.get("hypothesis_revision") or manifest.get("hypothesis_revision"),
        "materialization_file": start.get("materialization_file") or manifest.get("materialization_file"),
        "mode": start.get("mode") or manifest.get("mode"),
        "profile_id": start.get("profile_id") or manifest.get("profile_id"),
        "kind": str(manifest.get("kind") or "source"),
        "status": status,
        "submit_status": str(manifest.get("submit_status") or "not_submitted"),
        "local_score": metric,
        "public_score": manifest.get("public_score"),
        "public_rank": manifest.get("public_rank"),
        "metric": manifest.get("eval_metric")
        or result_payload.get("eval_metric")
        or run_stats.get("eval_metric")
        or payload_stats.get("eval_metric"),
        "code_hash": code_hash or manifest.get("code_hash"),
        "run_seconds": run_seconds,
        "created_at": start.get("created_at"),
        "finished_at": finished_at,
        "artifact_dir": _project_path(project_dir, node_dir),
        "source_submission_sha256": start.get("source_submission_sha256") or manifest.get("source_submission_sha256"),
        "source_run_id": start.get("source_run_id") or manifest.get("source_run_id"),
        "source_node_id": start.get("source_node_id") or manifest.get("source_node_id"),
        "source_step": start.get("source_step") or manifest.get("source_step"),
        "source_profile_id": start.get("source_profile_id") or manifest.get("source_profile_id"),
    }


def upsert_submission(conn, row: dict[str, Any] | None) -> None:
    if row is None:
        return
    conn.execute(
        """
        INSERT INTO submissions(
          node_id, submission_path, submission_sha256, submission_size, submission_mtime_ns,
          run_id, step, hypothesis_id, hypothesis_revision, materialization_file, mode, profile_id, kind, status, submit_status,
          local_score, public_score, public_rank, metric, code_hash, run_seconds,
          created_at, finished_at, artifact_dir, source_submission_sha256, source_run_id,
          source_node_id, source_step, source_profile_id
        )
        VALUES (
          :node_id, :submission_path, :submission_sha256, :submission_size, :submission_mtime_ns,
          :run_id, :step, :hypothesis_id, :hypothesis_revision, :materialization_file, :mode, :profile_id, :kind, :status, :submit_status,
          :local_score, :public_score, :public_rank, :metric, :code_hash, :run_seconds,
          :created_at, :finished_at, :artifact_dir, :source_submission_sha256, :source_run_id,
          :source_node_id, :source_step, :source_profile_id
        )
        ON CONFLICT(node_id, submission_path) DO UPDATE SET
          submission_sha256=excluded.submission_sha256,
          submission_size=excluded.submission_size,
          submission_mtime_ns=excluded.submission_mtime_ns,
          run_id=excluded.run_id,
          step=excluded.step,
          hypothesis_id=excluded.hypothesis_id,
          hypothesis_revision=excluded.hypothesis_revision,
          materialization_file=excluded.materialization_file,
          mode=excluded.mode,
          profile_id=excluded.profile_id,
          kind=excluded.kind,
          status=excluded.status,
          submit_status=CASE
            WHEN submissions.submit_status='submitted' THEN submissions.submit_status
            ELSE excluded.submit_status
          END,
          local_score=excluded.local_score,
          public_score=COALESCE(submissions.public_score, excluded.public_score),
          public_rank=COALESCE(submissions.public_rank, excluded.public_rank),
          metric=excluded.metric,
          code_hash=excluded.code_hash,
          run_seconds=excluded.run_seconds,
          created_at=excluded.created_at,
          finished_at=excluded.finished_at,
          artifact_dir=excluded.artifact_dir,
          source_submission_sha256=excluded.source_submission_sha256,
          source_run_id=excluded.source_run_id,
          source_node_id=excluded.source_node_id,
          source_step=excluded.source_step,
          source_profile_id=excluded.source_profile_id
        """,
        row,
    )


def _project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        try:
            return path.relative_to(repo_root_for_project(project_dir)).as_posix()
        except ValueError:
            return path.as_posix()
