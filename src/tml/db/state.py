from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml

from .connect import connect
from .migrate import migrate


def project_db_path(project_dir: Path) -> Path:
    return project_dir / "tml.db"


def ensure_project_db(project_dir: Path) -> Path:
    db_path = project_db_path(project_dir)
    migrate(db_path)
    return db_path


def upsert_project(project_dir: Path) -> None:
    db_path = ensure_project_db(project_dir)
    config = read_yaml(project_dir / "project.yaml")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO projects(project_id, kind, path)
            VALUES (?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
              kind=excluded.kind,
              path=excluded.path
            """,
            (config.get("project_id"), config.get("kind", "kaggle"), "."),
        )
        conn.commit()


def upsert_hypothesis(project_dir: Path, hdir: Path) -> None:
    db_path = ensure_project_db(project_dir)
    payload = read_yaml(hdir / "hypothesis.yaml")
    hid = str(payload.get("hypothesis_id") or hdir.name)
    summary = _run_summary(hdir / "01-hypothesis.request.json", hdir / "01-hypothesis.response.json")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses(
              hypothesis_id, title, summary, created_at, model, reasoning_tokens,
              total_tokens, generation_seconds, enabled, path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
              title=excluded.title,
              summary=excluded.summary,
              created_at=excluded.created_at,
              model=excluded.model,
              reasoning_tokens=excluded.reasoning_tokens,
              total_tokens=excluded.total_tokens,
              generation_seconds=excluded.generation_seconds,
              enabled=excluded.enabled,
              path=excluded.path
            """,
            (
                hid,
                payload.get("title"),
                payload.get("summary"),
                payload.get("created_at"),
                summary.get("model"),
                summary.get("reasoning_tokens"),
                summary.get("total_tokens"),
                summary.get("generation_seconds"),
                1 if payload.get("enabled", True) else 0,
                _project_path(project_dir, hdir / "hypothesis.yaml"),
            ),
        )
        conn.commit()


def upsert_materialization(project_dir: Path, hdir: Path, mode: str, code_path: Path) -> None:
    db_path = ensure_project_db(project_dir)
    hid = hdir.name
    summary = _run_summary(
        code_path.parent / f"{mode}-001.request.json",
        code_path.parent / f"{mode}-001.response.json",
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO materializations(
              hypothesis_id, mode, file, code_hash, model, reasoning_tokens,
              total_tokens, generation_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id, mode, file) DO UPDATE SET
              code_hash=excluded.code_hash,
              model=excluded.model,
              reasoning_tokens=excluded.reasoning_tokens,
              total_tokens=excluded.total_tokens,
              generation_seconds=excluded.generation_seconds
            """,
            (
                hid,
                mode,
                code_path.name,
                sha256_file(code_path),
                summary.get("model"),
                summary.get("reasoning_tokens"),
                summary.get("total_tokens"),
                summary.get("generation_seconds"),
            ),
        )
        conn.commit()


def upsert_run(project_dir: Path, run_dir: Path) -> None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs(run_id, path)
            VALUES (?, ?)
            ON CONFLICT(run_id) DO UPDATE SET path=excluded.path
            """,
            (run_dir.name, _project_path(project_dir, run_dir)),
        )
        conn.commit()


def upsert_node_start(project_dir: Path, node_dir: Path, payload: dict[str, Any]) -> None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO nodes(node_id, run_id, step, hypothesis_id, mode, profile_id, status, created_at, path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              run_id=excluded.run_id,
              step=excluded.step,
              hypothesis_id=excluded.hypothesis_id,
              mode=excluded.mode,
              profile_id=excluded.profile_id,
              status=excluded.status,
              created_at=excluded.created_at,
              path=excluded.path
            """,
            (
                payload.get("node_id"),
                payload.get("run_id"),
                payload.get("step"),
                payload.get("hypothesis_id"),
                payload.get("mode"),
                payload.get("profile_id"),
                "started",
                payload.get("created_at"),
                _project_path(project_dir, node_dir),
            ),
        )
        _upsert_artifact_rows(conn, node_dir, ["node.start.yaml"])
        conn.commit()


def upsert_node_result(
    project_dir: Path,
    node_dir: Path,
    *,
    status: str,
    metric: float | None,
    code_hash: str,
    finished_at: str,
    error: str | None = None,
) -> None:
    _ = error
    db_path = ensure_project_db(project_dir)
    start = read_yaml(node_dir / "node.start.yaml")
    run_seconds = _elapsed_seconds(start.get("created_at"), finished_at)
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE nodes
            SET status=?, finished_at=?, run_seconds=?
            WHERE node_id=?
            """,
            (status, finished_at, run_seconds, node_dir.name),
        )
        conn.execute(
            """
            INSERT INTO evaluations(node_id, hypothesis_id, mode, profile_id, code_hash, metric, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              hypothesis_id=excluded.hypothesis_id,
              mode=excluded.mode,
              profile_id=excluded.profile_id,
              code_hash=excluded.code_hash,
              metric=excluded.metric,
              status=excluded.status
            """,
            (
                node_dir.name,
                start.get("hypothesis_id"),
                start.get("mode"),
                start.get("profile_id"),
                code_hash,
                metric,
                status,
            ),
        )
        _upsert_artifact_rows(
            conn,
            node_dir,
            [
                "01-hypothesis.yaml",
                "02-code.py",
                "started.yaml",
                "stdout.log",
                "stderr.log",
                "result.yaml",
                "artifact-manifest.yaml",
                "node.done.yaml",
                "failed.yaml",
            ],
        )
        conn.commit()


def next_hypothesis_number(project_dir: Path) -> int:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT max(CAST(hypothesis_id AS INTEGER)) AS max_id FROM hypotheses").fetchone()
    value = row["max_id"] if row else None
    return int(value or 0) + 1


def hypothesis_count(project_dir: Path) -> int:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        return int(conn.execute("SELECT count(*) FROM hypotheses").fetchone()[0])


def hypothesis_records(project_dir: Path) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM hypotheses ORDER BY hypothesis_id").fetchall()
    return [dict(row) for row in rows]


def root_hypothesis_rows(project_dir: Path) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT
          h.*,
          CASE
            WHEN h.enabled=0 THEN '⊘'
            WHEN EXISTS (
              SELECT 1 FROM nodes n
              WHERE n.hypothesis_id=h.hypothesis_id AND n.status='failed'
            ) THEN '⚠'
            WHEN EXISTS (
              SELECT 1 FROM nodes n
              WHERE n.hypothesis_id=h.hypothesis_id AND n.status='complete'
            ) THEN '▶'
            WHEN EXISTS (
              SELECT 1 FROM materializations m
              WHERE m.hypothesis_id=h.hypothesis_id
            ) THEN '⌘'
            ELSE '◇'
          END AS status_icon
        FROM hypotheses h
        ORDER BY h.hypothesis_id
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def materialization_candidates(project_dir: Path, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = "SELECT hypothesis_id, path FROM hypotheses"
    params: tuple[Any, ...] = ()
    if hypothesis_id:
        sql += " WHERE hypothesis_id=?"
        params = (hypothesis_id.zfill(6),)
    sql += " ORDER BY hypothesis_id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def run_candidates(project_dir: Path, mode: str, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT h.hypothesis_id, h.path, m.file, m.code_hash
        FROM hypotheses h
        JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=?
    """
    params: list[Any] = [mode]
    if hypothesis_id:
        sql += " WHERE h.hypothesis_id=?"
        params.append(hypothesis_id.zfill(6))
    sql += " ORDER BY h.hypothesis_id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def already_evaluated(project_dir: Path, *, hypothesis_id: str, mode: str, profile_id: str, code_hash: str) -> bool:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM evaluations
            WHERE hypothesis_id=? AND mode=? AND profile_id=? AND code_hash=?
            LIMIT 1
            """,
            (hypothesis_id, mode, profile_id, code_hash),
        ).fetchone()
    return row is not None


def active_or_create_run(project_dir: Path) -> Path:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if row:
        run_dir = project_dir / "runs" / str(row["run_id"])
    else:
        from tml.core.ids import run_id

        run_dir = project_dir / "runs" / run_id()
        run_dir.mkdir(parents=True, exist_ok=True)
        from tml.utils.yaml_io import write_yaml

        write_yaml(run_dir / "run.yaml", {"schema_version": 1, "run_id": run_dir.name, "created_at": datetime.now().isoformat(timespec="seconds")})
        (run_dir / "artifacts").mkdir(exist_ok=True)
    upsert_run(project_dir, run_dir)
    return run_dir


def next_node_step(project_dir: Path, run_id_value: str) -> int:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT max(step) AS max_step FROM nodes WHERE run_id=?", (run_id_value,)).fetchone()
    return int(row["max_step"] or 0) + 1


def root_counts(project_dir: Path) -> dict[str, int]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        return {
            "hypotheses": int(conn.execute("SELECT count(*) FROM hypotheses").fetchone()[0]),
            "materialized": int(conn.execute("SELECT count(DISTINCT hypothesis_id) FROM materializations").fetchone()[0]),
            "evaluated": int(conn.execute("SELECT count(*) FROM nodes WHERE status='complete'").fetchone()[0]),
            "incomplete": int(conn.execute("SELECT count(*) FROM nodes WHERE status NOT IN ('complete','failed')").fetchone()[0]),
        }


def best_score(project_dir: Path) -> float | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT max(metric) AS best FROM evaluations WHERE metric IS NOT NULL").fetchone()
    value = row["best"] if row else None
    return float(value) if value is not None else None


def materialization_rows(project_dir: Path, *, mode: str, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT h.hypothesis_id, h.summary, m.mode, m.file, m.model, m.reasoning_tokens,
               m.total_tokens, m.generation_seconds
        FROM materializations m
        JOIN hypotheses h ON h.hypothesis_id=m.hypothesis_id
        WHERE m.mode=?
    """
    params: list[Any] = [mode]
    if hypothesis_id:
        sql += " AND h.hypothesis_id=?"
        params.append(hypothesis_id.zfill(6))
    sql += " ORDER BY h.hypothesis_id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def root_run_rows(project_dir: Path, *, mode: str, profile_id: str) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT
          h.hypothesis_id, h.summary, h.created_at, h.model, h.reasoning_tokens,
          h.total_tokens, h.generation_seconds, h.enabled,
          m.code_hash,
          n.node_id, n.status AS node_status, n.run_seconds,
          e.metric
        FROM hypotheses h
        LEFT JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=?
        LEFT JOIN evaluations e ON e.hypothesis_id=h.hypothesis_id
          AND e.mode=? AND e.profile_id=? AND e.code_hash=m.code_hash
        LEFT JOIN nodes n ON n.node_id=e.node_id
        ORDER BY h.hypothesis_id
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql, (mode, mode, profile_id)).fetchall()
    return [dict(row) for row in rows]


def _run_summary(request_path: Path, response_path: Path) -> dict[str, Any]:
    if not request_path.exists() or not response_path.exists():
        return {}
    request = read_yaml(request_path)
    response = read_yaml(response_path)
    if not isinstance(request, dict) or not isinstance(response, dict):
        return {}
    total = _token_total(response)
    wall_ms = response.get("wall_ms")
    return {
        "model": str(request.get("model") or response.get("model") or "") or None,
        "reasoning_tokens": total.get("reasoningOutputTokens"),
        "total_tokens": total.get("totalTokens"),
        "generation_seconds": round(wall_ms / 1000) if isinstance(wall_ms, int) else None,
    }


def _token_total(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    token_usage = usage.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return {}
    total = token_usage.get("total")
    return total if isinstance(total, dict) else {}


def _elapsed_seconds(start_value: Any, end_value: Any) -> int | None:
    started_at = _parse_datetime(start_value)
    finished_at = _parse_datetime(end_value)
    if started_at is None or finished_at is None:
        return None
    return max(0, round((finished_at - started_at).total_seconds()))


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _upsert_artifact_rows(conn, node_dir: Path, relative_paths: list[str]) -> None:
    for relative_path in relative_paths:
        path = node_dir / relative_path
        if not path.exists():
            continue
        conn.execute(
            "INSERT OR IGNORE INTO artifacts(node_id, path, kind) VALUES (?, ?, ?)",
            (node_dir.name, relative_path, path.suffix.lstrip(".") or "file"),
        )


def _project_path(project_dir: Path, path: Path) -> str:
    return path.relative_to(project_dir).as_posix()
