from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from tml.utils.hashing import sha256_file, sha256_text
from tml.utils.yaml_io import read_yaml
from tml.hypotheses.revisions import (
    latest_revision_record,
    load_revision,
    materialization_revision,
    migrate_hypothesis_dir,
)

from .connect import connect
from .migrate import migrate
from .submissions import build_submission_row, upsert_submission


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
    migrate_hypothesis_dir(project_dir, hdir)
    latest = latest_revision_record(hdir)
    payload = latest.payload
    hid = str(payload.get("hypothesis_id") or hdir.name)
    summary = _run_summary(hdir / f"{latest.prefix}.request.json", hdir / f"{latest.prefix}.response.json")
    web_search = _web_search_summary(hdir / f"{latest.prefix}.request.json", hdir / f"{latest.prefix}.web_search.md")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypotheses(
              hypothesis_id, title, summary, created_at, model, reasoning_tokens,
              total_tokens, generation_seconds, web_search_enabled,
              web_search_has_results, enabled, path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET
              title=excluded.title,
              summary=excluded.summary,
              created_at=excluded.created_at,
              model=excluded.model,
              reasoning_tokens=excluded.reasoning_tokens,
              total_tokens=excluded.total_tokens,
              generation_seconds=excluded.generation_seconds,
              web_search_enabled=excluded.web_search_enabled,
              web_search_has_results=excluded.web_search_has_results,
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
                1 if web_search["enabled"] else 0,
                1 if web_search["has_results"] else 0,
                1 if payload.get("enabled", True) else 0,
                _project_path(project_dir, latest.path),
            ),
        )
        conn.commit()
    upsert_hypothesis_revision(project_dir, hdir, latest.revision)


def upsert_hypothesis_revision(project_dir: Path, hdir: Path, revision: int) -> None:
    db_path = ensure_project_db(project_dir)
    record = load_revision(hdir, revision)
    payload = record.payload
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO hypothesis_revisions(
              hypothesis_id, revision, path, prefix, created_at, summary, change_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id, revision) DO UPDATE SET
              path=excluded.path,
              prefix=excluded.prefix,
              created_at=excluded.created_at,
              summary=excluded.summary,
              change_summary=excluded.change_summary
            """,
            (
                hdir.name,
                revision,
                _project_path(project_dir, record.path),
                record.prefix,
                payload.get("created_at"),
                payload.get("summary"),
                payload.get("change_summary"),
            ),
        )
        conn.commit()


def delete_hypothesis_revision(project_dir: Path, *, hypothesis_id: str, revision: int) -> None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM hypothesis_revisions WHERE hypothesis_id=? AND revision=?",
            (hypothesis_id, int(revision)),
        )
        conn.commit()


def upsert_materialization(
    project_dir: Path,
    hdir: Path,
    mode: str,
    code_path: Path,
    *,
    status: str = "active",
    active: bool = True,
    source_node_id: str | None = None,
    fixed_from_file: str | None = None,
    fixed_from_code_hash: str | None = None,
    hypothesis_revision: int | None = None,
) -> None:
    db_path = ensure_project_db(project_dir)
    hid = hdir.name
    revision = int(hypothesis_revision or materialization_revision(hdir, mode, code_path.name))
    summary = _run_summary(
        code_path.parent / f"{code_path.stem}.request.json",
        code_path.parent / f"{code_path.stem}.response.json",
    )
    with connect(db_path) as conn:
        if active:
            conn.execute(
                """
                UPDATE materializations
                SET active=0
                WHERE hypothesis_id=? AND mode=? AND file<>?
                """,
                (hid, mode, code_path.name),
            )
        if fixed_from_file:
            conn.execute(
                """
                UPDATE materializations
                SET status='superseded', active=0, source_node_id=COALESCE(source_node_id, ?)
                WHERE hypothesis_id=? AND mode=? AND file=?
                """,
                (source_node_id, hid, mode, fixed_from_file),
            )
        conn.execute(
            """
            INSERT INTO materializations(
              hypothesis_id, mode, file, code_hash, hypothesis_revision, status, active, source_node_id,
              fixed_from_file, fixed_from_code_hash, model, reasoning_tokens,
              total_tokens, generation_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id, mode, file) DO UPDATE SET
              code_hash=excluded.code_hash,
              hypothesis_revision=excluded.hypothesis_revision,
              status=excluded.status,
              active=excluded.active,
              source_node_id=excluded.source_node_id,
              fixed_from_file=excluded.fixed_from_file,
              fixed_from_code_hash=excluded.fixed_from_code_hash,
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
                revision,
                status,
                1 if active else 0,
                source_node_id,
                fixed_from_file,
                fixed_from_code_hash,
                summary.get("model"),
                summary.get("reasoning_tokens"),
                summary.get("total_tokens"),
                summary.get("generation_seconds"),
            ),
        )
        conn.commit()


def upsert_failed_materialization(
    project_dir: Path,
    hdir: Path,
    mode: str,
    file_name: str,
    *,
    code_text: str,
) -> None:
    db_path = ensure_project_db(project_dir)
    hid = hdir.name
    stem = Path(file_name).stem
    summary = _run_summary(
        hdir / "materializations" / f"{stem}.request.json",
        hdir / "materializations" / f"{stem}.response.json",
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO materializations(
              hypothesis_id, mode, file, code_hash, hypothesis_revision, status, active, source_node_id,
              fixed_from_file, fixed_from_code_hash, model, reasoning_tokens,
              total_tokens, generation_seconds
            )
            VALUES (?, ?, ?, ?, ?, 'failed', 0, NULL, NULL, NULL, ?, ?, ?, ?)
            ON CONFLICT(hypothesis_id, mode, file) DO UPDATE SET
              code_hash=excluded.code_hash,
              hypothesis_revision=excluded.hypothesis_revision,
              status='failed',
              active=0,
              source_node_id=NULL,
              fixed_from_file=NULL,
              fixed_from_code_hash=NULL,
              model=excluded.model,
              reasoning_tokens=excluded.reasoning_tokens,
              total_tokens=excluded.total_tokens,
              generation_seconds=excluded.generation_seconds
            """,
            (
                hid,
                mode,
                file_name,
                sha256_text(code_text),
                materialization_revision(hdir, mode, file_name),
                summary.get("model"),
                summary.get("reasoning_tokens"),
                summary.get("total_tokens"),
                summary.get("generation_seconds"),
            ),
        )
        conn.commit()


def materialization_status(
    project_dir: Path,
    *,
    hypothesis_id: str,
    mode: str,
    file_name: str,
) -> str | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT status
            FROM materializations
            WHERE hypothesis_id=? AND mode=? AND file=?
            """,
            (hypothesis_id.zfill(6), mode, file_name),
        ).fetchone()
    if row is None:
        return None
    return str(row["status"])


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


def next_branch_id(project_dir: Path) -> str:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT max(CAST(substr(branch_id, 2) AS INTEGER)) AS max_id
            FROM branches
            WHERE branch_id GLOB 'B[0-9][0-9][0-9][0-9][0-9][0-9]'
            """
        ).fetchone()
    return f"B{int(row['max_id'] or 0) + 1:06d}"


def branch_by_composition(project_dir: Path, *, mode: str, composition_hash: str) -> dict[str, Any] | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM branches
            WHERE mode=? AND composition_hash=?
            LIMIT 1
            """,
            (mode, composition_hash),
        ).fetchone()
    return dict(row) if row else None


def branch_by_id(project_dir: Path, branch_id: str) -> dict[str, Any] | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM branches WHERE branch_id=?",
            (_normalize_branch_id(branch_id),),
        ).fetchone()
    return dict(row) if row else None


def branch_algorithm_candidate_pairs(
    project_dir: Path,
    *,
    mode: str,
    profile_id: str,
    parent_kinds: list[str],
    source_kinds: list[str],
    max_children: int,
    limit: int,
) -> list[dict[str, Any]]:
    parent_ref_kinds = [_branch_algorithm_db_kind(kind) for kind in parent_kinds]
    source_ref_kinds = [_branch_algorithm_db_kind(kind) for kind in source_kinds]
    if not parent_ref_kinds or not source_ref_kinds:
        return []

    parent_placeholders = ", ".join("?" for _ in parent_ref_kinds)
    source_placeholders = ", ".join("?" for _ in source_ref_kinds)
    parent_order = _branch_algorithm_kind_order_sql("p.ref_kind", parent_ref_kinds)
    source_order = _branch_algorithm_kind_order_sql("s.ref_kind", source_ref_kinds)
    db_path = ensure_project_db(project_dir)
    params: list[Any] = [
        mode,
        profile_id,
        profile_id,
        mode,
        mode,
        mode,
        mode,
        *parent_ref_kinds,
        max_children,
        *source_ref_kinds,
        limit,
    ]
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            WITH ranked_nodes AS (
                SELECT
                    'hypothesis' AS ref_kind,
                    h.hypothesis_id AS ref_id,
                    h.hypothesis_id AS ref,
                    e.metric AS score,
                    e.node_id AS node_id,
                    m.file AS file,
                    m.code_hash AS code_hash
                FROM hypotheses h
                JOIN materializations m
                  ON m.hypothesis_id = h.hypothesis_id
                 AND m.mode = ?
                 AND m.active = 1
                JOIN evaluations e
                  ON e.kind = 'root'
                 AND e.hypothesis_id = h.hypothesis_id
                 AND e.mode = m.mode
                 AND e.profile_id = ?
                 AND e.code_hash = m.code_hash
                 AND e.status = 'complete'
                WHERE h.hypothesis_id <> '000000'
                  AND e.metric IS NOT NULL

                UNION ALL

                SELECT
                    'branch' AS ref_kind,
                    b.branch_id AS ref_id,
                    b.branch_id AS ref,
                    e.metric AS score,
                    e.node_id AS node_id,
                    b.materialization_file AS file,
                    b.code_hash AS code_hash
                FROM branches b
                JOIN evaluations e
                  ON e.kind = 'branch'
                 AND e.branch_id = b.branch_id
                 AND e.mode = b.mode
                 AND e.profile_id = ?
                 AND e.code_hash = b.code_hash
                 AND e.status = 'complete'
                WHERE b.mode = ?
                  AND b.status = 'materialized'
                  AND e.metric IS NOT NULL
            ),
            node_components AS (
                SELECT
                    'hypothesis' AS owner_kind,
                    m.hypothesis_id AS owner_id,
                    'hypothesis' AS source_type,
                    m.hypothesis_id AS source_id,
                    m.mode AS mode,
                    m.file AS file,
                    m.code_hash AS code_hash
                FROM materializations m
                WHERE m.mode = ?
                  AND m.active = 1

                UNION ALL

                SELECT
                    'branch' AS owner_kind,
                    bc.branch_id AS owner_id,
                    bc.source_type,
                    bc.source_id,
                    bc.mode,
                    bc.file,
                    bc.code_hash
                FROM branch_components bc
                WHERE bc.mode = ?
            ),
            parent_candidates AS (
                SELECT
                    p.*,
                    (
                        SELECT COUNT(*)
                        FROM branch_edges be
                        JOIN branches child
                          ON child.branch_id = be.branch_id
                         AND child.mode = ?
                        WHERE be.parent_kind = p.ref_kind
                          AND be.parent_id = p.ref_id
                          AND be.edge_kind = 'add_existing_groups'
                    ) AS child_count
                FROM ranked_nodes p
            )
            SELECT
                p.ref_kind AS parent_kind,
                p.ref_id AS parent_id,
                p.ref AS parent_ref,
                s.ref_kind AS source_kind,
                s.ref_id AS source_id,
                s.ref AS source_ref,
                p.score AS parent_score,
                s.score AS source_score,
                p.child_count AS parent_child_count
            FROM parent_candidates p
            JOIN ranked_nodes s
              ON NOT (s.ref_kind = p.ref_kind AND s.ref_id = p.ref_id)
            WHERE p.ref_kind IN ({parent_placeholders})
              AND p.child_count < ?
              AND s.ref_kind IN ({source_placeholders})
              AND EXISTS (
                  SELECT 1
                  FROM node_components sc
                  WHERE sc.owner_kind = s.ref_kind
                    AND sc.owner_id = s.ref_id
                    AND NOT EXISTS (
                        SELECT 1
                        FROM node_components pc
                        WHERE pc.owner_kind = p.ref_kind
                          AND pc.owner_id = p.ref_id
                          AND pc.source_type = sc.source_type
                          AND pc.source_id = sc.source_id
                          AND pc.mode = sc.mode
                          AND pc.file = sc.file
                          AND pc.code_hash = sc.code_hash
                    )
              )
            ORDER BY
                p.score DESC,
                {parent_order},
                p.ref_id ASC,
                s.score DESC,
                {source_order},
                s.ref_id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def existing_branch_composition_hashes(project_dir: Path, *, mode: str) -> set[str]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT composition_hash FROM branches WHERE mode=?",
            (mode,),
        ).fetchall()
    return {str(row["composition_hash"]) for row in rows if row["composition_hash"]}


def direct_branch_child_counts(project_dir: Path, *, mode: str) -> dict[tuple[str, str], int]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT be.parent_kind, be.parent_id, COUNT(*) AS child_count
            FROM branch_edges be
            JOIN branches child
              ON child.branch_id = be.branch_id
             AND child.mode = ?
            WHERE be.edge_kind = 'add_existing_groups'
            GROUP BY be.parent_kind, be.parent_id
            """,
            (mode,),
        ).fetchall()
    return {(str(row["parent_kind"]), str(row["parent_id"])): int(row["child_count"]) for row in rows}


def branch_node_count(project_dir: Path, branch_id: str) -> int:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT count(*) FROM nodes WHERE kind='branch' AND branch_id=?",
            (_normalize_branch_id(branch_id),),
        ).fetchone()
    return int(row[0]) if row else 0


def branch_node_paths(project_dir: Path, branch_id: str) -> list[str]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path FROM nodes WHERE kind='branch' AND branch_id=?",
            (_normalize_branch_id(branch_id),),
        ).fetchall()
    return [str(row["path"]) for row in rows]


def node_record(project_dir: Path, node_id: str) -> dict[str, Any]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    if row is None:
        raise ValueError(f"No node record found: {node_id}")
    return dict(row)


def root_materialization_by_code_hash(
    project_dir: Path,
    *,
    hypothesis_id: str,
    mode: str,
    code_hash: str,
) -> dict[str, Any]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT h.hypothesis_id, r.path AS hypothesis_path, m.mode, m.file, m.code_hash
            FROM hypotheses h
            JOIN materializations m ON m.hypothesis_id=h.hypothesis_id
            LEFT JOIN hypothesis_revisions r
              ON r.hypothesis_id=m.hypothesis_id
             AND r.revision=m.hypothesis_revision
            WHERE h.hypothesis_id=? AND m.mode=? AND m.code_hash=?
            ORDER BY m.active DESC, m.file
            LIMIT 1
            """,
            (hypothesis_id.zfill(6), mode, code_hash),
        ).fetchone()
    if row is None:
        raise ValueError(f"No {mode} materialization for hypothesis {hypothesis_id} with code hash {code_hash[:12]}")
    return dict(row)


def delete_branch_records(project_dir: Path, branch_id: str, *, force: bool = False) -> None:
    normalized = _normalize_branch_id(branch_id)
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        if not force:
            row = conn.execute(
                "SELECT count(*) FROM nodes WHERE kind='branch' AND branch_id=?",
                (normalized,),
            ).fetchone()
            if int(row[0] if row else 0):
                raise ValueError(f"Branch {normalized} has run nodes; use force=true.")
        node_rows = conn.execute(
            "SELECT node_id FROM nodes WHERE kind='branch' AND branch_id=?",
            (normalized,),
        ).fetchall()
        node_ids = [str(row["node_id"]) for row in node_rows]
        for node_id_value in node_ids:
            conn.execute("DELETE FROM submissions WHERE node_id=?", (node_id_value,))
            conn.execute("DELETE FROM artifacts WHERE node_id=?", (node_id_value,))
            conn.execute("DELETE FROM evaluations WHERE node_id=?", (node_id_value,))
            conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id_value,))
        conn.execute("DELETE FROM branch_edges WHERE branch_id=? OR child_id=?", (normalized, normalized))
        conn.execute("DELETE FROM branch_components WHERE branch_id=?", (normalized,))
        conn.execute("DELETE FROM branches WHERE branch_id=?", (normalized,))
        conn.commit()


def upsert_branch(
    project_dir: Path,
    branch_dir: Path,
    *,
    parent_ref: str,
    source_ref: str,
    parent_kind: str,
    parent_id: str,
    mode: str,
    materialization_file: str,
    code_hash: str,
    composition_hash: str,
    summary: str,
    components: list[dict[str, Any]],
) -> None:
    db_path = ensure_project_db(project_dir)
    payload = read_yaml(branch_dir / "branch.yaml")
    branch_id = str(payload.get("branch_id") or branch_dir.name)
    created_at = str(payload.get("created_at") or "")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO branches(
              branch_id, parent_ref, source_ref, mode, status, created_at, path,
              materialization_file, code_hash, composition_hash, summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
              parent_ref=excluded.parent_ref,
              source_ref=excluded.source_ref,
              mode=excluded.mode,
              status=excluded.status,
              created_at=excluded.created_at,
              path=excluded.path,
              materialization_file=excluded.materialization_file,
              code_hash=excluded.code_hash,
              composition_hash=excluded.composition_hash,
              summary=excluded.summary
            """,
            (
                branch_id,
                parent_ref,
                source_ref,
                mode,
                "materialized",
                created_at,
                _project_path(project_dir, branch_dir / "branch.yaml"),
                materialization_file,
                code_hash,
                composition_hash,
                summary,
            ),
        )
        conn.execute("DELETE FROM branch_components WHERE branch_id=?", (branch_id,))
        for component in components:
            conn.execute(
                """
                INSERT INTO branch_components(
                  branch_id, role, source_type, source_id, mode, file, code_hash, path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
                    component.get("role"),
                    component.get("source_type"),
                    component.get("source_id"),
                    component.get("mode"),
                    component.get("file"),
                    component.get("code_hash"),
                    component.get("path"),
                ),
            )
        conn.execute(
            """
            INSERT OR REPLACE INTO branch_edges(
              branch_id, parent_kind, parent_id, child_kind, child_id, edge_kind, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (branch_id, parent_kind, parent_id, "branch", branch_id, "add_existing_groups", created_at),
        )
        conn.commit()


def upsert_node_start(project_dir: Path, node_dir: Path, payload: dict[str, Any]) -> None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO nodes(
              node_id, run_id, step, kind, hypothesis_id, hypothesis_revision,
              materialization_file, branch_id, mode, profile_id, status, created_at, path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              run_id=excluded.run_id,
              step=excluded.step,
              kind=excluded.kind,
              hypothesis_id=excluded.hypothesis_id,
              hypothesis_revision=excluded.hypothesis_revision,
              materialization_file=excluded.materialization_file,
              branch_id=excluded.branch_id,
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
                payload.get("kind") or "root",
                payload.get("hypothesis_id"),
                payload.get("hypothesis_revision"),
                payload.get("materialization_file"),
                payload.get("branch_id"),
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
    manifest = read_yaml(node_dir / "artifact-manifest.yaml")
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
            INSERT INTO evaluations(
              node_id, kind, hypothesis_id, hypothesis_revision, materialization_file,
              branch_id, mode, profile_id, code_hash, metric, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              kind=excluded.kind,
              hypothesis_id=excluded.hypothesis_id,
              hypothesis_revision=excluded.hypothesis_revision,
              materialization_file=excluded.materialization_file,
              branch_id=excluded.branch_id,
              mode=excluded.mode,
              profile_id=excluded.profile_id,
              code_hash=excluded.code_hash,
              metric=excluded.metric,
              status=excluded.status
            """,
            (
                node_dir.name,
                start.get("kind") or "root",
                start.get("hypothesis_id"),
                start.get("hypothesis_revision") or manifest.get("hypothesis_revision"),
                start.get("materialization_file") or manifest.get("materialization_file"),
                start.get("branch_id"),
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
                "01-branch.yaml",
                "02-code.py",
                "started.yaml",
                "stdout.log",
                "stderr.log",
                "result.yaml",
                "artifact-manifest.yaml",
                "node.done.yaml",
                "failed.yaml",
                "artifacts/submission.csv.gz",
                "artifacts/submission.csv",
                "artifacts/test_predictions.csv.gz",
                "artifacts/validation_predictions.csv.gz",
            ],
        )
        upsert_submission(
            conn,
            build_submission_row(
                project_dir=project_dir,
                node_dir=node_dir,
                start=start,
                manifest=manifest,
                status=status,
                metric=metric,
                code_hash=code_hash,
                finished_at=finished_at,
                run_seconds=run_seconds,
            ),
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
              SELECT 1
              FROM materializations m
              JOIN evaluations e ON e.hypothesis_id=m.hypothesis_id
                AND e.mode=m.mode AND e.code_hash=m.code_hash
              WHERE m.hypothesis_id=h.hypothesis_id
                AND m.active=1
                AND e.status='complete'
            ) THEN '▶'
            WHEN EXISTS (
              SELECT 1
              FROM materializations m
              JOIN evaluations e ON e.hypothesis_id=m.hypothesis_id
                AND e.mode=m.mode AND e.code_hash=m.code_hash
              WHERE m.hypothesis_id=h.hypothesis_id
                AND m.active=1
                AND e.status='failed'
            ) THEN '⚠'
            WHEN EXISTS (
              SELECT 1 FROM materializations m
              WHERE m.hypothesis_id=h.hypothesis_id AND m.active=1
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


def run_candidates(project_dir: Path, mode: str, hypothesis_id: str | None = None, revision: int | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT h.hypothesis_id, h.path, m.file, m.code_hash, m.hypothesis_revision
        FROM hypotheses h
        JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=?
          AND m.status='active'
    """
    params: list[Any] = [mode]
    conditions: list[str] = []
    if hypothesis_id:
        conditions.append("h.hypothesis_id=?")
        params.append(hypothesis_id.zfill(6))
    if revision is not None:
        conditions.append("m.hypothesis_revision=?")
        params.append(int(revision))
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY h.hypothesis_id, m.hypothesis_revision, m.file"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def branch_run_candidates(project_dir: Path, mode: str, branch_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT branch_id, path, materialization_file AS file, code_hash, composition_hash, summary, created_at
        FROM branches
        WHERE mode=? AND status='materialized'
    """
    params: list[Any] = [mode]
    if branch_id:
        sql += " AND branch_id=?"
        params.append(_normalize_branch_id(branch_id))
    sql += " ORDER BY branch_id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def pending_branch_run_candidates(project_dir: Path, *, mode: str, profile_id: str, branch_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT branch_id, path, materialization_file AS file, code_hash, composition_hash, summary, created_at
        FROM branches b
        WHERE b.mode=? AND b.status='materialized'
          AND NOT EXISTS (
            SELECT 1
            FROM evaluations e
            WHERE e.kind='branch'
              AND e.branch_id=b.branch_id
              AND e.mode=b.mode
              AND e.profile_id=?
              AND e.code_hash=b.code_hash
          )
    """
    params: list[Any] = [mode, profile_id]
    if branch_id:
        sql += " AND b.branch_id=?"
        params.append(_normalize_branch_id(branch_id))
    sql += " ORDER BY b.branch_id"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def bugfix_candidates(project_dir: Path, mode: str, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT *
        FROM (
          SELECT
            h.hypothesis_id,
            h.path,
            h.summary,
            m.file,
            m.code_hash,
            m.status AS materialization_status,
            n.node_id,
            n.run_id,
            n.step,
            n.path AS node_path,
            n.finished_at,
            n.run_seconds,
            ROW_NUMBER() OVER (
              PARTITION BY h.hypothesis_id, m.mode, m.file
              ORDER BY COALESCE(n.finished_at, n.created_at, '') DESC, n.step DESC
            ) AS rn
          FROM hypotheses h
          JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=? AND m.active=1
          JOIN evaluations e ON e.hypothesis_id=h.hypothesis_id
            AND e.mode=m.mode AND e.code_hash=m.code_hash AND e.status='failed'
          JOIN nodes n ON n.node_id=e.node_id
        )
        WHERE rn=1 AND hypothesis_id<>'000000'
    """
    params: list[Any] = [mode]
    if hypothesis_id:
        sql += " AND hypothesis_id=?"
        params.append(hypothesis_id.zfill(6))
    sql += " ORDER BY hypothesis_id"
    validation_sql = """
        SELECT
          h.hypothesis_id,
          h.path,
          h.summary,
          m.file,
          m.code_hash,
          m.status AS materialization_status,
          NULL AS node_id,
          NULL AS run_id,
          NULL AS step,
          NULL AS node_path,
          NULL AS finished_at,
          NULL AS run_seconds
        FROM hypotheses h
        JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=?
        WHERE m.status='failed' AND m.active=0 AND h.hypothesis_id<>'000000'
    """
    validation_params: list[Any] = [mode]
    if hypothesis_id:
        validation_sql += " AND h.hypothesis_id=?"
        validation_params.append(hypothesis_id.zfill(6))
    validation_sql += " ORDER BY h.hypothesis_id, m.file"
    with connect(db_path) as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        rows.extend(dict(row) for row in conn.execute(validation_sql, validation_params).fetchall())
    return sorted(rows, key=lambda row: (str(row["hypothesis_id"]), str(row["file"])))


def run_request_status(project_dir: Path, mode: str, hypothesis_id: str | None) -> list[dict[str, Any]]:
    if not hypothesis_id:
        return []
    db_path = ensure_project_db(project_dir)
    hid = hypothesis_id.zfill(6)
    sql = """
        SELECT
          h.hypothesis_id,
          CASE
            WHEN h.hypothesis_id IS NULL THEN 'missing_hypothesis'
            WHEN m.file IS NULL THEN 'missing_materialization'
            ELSE 'ready'
          END AS status
        FROM (SELECT ? AS requested_id) r
        LEFT JOIN hypotheses h ON h.hypothesis_id=r.requested_id
        LEFT JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=? AND m.active=1
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql, (hid, mode)).fetchall()
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


def branch_already_evaluated(project_dir: Path, *, branch_id: str, mode: str, profile_id: str, code_hash: str) -> bool:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM evaluations
            WHERE kind='branch' AND branch_id=? AND mode=? AND profile_id=? AND code_hash=?
            LIMIT 1
            """,
            (_normalize_branch_id(branch_id), mode, profile_id, code_hash),
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


def latest_run_id(project_dir: Path) -> str | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
    return str(row["run_id"]) if row else None


def next_node_step(project_dir: Path, run_id_value: str) -> int:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT max(step) AS max_step FROM nodes WHERE run_id=?", (run_id_value,)).fetchone()
    return int(row["max_step"] or 0) + 1


def root_counts(project_dir: Path) -> dict[str, int]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        return {
            "hypotheses": int(conn.execute("SELECT count(*) FROM hypotheses WHERE hypothesis_id<>'000000'").fetchone()[0]),
            "materialized": int(
                conn.execute("SELECT count(DISTINCT hypothesis_id) FROM materializations WHERE hypothesis_id<>'000000'").fetchone()[0]
            ),
            "evaluated": int(
                conn.execute(
                    "SELECT count(DISTINCT hypothesis_id) FROM nodes WHERE kind='root' AND status='complete' AND hypothesis_id<>'000000'"
                ).fetchone()[0]
            ),
            "incomplete": int(
                conn.execute(
                    "SELECT count(*) FROM nodes WHERE kind='root' AND status NOT IN ('complete','failed') AND hypothesis_id<>'000000'"
                ).fetchone()[0]
            ),
        }


def best_score(project_dir: Path) -> float | None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        row = conn.execute("SELECT max(metric) AS best FROM evaluations WHERE kind='root' AND metric IS NOT NULL").fetchone()
    value = row["best"] if row else None
    return float(value) if value is not None else None


def submission_rows(project_dir: Path) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT
          s.*,
          COALESCE(n.branch_id, s.hypothesis_id) AS source_id,
          RANK() OVER (
            ORDER BY CASE WHEN s.local_score IS NULL THEN 1 ELSE 0 END, s.local_score DESC
          ) AS cv_rank,
          RANK() OVER (
            ORDER BY CASE WHEN s.public_score IS NULL THEN 1 ELSE 0 END, s.public_score DESC
          ) AS computed_public_rank
        FROM submissions s
        LEFT JOIN nodes n ON n.node_id=s.node_id
        ORDER BY
          CASE WHEN s.local_score IS NULL THEN 1 ELSE 0 END,
          s.local_score DESC,
          s.created_at DESC
    """
    with connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def submission_by_sha_prefix(project_dir: Path, sha_prefix: str) -> dict[str, Any]:
    db_path = ensure_project_db(project_dir)
    prefix = sha_prefix.strip().lower()
    if not prefix:
        raise ValueError("Missing submission sha prefix.")
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM submissions
            WHERE lower(submission_sha256) LIKE ?
            ORDER BY local_score DESC
            """,
            (prefix + "%",),
        ).fetchall()
    if not rows:
        raise ValueError(f"No submission matches sha prefix: {sha_prefix}")
    distinct_hashes = {str(row["submission_sha256"]) for row in rows}
    if len(distinct_hashes) > 1:
        preview = ", ".join(sorted(value[:10] for value in distinct_hashes))
        raise ValueError(f"Ambiguous sha prefix {sha_prefix}; matches: {preview}")
    return dict(rows[0])


def mark_submission_submitted(
    project_dir: Path,
    *,
    node_id: str,
    submission_path: str,
    submitted_at: str,
    kaggle_message: str,
    kaggle_response: Any,
    upload_path: str,
    uploaded_filename: str,
) -> None:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE submissions
            SET submit_status='uploaded',
                submitted_at=?,
                kaggle_message=?,
                kaggle_response_json=?,
                kaggle_ref=?,
                upload_path=?,
                uploaded_filename=?
            WHERE node_id=? AND submission_path=?
            """,
            (
                submitted_at,
                kaggle_message,
                json.dumps(_jsonable(kaggle_response), sort_keys=True),
                _response_ref(kaggle_response),
                upload_path,
                uploaded_filename,
                node_id,
                submission_path,
            ),
        )
        conn.commit()


def sync_submission_remote_rows(project_dir: Path, remote_submissions: list[object]) -> int:
    db_path = ensure_project_db(project_dir)
    synced_at = datetime.now().isoformat(timespec="seconds")
    changed = 0
    with connect(db_path) as conn:
        changed += _normalize_remote_error_submit_status(conn)
        rows = [dict(row) for row in conn.execute("SELECT * FROM submissions").fetchall()]
        remote_rows = [_remote_submission_fields(remote, synced_at=synced_at) for remote in remote_submissions]
        for row in rows:
            candidates = [remote for remote in remote_rows if _remote_matches_submission_row(row, remote)]
            if not candidates:
                continue
            remote_fields = max(candidates, key=_remote_preference_key)
            update = {
                "submit_status": _submit_status_from_remote(remote_fields.get("remote_status")),
                **remote_fields,
            }
            persisted_update = {
                key: update[key]
                for key in (
                    "submit_status",
                    "kaggle_ref",
                    "remote_status",
                    "remote_date",
                    "remote_url",
                    "public_score",
                    "private_score",
                )
            }
            before = {key: row.get(key) for key in persisted_update}
            if before == persisted_update:
                continue
            conn.execute(
                """
                UPDATE submissions
                SET submit_status=?,
                    kaggle_ref=?,
                    remote_status=?,
                    remote_date=?,
                    remote_url=?,
                    public_score=?,
                    private_score=?
                WHERE node_id=? AND submission_path=?
                """,
                (
                    update["submit_status"],
                    update["kaggle_ref"],
                    update["remote_status"],
                    update["remote_date"],
                    update["remote_url"],
                    update["public_score"],
                    update["private_score"],
                    row["node_id"],
                    row["submission_path"],
                ),
            )
            row.update(persisted_update)
            changed += 1
        conn.commit()
    return changed


def _normalize_remote_error_submit_status(conn) -> int:
    cursor = conn.execute(
        """
        UPDATE submissions
        SET submit_status='uploaded'
        WHERE submit_status='failed'
          AND upper(COALESCE(remote_status, '')) IN ('ERROR', 'FAILED', 'FAILURE')
        """
    )
    return int(cursor.rowcount or 0)


def _remote_submission_fields(remote: object, *, synced_at: str) -> dict[str, Any]:
    description = _remote_attr(remote, "description")
    return {
        "kaggle_ref": _remote_ref(remote),
        "remote_filename": _remote_attr(remote, "file_name"),
        "remote_date": _date_to_string(_remote_attr(remote, "date")),
        "remote_description": description,
        "remote_status": _status_to_string(_remote_attr(remote, "status")),
        "public_score": _score_or_none(_remote_attr(remote, "public_score")),
        "private_score": _score_or_none(_remote_attr(remote, "private_score")),
        "remote_url": _remote_attr(remote, "url"),
        "synced_at": synced_at,
        "parsed_description": _parse_submission_description(description),
    }


def _remote_matches_submission_row(row: dict[str, Any], remote_fields: dict[str, Any]) -> bool:
    remote_ref = str(remote_fields.get("kaggle_ref") or "")
    if remote_ref and str(row.get("kaggle_ref") or "") == remote_ref:
        return True
    filename = str(remote_fields.get("remote_filename") or "")
    if filename and str(row.get("uploaded_filename") or "") == filename:
        return True
    parsed = remote_fields.get("parsed_description")
    parsed_description = parsed if isinstance(parsed, dict) else {}
    sha = str(parsed_description.get("sha") or "").lower()
    if not sha or not str(row.get("submission_sha256") or "").lower().startswith(sha):
        return False
    run = str(parsed_description.get("run") or "")
    if run and str(row.get("run_id") or "") != run:
        return False
    step = str(parsed_description.get("step") or "")
    if step and str(row.get("step") or "") != step:
        return False
    return True


def _remote_preference_key(remote_fields: dict[str, Any]) -> tuple[int, str, int]:
    status = str(remote_fields.get("remote_status") or "").strip().upper()
    status_rank = {
        "COMPLETE": 4,
        "SUBMITTED": 3,
        "PENDING": 2,
        "RUNNING": 2,
        "ERROR": 1,
        "FAILED": 1,
        "FAILURE": 1,
    }.get(status, 0)
    ref_text = str(remote_fields.get("kaggle_ref") or "")
    try:
        ref = int(ref_text)
    except ValueError:
        ref = 0
    return status_rank, str(remote_fields.get("remote_date") or ""), ref


def _remote_attr(remote: object, snake_name: str) -> Any:
    if isinstance(remote, dict):
        return remote.get(snake_name) or remote.get(_snake_to_camel(snake_name))
    if hasattr(remote, snake_name):
        return getattr(remote, snake_name)
    private_name = f"_{snake_name}"
    if hasattr(remote, private_name):
        return getattr(remote, private_name)
    camel_name = _snake_to_camel(snake_name)
    if hasattr(remote, camel_name):
        return getattr(remote, camel_name)
    return None


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _remote_ref(remote: object) -> str | None:
    ref = _remote_attr(remote, "ref")
    return str(ref) if ref is not None else None


def _status_to_string(status: object) -> str | None:
    if status is None:
        return None
    if hasattr(status, "name"):
        return str(status.name)
    return str(status)


def _date_to_string(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _score_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_submission_description(description: object) -> dict[str, str]:
    import re

    if not description:
        return {}
    parsed: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^|]+)", str(description)):
        parsed[match.group(1).strip()] = match.group(2).strip()
    return parsed


def _submit_status_from_remote(remote_status: object) -> str:
    status = str(remote_status or "").strip().upper()
    if status in {"COMPLETE", "SUBMITTED"}:
        return "submitted"
    return "uploaded"


def materialization_rows(project_dir: Path, *, mode: str, hypothesis_id: str | None = None) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT h.hypothesis_id, h.summary, m.mode, m.file, m.status, m.active,
               m.hypothesis_revision, m.model, m.reasoning_tokens, m.total_tokens, m.generation_seconds
        FROM materializations m
        JOIN hypotheses h ON h.hypothesis_id=m.hypothesis_id
        WHERE m.mode=?
    """
    params: list[Any] = [mode]
    if hypothesis_id:
        sql += " AND h.hypothesis_id=?"
        params.append(hypothesis_id.zfill(6))
    sql += " ORDER BY h.hypothesis_id, m.file"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def root_run_rows(
    project_dir: Path,
    *,
    mode: str,
    profile_id: str,
    hypothesis_id: str | None = None,
) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT
          h.hypothesis_id, COALESCE(r.summary, h.summary) AS summary,
          COALESCE(r.created_at, h.created_at) AS created_at,
          m.model, m.reasoning_tokens, m.total_tokens, m.generation_seconds, h.enabled,
          m.file AS materialization_file, m.hypothesis_revision, m.code_hash, m.status AS materialization_status,
          n.node_id, n.status AS node_status, n.run_seconds,
          e.metric,
          (
            SELECT GROUP_CONCAT(status)
            FROM (
              SELECT DISTINCT cn.status AS status
              FROM run_components rc
              JOIN nodes cn ON cn.node_id=rc.node_id
              WHERE rc.source_type='hypothesis'
                AND rc.source_id=h.hypothesis_id
                AND rc.mode=m.mode
                AND rc.file=m.file
                AND rc.code_hash=m.code_hash
                AND cn.profile_id=?
              ORDER BY
                CASE cn.status
                  WHEN 'failed' THEN 0
                  WHEN 'execution_interrupted' THEN 1
                  WHEN 'started' THEN 2
                  WHEN 'missing_code' THEN 3
                  WHEN 'complete' THEN 4
                  ELSE 5
                END
            )
          ) AS component_statuses,
          (
            SELECT cn.status
            FROM run_components rc
            JOIN nodes cn ON cn.node_id=rc.node_id
            WHERE rc.source_type='hypothesis'
              AND rc.source_id=h.hypothesis_id
              AND rc.mode=m.mode
              AND rc.file=m.file
              AND rc.code_hash=m.code_hash
              AND cn.profile_id=?
            ORDER BY
              CASE cn.status
                WHEN 'failed' THEN 0
                WHEN 'execution_interrupted' THEN 1
                WHEN 'started' THEN 2
                WHEN 'missing_code' THEN 3
                WHEN 'complete' THEN 4
                ELSE 5
              END,
              cn.created_at DESC
            LIMIT 1
          ) AS component_status,
          (
            SELECT cn.node_id
            FROM run_components rc
            JOIN nodes cn ON cn.node_id=rc.node_id
            WHERE rc.source_type='hypothesis'
              AND rc.source_id=h.hypothesis_id
              AND rc.mode=m.mode
              AND rc.file=m.file
              AND rc.code_hash=m.code_hash
              AND cn.profile_id=?
            ORDER BY
              CASE cn.status
                WHEN 'failed' THEN 0
                WHEN 'execution_interrupted' THEN 1
                WHEN 'started' THEN 2
                WHEN 'missing_code' THEN 3
                WHEN 'complete' THEN 4
                ELSE 5
              END,
              cn.created_at DESC
            LIMIT 1
          ) AS component_node_id,
          (
            SELECT cn.run_seconds
            FROM run_components rc
            JOIN nodes cn ON cn.node_id=rc.node_id
            WHERE rc.source_type='hypothesis'
              AND rc.source_id=h.hypothesis_id
              AND rc.mode=m.mode
              AND rc.file=m.file
              AND rc.code_hash=m.code_hash
              AND cn.profile_id=?
            ORDER BY
              CASE cn.status
                WHEN 'failed' THEN 0
                WHEN 'execution_interrupted' THEN 1
                WHEN 'started' THEN 2
                WHEN 'missing_code' THEN 3
                WHEN 'complete' THEN 4
                ELSE 5
              END,
              cn.created_at DESC
            LIMIT 1
          ) AS component_run_seconds
        FROM hypotheses h
        LEFT JOIN materializations m ON m.hypothesis_id=h.hypothesis_id AND m.mode=?
        LEFT JOIN hypothesis_revisions r
          ON r.hypothesis_id=h.hypothesis_id
         AND r.revision=COALESCE(m.hypothesis_revision, 1)
        LEFT JOIN evaluations e ON e.hypothesis_id=h.hypothesis_id
          AND e.mode=? AND e.profile_id=? AND e.code_hash=m.code_hash
          AND COALESCE(e.materialization_file, m.file)=m.file
        LEFT JOIN nodes n ON n.node_id=e.node_id
    """
    params: list[Any] = [profile_id, profile_id, profile_id, profile_id, mode, mode, profile_id]
    if hypothesis_id:
        sql += " WHERE h.hypothesis_id=?"
        params.append(hypothesis_id.zfill(6))
    sql += " ORDER BY h.hypothesis_id, m.hypothesis_revision, m.file"
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def revision_status_rows(
    project_dir: Path,
    *,
    hypothesis_id: str,
    mode: str,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    db_path = project_db_path(project_dir)
    profile_filter = "AND n.profile_id=?" if profile_id else ""
    evaluation_profile_filter = "AND e.profile_id=?" if profile_id else ""
    params: list[Any] = []
    for _ in range(4):
        if profile_id:
            params.append(profile_id)
    params.extend([mode, mode])
    if profile_id:
        params.append(profile_id)
    params.append(hypothesis_id.zfill(6))
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              r.hypothesis_id,
              r.revision,
              r.path AS hypothesis_file,
              m.file AS materialization_file,
              m.active,
              m.status AS materialization_status,
              e.metric,
              e.status AS evaluation_status,
              en.created_at AS evaluation_created_at,
              (
                SELECT GROUP_CONCAT(status)
                FROM (
                  SELECT DISTINCT n.status AS status
                  FROM run_components rc
                  JOIN nodes n ON n.node_id=rc.node_id
                  WHERE rc.source_type='hypothesis'
                    AND rc.source_id=r.hypothesis_id
                    AND rc.mode=m.mode
                    AND rc.file=m.file
                    AND rc.code_hash=m.code_hash
                    {profile_filter}
                  ORDER BY
                    CASE n.status
                      WHEN 'failed' THEN 0
                      WHEN 'execution_interrupted' THEN 1
                      WHEN 'started' THEN 2
                      WHEN 'missing_code' THEN 3
                      WHEN 'complete' THEN 4
                      ELSE 5
                    END
                )
              ) AS component_statuses,
              (
                SELECT n.status
                FROM run_components rc
                JOIN nodes n ON n.node_id=rc.node_id
                WHERE rc.source_type='hypothesis'
                  AND rc.source_id=r.hypothesis_id
                  AND rc.mode=m.mode
                  AND rc.file=m.file
                  AND rc.code_hash=m.code_hash
                  {profile_filter}
                ORDER BY
                  CASE n.status
                    WHEN 'failed' THEN 0
                    WHEN 'execution_interrupted' THEN 1
                    WHEN 'started' THEN 2
                    WHEN 'missing_code' THEN 3
                    WHEN 'complete' THEN 4
                    ELSE 5
                  END,
                  n.created_at DESC
                LIMIT 1
              ) AS component_status,
              (
                SELECT n.created_at
                FROM run_components rc
                JOIN nodes n ON n.node_id=rc.node_id
                WHERE rc.source_type='hypothesis'
                  AND rc.source_id=r.hypothesis_id
                  AND rc.mode=m.mode
                  AND rc.file=m.file
                  AND rc.code_hash=m.code_hash
                  {profile_filter}
                ORDER BY
                  CASE n.status
                    WHEN 'failed' THEN 0
                    WHEN 'execution_interrupted' THEN 1
                    WHEN 'started' THEN 2
                    WHEN 'missing_code' THEN 3
                    WHEN 'complete' THEN 4
                    ELSE 5
                  END,
                  n.created_at DESC
                LIMIT 1
              ) AS component_created_at,
              (
                SELECT n.node_id
                FROM run_components rc
                JOIN nodes n ON n.node_id=rc.node_id
                WHERE rc.source_type='hypothesis'
                  AND rc.source_id=r.hypothesis_id
                  AND rc.mode=m.mode
                  AND rc.file=m.file
                  AND rc.code_hash=m.code_hash
                  {profile_filter}
                ORDER BY
                  CASE n.status
                    WHEN 'failed' THEN 0
                    WHEN 'execution_interrupted' THEN 1
                    WHEN 'started' THEN 2
                    WHEN 'missing_code' THEN 3
                    WHEN 'complete' THEN 4
                    ELSE 5
                  END,
                  n.created_at DESC
                LIMIT 1
              ) AS component_node_id
            FROM hypothesis_revisions r
            LEFT JOIN materializations m
              ON m.hypothesis_id=r.hypothesis_id
             AND m.hypothesis_revision=r.revision
             AND m.mode=?
            LEFT JOIN evaluations e
              ON e.hypothesis_id=r.hypothesis_id
             AND e.hypothesis_revision=r.revision
             AND e.mode=?
             {evaluation_profile_filter}
             AND COALESCE(e.materialization_file, m.file)=m.file
            LEFT JOIN nodes en ON en.node_id=e.node_id
            WHERE r.hypothesis_id=?
            ORDER BY r.revision, m.file
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def branch_rows(
    project_dir: Path,
    *,
    mode: str,
    profile_id: str,
    branch_id: str | None = None,
    sort_by: str = "score",
    sort_order: str | None = None,
) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    sql = """
        SELECT
          b.branch_id, b.parent_ref, b.source_ref, b.mode, b.status AS branch_status,
          b.created_at, b.materialization_file, b.code_hash, b.composition_hash, b.summary,
          n.node_id, n.status AS node_status, n.run_seconds,
          e.metric
        FROM branches b
        LEFT JOIN evaluations e ON e.kind='branch'
          AND e.branch_id=b.branch_id AND e.mode=b.mode AND e.profile_id=? AND e.code_hash=b.code_hash
        LEFT JOIN nodes n ON n.node_id=e.node_id
        WHERE b.mode=?
    """
    params: list[Any] = [profile_id, mode]
    if branch_id:
        sql += " AND b.branch_id=?"
        params.append(_normalize_branch_id(branch_id))
    sql += _branch_order_clause(sort_by, sort_order)
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def _branch_order_clause(sort_by: str, sort_order: str | None) -> str:
    sort_key = str(sort_by or "score").strip().lower()
    order_key = str(sort_order or ("desc" if sort_key == "score" else "asc")).strip().lower()
    if order_key not in {"asc", "desc"}:
        raise ValueError(f"Invalid branch status order: {sort_order}. Use asc or desc.")
    direction = order_key.upper()
    sort_columns = {
        "id": "b.branch_id",
        "branch": "b.branch_id",
        "branch_id": "b.branch_id",
        "score": "e.metric",
        "created": "b.created_at",
        "created_at": "b.created_at",
        "parent": "b.parent_ref",
        "source": "b.source_ref",
        "file": "b.materialization_file",
        "status": "b.status",
    }
    column = sort_columns.get(sort_key)
    if column is None:
        allowed = ", ".join(sorted(sort_columns))
        raise ValueError(f"Invalid branch status sort: {sort_by}. Use one of: {allowed}.")
    if sort_key == "score":
        return f" ORDER BY CASE WHEN e.metric IS NULL THEN 1 ELSE 0 END, e.metric {direction}, b.branch_id"
    if column == "b.branch_id":
        return f" ORDER BY {column} {direction}"
    return f" ORDER BY CASE WHEN {column} IS NULL THEN 1 ELSE 0 END, {column} {direction}, b.branch_id"


def solution_tree_root_rows(project_dir: Path, *, mode: str, profile_id: str) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              h.hypothesis_id, h.summary, h.created_at, h.model,
              h.reasoning_tokens, h.total_tokens, h.generation_seconds,
              m.code_hash, m.status AS materialization_status,
              n.node_id, n.status AS node_status, n.run_seconds,
              e.metric
            FROM hypotheses h
            LEFT JOIN materializations m
              ON m.hypothesis_id=h.hypothesis_id AND m.mode=? AND m.active=1
            LEFT JOIN evaluations e
              ON e.kind='root'
             AND e.hypothesis_id=h.hypothesis_id
             AND e.mode=?
             AND e.profile_id=?
             AND e.code_hash=m.code_hash
            LEFT JOIN nodes n ON n.node_id=e.node_id
            ORDER BY h.hypothesis_id
            """,
            (mode, mode, profile_id),
        ).fetchall()
    return [dict(row) for row in rows]


def solution_tree_branch_rows(project_dir: Path, *, mode: str, profile_id: str) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              b.branch_id, b.parent_ref, b.source_ref, b.mode,
              b.status AS branch_status, b.created_at, b.materialization_file,
              b.code_hash, b.composition_hash, b.summary,
              be.parent_kind, be.parent_id, be.edge_kind,
              n.node_id, n.status AS node_status, n.run_seconds,
              e.metric
            FROM branches b
            LEFT JOIN branch_edges be ON be.branch_id=b.branch_id
            LEFT JOIN evaluations e
              ON e.kind='branch'
             AND e.branch_id=b.branch_id
             AND e.mode=b.mode
             AND e.profile_id=?
             AND e.code_hash=b.code_hash
            LEFT JOIN nodes n ON n.node_id=e.node_id
            WHERE b.mode=?
            ORDER BY b.branch_id
            """,
            (profile_id, mode),
        ).fetchall()
    return [dict(row) for row in rows]


def branch_component_rows(project_dir: Path, branch_id: str) -> list[dict[str, Any]]:
    db_path = ensure_project_db(project_dir)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM branch_components
            WHERE branch_id=?
            ORDER BY role, source_type, source_id, file
            """,
            (_normalize_branch_id(branch_id),),
        ).fetchall()
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


def _web_search_summary(request_path: Path, summary_path: Path) -> dict[str, bool]:
    request = read_yaml(request_path) if request_path.exists() else {}
    metadata = request.get("metadata") if isinstance(request, dict) and isinstance(request.get("metadata"), dict) else {}
    enabled = bool(metadata.get("web_search_enabled"))
    has_results = summary_path.exists() and bool(summary_path.read_text(encoding="utf-8", errors="replace").strip())
    return {"enabled": enabled, "has_results": has_results}


def _token_total(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    token_usage = usage.get("tokenUsage")
    if not isinstance(token_usage, dict):
        return {}
    total = token_usage.get("total")
    return total if isinstance(total, dict) else {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(dict(value.__dict__))
    return repr(value)


def _response_ref(value: Any) -> str | None:
    payload = _jsonable(value)
    if isinstance(payload, dict):
        ref = payload.get("ref")
        return str(ref) if ref is not None else None
    return None


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


def _normalize_branch_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.upper().startswith("B"):
        suffix = text[1:]
        return f"B{int(suffix):06d}" if suffix.isdigit() else text.upper()
    return f"B{int(text):06d}" if text.isdigit() else text


def _branch_algorithm_db_kind(kind: str) -> str:
    text = str(kind).strip().lower()
    if text == "root":
        return "hypothesis"
    if text in {"hypothesis", "branch"}:
        return text
    raise ValueError(f"Invalid branch algorithm node kind: {kind}")


def _branch_algorithm_kind_order_sql(column: str, ordered_kinds: list[str]) -> str:
    clauses = " ".join(f"WHEN '{kind}' THEN {index}" for index, kind in enumerate(ordered_kinds))
    return f"CASE {column} {clauses} ELSE 99 END"
