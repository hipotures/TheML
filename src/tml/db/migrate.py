from __future__ import annotations

from importlib import resources
from pathlib import Path

from .connect import connect


def migrate(db_path: Path) -> None:
    schema = resources.files("tml.db").joinpath("schema.sql").read_text(encoding="utf-8")
    with connect(db_path) as conn:
        conn.executescript(schema)
        _ensure_column(conn, "hypotheses", "summary", "TEXT")
        _ensure_column(conn, "hypotheses", "created_at", "TEXT")
        _ensure_column(conn, "hypotheses", "model", "TEXT")
        _ensure_column(conn, "hypotheses", "reasoning_tokens", "INTEGER")
        _ensure_column(conn, "hypotheses", "total_tokens", "INTEGER")
        _ensure_column(conn, "hypotheses", "generation_seconds", "INTEGER")
        _ensure_column(conn, "hypotheses", "web_search_enabled", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "hypotheses", "web_search_has_results", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "materializations", "model", "TEXT")
        _ensure_column(conn, "materializations", "status", "TEXT NOT NULL DEFAULT 'active'")
        _ensure_column(conn, "materializations", "active", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(conn, "materializations", "source_node_id", "TEXT")
        _ensure_column(conn, "materializations", "fixed_from_file", "TEXT")
        _ensure_column(conn, "materializations", "fixed_from_code_hash", "TEXT")
        _ensure_column(conn, "materializations", "reasoning_tokens", "INTEGER")
        _ensure_column(conn, "materializations", "total_tokens", "INTEGER")
        _ensure_column(conn, "materializations", "generation_seconds", "INTEGER")
        _ensure_column(conn, "nodes", "kind", "TEXT NOT NULL DEFAULT 'root'")
        _ensure_column(conn, "nodes", "branch_id", "TEXT")
        _ensure_column(conn, "nodes", "created_at", "TEXT")
        _ensure_column(conn, "nodes", "finished_at", "TEXT")
        _ensure_column(conn, "nodes", "run_seconds", "INTEGER")
        _ensure_column(conn, "evaluations", "kind", "TEXT NOT NULL DEFAULT 'root'")
        _ensure_column(conn, "evaluations", "branch_id", "TEXT")
        _ensure_column(conn, "submissions", "submitted_at", "TEXT")
        _ensure_column(conn, "submissions", "kaggle_message", "TEXT")
        _ensure_column(conn, "submissions", "kaggle_response_json", "TEXT")
        _ensure_column(conn, "submissions", "kaggle_ref", "TEXT")
        _ensure_column(conn, "submissions", "upload_path", "TEXT")
        _ensure_column(conn, "submissions", "uploaded_filename", "TEXT")
        _ensure_column(conn, "submissions", "remote_status", "TEXT")
        _ensure_column(conn, "submissions", "remote_date", "TEXT")
        _ensure_column(conn, "submissions", "remote_url", "TEXT")
        _ensure_column(conn, "submissions", "private_score", "REAL")
        _ensure_column(conn, "submissions", "source_submission_sha256", "TEXT")
        _ensure_column(conn, "submissions", "source_run_id", "TEXT")
        _ensure_column(conn, "submissions", "source_node_id", "TEXT")
        _ensure_column(conn, "submissions", "source_step", "INTEGER")
        _ensure_column(conn, "submissions", "source_profile_id", "TEXT")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_branches_mode_composition
            ON branches(mode, composition_hash)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_submissions_source_sha256
            ON submissions(source_submission_sha256)
            """
        )


def _ensure_column(conn, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in rows):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
