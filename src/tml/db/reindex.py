from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from tml.core.config import load_project_config, repo_root_for_project
from tml.hypotheses.baseline import ensure_root_baseline
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml

from .connect import connect
from .migrate import migrate
from .submissions import build_submission_row, upsert_submission


def reindex_project(project_dir: Path, db_path: Path) -> dict[str, int]:
    ensure_root_baseline(project_dir)
    migrate(db_path)
    config = load_project_config(project_dir)
    with connect(db_path) as conn:
        for table in (
            "projects",
            "profiles",
            "hypotheses",
            "materializations",
            "runs",
            "nodes",
            "evaluations",
            "artifacts",
            "submissions",
            "prompt_calls",
        ):
            conn.execute(f"DELETE FROM {table}")
        conn.execute(
            "INSERT INTO projects(project_id, kind, path) VALUES (?, ?, ?)",
            (config["project_id"], config.get("kind", "kaggle"), "."),
        )
        _index_profiles(conn, project_dir)
        _index_hypotheses(conn, project_dir)
        _index_runs(conn, project_dir)
        conn.commit()
        counts = {
            "hypotheses": conn.execute("SELECT count(*) FROM hypotheses").fetchone()[0],
            "materializations": conn.execute("SELECT count(*) FROM materializations").fetchone()[0],
            "nodes": conn.execute("SELECT count(*) FROM nodes").fetchone()[0],
            "submissions": conn.execute("SELECT count(*) FROM submissions").fetchone()[0],
        }
    return counts


def _index_profiles(conn, project_dir: Path) -> None:
    profile_paths = list((repo_root_for_project(project_dir) / "profiles").glob("*/*.yaml"))
    profile_paths.extend((project_dir / "profiles").glob("**/*.yaml"))
    for path in sorted(profile_paths):
        payload = read_yaml(path)
        profile_id = str(payload.get("profile_id") or path.stem)
        conn.execute(
            "INSERT INTO profiles(profile_id, mode, path, profile_hash) VALUES (?, ?, ?, ?)",
            (profile_id, str(payload.get("mode") or "unknown"), _project_path(project_dir, path), sha256_file(path)),
        )


def _index_hypotheses(conn, project_dir: Path) -> None:
    for path in sorted((project_dir / "hypotheses").glob("*/hypothesis.yaml")):
        payload = read_yaml(path)
        hid = str(payload.get("hypothesis_id") or path.parent.name)
        run_summary = _run_summary(path.parent / "01-hypothesis.request.json", path.parent / "01-hypothesis.response.json")
        web_search = _web_search_summary(path.parent / "01-hypothesis.request.json", path.parent / "01-hypothesis.web_search.md")
        conn.execute(
            """
            INSERT INTO hypotheses(
              hypothesis_id, title, summary, created_at, model, reasoning_tokens,
              total_tokens, generation_seconds, web_search_enabled,
              web_search_has_results, enabled, path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hid,
                payload.get("title"),
                payload.get("summary"),
                payload.get("created_at"),
                run_summary.get("model"),
                run_summary.get("reasoning_tokens"),
                run_summary.get("total_tokens"),
                run_summary.get("generation_seconds"),
                1 if web_search["enabled"] else 0,
                1 if web_search["has_results"] else 0,
                1 if payload.get("enabled", True) else 0,
                _project_path(project_dir, path),
            ),
        )
        for code in sorted((path.parent / "materializations").glob("*.py")):
            mode = code.name.split("-", 1)[0]
            manifest = read_yaml(path.parent / "manifest.yaml")
            materializations = manifest.get("materializations") if isinstance(manifest.get("materializations"), dict) else {}
            mode_manifest = materializations.get(mode) if isinstance(materializations.get(mode), dict) else {}
            active_file = mode_manifest.get("active") if isinstance(mode_manifest.get("active"), str) else None
            active = active_file is None or active_file == code.name
            mat_summary = _run_summary(code.parent / f"{code.stem}.request.json", code.parent / f"{code.stem}.response.json")
            conn.execute(
                """
                INSERT INTO materializations(
                  hypothesis_id, mode, file, code_hash, status, active, model,
                  reasoning_tokens, total_tokens, generation_seconds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hid,
                    mode,
                    code.name,
                    sha256_file(code),
                    "active" if active else "inactive",
                    1 if active else 0,
                    mat_summary.get("model"),
                    mat_summary.get("reasoning_tokens"),
                    mat_summary.get("total_tokens"),
                    mat_summary.get("generation_seconds"),
                ),
            )


def _index_runs(conn, project_dir: Path) -> None:
    for run_yaml in sorted((project_dir / "runs").glob("*/run.yaml")):
        run_dir = run_yaml.parent
        run_id = run_dir.name
        conn.execute("INSERT INTO runs(run_id, path) VALUES (?, ?)", (run_id, _project_path(project_dir, run_dir)))
        for node_dir in sorted((run_dir / "artifacts").glob("*")):
            if not node_dir.is_dir():
                continue
            start = read_yaml(node_dir / "node.start.yaml")
            done = read_yaml(node_dir / "node.done.yaml")
            failed = read_yaml(node_dir / "failed.yaml")
            manifest = read_yaml(node_dir / "artifact-manifest.yaml")
            status = classify_node(node_dir)
            node_id = node_dir.name
            created_at = start.get("created_at")
            finished_at = done.get("created_at") or failed.get("created_at")
            conn.execute(
                """
                INSERT INTO nodes(
                  node_id, run_id, step, hypothesis_id, mode, profile_id, status,
                  created_at, finished_at, run_seconds, path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    run_id,
                    start.get("step"),
                    start.get("hypothesis_id") or done.get("hypothesis_id") or failed.get("hypothesis_id"),
                    start.get("mode") or done.get("mode"),
                    start.get("profile_id") or done.get("profile_id"),
                    status,
                    created_at,
                    finished_at,
                    _elapsed_seconds(created_at, finished_at),
                    _project_path(project_dir, node_dir),
                ),
            )
            if manifest or done or failed:
                conn.execute(
                    """
                    INSERT INTO evaluations(node_id, hypothesis_id, mode, profile_id, code_hash, metric, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        manifest.get("hypothesis_id") or start.get("hypothesis_id"),
                        manifest.get("mode") or start.get("mode"),
                        manifest.get("profile_id") or start.get("profile_id"),
                        manifest.get("code_hash"),
                        manifest.get("metric"),
                        status,
                    ),
                )
            for artifact in sorted(node_dir.glob("**/*")):
                if artifact.is_file():
                    conn.execute(
                        "INSERT OR IGNORE INTO artifacts(node_id, path, kind) VALUES (?, ?, ?)",
                        (node_id, str(artifact.relative_to(node_dir)), artifact.suffix.lstrip(".") or "file"),
                    )
            upsert_submission(
                conn,
                build_submission_row(
                    project_dir=project_dir,
                    node_dir=node_dir,
                    start=start,
                    manifest=manifest,
                    status=status,
                    metric=manifest.get("metric"),
                    code_hash=manifest.get("code_hash"),
                    finished_at=finished_at,
                    run_seconds=_elapsed_seconds(created_at, finished_at),
                ),
            )


def classify_node(node_dir: Path) -> str:
    if (node_dir / "node.done.yaml").exists():
        return "complete"
    if (node_dir / "failed.yaml").exists():
        return "failed"
    if (node_dir / "artifact-manifest.yaml").exists():
        return "ready_to_finalize"
    if not (node_dir / "01-hypothesis.yaml").exists():
        return "missing_hypothesis"
    if not (node_dir / "02-code.py").exists():
        return "missing_code"
    if (node_dir / "started.yaml").exists() or (node_dir / "stdout.log").exists() or (node_dir / "stderr.log").exists():
        return "execution_interrupted"
    if list((node_dir / "03-execute").glob("attempt-*")):
        return "execution_interrupted"
    return "aborted"


def _project_path(project_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(project_dir).as_posix()
    except ValueError:
        try:
            return path.relative_to(repo_root_for_project(project_dir)).as_posix()
        except ValueError:
            return path.as_posix()


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
