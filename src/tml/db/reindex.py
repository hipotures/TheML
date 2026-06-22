from __future__ import annotations

from pathlib import Path

from tml.core.config import load_project_config
from tml.utils.hashing import sha256_file
from tml.utils.yaml_io import read_yaml

from .connect import connect
from .migrate import migrate


def reindex_project(project_dir: Path, db_path: Path) -> dict[str, int]:
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
        }
    return counts


def _index_profiles(conn, project_dir: Path) -> None:
    for path in sorted((project_dir / "profiles").glob("**/*.yaml")):
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
        conn.execute(
            "INSERT INTO hypotheses(hypothesis_id, title, enabled, path) VALUES (?, ?, ?, ?)",
            (hid, payload.get("title"), 1 if payload.get("enabled", True) else 0, _project_path(project_dir, path)),
        )
        for code in sorted((path.parent / "materializations").glob("*.py")):
            mode = code.name.split("-", 1)[0]
            conn.execute(
                "INSERT INTO materializations(hypothesis_id, mode, file, code_hash) VALUES (?, ?, ?, ?)",
                (hid, mode, code.name, sha256_file(code)),
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
            conn.execute(
                """
                INSERT INTO nodes(node_id, run_id, step, hypothesis_id, mode, profile_id, status, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    run_id,
                    start.get("step"),
                    start.get("hypothesis_id") or done.get("hypothesis_id") or failed.get("hypothesis_id"),
                    start.get("mode") or done.get("mode"),
                    start.get("profile_id") or done.get("profile_id"),
                    status,
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
    return path.relative_to(project_dir).as_posix()
