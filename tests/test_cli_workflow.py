from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tml.cli.main import app


runner = CliRunner()


def invoke(root: Path, *args: str):
    return runner.invoke(app, list(args), env={"TML_CWD": str(root)})


def test_init_project_use_and_status_create_context_first_project(tmp_path: Path):
    result = invoke(tmp_path, "init", "project", "demo_project", "kind=local")
    assert result.exit_code == 0, result.output

    project_dir = tmp_path / "projects" / "local" / "demo_project"
    assert (tmp_path / ".gitignore").exists()
    assert (project_dir / "project.yaml").exists()
    assert (project_dir / "task.md").exists()
    assert (project_dir / "profiles" / "root" / "autogluon-root-start-v1.yaml").exists()
    assert not (project_dir / "code").exists()

    result = invoke(tmp_path, "project", "use", "demo_project")
    assert result.exit_code == 0, result.output
    assert "demo_project" in (tmp_path / "tml.yaml").read_text(encoding="utf-8")

    result = invoke(tmp_path, "root", "status")
    assert result.exit_code == 0, result.output
    assert "Active project: demo_project" in result.output
    assert "Hypotheses: 0" in result.output


def test_root_mock_workflow_reindexes_and_records_node_artifacts(tmp_path: Path):
    assert invoke(tmp_path, "init", "project", "demo_project", "kind=local").exit_code == 0
    assert invoke(tmp_path, "project", "use", "demo_project").exit_code == 0

    result = invoke(tmp_path, "root", "generate", "count=1")
    assert result.exit_code == 0, result.output
    hypothesis = (
        tmp_path
        / "projects"
        / "local"
        / "demo_project"
        / "hypotheses"
        / "000001"
        / "hypothesis.yaml"
    )
    assert "hypothesis_id: '000001'" in hypothesis.read_text(encoding="utf-8")

    result = invoke(tmp_path, "root", "materialize", "mode=legacy")
    assert result.exit_code == 0, result.output
    materialization = hypothesis.parent / "materializations" / "legacy-001.py"
    assert "TML_RESULT_JSON" in materialization.read_text(encoding="utf-8")

    result = invoke(tmp_path, "root", "run", "mode=legacy")
    assert result.exit_code == 0, result.output
    project_dir = tmp_path / "projects" / "local" / "demo_project"
    node_dirs = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs) == 1
    node_dir = node_dirs[0]
    assert (node_dir / "node.start.yaml").exists()
    assert (node_dir / "01-hypothesis.yaml").exists()
    assert (node_dir / "02-code.py").exists()
    assert (node_dir / "artifact-manifest.yaml").exists()
    assert (node_dir / "node.done.yaml").exists()

    result = invoke(tmp_path, "reindex")
    assert result.exit_code == 0, result.output
    db_path = project_dir / "tml.db"
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("select count(*) from hypotheses").fetchone()[0] == 1
        assert conn.execute("select count(*) from materializations").fetchone()[0] == 1
        assert conn.execute("select status from nodes").fetchone()[0] == "complete"

    result = invoke(tmp_path, "root", "status")
    assert result.exit_code == 0, result.output
    assert "Hypotheses: 1" in result.output
    assert "Evaluated: 1" in result.output


def test_prompt_render_probe_and_diff_do_not_create_nodes(tmp_path: Path):
    assert invoke(tmp_path, "init", "project", "demo_project", "kind=local").exit_code == 0
    assert invoke(tmp_path, "project", "use", "demo_project").exit_code == 0
    assert invoke(tmp_path, "root", "generate", "count=1").exit_code == 0
    assert invoke(tmp_path, "root", "materialize", "mode=autogluon").exit_code == 0

    render = invoke(tmp_path, "prompt", "render", "tmp=true")
    assert render.exit_code == 0, render.output
    rendered_path = Path(render.output.strip().splitlines()[-1])
    assert rendered_path.exists()
    assert "ROOT hypothesis" in rendered_path.read_text(encoding="utf-8")

    code_render = invoke(tmp_path, "prompt", "render", "1", "code", "tmp=true")
    assert code_render.exit_code == 0, code_render.output
    code_rendered_path = Path(code_render.output.strip().splitlines()[-1])
    assert "preprocess(df)" in code_rendered_path.read_text(encoding="utf-8")

    probe = invoke(tmp_path, "prompt", "probe", "tmp=true")
    assert probe.exit_code == 0, probe.output
    probe_path = Path(probe.output.strip().splitlines()[-1])
    assert (probe_path / "request.md").exists()
    assert (probe_path / "response.json").exists()

    assert not list((tmp_path / "projects" / "local" / "demo_project" / "runs").glob("*"))

    assert invoke(tmp_path, "root", "run").exit_code == 0
    diff = invoke(tmp_path, "prompt", "diff", "1", "code")
    assert diff.exit_code == 0, diff.output
    assert "saved request differs" in diff.output or "saved request matches" in diff.output


def test_root_ensure_is_idempotent_for_failed_autogluon_attempt(tmp_path: Path):
    assert invoke(tmp_path, "init", "project", "demo_project", "kind=local").exit_code == 0
    assert invoke(tmp_path, "project", "use", "demo_project").exit_code == 0

    first = invoke(tmp_path, "root", "ensure", "count=1")
    assert first.exit_code == 0, first.output
    project_dir = tmp_path / "projects" / "local" / "demo_project"
    node_dirs = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs) == 1
    assert (node_dirs[0] / "failed.yaml").exists()

    second = invoke(tmp_path, "root", "ensure", "count=1")
    assert second.exit_code == 0, second.output
    node_dirs_after_second = list((project_dir / "runs").glob("*/artifacts/*"))
    assert len(node_dirs_after_second) == 1
